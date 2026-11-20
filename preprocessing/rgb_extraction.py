import cv2
import numpy as np

from numbers import Integral
from typing import Any, Dict, List, Optional, Tuple

from preprocessing.roi import ROIExtractor


class RGBExtractor:
    """
    PURPOSE:
        Extract average RGB measurements from facial skin regions
        for the FALSEFACET rPPG preprocessing pipeline.

    MAIN RESPONSIBILITIES:
        1. Extract forehead and cheek ROIs.
        2. Apply ROI masks to select usable pixels.
        3. Calculate average red, green, and blue values.
        4. Combine available ROI measurements.
        5. Mark unusable frames with NaN values.
        6. Reject frames without actual valid face tracking.

    SUPPORTED INPUTS:

        A. Already-cropped face images:

            Cropped face
                -> ROI extraction
                -> RGB calculation
                -> Combined RGB measurement

        B. Original video frames:

            Original frame
                -> Validate detection and tracking
                -> ROI extraction
                -> RGB calculation
                -> Combined RGB measurement

    OUTPUT:

        NumPy array with shape (number_of_frames, 3).

        Each row contains:

            [red_mean, green_mean, blue_mean]

        Invalid measurements contain:

            [NaN, NaN, NaN]

    IMPORTANT:
        OpenCV images use BGR channel order.
        This class returns RGB channel order.

        Predicted tracking boxes must never be treated
        as valid RGB measurements.
    """

    def __init__(
        self,
        min_roi_width: int = 5,
        min_roi_height: int = 5,
        min_skin_pixels: int = 1,
        debug: bool = False,
    ):
        """
        Initialize RGB extraction settings and ROIExtractor.
        """

        # --------------------------------------------------
        # 1. VALIDATE CONFIGURATION
        # --------------------------------------------------

        for name, value in (
            ("min_roi_width", min_roi_width),
            ("min_roi_height", min_roi_height),
            ("min_skin_pixels", min_skin_pixels),
        ):

            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if not isinstance(debug, bool):
            raise TypeError(
                "debug must be a boolean."
            )

        self.min_roi_width = int(min_roi_width)
        self.min_roi_height = int(min_roi_height)
        self.min_skin_pixels = int(min_skin_pixels)
        self.debug = debug

        # --------------------------------------------------
        # 2. INITIALIZE ROI EXTRACTOR
        # --------------------------------------------------

        self.roi_extractor = ROIExtractor(
            debug=debug
        )

    # ======================================================
    # VALIDATION HELPERS
    # ======================================================

    @staticmethod
    def _invalid_rgb() -> np.ndarray:
        """
        Return an invalid RGB measurement.

        NaN indicates that no usable RGB measurement
        could be extracted from the frame.
        """

        return np.array(
            [np.nan, np.nan, np.nan],
            dtype=np.float32,
        )

    @staticmethod
    def _valid_image(image: Any) -> bool:
        """
        Validate an OpenCV BGR image.

        Expected:
            - NumPy array
            - Three color channels
            - uint8 data type
            - Nonempty image
        """

        if not isinstance(image, np.ndarray):
            return False

        if image.size == 0:
            return False

        if image.ndim != 3:
            return False

        if image.shape[2] != 3:
            return False

        if image.dtype != np.uint8:
            return False

        return True

    @staticmethod
    def _valid_bbox(bbox: Any) -> bool:
        """
        Validate a bounding box.

        Expected format:

            [x, y, width, height]
        """

        if not isinstance(
            bbox,
            (list, tuple, np.ndarray),
        ):
            return False

        if len(bbox) != 4:
            return False

        try:
            values = np.asarray(
                bbox,
                dtype=np.float64,
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return False

        if not np.all(np.isfinite(values)):
            return False

        _, _, width, height = values

        return bool(
            width > 0 and height > 0
        )

    @staticmethod
    def _valid_rgb_values(
        r_mean: float,
        g_mean: float,
        b_mean: float,
    ) -> bool:
        """
        Check that all RGB measurements are finite.
        """

        values = np.asarray(
            [r_mean, g_mean, b_mean],
            dtype=np.float64,
        )

        return bool(
            np.all(np.isfinite(values))
        )

    @staticmethod
    def _is_true(value: Any) -> bool:
        """
        Safely interpret Python and NumPy Boolean values.
        """

        if isinstance(
            value,
            (bool, np.bool_),
        ):
            return bool(value)

        return False

    # ======================================================
    # ROI RGB EXTRACTION
    # ======================================================

    def _calculate_roi_rgb(
        self,
        roi_name: str,
        roi_data: Any,
        roi_stats: Dict[str, int],
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[str],
    ]:
        """
        Calculate average RGB values from one facial ROI.

        ROIExtractor is expected to return:

            {
                "crop": facial region image,
                "mask": selected pixels
            }

        OpenCV stores channels in BGR order.

        This method returns RGB measurements.

        Returns:

            (r_mean, g_mean, b_mean, failure_reason)
        """

        roi_stats["total_checked"] += 1

        # --------------------------------------------------
        # 1. VALIDATE ROI STRUCTURE
        # --------------------------------------------------

        if not isinstance(roi_data, dict):

            roi_stats["invalid_data"] += 1

            return (
                None,
                None,
                None,
                f"{roi_name}: ROI is not a dictionary",
            )

        crop = roi_data.get("crop")
        mask = roi_data.get("mask")

        if (
            not isinstance(crop, np.ndarray)
            or not isinstance(mask, np.ndarray)
            or crop.size == 0
            or mask.size == 0
        ):

            roi_stats["invalid_data"] += 1

            return (
                None,
                None,
                None,
                f"{roi_name}: invalid crop or mask",
            )

        # --------------------------------------------------
        # 2. VALIDATE DIMENSIONS AND DATA TYPES
        # --------------------------------------------------

        if (
            crop.ndim != 3
            or crop.shape[2] != 3
            or crop.dtype != np.uint8
            or mask.ndim != 2
            or mask.shape != crop.shape[:2]
        ):

            roi_stats["invalid_dimensions"] += 1

            return (
                None,
                None,
                None,
                f"{roi_name}: incompatible crop/mask dimensions",
            )

        crop_height, crop_width = crop.shape[:2]

        # --------------------------------------------------
        # 3. REJECT SMALL ROIs
        # --------------------------------------------------

        if (
            crop_width < self.min_roi_width
            or crop_height < self.min_roi_height
        ):

            roi_stats["small_crop"] += 1

            return (
                None,
                None,
                None,
                f"{roi_name}: crop too small "
                f"({crop_width}x{crop_height})",
            )

        # --------------------------------------------------
        # 4. VALIDATE SELECTED PIXELS
        # --------------------------------------------------

        selected_pixels = mask > 0

        selected_count = int(
            np.count_nonzero(selected_pixels)
        )

        if selected_count == 0:

            roi_stats["empty_mask"] += 1

            return (
                None,
                None,
                None,
                f"{roi_name}: mask contains no selected pixels",
            )

        if selected_count < self.min_skin_pixels:

            roi_stats["insufficient_skin_pixels"] += 1

            return (
                None,
                None,
                None,
                f"{roi_name}: insufficient selected pixels "
                f"({selected_count})",
            )

        # --------------------------------------------------
        # 5. CALCULATE RGB MEANS
        # --------------------------------------------------

        try:
            # OpenCV channel order:
            # Channel 0 = Blue
            # Channel 1 = Green
            # Channel 2 = Red

            blue_channel = crop[:, :, 0]
            green_channel = crop[:, :, 1]
            red_channel = crop[:, :, 2]

            r_mean = float(
                np.mean(
                    red_channel[selected_pixels]
                )
            )

            g_mean = float(
                np.mean(
                    green_channel[selected_pixels]
                )
            )

            b_mean = float(
                np.mean(
                    blue_channel[selected_pixels]
                )
            )

        except (
            TypeError,
            ValueError,
            IndexError,
            FloatingPointError,
        ):

            roi_stats["invalid_data"] += 1

            return (
                None,
                None,
                None,
                f"{roi_name}: failed to calculate RGB means",
            )

        # --------------------------------------------------
        # 6. VALIDATE RGB VALUES
        # --------------------------------------------------

        if not self._valid_rgb_values(
            r_mean,
            g_mean,
            b_mean,
        ):

            roi_stats["invalid_data"] += 1

            return (
                None,
                None,
                None,
                f"{roi_name}: RGB values are not finite",
            )

        roi_stats["usable"] += 1

        if self.debug:

            print(
                f"[ROI OK] {roi_name}: "
                f"size={crop_width}x{crop_height}, "
                f"pixels={selected_count}, "
                f"RGB=({r_mean:.2f}, "
                f"{g_mean:.2f}, {b_mean:.2f})"
            )

        return (
            r_mean,
            g_mean,
            b_mean,
            None,
        )

    # ======================================================
    # COMMON RGB EXTRACTION
    # ======================================================

    def _extract_rgb_from_image(
        self,
        image: np.ndarray,
        bbox: List[int],
    ) -> np.ndarray:
        """
        Extract RGB measurements from facial ROIs.

        This is the common extraction method used by
        cropped face images and original video frames.

        Args:
            image:
                OpenCV BGR image.

            bbox:
                Face bounding box [x, y, width, height].

        Returns:
            Float32 array with shape (3,).
        """

        invalid_rgb = self._invalid_rgb()

        # --------------------------------------------------
        # 1. VALIDATE IMAGE
        # --------------------------------------------------

        if not self._valid_image(image):

            if self.debug:
                print("[RGB ERROR] Invalid image.")

            return invalid_rgb

        # --------------------------------------------------
        # 2. VALIDATE BOUNDING BOX
        # --------------------------------------------------

        if not self._valid_bbox(bbox):

            if self.debug:
                print("[RGB ERROR] Invalid bounding box.")

            return invalid_rgb

        # --------------------------------------------------
        # 3. EXTRACT FACIAL ROIs
        # --------------------------------------------------

        try:

            rois = self.roi_extractor.extract_rois(
                image,
                bbox,
            )

        except Exception as exc:

            if self.debug:

                print(
                    f"[ROI EXCEPTION] "
                    f"{type(exc).__name__}: {exc}"
                )

            return invalid_rgb

        if not isinstance(rois, dict) or not rois:

            if self.debug:
                print("[RGB ERROR] No valid ROI structure.")

            return invalid_rgb

        # --------------------------------------------------
        # 4. INITIALIZE ROI STATISTICS
        # --------------------------------------------------

        roi_stats = {
            "total_checked": 0,
            "invalid_data": 0,
            "invalid_dimensions": 0,
            "empty_mask": 0,
            "small_crop": 0,
            "insufficient_skin_pixels": 0,
            "usable": 0,
        }

        r_values = []
        g_values = []
        b_values = []

        # --------------------------------------------------
        # 5. PROCESS EACH ROI
        # --------------------------------------------------

        for roi_name, roi_data in rois.items():

            (
                r_mean,
                g_mean,
                b_mean,
                failure_reason,
            ) = self._calculate_roi_rgb(
                roi_name,
                roi_data,
                roi_stats,
            )

            if failure_reason is not None:

                if self.debug:
                    print(
                        f"[ROI FAILURE] {failure_reason}"
                    )

                continue

            r_values.append(r_mean)
            g_values.append(g_mean)
            b_values.append(b_mean)

        # --------------------------------------------------
        # 6. CHECK USABLE ROI COUNT
        # --------------------------------------------------

        if not r_values:

            if self.debug:
                print(
                    "[RGB WARNING] No usable facial ROIs."
                )

            return invalid_rgb

        # --------------------------------------------------
        # 7. COMBINE ROI MEASUREMENTS
        # --------------------------------------------------

        # Each usable ROI contributes equally.
        rgb = np.array(
            [
                np.mean(r_values),
                np.mean(g_values),
                np.mean(b_values),
            ],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(rgb)):
            return invalid_rgb

        return rgb

    # ======================================================
    # CROPPED FACE IMAGE PROCESSING
    # ======================================================

    def process_cropped_face(
        self,
        face_image: np.ndarray,
    ) -> np.ndarray:
        """
        Extract RGB measurements from an already-cropped face.

        PURPOSE:
            Process saved face crops from face_extraction.py.

        No FaceDetector or FaceTracker is required here.

        Args:
            face_image:
                Already-cropped OpenCV BGR face image.

        Returns:
            Float32 array with shape (3,):

                [red_mean, green_mean, blue_mean]

            Returns NaN values when no usable RGB measurement
            can be extracted.
        """

        if not self._valid_image(face_image):

            if self.debug:
                print(
                    "[RGB ERROR] Invalid cropped face image."
                )

            return self._invalid_rgb()

        height, width = face_image.shape[:2]

        # The entire image is already the face crop.
        full_face_bbox = [
            0,
            0,
            width,
            height,
        ]

        return self._extract_rgb_from_image(
            face_image,
            full_face_bbox,
        )

    # ======================================================
    # ORIGINAL VIDEO FRAME PROCESSING
    # ======================================================

    def process_frame(
        self,
        frame: np.ndarray,
        tracking_record: Dict[str, Any],
    ) -> np.ndarray:
        """
        Extract RGB measurements from an original video frame.

        PURPOSE:
            Support the original video-based preprocessing path.

        Requires:

            detected=True
            track_valid=True

        A predicted or missing tracking box is rejected.

        Args:
            frame:
                OpenCV BGR frame.

            tracking_record:
                Face detection and tracking information.

        Returns:
            Float32 array with shape (3,).
        """

        if not self._valid_image(frame):
            return self._invalid_rgb()

        if not isinstance(tracking_record, dict):
            return self._invalid_rgb()

        # --------------------------------------------------
        # 1. CHECK ACTUAL FACE DETECTION
        # --------------------------------------------------

        detected = self._is_true(
            tracking_record.get(
                "detected",
                False,
            )
        )

        # --------------------------------------------------
        # 2. CHECK TRACKING VALIDITY
        # --------------------------------------------------

        track_valid = self._is_true(
            tracking_record.get(
                "track_valid",
                detected,
            )
        )

        # Both conditions are required.
        if not detected or not track_valid:
            return self._invalid_rgb()

        # --------------------------------------------------
        # 3. VALIDATE TRACKING BOUNDING BOX
        # --------------------------------------------------

        bbox = tracking_record.get("bbox")

        if not self._valid_bbox(bbox):
            return self._invalid_rgb()

        try:

            x, y, width, height = (
                int(value) for value in bbox
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):

            return self._invalid_rgb()

        return self._extract_rgb_from_image(
            frame,
            [x, y, width, height],
        )

    # ======================================================
    # VIDEO PROCESSING
    # ======================================================

    def process_video_frames(
        self,
        video_path: str,
        tracking_results: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Extract one RGB measurement per tracking record.

        PURPOSE:
            Process original video frames using their
            corresponding face tracking results.

        IMPORTANT:
            Output length always matches the number of
            tracking records, even if the video ends early.

        Invalid or unavailable frames receive NaN values.
        """

        if not isinstance(tracking_results, list):

            raise TypeError(
                "tracking_results must be a list."
            )

        if hasattr(
            self.roi_extractor,
            "reset_stats",
        ):
            self.roi_extractor.reset_stats()

        # --------------------------------------------------
        # 1. OPEN VIDEO
        # --------------------------------------------------

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():

            cap.release()

            raise ValueError(
                f"Could not open video: {video_path}"
            )

        cap.set(
            cv2.CAP_PROP_ORIENTATION_AUTO,
            1,
        )

        rgb_signals = []

        stats = {
            "valid": 0,
            "face_not_detected": 0,
            "tracking_rejected": 0,
            "invalid_tracking_record": 0,
            "unavailable_video_frame": 0,
        }

        # --------------------------------------------------
        # 2. PROCESS VIDEO FRAMES
        # --------------------------------------------------

        try:

            for frame_idx, tracking_record in enumerate(
                tracking_results
            ):

                ret, frame = cap.read()

                # ------------------------------------------
                # VIDEO ENDED BEFORE TRACKING RECORDS
                # ------------------------------------------

                if not ret:

                    if self.debug:

                        print(
                            f"[WARNING] Video ended at frame "
                            f"{frame_idx}."
                        )

                    # Preserve sequence alignment.
                    remaining = (
                        len(tracking_results)
                        - frame_idx
                    )

                    rgb_signals.extend(
                        [
                            self._invalid_rgb()
                            for _ in range(remaining)
                        ]
                    )

                    stats[
                        "unavailable_video_frame"
                    ] += remaining

                    break

                # ------------------------------------------
                # INVALID TRACKING RECORD
                # ------------------------------------------

                if not isinstance(
                    tracking_record,
                    dict,
                ):

                    rgb_signals.append(
                        self._invalid_rgb()
                    )

                    stats[
                        "invalid_tracking_record"
                    ] += 1

                    continue

                # ------------------------------------------
                # CHECK DETECTION
                # ------------------------------------------

                detected = self._is_true(
                    tracking_record.get(
                        "detected",
                        False,
                    )
                )

                if not detected:

                    rgb_signals.append(
                        self._invalid_rgb()
                    )

                    stats[
                        "face_not_detected"
                    ] += 1

                    continue

                # ------------------------------------------
                # CHECK TRACKING VALIDITY
                # ------------------------------------------

                track_valid = self._is_true(
                    tracking_record.get(
                        "track_valid",
                        detected,
                    )
                )

                if not track_valid:

                    rgb_signals.append(
                        self._invalid_rgb()
                    )

                    stats[
                        "tracking_rejected"
                    ] += 1

                    continue

                # ------------------------------------------
                # EXTRACT RGB
                # ------------------------------------------

                rgb = self.process_frame(
                    frame,
                    tracking_record,
                )

                rgb_signals.append(rgb)

                if np.all(np.isfinite(rgb)):
                    stats["valid"] += 1

        finally:
            cap.release()

        # --------------------------------------------------
        # 3. CONVERT TO NUMPY ARRAY
        # --------------------------------------------------

        if not rgb_signals:

            result = np.empty(
                (0, 3),
                dtype=np.float32,
            )

        else:

            result = np.asarray(
                rgb_signals,
                dtype=np.float32,
            )

        # --------------------------------------------------
        # 4. PRINT SUMMARY
        # --------------------------------------------------

        total_frames = len(result)

        print(
            "\n========== RGB EXTRACTION SUMMARY =========="
        )

        print(f"Video: {video_path}")

        print(
            f"Tracking records: {len(tracking_results)}"
        )

        print(
            f"RGB records produced: {total_frames}"
        )

        print(
            f"Valid RGB measurements: {stats['valid']}"
        )

        print(
            f"Missing RGB measurements: "
            f"{total_frames - stats['valid']}"
        )

        if total_frames:

            print(
                f"Valid percentage: "
                f"{stats['valid'] / total_frames:.2%}"
            )

        print("\nFRAME DIAGNOSTICS")

        for name, count in stats.items():

            if name != "valid":
                print(f"{name}: {count}")

        if hasattr(
            self.roi_extractor,
            "print_stats",
        ):
            self.roi_extractor.print_stats()

        print(
            "============================================"
        )

        return result

    # ======================================================
    # RESOURCE CLEANUP
    # ======================================================

    def close(self):
        """
        Release resources held by ROIExtractor.
        """

        if hasattr(
            self.roi_extractor,
            "close",
        ):
            self.roi_extractor.close()

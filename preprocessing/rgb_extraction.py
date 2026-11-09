import cv2
import numpy as np

from typing import List, Dict, Any, Optional, Tuple

from preprocessing.roi import ROIExtractor


class RGBExtractor:
    """
    Extract average RGB measurements from facial regions.

    Supports two input types:

    1. Already-cropped face images:
        Cropped face image
            -> Extract forehead and cheek ROIs
            -> Calculate RGB means
            -> Combine usable ROIs

    2. Original video frames:
        Video frame
            -> Validate face detection/tracking
            -> Extract forehead and cheek ROIs
            -> Calculate RGB means
            -> Combine usable ROIs

    Output:
        NumPy array with shape (number_of_frames, 3).

        Each row contains:
            [red_mean, green_mean, blue_mean]

        Frames without usable RGB measurements contain NaN values.
    """

    def __init__(
        self,
        min_roi_width: int = 5,
        min_roi_height: int = 5,
        min_skin_pixels: int = 1,
        debug: bool = False,
    ):
        # -----------------------------------------------------
        # 1. Validate configuration
        # -----------------------------------------------------

        if (
            isinstance(min_roi_width, bool)
            or not isinstance(min_roi_width, int)
            or min_roi_width < 1
        ):
            raise ValueError(
                "min_roi_width must be a positive integer."
            )

        if (
            isinstance(min_roi_height, bool)
            or not isinstance(min_roi_height, int)
            or min_roi_height < 1
        ):
            raise ValueError(
                "min_roi_height must be a positive integer."
            )

        if (
            isinstance(min_skin_pixels, bool)
            or not isinstance(min_skin_pixels, int)
            or min_skin_pixels < 1
        ):
            raise ValueError(
                "min_skin_pixels must be a positive integer."
            )

        self.min_roi_width = min_roi_width
        self.min_roi_height = min_roi_height
        self.min_skin_pixels = min_skin_pixels
        self.debug = debug

        # Initialize the ROI extractor once.
        self.roi_extractor = ROIExtractor(
            debug=debug
        )

    # =========================================================
    # 2. VALIDATION HELPERS
    # =========================================================

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
            NumPy array
            Three color channels
            uint8 data type
            Nonempty image
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
        Check that all RGB values are finite.
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
        Safely interpret Python and NumPy booleans.
        """

        if isinstance(
            value,
            (bool, np.bool_),
        ):
            return bool(value)

        return False

    # =========================================================
    # 3. ROI RGB EXTRACTION
    # =========================================================

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

        ROIExtractor returns:
            {
                "crop": facial region image,
                "mask": selected pixels
            }

        OpenCV stores image channels as BGR.

        This method converts the channel order when calculating
        the measurements.

        Returns:
            (r_mean, g_mean, b_mean, failure_reason)
        """

        roi_stats["total_checked"] += 1

        # -----------------------------------------------------
        # Validate ROI structure
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Validate dimensions
        # -----------------------------------------------------

        if (
            crop.ndim != 3
            or crop.shape[2] != 3
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

        # -----------------------------------------------------
        # Reject small ROI crops
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Validate selected pixels
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Calculate RGB means
        # -----------------------------------------------------

        try:
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

    # =========================================================
    # 4. COMMON RGB EXTRACTION
    # =========================================================

    def _extract_rgb_from_image(
        self,
        image: np.ndarray,
        bbox: List[int],
    ) -> np.ndarray:
        """
        Extract RGB measurements from facial ROIs.

        This is the common extraction method used by both
        cropped face images and original video frames.

        Args:
            image:
                OpenCV BGR image.

            bbox:
                Face bounding box.

        Returns:
            Float32 array with shape (3,).
        """

        invalid_rgb = self._invalid_rgb()

        if not self._valid_image(image):
            if self.debug:
                print("[RGB ERROR] Invalid image.")

            return invalid_rgb

        if not self._valid_bbox(bbox):
            if self.debug:
                print("[RGB ERROR] Invalid bounding box.")

            return invalid_rgb

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

        # Calculate RGB measurements for every available ROI.
        for roi_name, roi_data in rois.items():

            r_mean, g_mean, b_mean, failure_reason = (
                self._calculate_roi_rgb(
                    roi_name,
                    roi_data,
                    roi_stats,
                )
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

        # No usable ROI means no usable RGB measurement.
        if not r_values:

            if self.debug:
                print(
                    "[RGB WARNING] No usable facial ROIs."
                )

            return invalid_rgb

        # Combine the available forehead and cheek measurements.
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

    # =========================================================
    # 5. NEW: CROPPED FACE IMAGE PROCESSING
    # =========================================================

    def process_cropped_face(
        self,
        face_image: np.ndarray,
    ) -> np.ndarray:
        """
        Extract RGB measurements from an already-cropped face.

        This is the main method for the new extracted_faces
        dataset.

        No FaceDetector or FaceTracker is required.

        Args:
            face_image:
                An already-cropped OpenCV BGR face image.

        Returns:
            Float32 array with shape (3,):

                [red_mean, green_mean, blue_mean]

            Returns NaN values if no usable RGB measurement
            can be extracted.
        """

        if not self._valid_image(face_image):
            if self.debug:
                print(
                    "[RGB ERROR] Invalid cropped face image."
                )

            return self._invalid_rgb()

        height, width = face_image.shape[:2]

        # The entire image is already the cropped face.
        # Therefore, use the full image as the bounding box.
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

    # =========================================================
    # 6. ORIGINAL VIDEO FRAME PROCESSING
    # =========================================================

    def process_frame(
        self,
        frame: np.ndarray,
        tracking_record: Dict[str, Any],
    ) -> np.ndarray:
        """
        Extract RGB measurements from an original video frame.

        This method is retained for backward compatibility.

        Args:
            frame:
                OpenCV BGR frame.

            tracking_record:
                Detection/tracking information.

        Returns:
            Float32 array with shape (3,).
        """

        if not self._valid_image(frame):
            return self._invalid_rgb()

        if not isinstance(tracking_record, dict):
            return self._invalid_rgb()

        detected = self._is_true(
            tracking_record.get("detected", False)
        )

        track_valid = self._is_true(
            tracking_record.get(
                "track_valid",
                detected,
            )
        )

        if not detected or not track_valid:
            return self._invalid_rgb()

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

    # =========================================================
    # 7. VIDEO PROCESSING
    # =========================================================

    def process_video_frames(
        self,
        video_path: str,
        tracking_results: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Extract one RGB measurement per video frame.

        Retained for compatibility with the old video-based
        preprocessing pipeline.

        The new cropped-face pipeline should instead call
        process_cropped_face() for each image.
        """

        if not isinstance(tracking_results, list):
            raise TypeError(
                "tracking_results must be a list."
            )

        if hasattr(self.roi_extractor, "reset_stats"):
            self.roi_extractor.reset_stats()

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
        }

        try:
            for frame_idx, tracking_record in enumerate(
                tracking_results
            ):

                ret, frame = cap.read()

                if not ret:
                    print(
                        f"[WARNING] Video ended at frame "
                        f"{frame_idx}."
                    )

                    break

                if not isinstance(tracking_record, dict):
                    rgb_signals.append(
                        self._invalid_rgb()
                    )

                    stats["invalid_tracking_record"] += 1

                    continue

                detected = self._is_true(
                    tracking_record.get(
                        "detected",
                        False,
                    )
                )

                track_valid = self._is_true(
                    tracking_record.get(
                        "track_valid",
                        detected,
                    )
                )

                if not detected:
                    rgb_signals.append(
                        self._invalid_rgb()
                    )

                    stats["face_not_detected"] += 1

                    continue

                if not track_valid:
                    rgb_signals.append(
                        self._invalid_rgb()
                    )

                    stats["tracking_rejected"] += 1

                    continue

                rgb = self.process_frame(
                    frame,
                    tracking_record,
                )

                rgb_signals.append(rgb)

                if np.all(np.isfinite(rgb)):
                    stats["valid"] += 1

        finally:
            cap.release()

        result = np.asarray(
            rgb_signals,
            dtype=np.float32,
        )

        total_frames = len(result)

        print("\n========== RGB EXTRACTION SUMMARY ==========")
        print(f"Video: {video_path}")
        print(f"RGB records produced: {total_frames}")
        print(f"Valid RGB measurements: {stats['valid']}")
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

        if hasattr(self.roi_extractor, "print_stats"):
            self.roi_extractor.print_stats()

        print("============================================")

        if result.size == 0:
            return np.empty(
                (0, 3),
                dtype=np.float32,
            )

        return result

    # =========================================================
    # 8. RESOURCE CLEANUP
    # =========================================================

    def close(self):
        """
        Release resources held by ROIExtractor.
        """

        if hasattr(self.roi_extractor, "close"):
            self.roi_extractor.close()
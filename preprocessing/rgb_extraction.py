import cv2
import numpy as np

from typing import List, Dict, Any

from preprocessing.roi import ROIExtractor


class RGBExtractor:
    """
    Extract average RGB measurements from facial regions.

    Input:
        Video path and face-tracking results.

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

        self.roi_extractor = ROIExtractor()

        self.min_roi_width = min_roi_width
        self.min_roi_height = min_roi_height
        self.min_skin_pixels = min_skin_pixels
        self.debug = debug

    @staticmethod
    def _valid_bbox(bbox: Any) -> bool:
        """
        Validate a bounding box in [x, y, width, height] format.
        """

        if not isinstance(
            bbox,
            (list, tuple, np.ndarray),
        ):
            return False

        if len(bbox) != 4:
            return False

        try:
            values = np.asarray(bbox, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return False

        if not np.all(np.isfinite(values)):
            return False

        _, _, width, height = values

        return width > 0 and height > 0

    @staticmethod
    def _valid_confidence_values(
        r_mean: float,
        g_mean: float,
        b_mean: float,
    ) -> bool:
        """
        Check that all calculated channel means are finite.
        """

        values = np.asarray(
            [r_mean, g_mean, b_mean],
            dtype=np.float64,
        )

        return bool(np.all(np.isfinite(values)))

    def process_video_frames(
        self,
        video_path: str,
        tracking_results: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Extract one RGB measurement for each corresponding video frame.

        Missing or unusable measurements are represented by NaN.
        """

        if not isinstance(tracking_results, list):
            raise TypeError(
                "tracking_results must be a list."
            )

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        rgb_signals = []

        # Frame-level counters.
        stats = {
            "valid": 0,
            "face_not_detected": 0,
            "tracking_rejected": 0,
            "invalid_bbox": 0,
            "roi_exception": 0,
            "invalid_roi_structure": 0,
            "no_valid_roi": 0,
            "no_rgb_values": 0,
            "frame_mismatch": 0,
        }

        # ROI-level counters.
        roi_stats = {
            "total_checked": 0,
            "invalid_data": 0,
            "invalid_dimensions": 0,
            "empty_mask": 0,
            "small_crop": 0,
            "insufficient_skin_pixels": 0,
            "usable": 0,
        }

        # Track why frames fail.
        failure_counts = {}
        failure_examples = []
        max_failure_examples = 10

        frames_decoded = 0
        tracking_records_processed = 0

        try:
            for frame_idx, info in enumerate(tracking_results):

                # Read the corresponding frame from the video.
                ret, frame = cap.read()

                if not ret:
                    print(
                        f"\nWarning: Video ended at frame {frame_idx}, "
                        f"but tracking results contain "
                        f"{len(tracking_results)} records."
                    )
                    break

                frames_decoded += 1
                tracking_records_processed += 1

                # A tracking record must be a dictionary.
                if not isinstance(info, dict):
                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["frame_mismatch"] += 1

                    if self.debug:
                        print(
                            f"[FRAME ERROR] Frame {frame_idx}: "
                            "tracking record is not a dictionary."
                        )

                    continue

                detected = info.get("detected", False)

                # Preserve compatibility with detection-only results:
                # if track_valid is absent, an explicit detection can
                # still be processed.
                track_valid = info.get(
                    "track_valid",
                    detected is True,
                )

                bbox = info.get("bbox")

                # -------------------------------------------------
                # CASE 1: No face was detected.
                # -------------------------------------------------
                if detected is not True:

                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["face_not_detected"] += 1

                    continue

                # -------------------------------------------------
                # CASE 2: The tracker rejected the face.
                # -------------------------------------------------
                if track_valid is not True:

                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["tracking_rejected"] += 1

                    if self.debug:
                        print(
                            f"[TRACKING REJECTED] Frame {frame_idx}: "
                            f"status={info.get('tracking_status')}, "
                            f"IoU={info.get('tracking_iou')}"
                        )

                    continue

                # -------------------------------------------------
                # CASE 3: Validate the bounding box.
                # -------------------------------------------------
                if not self._valid_bbox(bbox):

                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["invalid_bbox"] += 1

                    if self.debug:
                        print(
                            f"[INVALID BBOX] Frame {frame_idx}: "
                            f"{bbox}"
                        )

                    continue

                # Convert the validated box to integer coordinates.
                # Conversion errors are handled explicitly.
                try:
                    x, y, box_w, box_h = [
                        int(value)
                        for value in bbox
                    ]

                except (
                    TypeError,
                    ValueError,
                    OverflowError,
                ):
                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["invalid_bbox"] += 1

                    continue

                if box_w <= 0 or box_h <= 0:

                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["invalid_bbox"] += 1

                    continue

                # -------------------------------------------------
                # CASE 4: Extract the facial regions.
                # -------------------------------------------------
                try:
                    rois = self.roi_extractor.extract_rois(
                        frame,
                        [x, y, box_w, box_h],
                    )

                except Exception as exc:

                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["roi_exception"] += 1

                    if self.debug:
                        print(
                            f"[ROI EXCEPTION] Frame {frame_idx}: "
                            f"{type(exc).__name__}: {exc}"
                        )

                    continue

                if not isinstance(rois, dict) or not rois:

                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["invalid_roi_structure"] += 1

                    if self.debug:
                        print(
                            f"[INVALID ROI STRUCTURE] "
                            f"Frame {frame_idx}"
                        )

                    continue

                # -------------------------------------------------
                # CASE 5: Calculate RGB values for usable ROIs.
                # -------------------------------------------------
                r_vals = []
                g_vals = []
                b_vals = []

                frame_failure_reasons = []

                for roi_name, roi_data in rois.items():

                    roi_stats["total_checked"] += 1

                    # The ROI must contain a crop and a mask.
                    if not isinstance(roi_data, dict):

                        roi_stats["invalid_data"] += 1

                        frame_failure_reasons.append(
                            f"{roi_name}: invalid ROI data"
                        )

                        continue

                    crop = roi_data.get("crop")
                    mask = roi_data.get("mask")

                    # Check that both values are valid arrays.
                    if (
                        not isinstance(crop, np.ndarray)
                        or not isinstance(mask, np.ndarray)
                        or crop.size == 0
                        or mask.size == 0
                    ):

                        roi_stats["invalid_data"] += 1

                        frame_failure_reasons.append(
                            f"{roi_name}: invalid crop or mask"
                        )

                        continue

                    # The crop must have three color channels.
                    # The mask must be a 2D image matching the crop.
                    if (
                        crop.ndim != 3
                        or crop.shape[2] != 3
                        or mask.ndim != 2
                        or mask.shape != crop.shape[:2]
                    ):

                        roi_stats["invalid_dimensions"] += 1

                        frame_failure_reasons.append(
                            f"{roi_name}: incompatible dimensions"
                        )

                        continue

                    crop_h, crop_w = crop.shape[:2]

                    # Ignore extremely small crops.
                    if (
                        crop_w < self.min_roi_width
                        or crop_h < self.min_roi_height
                    ):

                        roi_stats["small_crop"] += 1

                        frame_failure_reasons.append(
                            f"{roi_name}: small crop "
                            f"{crop_w}x{crop_h}"
                        )

                        continue

                    # A positive mask value identifies a selected pixel.
                    skin_pixels = mask > 0

                    skin_pixel_count = int(
                        np.count_nonzero(skin_pixels)
                    )

                    # Empty masks contain no usable pixels.
                    if skin_pixel_count == 0:

                        roi_stats["empty_mask"] += 1

                        frame_failure_reasons.append(
                            f"{roi_name}: empty mask"
                        )

                        continue

                    # Avoid measurements based on too few pixels.
                    if (
                        skin_pixel_count < self.min_skin_pixels
                    ):

                        roi_stats[
                            "insufficient_skin_pixels"
                        ] += 1

                        frame_failure_reasons.append(
                            f"{roi_name}: only "
                            f"{skin_pixel_count} selected pixels"
                        )

                        continue

                    # OpenCV images use BGR channel ordering.
                    # Convert the extracted measurements to RGB order.
                    b_channel = crop[:, :, 0]
                    g_channel = crop[:, :, 1]
                    r_channel = crop[:, :, 2]

                    # Calculate the mean color of selected skin pixels.
                    r_mean = float(
                        np.mean(r_channel[skin_pixels])
                    )

                    g_mean = float(
                        np.mean(g_channel[skin_pixels])
                    )

                    b_mean = float(
                        np.mean(b_channel[skin_pixels])
                    )

                    # Reject NaN and infinite measurements.
                    if not self._valid_confidence_values(
                        r_mean,
                        g_mean,
                        b_mean,
                    ):

                        roi_stats["invalid_data"] += 1

                        frame_failure_reasons.append(
                            f"{roi_name}: non-finite RGB values"
                        )

                        continue

                    # This ROI produced a usable measurement.
                    r_vals.append(r_mean)
                    g_vals.append(g_mean)
                    b_vals.append(b_mean)

                    roi_stats["usable"] += 1

                    if self.debug:
                        print(
                            f"[ROI OK] Frame {frame_idx}, "
                            f"{roi_name}: "
                            f"crop={crop_w}x{crop_h}, "
                            f"skin_pixels={skin_pixel_count}, "
                            f"RGB=({r_mean:.2f}, "
                            f"{g_mean:.2f}, {b_mean:.2f})"
                        )

                # -------------------------------------------------
                # CASE 6: Combine the usable facial regions.
                # -------------------------------------------------
                if r_vals:

                    # Average the usable ROI measurements to produce
                    # one RGB measurement for this video frame.
                    rgb_signals.append([
                        float(np.mean(r_vals)),
                        float(np.mean(g_vals)),
                        float(np.mean(b_vals)),
                    ])

                    stats["valid"] += 1

                else:

                    # No ROI provided usable RGB values.
                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["no_valid_roi"] += 1

                    # Record the reasons for debugging.
                    for reason in frame_failure_reasons:

                        failure_counts[reason] = (
                            failure_counts.get(reason, 0) + 1
                        )

                    if (
                        len(failure_examples)
                        < max_failure_examples
                    ):
                        failure_examples.append(
                            (frame_idx, frame_failure_reasons)
                        )

        finally:
            cap.release()

        # -----------------------------------------------------
        # Final summary.
        # -----------------------------------------------------
        total = len(rgb_signals)
        valid = stats["valid"]
        missing = total - valid

        print("\n========== RGB EXTRACTION SUMMARY ==========")
        print(f"Video: {video_path}")
        print(f"Decoded frames: {frames_decoded}")
        print(
            f"Tracking records supplied: {len(tracking_results)}"
        )
        print(
            f"Tracking records processed: "
            f"{tracking_records_processed}"
        )
        print(f"RGB records produced: {total}")
        print(f"Valid RGB measurements: {valid}")
        print(f"Missing RGB measurements: {missing}")

        if total > 0:
            print(
                f"Valid percentage: {valid / total:.2%}"
            )
        else:
            print("Valid percentage: N/A")

        print("\nFrame-level failure categories:")
        print(
            f"Face not detected: "
            f"{stats['face_not_detected']}"
        )
        print(
            f"Tracking rejected: "
            f"{stats['tracking_rejected']}"
        )
        print(
            f"Invalid bounding box: "
            f"{stats['invalid_bbox']}"
        )
        print(
            f"ROI exceptions: "
            f"{stats['roi_exception']}"
        )
        print(
            f"Invalid ROI structure: "
            f"{stats['invalid_roi_structure']}"
        )
        print(
            f"No usable ROI: "
            f"{stats['no_valid_roi']}"
        )

        print("\nROI-level diagnostics:")
        print(
            f"ROIs checked: "
            f"{roi_stats['total_checked']}"
        )
        print(
            f"Invalid ROI data: "
            f"{roi_stats['invalid_data']}"
        )
        print(
            f"Incompatible dimensions: "
            f"{roi_stats['invalid_dimensions']}"
        )
        print(
            f"Empty masks: "
            f"{roi_stats['empty_mask']}"
        )
        print(
            f"Small crops: "
            f"{roi_stats['small_crop']}"
        )
        print(
            f"Insufficient skin pixels: "
            f"{roi_stats['insufficient_skin_pixels']}"
        )
        print(
            f"Usable ROIs: "
            f"{roi_stats['usable']}"
        )

        if failure_counts:
            print("\nMost frequent ROI failure reasons:")

            sorted_failures = sorted(
                failure_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            for reason, count in sorted_failures[:10]:
                print(f"{count} occurrence(s): {reason}")

        if failure_examples:
            print("\nExamples of frames without usable RGB:")

            for frame_idx, reasons in failure_examples:
                print(f"Frame {frame_idx}:")

                if reasons:
                    for reason in reasons:
                        print(f"  - {reason}")
                else:
                    print("  - No specific ROI failure recorded.")

        # Detect a mismatch between decoded frames and tracking records.
        if frames_decoded != len(tracking_results):
            print(
                "\nWARNING: The number of decoded video frames "
                "does not match the number of tracking records. "
                "Check that both stages process the same video "
                "and frame sequence."
            )

        print("============================================")

        if not rgb_signals:
            return np.empty((0, 3), dtype=np.float32)

        return np.asarray(
            rgb_signals,
            dtype=np.float32,
        )

    def close(self):
        """
        Release resources used by the ROI extractor.
        """

        if hasattr(self.roi_extractor, "close"):
            self.roi_extractor.close()
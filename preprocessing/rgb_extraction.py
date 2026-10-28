
import cv2
import numpy as np

from typing import List, Dict, Any

from preprocessing.roi import ROIExtractor


class RGBExtractor:
    """
    Extract average RGB measurements from facial regions.

    Pipeline:
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
        # Validate configuration.
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

    # ---------------------------------------------------------
    # VALIDATION HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _valid_bbox(bbox: Any) -> bool:
        """
        Check that a bounding box contains four finite values
        and has positive width and height.

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

        return bool(width > 0 and height > 0)

    @staticmethod
    def _valid_rgb_values(
        r_mean: float,
        g_mean: float,
        b_mean: float,
    ) -> bool:
        """
        Check that all RGB means are finite numbers.
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
        Safely interpret Python and NumPy boolean values.

        Non-boolean values are rejected instead of being
        interpreted as True merely because they are nonzero.
        """

        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        return False

    # ---------------------------------------------------------
    # ROI VALIDATION
    # ---------------------------------------------------------

    def _calculate_roi_rgb(
        self,
        roi_name: str,
        roi_data: Any,
        roi_stats: Dict[str, int],
    ):
        """
        Validate one ROI and calculate its RGB means.

        Returns:
            (r_mean, g_mean, b_mean, failure_reason)

        If the ROI is invalid, the RGB means are None and
        failure_reason explains why it was rejected.
        """

        roi_stats["total_checked"] += 1

        # The ROI must be represented by a dictionary.
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

        # The crop and mask must be nonempty NumPy arrays.
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

        # A color crop must have three channels.
        # Its mask must be a matching two-dimensional array.
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

        # Reject crops that are too small to be useful.
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

        # The mask identifies which pixels will be measured.
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

        # OpenCV stores color channels in BGR order.
        # Return the means in RGB order.
        try:
            blue_channel = crop[:, :, 0]
            green_channel = crop[:, :, 1]
            red_channel = crop[:, :, 2]

            r_mean = float(
                np.mean(red_channel[selected_pixels])
            )

            g_mean = float(
                np.mean(green_channel[selected_pixels])
            )

            b_mean = float(
                np.mean(blue_channel[selected_pixels])
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

        # Reject NaN and infinite measurements.
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
                f"crop={crop_width}x{crop_height}, "
                f"selected_pixels={selected_count}, "
                f"RGB=({r_mean:.2f}, "
                f"{g_mean:.2f}, {b_mean:.2f})"
            )

        return r_mean, g_mean, b_mean, None

    # ---------------------------------------------------------
    # VIDEO PROCESSING
    # ---------------------------------------------------------

    def process_video_frames(
        self,
        video_path: str,
        tracking_results: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Extract one RGB measurement per corresponding video frame.

        Args:
            video_path:
                Path to the video.

            tracking_results:
                One detection/tracking record per video frame.

        Returns:
            Float32 NumPy array of shape (N, 3).

            Frames without usable measurements contain NaN values.
        """

        if not isinstance(tracking_results, list):
            raise TypeError(
                "tracking_results must be a list."
            )

        # Reset ROI statistics so they describe this video only.
        if hasattr(self.roi_extractor, "reset_stats"):
            self.roi_extractor.reset_stats()

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()

            raise ValueError(
                f"Could not open video: {video_path}"
            )

        rgb_signals = []

        # -----------------------------------------------------
        # Frame-level statistics.
        # -----------------------------------------------------

        stats = {
            "valid": 0,
            "face_not_detected": 0,
            "tracking_rejected": 0,
            "invalid_bbox": 0,
            "roi_exception": 0,
            "invalid_roi_structure": 0,
            "no_valid_roi": 0,
            "invalid_tracking_record": 0,
        }

        # -----------------------------------------------------
        # ROI-level statistics.
        # -----------------------------------------------------

        roi_stats = {
            "total_checked": 0,
            "invalid_data": 0,
            "invalid_dimensions": 0,
            "empty_mask": 0,
            "small_crop": 0,
            "insufficient_skin_pixels": 0,
            "usable": 0,
        }

        failure_counts = {}
        failure_examples = []

        max_failure_examples = 10

        frames_decoded = 0
        tracking_records_processed = 0

        try:
            for frame_idx, info in enumerate(
                tracking_results
            ):
                # Read one video frame for each tracking record.
                ret, frame = cap.read()

                if not ret:
                    print(
                        f"\n[WARNING] Video ended at frame "
                        f"{frame_idx}, but tracking results "
                        f"contain {len(tracking_results)} records."
                    )

                    break

                frames_decoded += 1
                tracking_records_processed += 1

                # Each tracking result must be a dictionary.
                if not isinstance(info, dict):
                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["invalid_tracking_record"] += 1

                    if self.debug:
                        print(
                            f"[TRACKING ERROR] Frame {frame_idx}: "
                            "record is not a dictionary."
                        )

                    continue

                detected = self._is_true(
                    info.get("detected", False)
                )

                # Detection-only results may not contain track_valid.
                # In that case, use the detection status.
                track_valid = self._is_true(
                    info.get(
                        "track_valid",
                        detected,
                    )
                )

                # -------------------------------------------------
                # CASE 1: No face detected.
                # -------------------------------------------------

                if not detected:
                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["face_not_detected"] += 1

                    continue

                # -------------------------------------------------
                # CASE 2: Face tracking rejected the detection.
                # -------------------------------------------------

                if not track_valid:
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
                # CASE 3: Validate the face bounding box.
                # -------------------------------------------------

                bbox = info.get("bbox")

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

                try:
                    x, y, box_width, box_height = (
                        int(value) for value in bbox
                    )

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

                if box_width <= 0 or box_height <= 0:
                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["invalid_bbox"] += 1

                    continue

                # -------------------------------------------------
                # CASE 4: Extract forehead and cheek ROIs.
                # -------------------------------------------------

                try:
                    rois = self.roi_extractor.extract_rois(
                        frame,
                        [x, y, box_width, box_height],
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
                # CASE 5: Calculate RGB for each usable ROI.
                # -------------------------------------------------

                r_values = []
                g_values = []
                b_values = []

                frame_failure_reasons = []

                for roi_name, roi_data in rois.items():
                    r_mean, g_mean, b_mean, failure_reason = (
                        self._calculate_roi_rgb(
                            roi_name,
                            roi_data,
                            roi_stats,
                        )
                    )

                    if failure_reason is not None:
                        frame_failure_reasons.append(
                            failure_reason
                        )

                        if self.debug:
                            print(
                                f"[ROI FAILURE] Frame {frame_idx}: "
                                f"{failure_reason}"
                            )

                        continue

                    r_values.append(r_mean)
                    g_values.append(g_mean)
                    b_values.append(b_mean)

                # -------------------------------------------------
                # CASE 6: Combine successful ROI measurements.
                # -------------------------------------------------

                if r_values:
                    # A frame can still be usable if only one or two
                    # facial regions were successfully extracted.
                    rgb_signals.append([
                        float(np.mean(r_values)),
                        float(np.mean(g_values)),
                        float(np.mean(b_values)),
                    ])

                    stats["valid"] += 1

                else:
                    # No facial region provided a valid measurement.
                    rgb_signals.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["no_valid_roi"] += 1

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
        # FINAL RGB SUMMARY.
        # -----------------------------------------------------

        total_records = len(rgb_signals)
        valid_frames = stats["valid"]
        missing_frames = total_records - valid_frames

        print("\n========== RGB EXTRACTION SUMMARY ==========")
        print(f"Video: {video_path}")
        print(f"Decoded frames: {frames_decoded}")

        print(
            f"Tracking records supplied: "
            f"{len(tracking_results)}"
        )

        print(
            f"Tracking records processed: "
            f"{tracking_records_processed}"
        )

        print(f"RGB records produced: {total_records}")
        print(f"Valid RGB measurements: {valid_frames}")
        print(f"Missing RGB measurements: {missing_frames}")

        if total_records > 0:
            print(
                f"Valid percentage: "
                f"{valid_frames / total_records:.2%}"
            )
        else:
            print("Valid percentage: N/A")

        print("\nFRAME-LEVEL DIAGNOSTICS")

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
            f"Frames without usable ROIs: "
            f"{stats['no_valid_roi']}"
        )

        print(
            f"Invalid tracking records: "
            f"{stats['invalid_tracking_record']}"
        )

        print("\nROI-LEVEL DIAGNOSTICS")

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
            print("\nMOST FREQUENT FRAME-LEVEL ROI FAILURES")

            for reason, count in sorted(
                failure_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:10]:
                print(
                    f"{count} occurrence(s): {reason}"
                )

        if failure_examples:
            print("\nEXAMPLES OF FRAMES WITHOUT USABLE RGB")

            for frame_idx, reasons in failure_examples:
                print(f"Frame {frame_idx}:")

                if reasons:
                    for reason in reasons:
                        print(f"  - {reason}")
                else:
                    print("  - No specific failure reason recorded.")

        # Print the detailed statistics collected inside roi.py.
        if hasattr(self.roi_extractor, "print_stats"):
            self.roi_extractor.print_stats()

        # Check that video frames and tracking records correspond.
        if frames_decoded != len(tracking_results):
            print(
                "\n[WARNING] The number of decoded video frames "
                "does not match the number of tracking records. "
                "Confirm that detection/tracking and RGB extraction "
                "use the same video and frame sequence."
            )

        print("============================================")

        # Always return a two-dimensional array.
        if not rgb_signals:
            return np.empty(
                (0, 3),
                dtype=np.float32,
            )

        return np.asarray(
            rgb_signals,
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # RESOURCE CLEANUP
    # ---------------------------------------------------------

    def close(self):
        """
        Release resources held by ROIExtractor.
        """

        if hasattr(self.roi_extractor, "close"):
            self.roi_extractor.close()

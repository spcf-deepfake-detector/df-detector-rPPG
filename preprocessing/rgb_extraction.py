import cv2
import numpy as np
from typing import Dict, List, Any
from preprocessing.roi import ROIExtractor


class RGBExtractor:
    """Extract temporal RGB signals from facial regions."""

    def __init__(self):
        self.roi_extractor = ROIExtractor()

    def process_video_frames(
        self,
        video_path: str,
        tracking_results: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Extract RGB signals and report why measurements are missing.
        """

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video: {video_path}")

        rgb_signals = []

        # Count why RGB measurements are missing.
        stats = {
            "valid": 0,
            "face_not_detected": 0,
            "roi_failure": 0,
            "invalid_roi": 0,
            "empty_mask": 0,
        }

        try:
            for frame_idx, info in enumerate(tracking_results):

                ret, frame = cap.read()

                if not ret:
                    print(
                        f"[WARNING] Video ended at frame {frame_idx}, "
                        f"but tracking results contain "
                        f"{len(tracking_results)} entries."
                    )
                    break

                # Check whether a face was detected.
                # Check whether a face was detected.
                detected = info.get("detected", False)
                bbox = info.get("bbox")

                # Print details for the first 10 frames.
                if frame_idx < 10:
                    print(
                        f"[RGB DEBUG] Frame {frame_idx}: "
                        f"detected={detected}, bbox={bbox}"
                    )

                if not detected:
                    rgb_signals.append([np.nan, np.nan, np.nan])
                    stats["face_not_detected"] += 1
                    continue

                if (
                    not isinstance(bbox, (list, tuple))
                    or len(bbox) != 4
                    or bbox[2] <= 0
                    or bbox[3] <= 0
                ):
                    rgb_signals.append([np.nan, np.nan, np.nan])
                    stats["roi_failure"] += 1
                    continue

                # Extract forehead and cheek regions.
                # Extract forehead and cheek regions.
                try:
                    rois = self.roi_extractor.extract_rois(frame, bbox)

                except Exception as exc:
                    print(f"[ROI ERROR] Frame {frame_idx}: {exc}")

                    rgb_signals.append([np.nan, np.nan, np.nan])
                    stats["roi_failure"] += 1
                    continue

                # Debug the first 10 frames.
                if frame_idx < 10:
                    print(f"\n[ROI DEBUG] Frame: {frame_idx}")
                    print(f"Bounding box: {bbox}")
                    print(f"Frame dimensions: {frame.shape}")
                    print(f"ROIs type: {type(rois)}")

                    if isinstance(rois, dict):
                        print(f"ROI names: {list(rois.keys())}")

                        for name, roi_data in rois.items():
                            if isinstance(roi_data, (tuple, list)) and len(roi_data) == 2:
                                crop, mask = roi_data

                                print(f"  ROI: {name}")

                                if isinstance(crop, np.ndarray):
                                    print(f"    Crop shape: {crop.shape}")
                                else:
                                    print(f"    Crop type: {type(crop)}")

                                if isinstance(mask, np.ndarray):
                                    print(f"    Mask shape: {mask.shape}")
                                    print(f"    Nonzero mask pixels: {np.count_nonzero(mask)}")
                                else:
                                    print(f"    Mask type: {type(mask)}")

                r_vals = []
                g_vals = []
                b_vals = []

                had_valid_roi = False
                had_nonempty_mask = False

                for crop, mask in rois.values():

                    if (
                        crop is None
                        or mask is None
                        or not isinstance(crop, np.ndarray)
                        or not isinstance(mask, np.ndarray)
                        or crop.size == 0
                        or mask.size == 0
                    ):
                        continue

                    if (
                        crop.ndim != 3
                        or crop.shape[2] != 3
                        or mask.shape != crop.shape[:2]
                    ):
                        continue

                    had_valid_roi = True

                    skin_pixels = mask > 0

                    if not np.any(skin_pixels):
                        continue

                    had_nonempty_mask = True

                    # OpenCV channel order: B, G, R.
                    b_channel = crop[:, :, 0]
                    g_channel = crop[:, :, 1]
                    r_channel = crop[:, :, 2]

                    b_vals.append(
                        float(np.mean(b_channel[skin_pixels]))
                    )
                    g_vals.append(
                        float(np.mean(g_channel[skin_pixels]))
                    )
                    r_vals.append(
                        float(np.mean(r_channel[skin_pixels]))
                    )

                if r_vals:
                    rgb_signals.append([
                        float(np.mean(r_vals)),
                        float(np.mean(g_vals)),
                        float(np.mean(b_vals)),
                    ])
                    stats["valid"] += 1

                else:
                    rgb_signals.append([np.nan, np.nan, np.nan])

                    if not had_valid_roi:
                        stats["invalid_roi"] += 1
                    elif not had_nonempty_mask:
                        stats["empty_mask"] += 1
                    else:
                        stats["invalid_roi"] += 1

        finally:
            cap.release()

        # Print diagnostic results.
        total = len(rgb_signals)
        valid = stats["valid"]
        missing = total - valid

        print("\n========== RGB EXTRACTION REPORT ==========")
        print(f"Video: {video_path}")
        print(f"Frames processed: {total}")
        print(f"Valid RGB measurements: {valid}")
        print(f"Missing RGB measurements: {missing}")

        if total > 0:
            print(f"Valid percentage: {valid / total:.2%}")
        else:
            print("Valid percentage: N/A (no frames processed)")

        print("\nReasons for missing measurements:")
        print(f"Face not detected: {stats['face_not_detected']}")
        print(f"ROI extraction failure: {stats['roi_failure']}")
        print(f"Invalid ROI: {stats['invalid_roi']}")
        print(f"Empty skin mask: {stats['empty_mask']}")
        print("===========================================\n")

        if not rgb_signals:
            return np.empty((0, 3), dtype=np.float32)

        return np.asarray(rgb_signals, dtype=np.float32)
    def close(self):
        """Release the MediaPipe Face Mesh resources."""
        self.roi_extractor.close()
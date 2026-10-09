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

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video: {video_path}")

        rgb_signals = []

        try:
            for info in tracking_results:

                ret, frame = cap.read()

                if not ret:
                    break

                # Preserve the frame's position in the signal.
                # Do not measure a stale bounding box.
                if not info.get("detected", True):
                    rgb_signals.append([np.nan, np.nan, np.nan])
                    continue

                bbox = info["bbox"]

                # Each ROI is a (crop, mask) pair.
                rois = self.roi_extractor.extract_rois(frame, bbox)

                r_vals = []
                g_vals = []
                b_vals = []

                for crop, mask in rois.values():

                    # Ignore missing or invalid regions.
                    if crop.size == 0 or mask.size == 0:
                        continue

                    if crop.shape[:2] != mask.shape:
                        continue

                    # Select only pixels inside the ROI polygon.
                    skin_pixels = mask > 0

                    if not np.any(skin_pixels):
                        continue

                    # OpenCV images use BGR channel order.
                    b_channel = crop[:, :, 0]
                    g_channel = crop[:, :, 1]
                    r_channel = crop[:, :, 2]

                    # Calculate the mean of masked pixels only.
                    b_mean = float(np.mean(b_channel[skin_pixels]))
                    g_mean = float(np.mean(g_channel[skin_pixels]))
                    r_mean = float(np.mean(r_channel[skin_pixels]))

                    b_vals.append(b_mean)
                    g_vals.append(g_mean)
                    r_vals.append(r_mean)

                # Combine the valid facial regions.
                if r_vals:
                    r_mean = float(np.mean(r_vals))
                    g_mean = float(np.mean(g_vals))
                    b_mean = float(np.mean(b_vals))

                    rgb_signals.append([
                        r_mean,
                        g_mean,
                        b_mean
                    ])

                else:
                    # Preserve the frame but mark its measurement missing.
                    rgb_signals.append([
                        np.nan,
                        np.nan,
                        np.nan
                    ])

        finally:
            cap.release()

        # Return shape (number_of_frames, 3).
        # An empty video result becomes shape (0, 3).
        if not rgb_signals:
            return np.empty((0, 3), dtype=np.float32)

        return np.asarray(rgb_signals, dtype=np.float32)

    def close(self):
        """Release the MediaPipe Face Mesh resources."""
        self.roi_extractor.close()
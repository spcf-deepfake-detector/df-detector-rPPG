import cv2
import numpy as np
from typing import Dict, List, Any
from preprocessing.roi import ROIExtractor

class RGBExtractor:
    """Extracts spatial spatial-mean RGB temporal traces from region of interest clips."""
    def __init__(self):
        self.roi_extractor = ROIExtractor()

    def process_video_frames(self, video_path: str, tracking_results: List[Dict[str, Any]]) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        rgb_signals = []

        for info in tracking_results:
            ret, frame = cap.read()
            if not ret:
                break

            bbox = info["bbox"]
            rois = self.roi_extractor.extract_rois(frame, bbox)

            r_vals, g_vals, b_vals = [], [], []
            for roi in rois.values():
                if roi.size > 0 and np.sum(roi) > 0:
                    b_vals.append(np.mean(roi[:, :, 0]))
                    g_vals.append(np.mean(roi[:, :, 1]))
                    r_vals.append(np.mean(roi[:, :, 2]))

            if r_vals:
                r_mean = float(np.mean(r_vals))
                g_mean = float(np.mean(g_vals))
                b_mean = float(np.mean(b_vals))
            else:
                r_mean, g_mean, b_mean = 0.0, 0.0, 0.0

            rgb_signals.append([r_mean, g_mean, b_mean])

        cap.release()
        self.roi_extractor.close()
        
        rgb_arr = np.array(rgb_signals, dtype=np.float32)
        
        # Zero-pad or extrapolate single-frame/empty anomalies
        if len(rgb_arr) == 0:
            return np.zeros((10, 3), dtype=np.float32)
            
        for c in range(3):
            if np.all(rgb_arr[:, c] == 0):
                rgb_arr[:, c] = 1.0
            else:
                mask = rgb_arr[:, c] == 0
                rgb_arr[mask, c] = np.mean(rgb_arr[~mask, c])
                
        return rgb_arr
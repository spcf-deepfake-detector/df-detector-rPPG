import cv2
import numpy as np
from preprocessing.roi import ROIExtractor
from typing import List, Dict, Any


class RGBExtractor:
    def __init__(self):
        self.roi_extractor = RGBExtractor()

        self.min_roi_width = 5
        self.min_roi_height = 5

    def process_video_frames(
            self,
            video_path: str,
            tracking_results: List[Dict[str,Any]]
    )->np.ndarray:

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Couldnot find or open the video in this path: {video_path}")

        rgb_signals = []

        stats = {
            "valid": 0,
            "face_not_detected": 0,
            "invalid_bbox": 0,
            "roi_exception": 0,
            "invalid_roi_structure": 0,
            "no_valid_roi": 0,
            "empty_masks_only": 0,
            "no_rgb_values": 0,
        }

        roi_stats = {
            "total_checked": 0,
            "invalid_data": 0,
            "invalid_dimensions": 0,
            "empty_mask": 0,
            "small_crop": 0,
            "usable": 0,
        }

        failure_examples = []

        max_faliure_examples = 20




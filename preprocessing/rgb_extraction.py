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

        



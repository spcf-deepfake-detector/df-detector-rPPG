import os
import numpy as np
import argparse
import torch
import numpy as np

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker
from preprocessing.rgb_extraction import RGBExtractor
from preprocessing.roi import ROIExtractor
from preprocessing.signal_quality import evaluate_signal_quality
from model.rppg_model import RPPG1DCNN

def run_inference(video_path:str, checkpoint_path: str) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = FaceDetector()
    summary = detector.process_video(video_path)
    detector.close()

    tracker = FaceTracker()
    tracked_dets = tracker.track(summary["detections"])

    extractor = RGBExtractor()
    rgb_signals = extractor.process_video_frames(video_path, tracked_dets)

    fps = summary["fps"]
    if len(rgb_signals)<10:
        rppg_wave = np.zeros(300, )
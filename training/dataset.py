import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Dict

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker
from preprocessing.rgb_extraction import RGBExtractor
from preprocessing.rppg import pos_method

class RPPGDataset(Dataset):
    """Loads video files, extracts POS rPPG signals, and formats sequence tensors."""
    def __init__(self, root_dir: str, target_length: int = 300, fps: float = 30.0):
        self.root_dir = root_dir
        self.target_length = target_length
        self.fps = fps
        self.samples = []

        real_videos = glob.glob(os.path.join(root_dir, "real", "*.mp4"))
        fake_videos = glob.glob(os.path.join(root_dir, "fake", "*.mp4"))

        for path in real_videos:
            self.samples.append((path, 0))  # REAL = 0
        for path in fake_videos:
            self.samples.append((path, 1))  # FAKE = 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.samples[idx]
        
        try:
            detector = FaceDetector()
            summary = detector.process_video(video_path)
            detector.close()

            tracker = FaceTracker()
            tracked_dets = tracker.track(summary["detections"])

            extractor = RGBExtractor()
            rgb_signals = extractor.process_video_frames(video_path, tracked_dets)

            if len(rgb_signals) < 10:
                rppg_wave = np.zeros(self.target_length, dtype=np.float32)
            else:
                rppg_wave = pos_method(rgb_signals, fps=summary["fps"])

        except Exception:
            rppg_wave = np.zeros(self.target_length, dtype=np.float32)

        # Pad or truncate signal to match target_length
        if len(rppg_wave) < self.target_length:
            pad = self.target_length - len(rppg_wave)
            rppg_wave = np.pad(rppg_wave, (0, pad), mode='constant')
        else:
            rppg_wave = rppg_wave[:self.target_length]

        x_tensor = torch.tensor(rppg_wave, dtype=torch.float32)
        y_tensor = torch.tensor(label, dtype=torch.float32)
        return x_tensor, y_tensor
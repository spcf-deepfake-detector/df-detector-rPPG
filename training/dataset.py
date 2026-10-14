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
    def __init__(
            self, 
            root_dir: str, 
            target_length: int = 300, 
            fps: float = 30.0
            ):
        
        self.root_dir = root_dir
        self.target_length = target_length
        self.fps = fps
        self.samples = []

        if not isinstance(target_length, int) or target_length < 1:
            raise ValueError(
                "target_length must  be a positive integer."
            )
        if not np.isfinit(fps) or fps <= 0:
            raise ValueError(
                "fps must be finite and positive"
            )
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {root_dir}"
            )

        real_videos = sorted(
            glob.glob(os.path.join(root_dir, "real","*.mp4" ))
        )

        fake_videos = sorted(
            glob.glob(os.path.join(root_dir, "fake", "*.mp4"))
        )


        for path in real_videos:
            self.samples.append((path, 0))  # REAL = 0
        for path in fake_videos:
            self.samples.append((path, 1))  # FAKE = 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
            self, 
            idx: int
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        video_path, label = self.samples[idx]

        detector = FaceDetector()
        
        try:
            # Step 1: Detect faces in the video.
            summary = detector.process_video(video_path)

        finally:
            detector.close()

        try:
            # Step 2: Track the selected face.
            tracker = FaceTracker()

            tracked_dets = tracker.track(
                summary["detections"]
            )

            # Step 3: Extract RGB measurements.
            extractor = RGBExtractor()

            rgb_signals = extractor.process_video_frames(
                video_path,
                tracked_dets
            )

            rgb_signals = np.asarray(
                rgb_signals,
                dtype=np.float64
            )

            # Step 4: Validate extracted RGB data.
            if (
                rgb_signals.ndim != 2
                or rgb_signals.shape[1] != 3
            ):
                raise ValueError(
                    "RGBExtractor must return an array "
                    "with shape (number_of_frames, 3)."
                )

            if len(rgb_signals) < 10:
                raise ValueError(
                    "Too few RGB measurements were extracted."
                )

            # Step 5: Extract the POS rPPG signal.
            video_fps = summary.get("fps", self.fps)

            if not np.isfinite(video_fps) or video_fps <= 0:
                video_fps = self.fps

            rppg_wave = pos_method(
                rgb_signals,
                fps=video_fps
            )

            rppg_wave = np.asarray(
                rppg_wave,
                dtype=np.float32
            )

            # Step 6: Validate the resulting signal.
            if rppg_wave.ndim != 1:
                raise ValueError(
                    "The extracted rPPG signal must be one-dimensional."
                )

            if len(rppg_wave) == 0:
                raise ValueError(
                    "The extracted rPPG signal is empty."
                )

            if not np.all(np.isfinite(rppg_wave)):
                raise ValueError(
                    "The extracted rPPG signal contains invalid values."
                )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to preprocess video: {video_path}"
            ) from exc

        # Step 7: Pad or truncate to the target length.
        if len(rppg_wave) < self.target_length:

            pad_length = self.target_length - len(rppg_wave)

            rppg_wave = np.pad(
                rppg_wave,
                (0, pad_length),
                mode="constant"
            )

        else:
            rppg_wave = rppg_wave[:self.target_length]

        # Step 8: Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(
            np.ascontiguousarray(rppg_wave, dtype=np.float32)
        )

        y_tensor = torch.tensor(
            label,
            dtype=torch.float32
        )

        return x_tensor, y_tensor
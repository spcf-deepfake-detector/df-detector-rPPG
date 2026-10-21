import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker
from preprocessing.rgb_extraction import RGBExtractor
from preprocessing.rppg import pos_method


class RPPGDataset(Dataset):
    """Loads videos, extracts POS rPPG signals, and formats sequence tensors."""

    def __init__(
        self,
        root_dir: str,
        target_length: int = 300,
        fps: float = 30.0
    ):
        self.root_dir = root_dir
        self.target_length = target_length
        self.fps = fps
        self.samples: List[Tuple[str, int]] = []

        # Validate configuration.
        if not isinstance(target_length, int) or isinstance(target_length, bool) or target_length < 1:
            raise ValueError("target_length must be a positive integer.")

        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive.")

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {root_dir}"
            )

        # Load REAL videos: label 0.
        real_dir = os.path.join(root_dir, "real")
        real_videos = sorted(
            glob.glob(os.path.join(real_dir, "*.mp4"))
        )

        # Load FAKE videos: label 1.
        fake_dir = os.path.join(root_dir, "fake")
        fake_videos = sorted(
            glob.glob(os.path.join(fake_dir, "*.mp4"))
        )

        for path in real_videos:
            self.samples.append((path, 0))

        for path in fake_videos:
            self.samples.append((path, 1))

        if not self.samples:
            raise ValueError(
                f"No .mp4 videos found in '{real_dir}' or '{fake_dir}'."
            )

        print(f"[DATASET] Root directory: {root_dir}")
        print(f"[DATASET] Real videos: {len(real_videos)}")
        print(f"[DATASET] Fake videos: {len(fake_videos)}")
        print(f"[DATASET] Total videos: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        video_path, label = self.samples[idx]

        # --------------------------------------------------
        # STEP 1: Detect faces in the video.
        # --------------------------------------------------
        try:
            detector = FaceDetector()

            try:
                summary = detector.process_video(video_path)
            finally:
                detector.close()

            if not isinstance(summary, dict):
                raise ValueError(
                    "FaceDetector must return a dictionary."
                )

            if "detections" not in summary:
                raise ValueError(
                    "Face detection results do not contain 'detections'."
                )

            detections = summary["detections"]

            if detections is None:
                raise ValueError(
                    "Face detection returned no detection results."
                )

        except Exception as exc:
            raise RuntimeError(
                f"Face detection failed for video: {video_path}"
            ) from exc

        # --------------------------------------------------
        # STEP 2: Track the detected face.
        # --------------------------------------------------
        try:
            tracker = FaceTracker()

            tracked_dets = tracker.track_faces(detections)

            if tracked_dets is None:
                raise ValueError(
                    "FaceTracker returned no tracking results."
                )

            if len(tracked_dets) == 0:
                raise ValueError(
                    "No face-tracking results were produced."
                )

        except Exception as exc:
            raise RuntimeError(
                f"Face tracking failed for video: {video_path}"
            ) from exc

        # --------------------------------------------------
        # STEP 3: Extract RGB measurements.
        # --------------------------------------------------
        try:
            extractor = RGBExtractor()

            try:
                rgb_signals = extractor.process_video_frames(
                    video_path,
                    tracked_dets
                )
            finally:
                extractor.close()

            rgb_signals = np.asarray(
                rgb_signals,
                dtype=np.float64
            )

            # Expected shape: (number_of_frames, 3).
            if (
                rgb_signals.ndim != 2
                or rgb_signals.shape[1] != 3
            ):
                raise ValueError(
                    "RGBExtractor must return an array with shape "
                    "(number_of_frames, 3)."
                )

            if len(rgb_signals) < 10:
                raise ValueError(
                    f"Too few RGB measurements were extracted: "
                    f"{len(rgb_signals)}."
                )

            # --------------------------------------------------
            # STEP 4: Diagnose missing RGB measurements.
            # --------------------------------------------------
            valid_rows = np.all(
                np.isfinite(rgb_signals),
                axis=1
            )

            valid_count = int(np.sum(valid_rows))
            total_count = len(rgb_signals)
            missing_count = total_count - valid_count
            valid_fraction = valid_count / total_count

            print(
                f"\n[RGB DEBUG] Video: {video_path}"
                f"\n[RGB DEBUG] Total measurements: {total_count}"
                f"\n[RGB DEBUG] Valid measurements: {valid_count}"
                f"\n[RGB DEBUG] Missing measurements: {missing_count}"
                f"\n[RGB DEBUG] Valid fraction: {valid_fraction:.2%}"
            )

            # Identify consecutive runs of missing measurements.
            invalid = ~valid_rows
            missing_runs = []
            run_start = None

            for i, is_missing in enumerate(invalid):
                if is_missing and run_start is None:
                    run_start = i

                elif not is_missing and run_start is not None:
                    missing_runs.append((run_start, i - 1))
                    run_start = None

            if run_start is not None:
                missing_runs.append(
                    (run_start, len(invalid) - 1)
                )

            print(
                f"[RGB DEBUG] Missing runs: {missing_runs}"
            )

            # --------------------------------------------------
            # STEP 5: Determine the video's frame rate.
            # --------------------------------------------------
            video_fps = summary.get("fps", self.fps)

            try:
                video_fps = float(video_fps)
            except (TypeError, ValueError):
                video_fps = self.fps

            if not np.isfinite(video_fps) or video_fps <= 0:
                video_fps = self.fps

            # --------------------------------------------------
            # STEP 6: Extract the POS rPPG signal.
            # --------------------------------------------------
            rppg_wave = pos_method(
                rgb_signals,
                fps=video_fps
            )

            rppg_wave = np.asarray(
                rppg_wave,
                dtype=np.float32
            )

            # Validate the resulting signal.
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
                f"Failed to preprocess video: {video_path}. "
                f"Reason: {exc}"
            ) from exc

        # --------------------------------------------------
        # STEP 7: Pad or truncate to the target length.
        # --------------------------------------------------
        if len(rppg_wave) < self.target_length:
            pad_length = self.target_length - len(rppg_wave)

            rppg_wave = np.pad(
                rppg_wave,
                (0, pad_length),
                mode="constant"
            )

        else:
            rppg_wave = rppg_wave[:self.target_length]

        # --------------------------------------------------
        # STEP 8: Convert the signal and label to tensors.
        # --------------------------------------------------
        x_tensor = torch.from_numpy(
            np.ascontiguousarray(
                rppg_wave,
                dtype=np.float32
            )
        )

        y_tensor = torch.tensor(
            label,
            dtype=torch.float32
        )

        return x_tensor, y_tensor
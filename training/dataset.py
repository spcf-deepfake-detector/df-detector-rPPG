import os
import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker
from preprocessing.rgb_extraction import RGBExtractor
from preprocessing.rppg import pos_method


class RPPGDataset(Dataset):
    """
    Load real and fake videos, extract POS rPPG signals,
    and return fixed-length PyTorch tensors.

    Expected directory structure:

        root_dir/
            real/
                video1.mp4
                video2.mp4
            fake/
                video1.mp4
                video2.mp4

    Labels:
        Real = 0
        Fake = 1

    Returns:
        x_tensor: float32 tensor with shape (target_length,)
        y_tensor: float32 scalar tensor

    Important:
        This dataset currently processes one face per video frame.
        It does not implement multi-face tracking.
    """

    def __init__(
        self,
        root_dir: str,
        target_length: int = 300,
        fps: float = 30.0,
        min_valid_rgb_fraction: float = 0.80,
        min_rgb_frames: int = 10,
        detection_confidence: float = 0.3,
        model_selection: int = 1,
        frame_scale: float = 1.0,
        debug: bool = True,
    ):
        # --------------------------------------------------
        # STEP 1: Validate configuration
        # --------------------------------------------------

        if (
            isinstance(target_length, bool)
            or not isinstance(target_length, int)
            or target_length < 1
        ):
            raise ValueError(
                "target_length must be a positive integer."
            )

        if (
            isinstance(fps, bool)
            or not np.isscalar(fps)
            or not np.isfinite(fps)
            or fps <= 0
        ):
            raise ValueError(
                "fps must be a finite positive number."
            )

        if (
            isinstance(min_valid_rgb_fraction, bool)
            or not np.isscalar(min_valid_rgb_fraction)
            or not np.isfinite(min_valid_rgb_fraction)
            or not 0.0 < min_valid_rgb_fraction <= 1.0
        ):
            raise ValueError(
                "min_valid_rgb_fraction must be greater than 0 "
                "and at most 1."
            )

        if (
            isinstance(min_rgb_frames, bool)
            or not isinstance(min_rgb_frames, int)
            or min_rgb_frames < 10
        ):
            raise ValueError(
                "min_rgb_frames must be an integer of at least 10."
            )

        if (
            isinstance(detection_confidence, bool)
            or not np.isscalar(detection_confidence)
            or not np.isfinite(detection_confidence)
            or not 0.0 < detection_confidence <= 1.0
        ):
            raise ValueError(
                "detection_confidence must be greater than 0 "
                "and at most 1."
            )

        if (
            isinstance(model_selection, bool)
            or model_selection not in (0, 1)
        ):
            raise ValueError(
                "model_selection must be 0 or 1."
            )

        if (
            isinstance(frame_scale, bool)
            or not np.isscalar(frame_scale)
            or not np.isfinite(frame_scale)
            or frame_scale < 1.0
        ):
            raise ValueError(
                "frame_scale must be finite and at least 1.0."
            )

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {root_dir}"
            )

        self.root_dir = os.path.abspath(root_dir)
        self.target_length = target_length
        self.fps = float(fps)
        self.min_valid_rgb_fraction = float(
            min_valid_rgb_fraction
        )
        self.min_rgb_frames = min_rgb_frames
        self.detection_confidence = float(
            detection_confidence
        )
        self.model_selection = model_selection
        self.frame_scale = float(frame_scale)
        self.debug = debug

        # --------------------------------------------------
        # STEP 2: Discover videos and assign labels
        # --------------------------------------------------

        real_dir = os.path.join(self.root_dir, "real")
        fake_dir = os.path.join(self.root_dir, "fake")

        real_videos = sorted(
            glob.glob(os.path.join(real_dir, "*.mp4"))
        )

        fake_videos = sorted(
            glob.glob(os.path.join(fake_dir, "*.mp4"))
        )

        self.samples: List[Tuple[str, int]] = []

        self.samples.extend(
            (path, 0) for path in real_videos
        )

        self.samples.extend(
            (path, 1) for path in fake_videos
        )

        if not self.samples:
            raise ValueError(
                "No .mp4 videos were found. Expected videos in "
                f"'{real_dir}' and/or '{fake_dir}'."
            )

        print("\n========== RPPG DATASET ==========")
        print(f"Root directory: {self.root_dir}")
        print(f"Real videos: {len(real_videos)}")
        print(f"Fake videos: {len(fake_videos)}")
        print(f"Total videos: {len(self.samples)}")
        print(f"Target signal length: {self.target_length}")
        print(
            "Minimum valid RGB fraction: "
            f"{self.min_valid_rgb_fraction:.1%}"
        )
        print("==================================\n")

    def __len__(self) -> int:
        """Return the number of videos in the dataset."""
        return len(self.samples)

    @staticmethod
    def _validate_detections(summary, video_path):
        """Validate the detector output and its frame records."""

        if not isinstance(summary, dict):
            raise ValueError(
                f"FaceDetector returned an invalid result for "
                f"'{video_path}': expected a dictionary."
            )

        detections = summary.get("detections")

        if not isinstance(detections, list) or not detections:
            raise ValueError(
                f"No frame detection records were produced for "
                f"'{video_path}'."
            )

        # The detector is expected to return one record per decoded
        # frame, including frames where no face was found.
        for index, record in enumerate(detections):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid detection record at position {index} "
                    f"in '{video_path}'."
                )

            frame_idx = record.get("frame_idx")

            if frame_idx is not None and frame_idx != index:
                raise ValueError(
                    f"Frame-index mismatch in '{video_path}': "
                    f"record {index} refers to frame {frame_idx}."
                )

        return detections

    def _extract_tracked_detections(self, video_path):
        """Detect and track the primary face in a video."""

        detector = None

        try:
            detector = FaceDetector(
                min_detection_confidence=self.detection_confidence,
                model_selection=self.model_selection,
                frame_scale=self.frame_scale,
            )

            summary = detector.process_video(
                video_path,
                multi_face=False,
                verbose=self.debug,
            )

            detections = self._validate_detections(
                summary,
                video_path,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Face detection failed for '{video_path}': {exc}"
            ) from exc

        finally:
            if detector is not None:
                detector.close()

        try:
            tracker = FaceTracker()
            tracked_detections = tracker.track_faces(detections)

            if not isinstance(tracked_detections, list):
                raise ValueError(
                    "FaceTracker must return a list."
                )

            if len(tracked_detections) != len(detections):
                raise ValueError(
                    "The number of tracking records does not match "
                    "the number of detection records. "
                    f"Detections={len(detections)}, "
                    f"tracking={len(tracked_detections)}."
                )

            return summary, tracked_detections

        except Exception as exc:
            raise RuntimeError(
                f"Face tracking failed for '{video_path}': {exc}"
            ) from exc

    def _extract_rgb(self, video_path, tracked_detections):
        """Extract one RGB measurement per corresponding frame."""

        extractor = None

        try:
            extractor = RGBExtractor()

            rgb_signals = extractor.process_video_frames(
                video_path,
                tracked_detections,
            )

        except Exception as exc:
            raise RuntimeError(
                f"RGB extraction failed for '{video_path}': {exc}"
            ) from exc

        finally:
            if extractor is not None:
                extractor.close()

        try:
            rgb_signals = np.asarray(
                rgb_signals,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"RGB extraction returned non-numeric data for "
                f"'{video_path}'."
            ) from exc

        if (
            rgb_signals.ndim != 2
            or rgb_signals.shape[1] != 3
        ):
            raise ValueError(
                f"Invalid RGB array shape for '{video_path}': "
                f"{rgb_signals.shape}. Expected (frames, 3)."
            )

        if len(rgb_signals) != len(tracked_detections):
            raise ValueError(
                f"Frame-count mismatch for '{video_path}': "
                f"tracking produced {len(tracked_detections)} records, "
                f"but RGB extraction produced {len(rgb_signals)}."
            )

        return rgb_signals

    def _diagnose_rgb(self, rgb_signals, video_path):
        """Report missing RGB measurements and validate data quality."""

        total_count = len(rgb_signals)

        if total_count == 0:
            raise ValueError(
                f"No RGB measurements were produced for '{video_path}'."
            )

        valid_rows = np.all(
            np.isfinite(rgb_signals),
            axis=1,
        )

        valid_count = int(np.count_nonzero(valid_rows))
        missing_count = total_count - valid_count
        valid_fraction = valid_count / total_count

        if self.debug:
            print(f"\n[RGB DEBUG] Video: {video_path}")
            print(f"[RGB DEBUG] Total frames: {total_count}")
            print(f"[RGB DEBUG] Valid RGB frames: {valid_count}")
            print(f"[RGB DEBUG] Missing RGB frames: {missing_count}")
            print(
                f"[RGB DEBUG] Valid fraction: {valid_fraction:.2%}"
            )

        # Identify consecutive runs of missing RGB measurements.
        if self.debug:
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

            print(f"[RGB DEBUG] Missing runs: {missing_runs}")

        # Enforce the validity threshold before POS processing.
        if valid_fraction < self.min_valid_rgb_fraction:
            raise ValueError(
                f"Insufficient valid RGB measurements for "
                f"'{video_path}': {valid_fraction:.2%} valid, "
                f"but at least "
                f"{self.min_valid_rgb_fraction:.2%} is required. "
                f"Valid={valid_count}, total={total_count}."
            )

        if valid_count < self.min_rgb_frames:
            raise ValueError(
                f"Too few valid RGB measurements for '{video_path}': "
                f"{valid_count}; at least {self.min_rgb_frames} "
                "are required."
            )

        return valid_fraction

    @staticmethod
    def _resolve_video_fps(summary, fallback_fps):
        """Use valid video FPS metadata or a configured fallback."""

        try:
            video_fps = float(summary.get("fps", fallback_fps))
        except (TypeError, ValueError, OverflowError):
            video_fps = fallback_fps

        if not np.isfinite(video_fps) or video_fps <= 0:
            video_fps = fallback_fps

        return float(video_fps)

    def _extract_pos_signal(self, rgb_signals, video_fps, video_path):
        """Convert RGB measurements into a validated POS waveform."""

        try:
            # pos_method must handle or validate missing values before
            # performing POS calculations. We deliberately do not
            # silently interpolate here because that behavior belongs
            # to the signal-preparation implementation.
            rppg_wave = pos_method(
                rgb_signals,
                fps=video_fps,
            )

            rppg_wave = np.asarray(
                rppg_wave,
                dtype=np.float32,
            )

        except Exception as exc:
            raise RuntimeError(
                f"POS signal extraction failed for '{video_path}': "
                f"{exc}"
            ) from exc

        if rppg_wave.ndim != 1:
            raise ValueError(
                f"POS output for '{video_path}' must be one-dimensional; "
                f"received shape {rppg_wave.shape}."
            )

        if len(rppg_wave) == 0:
            raise ValueError(
                f"POS produced an empty signal for '{video_path}'."
            )

        if not np.all(np.isfinite(rppg_wave)):
            raise ValueError(
                f"POS produced NaN or infinite values for "
                f"'{video_path}'."
            )

        return rppg_wave

    def _format_signal(self, rppg_wave, video_path):
        """Truncate or pad a valid waveform to the configured length."""

        original_length = len(rppg_wave)

        if original_length > self.target_length:
            # Use the first target_length samples.
            rppg_wave = rppg_wave[:self.target_length]

        elif original_length < self.target_length:
            pad_length = self.target_length - original_length

            # Zero padding is retained for compatibility with the
            # existing model. The debug output makes it visible.
            rppg_wave = np.pad(
                rppg_wave,
                (0, pad_length),
                mode="constant",
                constant_values=0.0,
            )

            if self.debug:
                print(
                    f"[PADDING] {os.path.basename(video_path)}: "
                    f"{original_length} signal samples + "
                    f"{pad_length} zero samples."
                )

        # Return a contiguous float32 array for PyTorch.
        return np.ascontiguousarray(
            rppg_wave,
            dtype=np.float32,
        )

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one video and return its signal and label."""

        if (
            isinstance(idx, bool)
            or not isinstance(idx, (int, np.integer))
        ):
            raise TypeError("Dataset index must be an integer.")

        if idx < 0:
            idx += len(self.samples)

        if idx < 0 or idx >= len(self.samples):
            raise IndexError(
                f"Dataset index {idx} is out of range."
            )

        video_path, label = self.samples[idx]

        # STEP 1 and 2: Detect and track the primary face.
        summary, tracked_detections = (
            self._extract_tracked_detections(video_path)
        )

        # STEP 3: Extract RGB measurements.
        rgb_signals = self._extract_rgb(
            video_path,
            tracked_detections,
        )

        # STEP 4: Check RGB validity before POS processing.
        self._diagnose_rgb(
            rgb_signals,
            video_path,
        )

        # STEP 5: Resolve the video's frame rate.
        video_fps = self._resolve_video_fps(
            summary,
            self.fps,
        )

        # STEP 6: Extract and validate the POS signal.
        rppg_wave = self._extract_pos_signal(
            rgb_signals,
            video_fps,
            video_path,
        )

        # STEP 7: Pad or truncate the waveform.
        rppg_wave = self._format_signal(
            rppg_wave,
            video_path,
        )

        # STEP 8: Return tensors in the original format.
        x_tensor = torch.from_numpy(rppg_wave)

        y_tensor = torch.tensor(
            label,
            dtype=torch.float32,
        )

        return x_tensor, y_tensor
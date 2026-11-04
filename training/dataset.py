import os
import glob
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker
from preprocessing.rgb_extraction import RGBExtractor
from preprocessing.rppg import pos_method


class VideoQualityError(ValueError):
    """Raised when a video cannot produce a sufficiently reliable signal."""


class RPPGDataset(Dataset):
    """
    Load real and fake videos and extract their rPPG signals.

    Pipeline:
        Video
          -> Face Detection
          -> Face Tracking
          -> RGB Extraction
          -> RGB Validation
          -> POS
          -> Fixed-Length Signal
          -> PyTorch Tensor

    Labels:
        Real = 0
        Fake = 1

    Returns:
        x_tensor: float32 tensor, shape (target_length,)
        y_tensor: float32 scalar tensor
    """

    def __init__(
        self,
        root_dir: str,
        target_length: int = 300,
        fps: float = 30.0,
        min_valid_rgb_fraction: float = 0.80,
        min_rgb_frames: int = 10,
        detection_confidence: float = 0.3,
        model_selection: int = 0,
        frame_scale: float = 1.0,
        debug: bool = True,
    ):

        # ==================================================
        # STEP 1: VALIDATE CONFIGURATION
        # ==================================================

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
            or not 0 < min_valid_rgb_fraction <= 1
        ):
            raise ValueError(
                "min_valid_rgb_fraction must be between 0 and 1."
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
            or not 0 < detection_confidence <= 1
        ):
            raise ValueError(
                "detection_confidence must be between 0 and 1."
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

        # ==================================================
        # STEP 2: FIND VIDEOS
        # ==================================================

        real_dir = os.path.join(self.root_dir, "real")
        fake_dir = os.path.join(self.root_dir, "fake")

        real_videos = sorted(
            glob.glob(os.path.join(real_dir, "*.mp4"))
        )

        fake_videos = sorted(
            glob.glob(os.path.join(fake_dir, "*.mp4"))
        )

        self.samples: List[Tuple[str, int]] = []

        # Real = 0
        for path in real_videos:
            self.samples.append((path, 0))

        # Fake = 1
        for path in fake_videos:
            self.samples.append((path, 1))

        if not self.samples:
            raise ValueError(
                "No MP4 videos found.\n"
                f"Expected folders:\n"
                f"{real_dir}\n"
                f"{fake_dir}"
            )

        # ==================================================
        # STEP 3: DISPLAY DATASET INFORMATION
        # ==================================================

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

        print(
            f"Minimum valid RGB frames: {self.min_rgb_frames}"
        )

        print("==================================\n")

    # ======================================================
    # DATASET LENGTH
    # ======================================================

    def __len__(self) -> int:
        return len(self.samples)

    # ======================================================
    # STEP 4: VALIDATE FACE DETECTION RESULTS
    # ======================================================

    @staticmethod
    def _validate_detections(summary, video_path):

        if not isinstance(summary, dict):
            raise ValueError(
                f"Invalid FaceDetector output: {video_path}"
            )

        detections = summary.get("detections")

        if not isinstance(detections, list):
            raise ValueError(
                f"FaceDetector did not return a detection list: "
                f"{video_path}"
            )

        if len(detections) == 0:
            raise ValueError(
                f"No frame records were returned: {video_path}"
            )

        for index, record in enumerate(detections):

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid detection record at frame {index}: "
                    f"{video_path}"
                )

            frame_idx = record.get("frame_idx")

            if frame_idx is not None and frame_idx != index:
                raise ValueError(
                    f"Frame index mismatch in {video_path}. "
                    f"Expected {index}, received {frame_idx}."
                )

        return detections

    # ======================================================
    # STEP 5: FACE DETECTION AND TRACKING
    # ======================================================

    def _extract_tracked_detections(self, video_path):

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
                f"Face detection failed:\n"
                f"Video: {video_path}\n"
                f"Reason: {exc}"
            ) from exc

        finally:

            if detector is not None:
                detector.close()

        # ----------------------------------------------
        # TRACK DETECTED FACE
        # ----------------------------------------------

        try:

            tracker = FaceTracker()

            tracked_detections = tracker.track_faces(
                detections
            )

            if not isinstance(tracked_detections, list):
                raise ValueError(
                    "FaceTracker must return a list."
                )

            if len(tracked_detections) != len(detections):
                raise ValueError(
                    "Detection and tracking frame counts differ.\n"
                    f"Detection records: {len(detections)}\n"
                    f"Tracking records: {len(tracked_detections)}"
                )

            # ------------------------------------------
            # TRACKING DIAGNOSTICS
            # ------------------------------------------

            if self.debug:

                status_counts = Counter(
                    record.get("status", "unknown")
                    for record in tracked_detections
                )

                valid_count = sum(
                    bool(record.get("track_valid", False))
                    for record in tracked_detections
                )

                print("\n[TRACKING SUMMARY]")
                print(f"Video: {os.path.basename(video_path)}")

                print(
                    f"Total tracking frames: "
                    f"{len(tracked_detections)}"
                )

                print(
                    f"Tracking-valid frames: {valid_count}"
                )

                print(
                    f"Tracking-rejected frames: "
                    f"{len(tracked_detections) - valid_count}"
                )

                print("Tracking statuses:")

                for status, count in sorted(status_counts.items()):
                    print(f"  {status}: {count}")

            return summary, tracked_detections

        except Exception as exc:

            raise RuntimeError(
                f"Face tracking failed:\n"
                f"Video: {video_path}\n"
                f"Reason: {exc}"
            ) from exc

    # ======================================================
    # STEP 6: RGB EXTRACTION
    # ======================================================

    def _extract_rgb(self, video_path, tracked_detections):

        extractor = None

        try:

            extractor = RGBExtractor()

            rgb_signals = extractor.process_video_frames(
                video_path,
                tracked_detections,
            )

        except Exception as exc:

            raise RuntimeError(
                f"RGB extraction failed:\n"
                f"Video: {video_path}\n"
                f"Reason: {exc}"
            ) from exc

        finally:

            if extractor is not None:
                extractor.close()

        # ----------------------------------------------
        # CONVERT TO NUMPY
        # ----------------------------------------------

        try:

            rgb_signals = np.asarray(
                rgb_signals,
                dtype=np.float64,
            )

        except (TypeError, ValueError) as exc:

            raise ValueError(
                f"RGB extraction returned invalid data:\n"
                f"{video_path}"
            ) from exc

        # ----------------------------------------------
        # VALIDATE ARRAY SHAPE
        # ----------------------------------------------

        if rgb_signals.ndim != 2:
            raise ValueError(
                f"RGB array must have 2 dimensions.\n"
                f"Received shape: {rgb_signals.shape}\n"
                f"Video: {video_path}"
            )

        if rgb_signals.shape[1] != 3:
            raise ValueError(
                f"RGB array must contain 3 channels (R, G, B).\n"
                f"Received shape: {rgb_signals.shape}\n"
                f"Video: {video_path}"
            )

        if len(rgb_signals) != len(tracked_detections):
            raise ValueError(
                f"Frame count mismatch:\n"
                f"Tracking: {len(tracked_detections)}\n"
                f"RGB: {len(rgb_signals)}\n"
                f"Video: {video_path}"
            )

        return rgb_signals

    # ======================================================
    # STEP 7: RGB QUALITY DIAGNOSTICS
    # ======================================================

    def _diagnose_rgb(
        self,
        rgb_signals,
        tracked_detections,
        video_path,
    ):

        total_count = len(rgb_signals)

        if total_count == 0:
            raise VideoQualityError(
                f"No RGB measurements found: {video_path}"
            )

        # A row is valid only when all 3 channels are finite.
        valid_rows = np.all(
            np.isfinite(rgb_signals),
            axis=1,
        )

        valid_count = int(np.count_nonzero(valid_rows))

        missing_count = total_count - valid_count

        valid_fraction = valid_count / total_count

        # ----------------------------------------------
        # TRACKING STATUS OF MISSING RGB FRAMES
        # ----------------------------------------------

        missing_statuses = Counter()

        for index, is_valid in enumerate(valid_rows):

            if not is_valid:

                status = tracked_detections[index].get(
                    "tracking_status",
                    "unknown",
                )
                missing_statuses[status] += 1

        # ----------------------------------------------
        # PRINT DIAGNOSTICS
        # ----------------------------------------------

        if self.debug:

            print("\n========== RGB QUALITY ==========")

            print(f"Video: {os.path.basename(video_path)}")

            print(f"Total frames: {total_count}")
            print(f"Valid RGB frames: {valid_count}")
            print(f"Missing RGB frames: {missing_count}")

            print(
                f"Valid RGB percentage: {valid_fraction:.2%}"
            )

            print("\nMissing RGB by tracking status:")

            if missing_statuses:

                for status, count in sorted(
                    missing_statuses.items()
                ):
                    print(f"  {status}: {count}")

            else:
                print("  No missing RGB frames.")

            # ------------------------------------------
            # FIND CONSECUTIVE MISSING FRAME GROUPS
            # ------------------------------------------

            missing_runs = []

            run_start = None

            for index, is_valid in enumerate(valid_rows):

                is_missing = not is_valid

                if is_missing and run_start is None:
                    run_start = index

                elif not is_missing and run_start is not None:

                    missing_runs.append(
                        (run_start, index - 1)
                    )

                    run_start = None

            if run_start is not None:

                missing_runs.append(
                    (run_start, total_count - 1)
                )

            print(f"\nMissing frame groups: {missing_runs}")

            print("==================================\n")

        # ----------------------------------------------
        # ENFORCE QUALITY REQUIREMENTS
        # ----------------------------------------------

        if valid_fraction < self.min_valid_rgb_fraction:

            raise VideoQualityError(
                "\nInsufficient valid RGB measurements.\n"
                f"Video: {video_path}\n"
                f"Valid: {valid_count}/{total_count}\n"
                f"Valid percentage: {valid_fraction:.2%}\n"
                f"Required percentage: "
                f"{self.min_valid_rgb_fraction:.2%}\n"
                f"Missing: {missing_count}\n"
                "Check face tracking and ROI extraction."
            )

        if valid_count < self.min_rgb_frames:

            raise VideoQualityError(
                f"Too few valid RGB measurements.\n"
                f"Video: {video_path}\n"
                f"Valid frames: {valid_count}\n"
                f"Required frames: {self.min_rgb_frames}"
            )

        return valid_fraction

    # ======================================================
    # STEP 8: GET VIDEO FPS
    # ======================================================

    @staticmethod
    def _resolve_video_fps(summary, fallback_fps):

        try:

            video_fps = float(
                summary.get("fps", fallback_fps)
            )

        except (TypeError, ValueError, OverflowError):

            video_fps = fallback_fps

        if not np.isfinite(video_fps) or video_fps <= 0:
            video_fps = fallback_fps

        return float(video_fps)

    # ======================================================
    # STEP 9: RGB TO POS SIGNAL
    # ======================================================

    def _extract_pos_signal(
        self,
        rgb_signals,
        video_fps,
        video_path,
    ):

        try:

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
                f"POS processing failed:\n"
                f"Video: {video_path}\n"
                f"Reason: {exc}"
            ) from exc

        # ----------------------------------------------
        # VALIDATE POS OUTPUT
        # ----------------------------------------------

        if rppg_wave.ndim != 1:

            raise ValueError(
                f"POS output must be one-dimensional.\n"
                f"Received shape: {rppg_wave.shape}\n"
                f"Video: {video_path}"
            )

        if len(rppg_wave) == 0:

            raise ValueError(
                f"POS produced an empty signal: {video_path}"
            )

        if not np.all(np.isfinite(rppg_wave)):

            raise ValueError(
                f"POS produced NaN or infinite values: {video_path}"
            )

        return rppg_wave

    # ======================================================
    # STEP 10: FORMAT SIGNAL LENGTH
    # ======================================================

    def _format_signal(self, rppg_wave, video_path):

        original_length = len(rppg_wave)

        print(
            f"[SIGNAL LENGTH] "
            f"{os.path.basename(video_path)}: "
            f"{original_length} samples"
        )

        # ----------------------------------------------
        # TRUNCATE LONG SIGNALS
        # ----------------------------------------------

        if original_length > self.target_length:

            rppg_wave = rppg_wave[:self.target_length]

        # ----------------------------------------------
        # PAD SHORT SIGNALS
        # ----------------------------------------------

        elif original_length < self.target_length:

            pad_length = self.target_length - original_length

            rppg_wave = np.pad(
                rppg_wave,
                (0, pad_length),
                mode="constant",
                constant_values=0.0,
            )

            if self.debug:

                print(
                    f"[PADDING] {os.path.basename(video_path)}\n"
                    f"Original samples: {original_length}\n"
                    f"Added zero samples: {pad_length}"
                )

        return np.ascontiguousarray(
            rppg_wave,
            dtype=np.float32,
        )

    # ======================================================
    # STEP 11: LOAD ONE DATASET SAMPLE
    # ======================================================

    def __getitem__(self, idx: int):

        # ----------------------------------------------
        # VALIDATE INDEX
        # ----------------------------------------------

        if (
            isinstance(idx, bool)
            or not isinstance(idx, (int, np.integer))
        ):

            raise TypeError(
                "Dataset index must be an integer."
            )

        if idx < 0:
            idx += len(self.samples)

        if idx < 0 or idx >= len(self.samples):

            raise IndexError(
                f"Dataset index {idx} is out of range."
            )

        video_path, label = self.samples[idx]

        print(
            f"\n[PROCESSING VIDEO] "
            f"{os.path.basename(video_path)}"
        )

        # ----------------------------------------------
        # 1. FACE DETECTION + TRACKING
        # ----------------------------------------------

        summary, tracked_detections = (
            self._extract_tracked_detections(
                video_path
            )
        )

        # ----------------------------------------------
        # 2. RGB EXTRACTION
        # ----------------------------------------------

        rgb_signals = self._extract_rgb(
            video_path,
            tracked_detections,
        )

        # ----------------------------------------------
        # 3. RGB QUALITY CHECK
        # ----------------------------------------------

        self._diagnose_rgb(
            rgb_signals,
            tracked_detections,
            video_path,
        )

        # ----------------------------------------------
        # 4. VIDEO FPS
        # ----------------------------------------------

        video_fps = self._resolve_video_fps(
            summary,
            self.fps,
        )

        # ----------------------------------------------
        # 5. POS PROCESSING
        # ----------------------------------------------

        rppg_wave = self._extract_pos_signal(
            rgb_signals,
            video_fps,
            video_path,
        )

        # ----------------------------------------------
        # 6. FIXED-LENGTH SIGNAL
        # ----------------------------------------------

        rppg_wave = self._format_signal(
            rppg_wave,
            video_path,
        )

        # ----------------------------------------------
        # 7. CONVERT TO PYTORCH TENSORS
        # ----------------------------------------------

        x_tensor = torch.from_numpy(
            rppg_wave
        )

        y_tensor = torch.tensor(
            label,
            dtype=torch.float32,
        )

        return x_tensor, y_tensor
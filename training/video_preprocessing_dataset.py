import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from scipy.signal import resample
from torch.utils.data import Dataset

from preprocessing.rgb_extraction import RGBExtractor
from preprocessing.rppg import pos_method


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class VideoQualityError(ValueError):
    """Raised when a video window cannot produce reliable rPPG."""


class MetadataError(ValueError):
    """Raised when metadata is invalid."""


# ============================================================
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# RPPG VIDEO PREPROCESSING DATASET
# ============================================================

class RPPGDataset(Dataset):
    """
    FALSEFACET rPPG video preprocessing dataset.

    Pipeline:

        Extracted face images
            -> metadata.json
            -> RGB extraction
            -> RGB validation
            -> 10-second windows
            -> POS
            -> Fixed-length 300-sample signals

    Each source video can generate multiple training samples.

    Labels:
        Real = 0
        Fake = 1

    Expected structure:

        extracted_faces/
        ├── train/
        │   ├── real/
        │   │   ├── video_1/
        │   │   │   ├── metadata.json
        │   │   │   ├── frame_000000.jpg
        │   │   │   └── ...
        │   └── fake/
        │       └── ...
        ├── val/
        │   ├── real/
        │   └── fake/
        └── test/
            ├── real/
            └── fake/

    Output from process_sequence():

        X: (number_of_valid_windows, 300)
        Y: (number_of_valid_windows,)

    """

    def __init__(
        self,
        root_dir: str,
        target_length: int = 300,
        fps: float = 30.0,
        window_seconds: float = 10.0,
        min_valid_rgb_fraction: float = 0.80,
        min_rgb_frames: int = 10,
        debug: bool = True,
    ):

        # ==================================================
        # 1. VALIDATE CONFIGURATION
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
                "fps must be finite and positive."
            )

        if (
            not np.isscalar(window_seconds)
            or not np.isfinite(window_seconds)
            or window_seconds <= 0
        ):
            raise ValueError(
                "window_seconds must be positive."
            )

        if (
            not np.isscalar(min_valid_rgb_fraction)
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
                "min_rgb_frames must be at least 10."
            )

        # ==================================================
        # 2. INITIALIZE CONFIGURATION
        # ==================================================

        self.root_dir = Path(root_dir).resolve()

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        self.target_length = target_length
        self.fps = float(fps)

        self.window_seconds = float(window_seconds)

        self.min_valid_rgb_fraction = float(
            min_valid_rgb_fraction
        )

        self.min_rgb_frames = min_rgb_frames
        self.debug = bool(debug)

        # ==================================================
        # 3. FIND SOURCE VIDEO SEQUENCES
        # ==================================================

        self.samples: List[Tuple[Path, int]] = []

        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"

        if not real_dir.is_dir():
            raise FileNotFoundError(
                f"Real directory not found: {real_dir}"
            )

        if not fake_dir.is_dir():
            raise FileNotFoundError(
                f"Fake directory not found: {fake_dir}"
            )

        real_sequences = sorted(
            path
            for path in real_dir.iterdir()
            if path.is_dir()
        )

        fake_sequences = sorted(
            path
            for path in fake_dir.iterdir()
            if path.is_dir()
        )

        # Real = 0
        for sequence_dir in real_sequences:
            self.samples.append((sequence_dir, 0))

        # Fake = 1
        for sequence_dir in fake_sequences:
            self.samples.append((sequence_dir, 1))

        if not self.samples:
            raise ValueError(
                "No source-video sequence folders found."
            )

        # ==================================================
        # 4. DISPLAY CONFIGURATION
        # ==================================================

        print("\n========== RPPG PREPROCESSING DATASET ==========")

        print(f"Root directory: {self.root_dir}")

        print(f"Real videos: {len(real_sequences)}")
        print(f"Fake videos: {len(fake_sequences)}")

        print(f"Total videos: {len(self.samples)}")

        print(f"Window duration: {self.window_seconds:.2f} seconds")
        print(f"Target FPS: {self.fps:.2f}")
        print(f"Target signal length: {self.target_length}")

        print(
            f"Minimum valid RGB: "
            f"{self.min_valid_rgb_fraction:.0%}"
        )

        print(f"Minimum valid frames: {self.min_rgb_frames}")

        print("================================================\n")

    # ======================================================
    # 5. DATASET LENGTH
    # ======================================================

    def __len__(self):
        return len(self.samples)

    # ======================================================
    # 6. LOAD METADATA
    # ======================================================

    @staticmethod
    def _load_metadata(
        sequence_dir: Path,
    ) -> List[Dict[str, Any]]:

        metadata_path = sequence_dir / "metadata.json"

        if not metadata_path.is_file():
            raise MetadataError(
                f"metadata.json not found: {metadata_path}"
            )

        try:
            with metadata_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:

            raise MetadataError(
                f"Could not read metadata: {metadata_path}. "
                f"Reason: {exc}"
            ) from exc

        if isinstance(data, list):

            records = data

        elif isinstance(data, dict):

            records = data.get(
                "frames",
                data.get(
                    "metadata",
                    data.get("records"),
                ),
            )

        else:
            records = None

        if not isinstance(records, list) or not records:
            raise MetadataError(
                f"No frame records found: {metadata_path}"
            )

        validated_records = []

        for position, record in enumerate(records):

            if not isinstance(record, dict):
                raise MetadataError(
                    f"Invalid record at position {position}."
                )

            if "frame_index" not in record:
                raise MetadataError(
                    f"Missing frame_index at position {position}."
                )

            if "timestamp" not in record:
                raise MetadataError(
                    f"Missing timestamp at position {position}."
                )

            frame_index = record["frame_index"]

            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index < 0
            ):
                raise MetadataError(
                    f"Invalid frame_index at position {position}."
                )

            try:
                timestamp = float(record["timestamp"])

            except (
                TypeError,
                ValueError,
                OverflowError,
            ) as exc:

                raise MetadataError(
                    f"Invalid timestamp at position {position}."
                ) from exc

            if not np.isfinite(timestamp):
                raise MetadataError(
                    f"Non-finite timestamp at position {position}."
                )

            validated_record = dict(record)
            validated_record["timestamp"] = timestamp

            validated_records.append(validated_record)

        validated_records.sort(
            key=lambda record: record["frame_index"]
        )

        frame_indices = [
            record["frame_index"]
            for record in validated_records
        ]

        if len(frame_indices) != len(set(frame_indices)):
            raise MetadataError(
                f"Duplicate frame indices: {metadata_path}"
            )

        for previous, current in zip(
            frame_indices,
            frame_indices[1:],
        ):

            if current != previous + 1:
                raise MetadataError(
                    f"Missing frame positions in {sequence_dir}. "
                    f"Found {previous} followed by {current}."
                )

        return validated_records

    # ======================================================
    # 7. ESTIMATE ACTUAL FPS
    # ======================================================

    def _estimate_fps(
        self,
        records: List[Dict[str, Any]],
        sequence_dir: Path,
    ) -> float:

        timestamps = np.asarray(
            [
                record["timestamp"]
                for record in records
            ],
            dtype=np.float64,
        )

        if len(timestamps) < 2:
            raise MetadataError(
                f"At least two timestamps are required: {sequence_dir}"
            )

        intervals = np.diff(timestamps)

        if not np.all(np.isfinite(intervals)):
            raise MetadataError(
                f"Invalid timestamp intervals: {sequence_dir}"
            )

        if np.any(intervals <= 0):
            raise MetadataError(
                f"Timestamps must be strictly increasing: {sequence_dir}"
            )

        median_interval = float(np.median(intervals))

        estimated_fps = 1.0 / median_interval

        relative_deviations = (
            np.abs(intervals - median_interval)
            / median_interval
        )

        max_deviation = float(
            np.max(relative_deviations)
        )

        if max_deviation > 0.10:
            raise MetadataError(
                f"Irregular timestamps: {sequence_dir}. "
                f"Maximum deviation: {max_deviation:.2%}"
            )

        if not np.isfinite(estimated_fps) or estimated_fps <= 0:
            raise MetadataError(
                f"Invalid estimated FPS: {sequence_dir}"
            )

        if self.debug:
            print(
                f"[FPS] {sequence_dir.name}: "
                f"{estimated_fps:.3f}"
            )

        return estimated_fps

    # ======================================================
    # 8. RESOLVE IMAGE PATH
    # ======================================================

    @staticmethod
    def _resolve_image_path(
        sequence_dir: Path,
        image_path: str,
    ) -> Path | None:

        if (
            not isinstance(image_path, str)
            or not image_path.strip()
        ):
            return None

        path = Path(image_path)

        if path.is_absolute():

            if (
                path.suffix.lower() in IMAGE_EXTENSIONS
                and path.is_file()
            ):
                return path.resolve()

            return None

        candidates = [
            Path.cwd() / path,
            sequence_dir / path,
            sequence_dir / path.name,
        ]

        for candidate in candidates:

            if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if candidate.is_file():
                return candidate.resolve()

        return None

    # ======================================================
    # 9. EXTRACT RGB FROM ONE CROPPED FACE
    # ======================================================

    def _extract_single_rgb(
        self,
        rgb_extractor: RGBExtractor,
        face_image: np.ndarray,
    ) -> np.ndarray:

        if (
            not isinstance(face_image, np.ndarray)
            or face_image.size == 0
            or face_image.ndim != 3
            or face_image.shape[2] != 3
        ):
            raise ValueError(
                "Invalid cropped face image."
            )

        if not hasattr(
            rgb_extractor,
            "process_cropped_face",
        ):
            raise AttributeError(
                "RGBExtractor must implement "
                "process_cropped_face()."
            )

        rgb = rgb_extractor.process_cropped_face(
            face_image
        )

        rgb = np.asarray(
            rgb,
            dtype=np.float32,
        )

        if rgb.shape != (3,):
            raise ValueError(
                f"Expected RGB shape (3,), got {rgb.shape}."
            )

        if not np.all(np.isfinite(rgb)):
            raise ValueError(
                "RGB measurement contains invalid values."
            )

        return rgb

    # ======================================================
    # 10. EXTRACT RGB FROM COMPLETE VIDEO SEQUENCE
    # ======================================================

    def _extract_rgb_from_sequence(
        self,
        sequence_dir: Path,
        records: List[Dict[str, Any]],
    ) -> np.ndarray:

        rgb_values = []

        stats = {
            "total_frames": len(records),
            "images_loaded": 0,
            "missing_image_paths": 0,
            "unreadable_images": 0,
            "invalid_rgb": 0,
            "valid_rgb": 0,
        }

        rgb_extractor = None

        try:

            rgb_extractor = RGBExtractor(
                min_roi_width=5,
                min_roi_height=5,
                min_skin_pixels=1,
                debug=False,
            )

            for position, record in enumerate(records):

                image_path = record.get("image_path")
                detected = record.get("detected", False)

                # A tracker prediction is not valid RGB.
                # Only an explicitly detected face is accepted.
                if detected is not True or not image_path:

                    rgb_values.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["missing_image_paths"] += 1

                    continue

                resolved_path = self._resolve_image_path(
                    sequence_dir,
                    image_path,
                )

                if resolved_path is None:

                    rgb_values.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["missing_image_paths"] += 1

                    continue

                face_image = cv2.imread(
                    str(resolved_path),
                    cv2.IMREAD_COLOR,
                )

                if (
                    face_image is None
                    or face_image.size == 0
                ):

                    rgb_values.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["unreadable_images"] += 1

                    continue

                stats["images_loaded"] += 1

                try:

                    rgb = self._extract_single_rgb(
                        rgb_extractor,
                        face_image,
                    )

                except Exception as exc:

                    rgb_values.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["invalid_rgb"] += 1

                    if self.debug:
                        print(
                            f"[INVALID RGB] "
                            f"Frame {record['frame_index']}: {exc}"
                        )

                    continue

                rgb_values.append(rgb)

                stats["valid_rgb"] += 1

                if (
                    self.debug
                    and (position + 1) % 100 == 0
                ):
                    print(
                        f"[RGB PROGRESS] "
                        f"{position + 1}/{len(records)}"
                    )

        finally:

            if (
                rgb_extractor is not None
                and hasattr(rgb_extractor, "close")
            ):
                rgb_extractor.close()

        rgb_signals = np.asarray(
            rgb_values,
            dtype=np.float32,
        )

        if rgb_signals.shape != (len(records), 3):
            raise RuntimeError(
                "RGB output does not match metadata length. "
                f"Expected {(len(records), 3)}, "
                f"received {rgb_signals.shape}."
            )

        print("\n========== RGB EXTRACTION SUMMARY ==========")

        print(f"Sequence: {sequence_dir.name}")

        for name, value in stats.items():
            print(f"{name}: {value}")

        valid_fraction = (
            stats["valid_rgb"] / stats["total_frames"]
        )

        print(f"Valid RGB percentage: {valid_fraction:.2%}")

        print("============================================")

        return rgb_signals

    # ======================================================
    # 11. CHECK RGB QUALITY FOR ONE WINDOW
    # ======================================================

    def _diagnose_rgb(
        self,
        rgb_signals: np.ndarray,
        sequence_dir: Path,
    ) -> float:

        if (
            not isinstance(rgb_signals, np.ndarray)
            or rgb_signals.ndim != 2
            or rgb_signals.shape[1] != 3
        ):
            raise VideoQualityError(
                f"Invalid RGB array: {sequence_dir}"
            )

        total_frames = len(rgb_signals)

        if total_frames == 0:
            raise VideoQualityError(
                f"Empty RGB window: {sequence_dir}"
            )

        valid_rows = np.all(
            np.isfinite(rgb_signals),
            axis=1,
        )

        valid_count = int(
            np.count_nonzero(valid_rows)
        )

        valid_fraction = valid_count / total_frames

        # Find consecutive missing-frame groups.
        missing_runs = []
        run_start = None

        for index, is_valid in enumerate(valid_rows):

            if not is_valid and run_start is None:
                run_start = index

            elif is_valid and run_start is not None:

                missing_runs.append(
                    (run_start, index - 1)
                )

                run_start = None

        if run_start is not None:
            missing_runs.append(
                (run_start, total_frames - 1)
            )

        print("\n========== RGB WINDOW QUALITY ==========")

        print(f"Window: {sequence_dir.name}")
        print(f"Total frames: {total_frames}")
        print(f"Valid RGB frames: {valid_count}")

        print(
            f"Valid RGB percentage: {valid_fraction:.2%}"
        )

        print(f"Missing frame groups: {missing_runs}")

        print("========================================")

        if valid_fraction < self.min_valid_rgb_fraction:

            raise VideoQualityError(
                f"Insufficient valid RGB: {sequence_dir}. "
                f"Got {valid_fraction:.2%}; "
                f"required {self.min_valid_rgb_fraction:.2%}."
            )

        if valid_count < self.min_rgb_frames:

            raise VideoQualityError(
                f"Too few valid RGB frames: {sequence_dir}. "
                f"Got {valid_count}; "
                f"required {self.min_rgb_frames}."
            )

        return float(valid_fraction)

    # ======================================================
    # 12. EXTRACT POS SIGNAL
    # ======================================================

    def _extract_pos_signal(
        self,
        rgb_signals: np.ndarray,
        sequence_fps: float,
        sequence_dir: Path,
    ) -> np.ndarray:

        try:

            rppg_wave = pos_method(
                rgb_signals,
                fps=sequence_fps,
            )

            rppg_wave = np.asarray(
                rppg_wave,
                dtype=np.float32,
            )

        except Exception as exc:

            raise RuntimeError(
                f"POS processing failed: {sequence_dir}. "
                f"Reason: {exc}"
            ) from exc

        if rppg_wave.ndim != 1:
            raise ValueError(
                f"POS output must be 1D, got {rppg_wave.shape}."
            )

        if len(rppg_wave) == 0:
            raise ValueError(
                f"POS output is empty: {sequence_dir}"
            )

        if not np.all(np.isfinite(rppg_wave)):
            raise ValueError(
                f"POS contains NaN or infinite values: {sequence_dir}"
            )

        return rppg_wave

    # ======================================================
    # 13. FORMAT ONE 10-SECOND SIGNAL
    # ======================================================

    def _format_signal(
        self,
        rppg_wave: np.ndarray,
        sequence_fps: float,
        sequence_dir: Path,
    ) -> np.ndarray:

        rppg_wave = np.asarray(
            rppg_wave,
            dtype=np.float32,
        )

        if rppg_wave.ndim != 1:
            raise ValueError(
                "POS waveform must be one-dimensional."
            )

        if not np.all(np.isfinite(rppg_wave)):
            raise VideoQualityError(
                f"POS waveform contains invalid values: {sequence_dir}"
            )

        expected_frames = int(
            round(self.window_seconds * sequence_fps)
        )

        if len(rppg_wave) != expected_frames:
            raise VideoQualityError(
                f"Unexpected POS length: {sequence_dir}. "
                f"Expected {expected_frames}, got {len(rppg_wave)}."
            )

        if len(rppg_wave) < self.min_rgb_frames:
            raise VideoQualityError(
                f"POS waveform too short: {sequence_dir}"
            )

        # Each waveform represents approximately 10 seconds.
        # Only this individual window is resampled.
        formatted_signal = resample(
            rppg_wave,
            self.target_length,
        )

        formatted_signal = np.asarray(
            formatted_signal,
            dtype=np.float32,
        )

        if not np.all(np.isfinite(formatted_signal)):
            raise ValueError(
                f"Invalid formatted signal: {sequence_dir}"
            )

        if np.std(formatted_signal) < 1e-8:
            raise VideoQualityError(
                f"POS waveform has almost no variation: {sequence_dir}"
            )

        if self.debug:

            print("\n========== SIGNAL FORMATTING ==========")

            print(f"Original samples: {len(rppg_wave)}")
            print(f"Actual FPS: {sequence_fps:.3f}")

            print(
                f"Window duration: {self.window_seconds:.2f}s"
            )

            print(
                f"Formatted samples: {len(formatted_signal)}"
            )

            print("=======================================")

        return np.ascontiguousarray(
            formatted_signal,
            dtype=np.float32,
        )

    # ======================================================
    # 14. PROCESS ONE 10-SECOND WINDOW
    # ======================================================

    def _process_window(
        self,
        rgb_window: np.ndarray,
        sequence_fps: float,
        sequence_dir: Path,
        window_index: int,
    ) -> np.ndarray:

        window_name = (
            f"{sequence_dir.name}_window_{window_index + 1:03d}"
        )

        window_dir = sequence_dir / window_name

        # 1. Validate RGB quality for this window.
        self._diagnose_rgb(
            rgb_window,
            window_dir,
        )

        # 2. Generate POS waveform.
        pos_wave = self._extract_pos_signal(
            rgb_window,
            sequence_fps,
            window_dir,
        )

        # 3. Convert to exactly 300 samples.
        formatted_signal = self._format_signal(
            pos_wave,
            sequence_fps,
            window_dir,
        )

        return formatted_signal

    # ======================================================
    # 15. PROCESS COMPLETE SOURCE VIDEO
    # ======================================================

    def process_sequence(
        self,
        sequence_dir: Path,
        label: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        print(
            f"\n[PROCESSING VIDEO] {sequence_dir.name}"
        )

        # ----------------------------------------------
        # A. Load metadata
        # ----------------------------------------------

        records = self._load_metadata(
            sequence_dir
        )

        print(
            f"[METADATA] Total frames: {len(records)}"
        )

        # ----------------------------------------------
        # B. Estimate actual FPS
        # ----------------------------------------------

        sequence_fps = self._estimate_fps(
            records,
            sequence_dir,
        )

        # ----------------------------------------------
        # C. Extract RGB for all frames
        # ----------------------------------------------

        rgb_signals = self._extract_rgb_from_sequence(
            sequence_dir,
            records,
        )

        # ----------------------------------------------
        # D. Calculate frames per window
        # ----------------------------------------------

        frames_per_window = int(
            round(self.window_seconds * sequence_fps)
        )

        if frames_per_window < self.min_rgb_frames:
            raise VideoQualityError(
                f"Window is too short: {sequence_dir}"
            )

        total_windows = (
            len(rgb_signals) // frames_per_window
        )

        if total_windows == 0:
            raise VideoQualityError(
                f"Video is shorter than "
                f"{self.window_seconds} seconds: {sequence_dir}"
            )

        print(f"[FPS] {sequence_fps:.3f}")
        print(f"[WINDOW] Duration: {self.window_seconds:.2f}s")
        print(f"[WINDOW] Frames: {frames_per_window}")
        print(f"[WINDOW] Available complete windows: {total_windows}")

        # ----------------------------------------------
        # E. Process each window independently
        # ----------------------------------------------

        formatted_windows = []

        rejected_windows = 0

        for window_index in range(total_windows):

            start = window_index * frames_per_window
            end = start + frames_per_window

            rgb_window = rgb_signals[start:end]

            print(
                f"\n[WINDOW {window_index + 1}/{total_windows}] "
                f"Frames {start}-{end - 1}"
            )

            try:

                formatted_signal = self._process_window(
                    rgb_window=rgb_window,
                    sequence_fps=sequence_fps,
                    sequence_dir=sequence_dir,
                    window_index=window_index,
                )

                formatted_windows.append(
                    formatted_signal
                )

                print(
                    f"[WINDOW ACCEPTED] "
                    f"{len(formatted_signal)} samples"
                )

            except VideoQualityError as exc:

                rejected_windows += 1

                print(
                    f"[WINDOW REJECTED] {exc}"
                )

            except Exception as exc:

                rejected_windows += 1

                print(
                    f"[WINDOW ERROR] {exc}"
                )

        # ----------------------------------------------
        # F. Ensure at least one valid window
        # ----------------------------------------------

        if not formatted_windows:
            raise VideoQualityError(
                f"No valid 10-second windows were produced: "
                f"{sequence_dir}. "
                f"Rejected windows: {rejected_windows}."
            )

        # ----------------------------------------------
        # G. Stack accepted signals
        # ----------------------------------------------

        signals_array = np.stack(
            formatted_windows,
            axis=0,
        ).astype(np.float32)

        labels_array = np.full(
            len(formatted_windows),
            fill_value=label,
            dtype=np.float32,
        )

        x_tensor = torch.from_numpy(
            signals_array
        )

        y_tensor = torch.from_numpy(
            labels_array
        )

        # ----------------------------------------------
        # H. Final validation
        # ----------------------------------------------

        if x_tensor.ndim != 2:
            raise RuntimeError(
                f"Unexpected signal tensor shape: {x_tensor.shape}"
            )

        if x_tensor.shape[1] != self.target_length:
            raise RuntimeError(
                f"Expected signal length {self.target_length}, "
                f"got {x_tensor.shape[1]}."
            )

        if not torch.isfinite(x_tensor).all():
            raise RuntimeError(
                "Processed signals contain invalid values."
            )

        if len(x_tensor) != len(y_tensor):
            raise RuntimeError(
                "Signal and label counts do not match."
            )

        print("\n========== VIDEO PROCESSING RESULT ==========")

        print(f"Video: {sequence_dir.name}")
        print(f"FPS: {sequence_fps:.3f}")

        print(f"Complete windows: {total_windows}")
        print(f"Accepted windows: {len(formatted_windows)}")
        print(f"Rejected windows: {rejected_windows}")

        print(f"Signals shape: {tuple(x_tensor.shape)}")
        print(f"Labels shape: {tuple(y_tensor.shape)}")

        print(f"Label: {label}")

        print("==============================================")

        return x_tensor, y_tensor

    # ======================================================
    # 16. LOAD ONE VIDEO
    # ======================================================

    def __getitem__(self, idx):

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

        sequence_dir, label = self.samples[idx]

        try:

            return self.process_sequence(
                sequence_dir,
                label,
            )

        except Exception as exc:

            raise RuntimeError(
                f"Video processing failed.\n"
                f"Sequence: {sequence_dir}\n"
                f"Label: {label}\n"
                f"Reason: {exc}"
            ) from exc

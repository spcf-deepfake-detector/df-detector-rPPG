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
    """Raised when a sequence cannot produce a reliable rPPG signal."""


class MetadataError(ValueError):
    """Raised when a sequence's metadata is invalid."""


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
# RPPG DATASET
# ============================================================

class RPPGDataset(Dataset):
    """
    FALSEFACET rPPG Dataset.

    Processing pipeline:

        Source-video folder
            -> metadata.json
            -> Cropped face images
            -> RGB extraction
            -> RGB validation
            -> POS
            -> Fixed-length signal
            -> PyTorch tensors

    Each source-video folder represents ONE sample.

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
    """

    def __init__(
        self,
        root_dir: str,
        target_length: int = 300,
        fps: float = 30.0,
        min_valid_rgb_fraction: float = 0.80,
        min_rgb_frames: int = 10,
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
                "fps must be finite and positive."
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

        self.root_dir = Path(root_dir).resolve()

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        self.target_length = target_length
        self.fps = float(fps)

        self.min_valid_rgb_fraction = float(
            min_valid_rgb_fraction
        )

        self.min_rgb_frames = min_rgb_frames
        self.debug = bool(debug)

        # Example:
        # 300 samples / 30 FPS = 10 seconds.

        self.target_duration = (
            self.target_length / self.fps
        )

        # ==================================================
        # STEP 2: FIND SOURCE-VIDEO SEQUENCES
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
            self.samples.append(
                (sequence_dir, 0)
            )

        # Fake = 1
        for sequence_dir in fake_sequences:
            self.samples.append(
                (sequence_dir, 1)
            )

        if not self.samples:
            raise ValueError(
                "No source-video sequence folders found."
            )

        # ==================================================
        # STEP 3: DISPLAY CONFIGURATION
        # ==================================================

        print("\n========== RPPG DATASET ==========")

        print(f"Root directory: {self.root_dir}")
        print(f"Real sequences: {len(real_sequences)}")
        print(f"Fake sequences: {len(fake_sequences)}")
        print(f"Total sequences: {len(self.samples)}")

        print(f"Target signal length: {self.target_length}")
        print(f"Reference FPS: {self.fps}")

        print(
            f"Target duration: "
            f"{self.target_duration:.2f} seconds"
        )

        print(
            f"Minimum valid RGB fraction: "
            f"{self.min_valid_rgb_fraction:.1%}"
        )

        print(
            f"Minimum valid RGB frames: "
            f"{self.min_rgb_frames}"
        )

        print("==================================\n")

    # ======================================================
    # STEP 4: DATASET LENGTH
    # ======================================================

    def __len__(self):
        """Return the number of source-video sequences."""

        return len(self.samples)

    # ======================================================
    # STEP 5: LOAD METADATA
    # ======================================================

    @staticmethod
    def _load_metadata(
        sequence_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Load metadata.json.

        Supported structures:

        1. Direct list:
           [ {...}, {...} ]

        2. Dictionary:
           {"frames": [...]}

        3. Dictionary:
           {"metadata": [...]}

        4. Dictionary:
           {"records": [...]}
        """

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
                f"Could not read metadata: {metadata_path}\n"
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

        # Validate records.

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

        # Sort by original frame position.

        records = sorted(
            records,
            key=lambda record: record["frame_index"],
        )

        frame_indices = [
            record["frame_index"]
            for record in records
        ]

        if len(frame_indices) != len(set(frame_indices)):
            raise MetadataError(
                f"Duplicate frame indices: {metadata_path}"
            )

        # Do not silently discard missing frame positions.

        for previous, current in zip(
            frame_indices,
            frame_indices[1:],
        ):

            if current != previous + 1:
                raise MetadataError(
                    f"Missing metadata frame positions in "
                    f"{sequence_dir}. "
                    f"Found {previous} followed by {current}."
                )

        return records

    # ======================================================
    # STEP 6: ESTIMATE FPS
    # ======================================================

    def _estimate_fps(
        self,
        records: List[Dict[str, Any]],
        sequence_dir: Path,
    ) -> float:
        """
        Estimate FPS using metadata timestamps.

        Uses the median timestamp interval.
        """

        timestamps = np.asarray(
            [
                float(record["timestamp"])
                for record in records
            ],
            dtype=np.float64,
        )

        if len(timestamps) < 2:
            raise MetadataError(
                f"At least two timestamps are required: "
                f"{sequence_dir}"
            )

        intervals = np.diff(timestamps)

        if not np.all(np.isfinite(intervals)):
            raise MetadataError(
                f"Invalid timestamp intervals: {sequence_dir}"
            )

        if np.any(intervals <= 0):
            raise MetadataError(
                f"Timestamps must be strictly increasing: "
                f"{sequence_dir}"
            )

        median_interval = float(
            np.median(intervals)
        )

        estimated_fps = 1.0 / median_interval

        # Check timestamp regularity.

        relative_deviations = (
            np.abs(intervals - median_interval)
            / median_interval
        )

        max_deviation = float(
            np.max(relative_deviations)
        )

        if max_deviation > 0.10:
            raise MetadataError(
                f"Irregular timestamp intervals: {sequence_dir}\n"
                f"Median interval: {median_interval:.6f}s\n"
                f"Maximum deviation: {max_deviation:.2%}"
            )

        if self.debug:
            print(
                f"[FPS] {sequence_dir.name}: "
                f"{estimated_fps:.3f}"
            )

        return float(estimated_fps)

    # ======================================================
    # STEP 7: RESOLVE IMAGE PATH
    # ======================================================

    @staticmethod
    def _resolve_image_path(
        sequence_dir: Path,
        image_path: str,
    ) -> Path | None:
        """
        Resolve paths stored in metadata.json.

        Supports:

        1. Absolute paths.
        2. Project-root-relative paths.
        3. Sequence-folder-relative paths.
        4. Image filenames directly inside the sequence folder.

        Your current metadata uses project-root-relative paths.

        Example:
            extracted_faces/test/real/1/frame_000000.jpg
        """

        if not image_path or not isinstance(image_path, str):
            return None

        path = Path(image_path)

        # Absolute path.

        if path.is_absolute():

            if (
                path.suffix.lower() in IMAGE_EXTENSIONS
                and path.is_file()
            ):
                return path

            return None

        # Relative path candidates.

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
    # STEP 8: EXTRACT RGB FROM ONE CROPPED FACE
    # ======================================================

    def _extract_single_rgb(
        self,
        rgb_extractor: RGBExtractor,
        face_image: np.ndarray,
    ) -> np.ndarray:
        """
        Extract RGB from one cropped face image.

        Requires RGBExtractor.process_cropped_face() to return
        a three-element RGB measurement.

        Expected output:
            [R, G, B]
        """

        if (
            not isinstance(face_image, np.ndarray)
            or face_image.size == 0
            or face_image.ndim != 3
            or face_image.shape[2] != 3
        ):
            raise ValueError(
                "Invalid cropped face image."
            )

        if not hasattr(rgb_extractor, "process_cropped_face"):
            raise AttributeError(
                "RGBExtractor does not have process_cropped_face(). "
                "Add this method to preprocessing/rgb_extraction.py "
                "before running the cropped-face pipeline."
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
                f"Expected RGB shape (3,), received {rgb.shape}."
            )

        if not np.all(np.isfinite(rgb)):
            raise ValueError(
                "RGB measurement contains NaN or infinite values."
            )

        return rgb

    # ======================================================
    # STEP 9: PROCESS CROPPED FACE IMAGES
    # ======================================================

    def _extract_rgb_from_sequence(
        self,
        sequence_dir: Path,
        records: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Extract RGB measurements for every metadata record.

        Missing or unreadable images remain in their original
        temporal positions as:

            [NaN, NaN, NaN]

        This prevents temporal compression.
        """

        rgb_extractor = None

        # IMPORTANT:
        # Initialize the list and counters BEFORE the loop.

        rgb_values = []

        stats = {
            "total_frames": len(records),
            "images_loaded": 0,
            "missing_image_paths": 0,
            "unreadable_images": 0,
            "invalid_rgb": 0,
            "valid_rgb": 0,
        }

        try:

            # One extractor for the entire sequence.

            rgb_extractor = RGBExtractor(
                min_roi_width=5,
                min_roi_height=5,
                min_skin_pixels=1,
                debug=False,
            )

            # Process metadata in chronological order.

            for position, record in enumerate(records):

                image_path = record.get("image_path")
                detected = record.get("detected", False)

                # ------------------------------------------
                # A. Missing face detection or image path
                # ------------------------------------------

                if not detected or not image_path:

                    rgb_values.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["missing_image_paths"] += 1

                    continue

                # ------------------------------------------
                # B. Resolve the image path
                # ------------------------------------------

                resolved_path = self._resolve_image_path(
                    sequence_dir,
                    image_path,
                )

                if resolved_path is None:

                    rgb_values.append(
                        [np.nan, np.nan, np.nan]
                    )

                    stats["missing_image_paths"] += 1

                    if self.debug:
                        print(
                            f"[MISSING IMAGE] "
                            f"Frame {record['frame_index']}: "
                            f"{image_path}"
                        )

                    continue

                # ------------------------------------------
                # C. Read the cropped face image
                # ------------------------------------------

                try:

                    face_image = cv2.imread(
                        str(resolved_path),
                        cv2.IMREAD_COLOR,
                    )

                except Exception as exc:

                    face_image = None

                    if self.debug:
                        print(
                            f"[IMAGE ERROR] "
                            f"Frame {record['frame_index']}: "
                            f"{exc}"
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

                # ------------------------------------------
                # D. Extract RGB measurement
                # ------------------------------------------

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
                            f"Frame {record['frame_index']}: "
                            f"{exc}"
                        )

                    continue

                # ------------------------------------------
                # E. Store RGB measurement
                # ------------------------------------------

                rgb_values.append(rgb)

                stats["valid_rgb"] += 1

                # ------------------------------------------
                # F. Progress
                # ------------------------------------------

                if (
                    self.debug
                    and (position + 1) % 100 == 0
                ):

                    print(
                        f"[RGB PROGRESS] "
                        f"{position + 1}/{len(records)} frames"
                    )

        finally:

            # Release RGB extractor resources if supported.

            if (
                rgb_extractor is not None
                and hasattr(rgb_extractor, "close")
            ):
                rgb_extractor.close()

        # Convert the list into a NumPy array.

        rgb_signals = np.asarray(
            rgb_values,
            dtype=np.float32,
        )

        # Every metadata record must have one RGB row.

        if rgb_signals.shape != (len(records), 3):
            raise RuntimeError(
                "RGB sequence length does not match metadata.\n"
                f"Expected: ({len(records)}, 3)\n"
                f"Received: {rgb_signals.shape}"
            )

        # ----------------------------------------------
        # Print extraction summary
        # ----------------------------------------------

        print(
            "\n========== RGB EXTRACTION SUMMARY =========="
        )

        print(f"Sequence: {sequence_dir.name}")

        for name, value in stats.items():
            print(f"{name}: {value}")

        if stats["total_frames"] > 0:

            valid_fraction = (
                stats["valid_rgb"] / stats["total_frames"]
            )

            print(
                f"Valid RGB percentage: {valid_fraction:.2%}"
            )

        print("============================================")

        return rgb_signals

    # ======================================================
    # STEP 10: RGB QUALITY DIAGNOSTICS
    # ======================================================

    def _diagnose_rgb(
        self,
        rgb_signals: np.ndarray,
        sequence_dir: Path,
    ) -> float:
        """
        Validate RGB measurements.

        A valid row must contain three finite values.
        """

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
                f"No RGB measurements: {sequence_dir}"
            )

        valid_rows = np.all(
            np.isfinite(rgb_signals),
            axis=1,
        )

        valid_count = int(
            np.count_nonzero(valid_rows)
        )

        missing_count = total_frames - valid_count

        valid_fraction = valid_count / total_frames

        # ----------------------------------------------
        # Find consecutive missing frame groups
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Print quality summary
        # ----------------------------------------------

        print("\n========== RGB QUALITY ==========")

        print(f"Sequence: {sequence_dir.name}")
        print(f"Total frames: {total_frames}")
        print(f"Valid RGB frames: {valid_count}")
        print(f"Missing RGB frames: {missing_count}")

        print(
            f"Valid RGB percentage: {valid_fraction:.2%}"
        )

        print(f"Missing frame groups: {missing_runs}")

        print("==================================")

        # ----------------------------------------------
        # Enforce minimum quality requirements
        # ----------------------------------------------

        if valid_fraction < self.min_valid_rgb_fraction:

            raise VideoQualityError(
                "Insufficient valid RGB measurements.\n"
                f"Sequence: {sequence_dir}\n"
                f"Valid: {valid_count}/{total_frames}\n"
                f"Valid percentage: {valid_fraction:.2%}\n"
                f"Required: {self.min_valid_rgb_fraction:.2%}"
            )

        if valid_count < self.min_rgb_frames:

            raise VideoQualityError(
                "Too few valid RGB measurements.\n"
                f"Sequence: {sequence_dir}\n"
                f"Valid frames: {valid_count}\n"
                f"Required: {self.min_rgb_frames}"
            )

        return float(valid_fraction)

    # ======================================================
    # STEP 11: EXTRACT POS SIGNAL
    # ======================================================

    def _extract_pos_signal(
        self,
        rgb_signals: np.ndarray,
        sequence_fps: float,
        sequence_dir: Path,
    ) -> np.ndarray:
        """
        Convert RGB measurements into a POS waveform.

        Uses the actual FPS estimated from timestamps.
        """

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
                "POS processing failed.\n"
                f"Sequence: {sequence_dir}\n"
                f"Reason: {exc}"
            ) from exc

        if rppg_wave.ndim != 1:

            raise ValueError(
                f"POS output must be one-dimensional. "
                f"Received: {rppg_wave.shape}"
            )

        if len(rppg_wave) == 0:

            raise ValueError(
                f"POS produced an empty waveform: {sequence_dir}"
            )

        if not np.all(np.isfinite(rppg_wave)):

            raise ValueError(
                f"POS produced NaN or infinite values: "
                f"{sequence_dir}"
            )

        return rppg_wave

    # ======================================================
    # STEP 12: FORMAT FIXED-LENGTH SIGNAL
    # ======================================================

    def _format_signal(
        self,
        rppg_wave: np.ndarray,
        sequence_fps: float,
        sequence_dir: Path,
    ) -> np.ndarray:
        """
        Convert a POS waveform into the model's expected length.

        Shorter sequences are allowed and resampled.

        No zero-padding is performed.
        """

        if len(rppg_wave) < self.min_rgb_frames:
            raise VideoQualityError(
                f"POS waveform is too short: {sequence_dir}\n"
                f"Available samples: {len(rppg_wave)}\n"
                f"Minimum required: {self.min_rgb_frames}"
            )

        # Calculate actual sequence duration.
        actual_duration = len(rppg_wave) / sequence_fps

        # Resample the complete waveform to the target length.
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
                f"Invalid values after resampling: {sequence_dir}"
            )

        if self.debug:
            print("\n========== SIGNAL RESAMPLING ==========")
            print(f"Original samples: {len(rppg_wave)}")
            print(f"Actual FPS: {sequence_fps:.3f}")
            print(f"Actual duration: {actual_duration:.2f}s")
            print(f"Target samples: {self.target_length}")
            print("========================================")

        return np.ascontiguousarray(
            formatted_signal,
            dtype=np.float32,
        )

    # ======================================================
    # STEP 13: PROCESS ONE SOURCE-VIDEO SEQUENCE
    # ======================================================

    def process_sequence(
        self,
        sequence_dir: Path,
        label: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one source-video sequence.
        """

        print(
            f"\n[PROCESSING SEQUENCE] {sequence_dir.name}"
        )

        # 1. Load metadata.

        records = self._load_metadata(
            sequence_dir
        )

        print(
            f"[METADATA] Frame records: {len(records)}"
        )

        # 2. Estimate actual FPS.

        sequence_fps = self._estimate_fps(
            records,
            sequence_dir,
        )

        # 3. Extract RGB measurements.

        rgb_signals = self._extract_rgb_from_sequence(
            sequence_dir,
            records,
        )

        # 4. Validate RGB quality.

        valid_fraction = self._diagnose_rgb(
            rgb_signals,
            sequence_dir,
        )

        # 5. Generate POS waveform.

        rppg_wave = self._extract_pos_signal(
            rgb_signals,
            sequence_fps,
            sequence_dir,
        )

        # 6. Format fixed-length signal.

        rppg_wave = self._format_signal(
            rppg_wave,
            sequence_fps,
            sequence_dir,
        )

        # 7. Create PyTorch tensors.

        x_tensor = torch.from_numpy(
            rppg_wave
        ).to(dtype=torch.float32)

        y_tensor = torch.tensor(
            label,
            dtype=torch.float32,
        )

        if self.debug:

            print("\n========== SEQUENCE RESULT ==========")

            print(f"Sequence: {sequence_dir.name}")
            print(f"FPS: {sequence_fps:.3f}")
            print(
                f"Valid RGB fraction: {valid_fraction:.2%}"
            )
            print(
                f"POS waveform length: {len(rppg_wave)}"
            )
            print(
                f"Tensor shape: {tuple(x_tensor.shape)}"
            )
            print(f"Label: {label}")

            print("=====================================")

        return x_tensor, y_tensor

    # ======================================================
    # STEP 14: LOAD ONE DATASET SAMPLE
    # ======================================================

    def __getitem__(self, idx):
        """
        Return one source-video sequence.

        Returns:
            x_tensor:
                Float32 POS waveform.
                Shape: (target_length,)

            y_tensor:
                Float32 label.

                Real = 0
                Fake = 1
        """

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
                "Sequence processing failed.\n"
                f"Sequence: {sequence_dir}\n"
                f"Label: {label}\n"
                f"Reason: {exc}"
            ) from exc
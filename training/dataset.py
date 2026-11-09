import numpy as np
import torch

from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset


class RPPGDataset(Dataset):
    """
    FALSEFACET rPPG Dataset.

    Loads preprocessed one-dimensional rPPG signals.

    Expected structure:

        processed/
        ├── real/
        │   ├── video_001.npy
        │   └── video_002.npy
        │
        └── fake/
            ├── video_003.npy
            └── video_004.npy

    Labels:
        Real = 0
        Fake = 1

    Expected signal shape:
        (target_length,)
    """

    def __init__(
        self,
        root_dir: str,
        target_length: int = 300,
        debug: bool = False,
    ):

        # ==============================================
        # STEP 1: VALIDATE CONFIGURATION
        # ==============================================

        if (
            not isinstance(target_length, int)
            or isinstance(target_length, bool)
            or target_length < 8
        ):
            raise ValueError(
                "target_length must be an integer >= 8."
            )

        self.root_dir = Path(root_dir).resolve()
        self.target_length = target_length
        self.debug = debug

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        # ==============================================
        # STEP 2: FIND REAL AND FAKE SIGNALS
        # ==============================================

        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"

        if not real_dir.is_dir():
            raise FileNotFoundError(
                f"Real dataset directory not found: {real_dir}"
            )

        if not fake_dir.is_dir():
            raise FileNotFoundError(
                f"Fake dataset directory not found: {fake_dir}"
            )

        real_signals = sorted(
            real_dir.rglob("*.npy")
        )

        fake_signals = sorted(
            fake_dir.rglob("*.npy")
        )

        # ==============================================
        # STEP 3: CREATE SAMPLE LIST
        # ==============================================

        self.samples: List[Tuple[Path, int]] = []

        for signal_path in real_signals:
            self.samples.append((signal_path, 0))

        for signal_path in fake_signals:
            self.samples.append((signal_path, 1))

        if not self.samples:
            raise ValueError(
                f"No NPY signals found in {self.root_dir}"
            )

        if not real_signals:
            raise ValueError(
                f"No real signals found in {real_dir}"
            )

        if not fake_signals:
            raise ValueError(
                f"No fake signals found in {fake_dir}"
            )

        # ==============================================
        # STEP 4: DISPLAY DATASET INFORMATION
        # ==============================================

        print("\n========== RPPG DATASET ==========")

        print(f"Root directory: {self.root_dir}")

        print(f"Real signals: {len(real_signals)}")
        print(f"Fake signals: {len(fake_signals)}")

        print(f"Total signals: {len(self.samples)}")

        print(f"Target length: {self.target_length}")

        print("Labels: Real=0, Fake=1")
        print("Format: NumPy (.npy)")

        print("==================================\n")

    # ==============================================
    # DATASET LENGTH
    # ==============================================

    def __len__(self):
        return len(self.samples)

    # ==============================================
    # LOAD ONE SIGNAL
    # ==============================================

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
                f"Index {idx} is out of range."
            )

        signal_path, label = self.samples[idx]

        # ------------------------------------------
        # LOAD SIGNAL
        # ------------------------------------------

        try:
            signal = np.load(
                signal_path,
                allow_pickle=False,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Could not load signal: {signal_path}\n"
                f"Reason: {exc}"
            ) from exc

        # ------------------------------------------
        # VALIDATE DIMENSIONS
        # ------------------------------------------

        if signal.ndim != 1:
            raise ValueError(
                f"Signal must be one-dimensional.\n"
                f"File: {signal_path}\n"
                f"Shape: {signal.shape}"
            )

        if len(signal) != self.target_length:
            raise ValueError(
                f"Incorrect signal length.\n"
                f"File: {signal_path}\n"
                f"Expected: {self.target_length}\n"
                f"Received: {len(signal)}"
            )

        # ------------------------------------------
        # VALIDATE DATA TYPE
        # ------------------------------------------

        if not np.issubdtype(
            signal.dtype,
            np.number,
        ):
            raise ValueError(
                f"Signal must be numeric: {signal_path}"
            )

        # ------------------------------------------
        # VALIDATE FINITE VALUES
        # ------------------------------------------

        if not np.all(np.isfinite(signal)):
            raise ValueError(
                f"Signal contains NaN or Inf: {signal_path}"
            )

        # ------------------------------------------
        # CONVERT TO FLOAT32
        # ------------------------------------------

        signal = np.ascontiguousarray(
            signal,
            dtype=np.float32,
        )

        # ------------------------------------------
        # CONVERT TO PYTORCH
        # ------------------------------------------

        x_tensor = torch.from_numpy(signal)

        y_tensor = torch.tensor(
            label,
            dtype=torch.float32,
        )

        if self.debug:
            print(
                f"[LOADED] {signal_path.name} "
                f"| Label: {label} "
                f"| Shape: {tuple(x_tensor.shape)}"
            )

        return x_tensor, y_tensor
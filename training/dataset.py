import os
import glob
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class RPPGDataset(Dataset):
    """
    FALSEFACET rPPG Dataset.

    Loads previously processed rPPG signals.

    Pipeline:
        .npy signal
          -> Validate signal
          -> Convert to PyTorch tensor
          -> Attach label
          -> Return sample

    Labels:
        Real = 0
        Fake = 1
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
            or target_length < 1
        ):
            raise ValueError(
                "target_length must be a positive integer."
            )

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {root_dir}"
            )

        self.root_dir = os.path.abspath(root_dir)
        self.target_length = target_length
        self.debug = debug

        # ==============================================
        # STEP 2: FIND PROCESSED SIGNALS
        # ==============================================

        real_dir = os.path.join(self.root_dir, "real")
        fake_dir = os.path.join(self.root_dir, "fake")

        real_signals = sorted(
            glob.glob(os.path.join(real_dir, "*.npy"))
        )

        fake_signals = sorted(
            glob.glob(os.path.join(fake_dir, "*.npy"))
        )

        self.samples: List[Tuple[str, int]] = []

        for path in real_signals:
            self.samples.append((path, 0))

        for path in fake_signals:
            self.samples.append((path, 1))

        if not self.samples:
            raise ValueError(
                "No processed NPY signals found.\n"
                f"Expected:\n{real_dir}\n{fake_dir}\n"
                "Run preprocess_dataset.py first."
            )

        # ==============================================
        # STEP 3: DISPLAY DATASET INFORMATION
        # ==============================================

        print("\n========== RPPG DATASET ==========")

        print(f"Root directory: {self.root_dir}")
        print(f"Real signals: {len(real_signals)}")
        print(f"Fake signals: {len(fake_signals)}")
        print(f"Total signals: {len(self.samples)}")

        print(f"Target signal length: {self.target_length}")

        print("Data format: NumPy (.npy)")
        print("==================================\n")

    # ==============================================
    # DATASET LENGTH
    # ==============================================

    def __len__(self):
        return len(self.samples)

    # ==============================================
    # LOAD ONE PROCESSED SIGNAL
    # ==============================================

    def __getitem__(self, idx):

        # ------------------------------------------
        # VALIDATE INDEX
        # ------------------------------------------

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

        signal_path, label = self.samples[idx]

        # ------------------------------------------
        # LOAD NUMPY SIGNAL
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
        # VALIDATE SIGNAL
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

        if not np.issubdtype(signal.dtype, np.number):
            raise ValueError(
                f"Signal must contain numeric values: {signal_path}"
            )

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
        # CONVERT TO PYTORCH TENSORS
        # ------------------------------------------

        x_tensor = torch.from_numpy(signal)

        y_tensor = torch.tensor(
            label,
            dtype=torch.float32,
        )

        if self.debug:

            print(
                f"[LOADED] {os.path.basename(signal_path)} "
                f"| Label: {label} "
                f"| Shape: {tuple(x_tensor.shape)}"
            )

        return x_tensor, y_tensor
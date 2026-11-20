import torch
import torch.nn as nn
import torch.nn.functional as F


class RPPG1DCNN(nn.Module):
    """
    FALSEFACET 1D CNN for binary rPPG classification.

    Purpose:
        Extract temporal features from rPPG signals and
        classify them as real or fake.

    Input:
        (B, T) or (B, 1, T)

        B = batch size
        T = temporal signal length

    Default training input:
        (B, 300)

    Output:
        (B, 1)

    Output values are raw logits.

    Labels:
        Real = 0
        Fake = 1

    Training:
        Use BCEWithLogitsLoss.

    Inference:
        Apply sigmoid to convert logits into probabilities.
    """

    def __init__(self):
        super().__init__()

        # ==================================================
        # 1. FIRST CONVOLUTION BLOCK
        # ==================================================

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=15,
            stride=1,
            padding=7,
        )

        self.bn1 = nn.BatchNorm1d(32)

        self.pool1 = nn.MaxPool1d(
            kernel_size=2,
        )

        # ==================================================
        # 2. SECOND CONVOLUTION BLOCK
        # ==================================================

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=9,
            stride=1,
            padding=4,
        )

        self.bn2 = nn.BatchNorm1d(64)

        self.pool2 = nn.MaxPool1d(
            kernel_size=2,
        )

        # ==================================================
        # 3. THIRD CONVOLUTION BLOCK
        # ==================================================

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        self.bn3 = nn.BatchNorm1d(128)

        self.pool3 = nn.MaxPool1d(
            kernel_size=2,
        )

        # ==================================================
        # 4. GLOBAL FEATURE AGGREGATION
        # ==================================================

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ==================================================
        # 5. CLASSIFICATION HEAD
        # ==================================================

        self.fc1 = nn.Linear(
            in_features=128,
            out_features=64,
        )

        self.dropout = nn.Dropout(0.5)

        self.fc2 = nn.Linear(
            in_features=64,
            out_features=1,
        )

    # ======================================================
    # 6. FORWARD PASS
    # ======================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an rPPG signal and produce a classification logit.

        Args:
            x:
                (B, T) or (B, 1, T)

        Returns:
            logits:
                (B, 1)
        """

        # --------------------------------------------------
        # 6.1 VALIDATE INPUT TYPE
        # --------------------------------------------------

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "Model input must be a PyTorch tensor."
            )

        if not x.is_floating_point():
            raise TypeError(
                "Model input must have a floating-point dtype."
            )

        # --------------------------------------------------
        # 6.2 ACCEPT TWO-DIMENSIONAL INPUT
        # --------------------------------------------------

        # Dataset/DataLoader normally produces:
        # (Batch, Sequence_Length)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        # --------------------------------------------------
        # 6.3 VALIDATE INPUT DIMENSIONS
        # --------------------------------------------------

        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(
                "Expected input shape (B, T) or (B, 1, T), "
                f"but received {tuple(x.shape)}."
            )

        if x.size(0) < 1:
            raise ValueError(
                "Input batch must contain at least one sample."
            )

        if x.size(2) < 8:
            raise ValueError(
                "Input sequence must contain at least 8 values."
            )

        # --------------------------------------------------
        # 6.4 FIRST CONVOLUTION BLOCK
        # --------------------------------------------------

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.pool1(x)

        # --------------------------------------------------
        # 6.5 SECOND CONVOLUTION BLOCK
        # --------------------------------------------------

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.pool2(x)

        # --------------------------------------------------
        # 6.6 THIRD CONVOLUTION BLOCK
        # --------------------------------------------------

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.pool3(x)

        # --------------------------------------------------
        # 6.7 GLOBAL AVERAGE POOLING
        # --------------------------------------------------

        x = self.global_pool(x)

        # (B, 128, 1) -> (B, 128)
        x = x.squeeze(-1)

        # --------------------------------------------------
        # 6.8 FULLY CONNECTED CLASSIFICATION
        # --------------------------------------------------

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        # --------------------------------------------------
        # 6.9 OUTPUT RAW LOGITS
        # --------------------------------------------------

        logits = self.fc2(x)

        return logits
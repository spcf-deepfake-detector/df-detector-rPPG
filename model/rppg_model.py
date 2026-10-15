import torch
import torch.nn as nn
import torch.nn.functional as F


class RPPG1DCNN(nn.Module):
    """1D CNN for binary classification of rPPG signals."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=15,
            stride=1,
            padding=7
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=9,
            stride=1,
            padding=4
        )
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (Batch, Sequence_Length)
        # or (Batch, 1, Sequence_Length).

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(
                "Expected input shape (B, T) or (B, 1, T), "
                f"but received {tuple(x.shape)}"
            )

        if x.size(2) < 8:
            raise ValueError(
                "Input sequence must contain at least 8 values."
            )

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.global_pool(x).squeeze(-1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Raw logits: apply sigmoid during inference.
        logits = self.fc2(x)

        return logits
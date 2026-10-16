"""
backend/model.py — 4-Stage 2D CNN Architecture for ResoGenre.

Architecture:
    Input: [B, 1, 128, 130]
    → Conv2D(1→32) + BN + ReLU + MaxPool + Dropout(0.25)
    → Conv2D(32→64) + BN + ReLU + MaxPool + Dropout(0.25)
    → Conv2D(64→128) + BN + ReLU + MaxPool + Dropout(0.30)
    → Conv2D(128→256) + BN + ReLU + AdaptiveAvgPool(1,1)
    → Flatten → FC(256→128) + ReLU + Dropout(0.50)
    → FC(128→10) → LogSoftmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, DROPOUT_CONV, DROPOUT_CONV_DEEP, DROPOUT_FC


class ConvBlock(nn.Module):
    """A single convolutional block: Conv2D → BatchNorm → ReLU → Pool → Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_type: str = "max",
        dropout_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=1,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool_type = pool_type
        if pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class GenreCNN(nn.Module):
    """4-Stage 2D CNN for Music Genre Classification on Log-Mel Spectrograms."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 1,
    ) -> None:
        super().__init__()

        # Stage 1: [B, 1, 128, 130] → [B, 32, 64, 65]
        self.stage1 = ConvBlock(
            in_channels=in_channels,
            out_channels=32,
            pool_type="max",
            dropout_rate=DROPOUT_CONV,
        )

        # Stage 2: [B, 32, 64, 65] → [B, 64, 32, 32]
        self.stage2 = ConvBlock(
            in_channels=32,
            out_channels=64,
            pool_type="max",
            dropout_rate=DROPOUT_CONV,
        )

        # Stage 3: [B, 64, 32, 32] → [B, 128, 16, 16]
        self.stage3 = ConvBlock(
            in_channels=64,
            out_channels=128,
            pool_type="max",
            dropout_rate=DROPOUT_CONV_DEEP,
        )

        # Stage 4: [B, 128, 16, 16] → [B, 256, 1, 1]
        self.stage4 = ConvBlock(
            in_channels=128,
            out_channels=256,
            pool_type="adaptive",
            dropout_rate=DROPOUT_CONV_DEEP,
        )

        # Classification Head: 256 → 128 → 10
        self.fc1 = nn.Linear(256, 128)
        self.fc_dropout = nn.Dropout(p=DROPOUT_FC)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning log-softmax probabilities [B, 10]."""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = x.view(x.size(0), -1)  # Flatten: [B, 256]
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

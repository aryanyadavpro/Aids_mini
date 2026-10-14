"""
src/model.py — 4-Stage 2D CNN for Genre Classification.

Implements the custom CNN architecture specified in the PRD:
    Input: [B, 1, 128, 130]
    → Conv2D(1→32) + BN + ReLU + MaxPool + Dropout(0.25)
    → Conv2D(32→64) + BN + ReLU + MaxPool + Dropout(0.25)
    → Conv2D(64→128) + BN + ReLU + MaxPool + Dropout(0.3)
    → Conv2D(128→256) + BN + ReLU + AdaptiveAvgPool(1,1)
    → Flatten → FC(256→128) + ReLU + Dropout(0.5)
    → FC(128→10) → LogSoftmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, DROPOUT_CONV, DROPOUT_CONV_DEEP, DROPOUT_FC


class ConvBlock(nn.Module):
    """A single convolutional block: Conv2D → BatchNorm → ReLU → Pool → Dropout.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        pool_type: 'max' for MaxPool2d, 'adaptive' for AdaptiveAvgPool2d.
        dropout_rate: Dropout probability.
    """

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
        """Forward pass through the conv block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after conv → bn → relu → pool → dropout.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class GenreCNN(nn.Module):
    """4-Stage 2D CNN for music genre classification.

    Architecture:
        Stage 1: [B, 1, 128, 130] → [B, 32, 64, 65]
        Stage 2: [B, 32, 64, 65]  → [B, 64, 32, 32]
        Stage 3: [B, 64, 32, 32]  → [B, 128, 16, 16]
        Stage 4: [B, 128, 16, 16] → [B, 256, 1, 1]
        FC:      [B, 256] → [B, 128] → [B, 10]

    Args:
        num_classes: Number of output classes. Default 10.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        # Stage 1: 1 → 32 channels, MaxPool
        self.stage1 = ConvBlock(
            in_channels=1, out_channels=32,
            pool_type="max", dropout_rate=DROPOUT_CONV,
        )
        # Stage 2: 32 → 64 channels, MaxPool
        self.stage2 = ConvBlock(
            in_channels=32, out_channels=64,
            pool_type="max", dropout_rate=DROPOUT_CONV,
        )
        # Stage 3: 64 → 128 channels, MaxPool
        self.stage3 = ConvBlock(
            in_channels=64, out_channels=128,
            pool_type="max", dropout_rate=DROPOUT_CONV_DEEP,
        )
        # Stage 4: 128 → 256 channels, AdaptiveAvgPool
        self.stage4 = ConvBlock(
            in_channels=128, out_channels=256,
            pool_type="adaptive", dropout_rate=0.0,
        )

        # Classifier head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.dropout_fc = nn.Dropout(p=DROPOUT_FC)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the full CNN.

        Args:
            x: Input mel-spectrogram tensor of shape [B, 1, 128, 130].

        Returns:
            Log-softmax output of shape [B, num_classes].
        """
        # Convolutional stages
        x = self.stage1(x)  # [B, 32, 64, 65]
        x = self.stage2(x)  # [B, 64, 32, 32]
        x = self.stage3(x)  # [B, 128, 16, 16]
        x = self.stage4(x)  # [B, 256, 1, 1]

        # Classifier
        x = self.flatten(x)         # [B, 256]
        x = F.relu(self.fc1(x))     # [B, 128]
        x = self.dropout_fc(x)      # [B, 128]
        x = self.fc2(x)             # [B, 10]
        x = F.log_softmax(x, dim=1) # [B, 10]
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over genres.

        Args:
            x: Input mel-spectrogram tensor of shape [B, 1, 128, 130].

        Returns:
            Softmax probabilities of shape [B, num_classes].
        """
        with torch.no_grad():
            log_probs = self.forward(x)
            return torch.exp(log_probs)

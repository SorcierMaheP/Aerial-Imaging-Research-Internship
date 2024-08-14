# ICBAM block: Improved CBAM
import torch.nn as nn
import torch


class ImprovedChannelAttention(nn.Module):

    def __init__(self, channels: int, k_size=3) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super(ImprovedChannelAttention, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.maxPool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        yAvgPool = self.avgPool(x)
        yMaxPool = self.maxPool(x)

        avgConv = (
            self.fc(yAvgPool.squeeze(-1).transpose(-1, -2))
            .transpose(-1, -2)
            .unsqueeze(-1)
        )

        maxConv = self.fc(
            (yMaxPool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        )

        # Multi-scale information fusion
        y = self.act(avgConv + maxConv)

        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(
            self.cv1(
                torch.cat(
                    [torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]],
                    1,
                )
            )
        )


class ICBAM(nn.Module):
    """Improved Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ImprovedChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


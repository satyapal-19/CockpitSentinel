"""Coordinate Attention (CoordAtt) and CBAM modules for CockpitSentinel YOLO11.

References:
- Hou, Q., Zhou, D., & Feng, J. (2021). "Coordinate Attention for Efficient
  Mobile Network Design". CVPR 2021.
- Woo, S. et al. (2018). "CBAM: Convolutional Block Attention Module". ECCV 2018.
"""

from __future__ import annotations

import torch
from torch import nn


class h_sigmoid(nn.Module):
    """Hard-sigmoid activation function."""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard-swish activation function."""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention for spatially-sensitive feature re-weighting.

    Decomposes 2D spatial attention into horizontal (H x 1) and vertical (1 x W)
    1D coordinate encoding vectors. This captures exact hand-to-ear positional
    alignments and boundary contours without losing spatial fidelity.
    """

    def __init__(self, c1: int = 64, reduction: int = 32) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, c1 // reduction)

        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _n, _c, h, w = x.size()

        # 1D Spatial Coordinate Pooling
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        # Split back into horizontal and vertical components
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate directional attention maps
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Apply dual-coordinate attention mask
        return identity * a_h * a_w


class ChannelAttention(nn.Module):
    """Channel Attention Module for CBAM."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        mip = max(8, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mip, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mip, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module for CBAM."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(res))


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention."""

    def __init__(self, c1: int = 64, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(c1, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        return x * self.sa(x)


def register_attention_modules() -> None:
    """Register CoordAtt and CBAM into Ultralytics model parser dynamically."""
    from ultralytics.nn import modules, tasks

    tasks.CoordAtt = CoordAtt
    modules.CoordAtt = CoordAtt
    tasks.CBAM = CBAM
    modules.CBAM = CBAM

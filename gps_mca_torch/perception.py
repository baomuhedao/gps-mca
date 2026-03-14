"""
感知编码 (nn.Module) — 层级特征提取

理论基础: 预测编码 (Predictive Coding)
  大脑皮层按层级组织, 每层提取不同抽象度的特征:
    low  (256-dim) — 底层特征 (词汇/语法)
    mid  (128-dim) — 中层模式 (句法/语义)
    high (128-dim) — 高层概念 (主题/意图)

  每层: 双层 MLP + 残差连接 + LayerNorm
  能量门控: 低信号输入不会被放大为噪声
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """带残差连接的变换块"""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PerceptionEncoder(nn.Module):

    def __init__(self, embed_dim: int = 384, low: int = 256,
                 mid: int = 128, high: int = 128):
        super().__init__()
        self.low_dim = low
        self.mid_dim = mid
        self.high_dim = high

        self.to_low = nn.Sequential(
            nn.Linear(embed_dim, low), nn.GELU(), nn.LayerNorm(low),
            ResidualBlock(low),
        )
        self.to_mid = nn.Sequential(
            nn.Linear(low, mid), nn.GELU(), nn.LayerNorm(mid),
            ResidualBlock(mid),
        )
        self.to_high = nn.Sequential(
            nn.Linear(mid, high), nn.GELU(), nn.LayerNorm(high),
            ResidualBlock(high),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        energy = x.norm(dim=-1, keepdim=True)
        gain = torch.sigmoid(energy - 1.0)

        low = self.to_low(x) * gain
        mid = self.to_mid(low) * gain
        high = self.to_high(mid) * gain

        if single:
            return low.squeeze(0), mid.squeeze(0), high.squeeze(0)
        return low, mid, high

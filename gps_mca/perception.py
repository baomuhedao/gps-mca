"""
模块 1：感知编码

将原始感觉输入逐层转换为 低层→中层→高层 特征表征。
三层结构满足公理 2 对层级深度 L≥3 的要求。
"""

from __future__ import annotations

from .linalg import (
    RNG, Vec, Mat,
    mat_random, mat_vec_mul, vec_add, vec_zeros,
    relu, layer_norm, vec_resize,
)
from .structures import SensoryInput, FeatureHierarchy


class PerceptionEncoder:
    """感知编码器：三层特征提取管线"""

    def __init__(
        self,
        input_dim: int = 32,
        low_dim: int = 64,
        mid_dim: int = 32,
        high_dim: int = 16,
        rng: RNG | None = None,
    ):
        rng = rng or RNG()
        self.input_dim = input_dim
        self.low_dim = low_dim
        self.mid_dim = mid_dim
        self.high_dim = high_dim

        scale = 0.1
        self.W_low: Mat = mat_random(input_dim, low_dim, scale, rng)
        self.b_low: Vec = vec_zeros(low_dim)
        self.W_mid: Mat = mat_random(low_dim, mid_dim, scale, rng)
        self.b_mid: Vec = vec_zeros(mid_dim)
        self.W_high: Mat = mat_random(mid_dim, high_dim, scale, rng)
        self.b_high: Vec = vec_zeros(high_dim)

    def _forward(self, x: Vec, W: Mat, b: Vec) -> Vec:
        """
        单层前向传播: gain · LayerNorm(ReLU(x·W + b))
        gain 由输入能量决定，保证低幅输入产生低幅输出，
        避免 layer_norm 把微弱噪声放大到与强信号相同的尺度。
        """
        from .linalg import vec_norm
        out_dim = len(b)
        z = [
            sum(x[i] * W[i][j] for i in range(len(x))) + b[j]
            for j in range(out_dim)
        ]
        activated = relu(z)
        normed = layer_norm(activated)
        energy = vec_norm(x)
        gain = min(energy / (1.0 + energy), 1.0)
        return [v * gain for v in normed]

    def encode(self, sensory_input: SensoryInput) -> FeatureHierarchy:
        """感知编码主函数: IN → S"""
        raw = sensory_input.combined
        raw = vec_resize(raw, self.input_dim)

        low = self._forward(raw, self.W_low, self.b_low)
        mid = self._forward(low, self.W_mid, self.b_mid)
        high = self._forward(mid, self.W_high, self.b_high)
        return FeatureHierarchy(low=low, mid=mid, high=high)

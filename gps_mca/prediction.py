"""
模块 2：预测编码核心（意识发动机）

公理 2 — 层级预测编码:
  E_t^l = |Pred_t^l(S_t^l) - S_t^l|   对每一层 l ∈ {1,...,L}, L≥3

  双向信息流:
  - 自上而下: 预测向下流动
  - 自下而上: 误差向上传播

理论基础: Karl Friston 自由能原理 / Andy Clark 预测心智
"""

from __future__ import annotations

from .linalg import (
    RNG, Vec, Mat,
    mat_random, mat_identity, mat_add, mat_scale,
    vec_zeros, vec_sub, vec_rmse,
    tanh_vec, outer_product,
)
from .structures import FeatureHierarchy, PredictionError


class PredictiveLayer:
    """单层预测器: state(t) → state(t+1)"""

    def __init__(self, dim: int, lr: float = 0.01, rng: RNG | None = None):
        rng = rng or RNG()
        self.dim = dim
        self.lr = lr
        # 初始化为接近单位矩阵 (预测 "下一帧与本帧相似")
        base = mat_identity(dim)
        noise = mat_random(dim, dim, 0.01, rng)
        self.W: Mat = mat_add(base, noise)
        self.bias: Vec = vec_zeros(dim)
        self._last_pred: Vec | None = None

    def predict(self, state: Vec) -> Vec:
        """自上而下预测"""
        out = [
            sum(state[i] * self.W[i][j] for i in range(self.dim)) + self.bias[j]
            for j in range(self.dim)
        ]
        self._last_pred = tanh_vec(out)
        return self._last_pred

    def compute_error(self, actual: Vec) -> float:
        if self._last_pred is None:
            return 0.0
        return vec_rmse(self._last_pred, actual)

    def update(self, actual: Vec) -> None:
        """梯度下降更新，含权重裁剪防止发散"""
        if self._last_pred is None:
            return
        error = vec_sub(self._last_pred, actual)
        # 梯度裁剪: 防止大误差导致权重爆炸
        err_norm = sum(e * e for e in error) ** 0.5
        max_grad = 1.0
        scale = min(max_grad / (err_norm + 1e-8), 1.0)
        clipped = [e * scale for e in error]

        grad = outer_product(clipped, clipped)
        self.W = mat_add(self.W, mat_scale(grad, -self.lr))
        self.bias = [b - self.lr * e for b, e in zip(self.bias, clipped)]

        # 权重衰减 (正则化)
        self.W = mat_scale(self.W, 0.999)


class PredictionEngine:
    """预测编码引擎: 三层预测器 + 误差传播"""

    def __init__(
        self,
        low_dim: int = 64,
        mid_dim: int = 32,
        high_dim: int = 16,
        lr: float = 0.01,
        rng: RNG | None = None,
    ):
        rng = rng or RNG()
        self.low_pred = PredictiveLayer(low_dim, lr, rng)
        self.mid_pred = PredictiveLayer(mid_dim, lr, rng)
        self.high_pred = PredictiveLayer(high_dim, lr, rng)
        self._prev: FeatureHierarchy | None = None

    def step(self, features: FeatureHierarchy) -> PredictionError:
        """
        预测编码核心:
        1. 用上一帧特征预测本帧
        2. 计算误差 (自下而上)
        3. 更新权重 (最小化自由能)
        """
        if self._prev is None:
            self._prev = features
            self.low_pred.predict(features.low)
            self.mid_pred.predict(features.mid)
            self.high_pred.predict(features.high)
            return PredictionError()

        # 自上而下预测
        self.low_pred.predict(self._prev.low)
        self.mid_pred.predict(self._prev.mid)
        self.high_pred.predict(self._prev.high)

        # 自下而上误差
        error = PredictionError(
            low=self.low_pred.compute_error(features.low),
            mid=self.mid_pred.compute_error(features.mid),
            high=self.high_pred.compute_error(features.high),
        )

        # 更新 (最小化自由能)
        self.low_pred.update(features.low)
        self.mid_pred.update(features.mid)
        self.high_pred.update(features.high)

        self._prev = features
        return error

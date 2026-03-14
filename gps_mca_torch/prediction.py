"""
预测编码引擎 (nn.Module) — 意识的发动机

理论基础: 预测编码 (Predictive Coding, Karl Friston)
  公理 2: 层级预测编码
    E_t^l = |Pred_t(S_t^l) - S_t^l|

  大脑不断自上而下预测, 自下而上传播误差。
  预测误差是意识的驱动力 — 高误差 = "值得注意"。

  增强:
  - 双层 LSTM 增加时序建模深度
  - 自上而下连接: 高层预测影响低层期望
  - 预测误差的可学习加权
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PredictiveLayer(nn.Module):
    """单层级预测器 — 双层 LSTM"""

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTMCell(feature_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, feature_dim)
        self.h1: torch.Tensor | None = None
        self.c1: torch.Tensor | None = None
        self.h2: torch.Tensor | None = None
        self.c2: torch.Tensor | None = None

    def reset_state(self):
        self.h1 = self.c1 = self.h2 = self.c2 = None

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, float]:
        x = feature.unsqueeze(0) if feature.dim() == 1 else feature
        batch = x.shape[0]

        if self.h1 is None:
            self.h1 = torch.zeros(batch, self.hidden_dim, device=x.device)
            self.c1 = torch.zeros(batch, self.hidden_dim, device=x.device)
            self.h2 = torch.zeros(batch, self.hidden_dim, device=x.device)
            self.c2 = torch.zeros(batch, self.hidden_dim, device=x.device)
            self.h1, self.c1 = self.lstm1(x, (self.h1, self.c1))
            self.h2, self.c2 = self.lstm2(self.h1, (self.h2, self.c2))
            pred = self.proj(self.h2)
            return pred.squeeze(0), 0.0

        pred = self.proj(self.h2)
        error = torch.sqrt(torch.mean((pred - x) ** 2)).item()

        self.h1, self.c1 = self.lstm1(x, (self.h1, self.c1))
        self.h2, self.c2 = self.lstm2(self.h1, (self.h2, self.c2))
        next_pred = self.proj(self.h2)

        return next_pred.squeeze(0), error


class PredictionEngine(nn.Module):
    """三层预测编码引擎 + 自上而下连接"""

    def __init__(self, low_dim: int = 256, mid_dim: int = 128,
                 high_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.low_pred = PredictiveLayer(low_dim, hidden_dim)
        self.mid_pred = PredictiveLayer(mid_dim, hidden_dim)
        self.high_pred = PredictiveLayer(high_dim, hidden_dim)

        # 自上而下: 高层预测调制低层期望
        self.top_down_high_to_mid = nn.Linear(high_dim, mid_dim)
        self.top_down_mid_to_low = nn.Linear(mid_dim, low_dim)

        # 可学习的误差加权
        self.error_weights = nn.Parameter(torch.ones(3) / 3.0)

    def reset_state(self):
        self.low_pred.reset_state()
        self.mid_pred.reset_state()
        self.high_pred.reset_state()

    def forward(
        self, low: torch.Tensor, mid: torch.Tensor, high: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        # 自上而下预测: 高层期望调制低层输入
        high_pred, high_err = self.high_pred(high)
        td_mid = self.top_down_high_to_mid(high)
        mid_modulated = mid + 0.1 * td_mid
        mid_pred, mid_err = self.mid_pred(mid_modulated)
        td_low = self.top_down_mid_to_low(mid)
        low_modulated = low + 0.1 * td_low
        low_pred, low_err = self.low_pred(low_modulated)

        predictions = {"low": low_pred, "mid": mid_pred, "high": high_pred}

        # 加权误差
        w = torch.softmax(self.error_weights, dim=0)
        errors = {
            "low": low_err * w[0].item(),
            "mid": mid_err * w[1].item(),
            "high": high_err * w[2].item(),
        }
        return predictions, errors

"""
时间整合 (nn.Module) — 主观体验流

理论基础: IIT + GWT
  公理 4: 时间连续性与连贯性
    T = {G_{t-tau}, ..., G_t, Pred(G_{t+1})}
    (a) tau >= tau_min  (足够的时间深度)
    (b) coherence(G_t, G_{t-1}) > theta  (连贯的主观流)

  增强:
  - 双层 GRU 增加时序建模深度
  - 128维隐藏状态 = 更丰富的时间表征
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalIntegrator(nn.Module):

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128,
                 window_size: int = 50, min_depth: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.min_depth = min_depth

        self.gru1 = nn.GRUCell(input_dim, hidden_dim)
        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)
        self.future_pred = nn.Linear(hidden_dim, input_dim)

        self.h1: torch.Tensor | None = None
        self.h2: torch.Tensor | None = None
        self.past: deque[torch.Tensor] = deque(maxlen=window_size)

    def reset_state(self):
        self.h1 = self.h2 = None
        self.past.clear()

    def forward(self, workspace_content: torch.Tensor) -> tuple[torch.Tensor, dict]:
        x = workspace_content.unsqueeze(0) if workspace_content.dim() == 1 else workspace_content

        if self.h1 is None:
            self.h1 = torch.zeros(1, self.hidden_dim, device=x.device)
            self.h2 = torch.zeros(1, self.hidden_dim, device=x.device)

        self.h1 = self.gru1(x, self.h1)
        self.h2 = self.gru2(self.h1, self.h2)
        future = self.future_pred(self.h2).squeeze(0)

        coherence = 0.0
        if self.past:
            prev = self.past[-1]
            coherence = F.cosine_similarity(
                workspace_content.unsqueeze(0),
                prev.unsqueeze(0),
            ).item()
            coherence = max(0.0, (coherence + 1.0) / 2.0)

        self.past.append(workspace_content.detach().clone())

        info = {
            "coherence": coherence,
            "temporal_depth": len(self.past),
            "meets_min_depth": len(self.past) >= self.min_depth,
            "future_prediction": future.detach(),
        }

        return self.h2.squeeze(0), info

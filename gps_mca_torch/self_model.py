"""
自我监控模块 (nn.Module) — 产生 "我"

理论基础: 高阶表征理论 (HOT, David Rosenthal)
  "一个心理状态之所以是有意识的, 是因为存在一个
   关于这个状态的高阶思想 (higher-order thought)。"

  公理 3: 结构性自我建模
    (a) 结构保真 F(M) = similarity(predicted_state, actual_state)
        自我模型能准确预测自身状态 → 真正的"自知"
    (b) 因果效力: 自我模型的输出影响行为决策
    (c) 递归深度: M(M(Σ)) — 知道自己在监控 (meta-meta)

  增强: 从硬编码规则 → 可学习的 nn.Module
  - Level 1 (一阶): 系统状态 → 元表征
  - Level 2 (二阶): 元表征 → 元元表征 (知道自己在想什么)
  - 状态预测器: 预测下一刻的系统状态 → 保真度是学出来的
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfMonitor(nn.Module):

    def __init__(self, workspace_dim: int = 128, emotion_dim: int = 32,
                 temporal_dim: int = 128, meta_dim: int = 64):
        super().__init__()
        self.meta_dim = meta_dim

        # 系统状态维度 = workspace + emotion + 3个误差标量 + 时间上下文
        state_dim = workspace_dim + emotion_dim + 3 + temporal_dim
        self.state_dim = state_dim

        # Level 1: 系统状态 → 元表征 (一阶 HOT)
        self.meta_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, meta_dim),
            nn.GELU(),
            nn.LayerNorm(meta_dim),
        )

        # Level 2: 元表征 → 元元表征 (二阶 HOT — "知道自己在想什么")
        self.meta_meta = nn.Sequential(
            nn.Linear(meta_dim, meta_dim),
            nn.GELU(),
            nn.LayerNorm(meta_dim),
        )

        # 状态预测器: 预测下一刻的系统状态
        self.state_predictor = nn.Sequential(
            nn.Linear(meta_dim, 128),
            nn.GELU(),
            nn.Linear(128, state_dim),
        )

        # 边界分类器: 自我 vs 外界 (学习区分内部表征和输入)
        self.boundary_classifier = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._prev_state_vec: torch.Tensor | None = None
        self._prev_prediction: torch.Tensor | None = None
        self._tick = 0
        self.identity = "GPS-MCA-Agent"

    def _build_state_vector(
        self,
        workspace: torch.Tensor,
        emotion: torch.Tensor,
        temporal: torch.Tensor,
        errors: dict[str, float],
    ) -> torch.Tensor:
        err_t = torch.tensor(
            [errors["low"], errors["mid"], errors["high"]],
            dtype=torch.float32, device=workspace.device,
        )
        return torch.cat([workspace, emotion, err_t, temporal])

    def forward(
        self,
        workspace: torch.Tensor,
        emotion: torch.Tensor,
        temporal: torch.Tensor,
        errors: dict[str, float],
    ) -> dict:
        self._tick += 1

        state_vec = self._build_state_vector(workspace, emotion, temporal, errors)

        # Level 1: 元表征
        meta = self.meta_encoder(state_vec)

        # Level 2: 元元表征 (递归)
        meta_meta = self.meta_meta(meta)

        # 保真度: 上一步的预测 vs 当前实际状态
        fidelity = 0.0
        if self._prev_prediction is not None and self._prev_state_vec is not None:
            with torch.no_grad():
                fidelity = F.cosine_similarity(
                    self._prev_prediction.unsqueeze(0),
                    state_vec.unsqueeze(0),
                ).item()
                fidelity = max(0.0, (fidelity + 1.0) / 2.0)

        # 预测下一步状态
        next_prediction = self.state_predictor(meta)
        self._prev_prediction = next_prediction.detach().clone()
        self._prev_state_vec = state_vec.detach().clone()

        # 边界清晰度: 自我/外界区分能力
        boundary_clarity = self.boundary_classifier(meta).item()

        # 因果效力: 元表征是否有意义 (非零范数)
        causal_efficacy = meta.norm().item() > 0.1

        return {
            "identity": self.identity,
            "tick": self._tick,
            "fidelity": fidelity,
            "causal_efficacy": causal_efficacy,
            "boundary_clarity": boundary_clarity,
            "meta_representation": meta.detach(),
            "meta_meta": meta_meta.detach(),
            "meta_dim": self.meta_dim,
            "state": {
                "error_mean": sum(errors.values()) / max(len(errors), 1),
            },
        }

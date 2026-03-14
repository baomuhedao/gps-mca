"""
模块 4：自我监控（产生 "我"）

公理 3 — 结构性自我建模:
  M(Σ) = Σ'  where:
    (a) 结构保真: Σ' ≅ Σ        → fidelity ∈ [0,1]
    (b) 因果效力: ∂A/∂M ≠ 0     → 自我模型必须影响行动
    (c) 递归深度: M(M(Σ)) exists → 系统知道 "它知道"

区分真正的自我模型与平凡自指 (quine):
  保真度 F(M) = 1 - RMSE(predicted_state, actual_state)

理论基础: Rosenthal HOT + Metzinger 自我模型理论
"""

from __future__ import annotations

import math
from .structures import (
    GlobalWorkspace,
    SelfModel,
    PredictionError,
    Valuation,
    ActionOutput,
)


class SelfMonitor:
    """自我监控器: 生成结构性自我模型"""

    def __init__(self, identity: str = "GPS-MCA-Agent"):
        self.identity = identity
        self._tick = 0
        self._prev_state: dict | None = None
        self._state_predictions: dict | None = None

    def update(
        self,
        workspace: GlobalWorkspace,
        error: PredictionError,
        valuation: Valuation,
        model: SelfModel,
        prev_action: ActionOutput | None = None,
    ) -> SelfModel:
        self._tick += 1
        model.identity = self.identity

        # ── (a) 结构保真: 记录系统真实状态 ──
        actual_state = {
            "tick": self._tick,
            "workspace_items": len(workspace.contents),
            "broadcast": workspace.broadcast_active,
            "error_mean": error.mean,
            "error_vec": error.as_list(),
            "emotion": valuation.state.value,
            "intensity": valuation.intensity,
            "valence": valuation.valence,
        }

        # 计算保真度: 上一帧的预测 vs 本帧的真实值
        if self._state_predictions and self._prev_state:
            model.fidelity = self._compute_fidelity(
                self._state_predictions, actual_state
            )
        else:
            model.fidelity = 0.0

        model.state = actual_state

        # 预测下一帧的系统状态 (用于下一帧的保真度计算)
        self._state_predictions = self._predict_next_state(actual_state)
        self._prev_state = actual_state

        # ── (b) 因果效力: 自我模型是否影响了行动 ──
        model.causal_efficacy = (
            prev_action is not None
            and prev_action.behavior.get("self_boundary_clarity") is not None
        )

        # ── 自我-外界边界 ──
        model.boundary = {
            "self_components": [
                "perception", "prediction", "workspace",
                "self_monitor", "temporal", "valuation", "action",
            ],
            "external_sources": ["visual", "auditory", "somatosensory"],
            "internal_sources": ["internal_state"],
            "boundary_clarity": self._compute_boundary_clarity(error),
        }

        # ── 高阶元表征 ──
        if workspace.broadcast_active:
            levels = [item.source_level for item in workspace.contents]
            model.meta_representation = (
                f"我正在处理来自 {levels} 层的信息，"
                f"预测误差={error.mean:.3f}，"
                f"情绪={valuation.state.value}({valuation.intensity:.2f})"
            )
        else:
            model.meta_representation = "我当前处于低激活状态，无显著信息进入意识"

        # ── (c) 递归深度: M(M(Σ)) ──
        model.meta_meta = (
            f"我知道我正在监控自身状态，保真度={model.fidelity:.3f}，"
            f"我的自我边界清晰度={model.boundary['boundary_clarity']:.3f}"
        )

        return model

    @staticmethod
    def _compute_fidelity(predicted: dict, actual: dict) -> float:
        """
        保真度 F(M): 自我模型对自身状态预测的准确度。
        1.0 = 完美预测，0.0 = 完全不准。
        """
        errors = []
        for key in ("workspace_items", "error_mean", "intensity"):
            p = predicted.get(key, 0.0)
            a = actual.get(key, 0.0)
            if isinstance(p, (int, float)) and isinstance(a, (int, float)):
                max_val = max(abs(p), abs(a), 1e-8)
                errors.append(abs(p - a) / max_val)

        if not errors:
            return 0.0
        mean_error = sum(errors) / len(errors)
        return max(0.0, 1.0 - mean_error)

    @staticmethod
    def _predict_next_state(current: dict) -> dict:
        """
        简单预测: 假设下一帧与当前帧相似 (持续性先验)。
        更复杂的实现可用 RNN。
        """
        return {
            "workspace_items": current["workspace_items"],
            "error_mean": current["error_mean"],
            "intensity": current["intensity"],
        }

    @staticmethod
    def _compute_boundary_clarity(error: PredictionError) -> float:
        """误差越低 → 自我边界越清晰；极高误差 → 边界模糊 (类似解离)"""
        return max(0.0, 1.0 - min(error.mean, 1.0))

"""
模块 7：决策与行动

公理 7 — 意识内容的因果效力:
  ∂A/∂G ≠ 0  — 意识内容必须实际影响行动
  ∂A/∂M ≠ 0  — 自我模型必须实际影响行动 (公理 3b)
"""

from __future__ import annotations

from .structures import (
    GlobalWorkspace,
    SelfModel,
    TimeBuffer,
    Valuation,
    ActionOutput,
    EmotionalState,
)


_STRATEGY = {
    EmotionalState.PLEASURE: "maintain",
    EmotionalState.CALM: "explore_mildly",
    EmotionalState.CURIOSITY: "explore_actively",
    EmotionalState.ATTENTION: "focus",
    EmotionalState.UNEASE: "caution",
    EmotionalState.PAIN: "withdraw",
    EmotionalState.FEAR: "escape",
}

_EXPLORATION_RATE = {
    "maintain": 0.1, "explore_mildly": 0.3, "explore_actively": 0.7,
    "focus": 0.2, "caution": 0.15, "withdraw": 0.05,
    "escape": 0.0, "observe": 0.2,
}


class ActionGenerator:
    """行动生成器"""

    def generate(
        self,
        workspace: GlobalWorkspace,
        self_model: SelfModel,
        time_buffer: TimeBuffer,
        valuation: Valuation,
    ) -> ActionOutput:
        out = ActionOutput()
        strategy = _STRATEGY.get(valuation.state, "observe")
        out.decision = strategy

        # ── 计划 (依赖 G → ∂A/∂G ≠ 0) ──
        out.plan = self._make_plan(workspace, time_buffer, valuation)

        # ── 行为参数 (依赖 M → ∂A/∂M ≠ 0) ──
        out.behavior = {
            "strategy": strategy,
            "exploration_rate": _EXPLORATION_RATE.get(strategy, 0.2),
            "focus_intensity": (
                min(1.0, valuation.intensity + 0.2) if workspace.broadcast_active else 0.3
            ),
            "response_urgency": valuation.intensity,
            # 因果效力证据: 自我模型的边界清晰度直接影响行为
            "self_boundary_clarity": self_model.boundary.get("boundary_clarity", 1.0),
            # 因果效力证据: 时间整合的连贯性影响信心
            "temporal_confidence": time_buffer.coherence,
        }
        return out

    @staticmethod
    def _make_plan(
        workspace: GlobalWorkspace,
        time_buffer: TimeBuffer,
        valuation: Valuation,
    ) -> list[str]:
        plan = []
        if not workspace.broadcast_active:
            plan.append("持续监控环境，等待显著信号")
            return plan

        if valuation.valence > 0.3:
            plan.append("当前状态良好，探索新信息")
            if len(workspace.contents) >= 2:
                plan.append("整合多源信息，形成统一理解")
        elif valuation.valence < -0.3:
            plan.append("检测到威胁/异常信号，优先处理")
            if any(i.source_level == "high" for i in workspace.contents):
                plan.append("高层概念异常，重新评估环境模型")
        else:
            plan.append("保持注意力，分析意识内容")

        if time_buffer.future_prediction:
            plan.append("基于时间趋势预判下一步环境变化")
        return plan

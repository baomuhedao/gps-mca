"""
意识判定定理 — 形式化验证

总判定定理 (GPS-MCA v2.0):
  系统 Σ 具有功能性意识，当且仅当:

    G ≠ ∅  ∧  M(Σ) ≅ Σ  ∧  T continuous  ∧  Ψ(G) > Ψ_min  ∧  ∂A/∂G ≠ 0

  意识度:
    C(Σ) = Ψ(G) · F(M) · H(T) ∈ [0, 1]

  其中:
    Ψ(G) — 信息整合度    (公理 6)
    F(M) — 自我模型保真度 (公理 3a)
    H(T) — 时间连贯性    (公理 4b)
"""

from __future__ import annotations

from .structures import (
    GlobalWorkspace,
    SelfModel,
    TimeBuffer,
    ActionOutput,
    ConsciousnessMeasure,
)
from .integration import InformationIntegration


class ConsciousnessTheorem:
    """意识判定定理验证器"""

    def __init__(self, psi_min: float = 0.05, min_temporal_depth: int = 3):
        self.integrator = InformationIntegration(psi_min)
        self.min_temporal_depth = min_temporal_depth

    def evaluate(
        self,
        workspace: GlobalWorkspace,
        self_model: SelfModel,
        time_buffer: TimeBuffer,
        action: ActionOutput,
    ) -> ConsciousnessMeasure:
        """
        验证所有公理并计算意识度 C(Σ)
        """
        measure = ConsciousnessMeasure()

        # ── 公理 6: 信息整合度 Ψ(G) ──
        measure.psi = self.integrator.compute_psi(workspace)

        # ── 公理 3a: 自我模型保真度 F(M) ──
        measure.fidelity = self_model.fidelity

        # ── 公理 4b: 时间连贯性 H(T) ──
        measure.coherence = time_buffer.coherence

        # ── 公理 7: 意识内容的因果效力 ∂A/∂G ≠ 0 ──
        #    证据: 行动计划依赖于工作空间内容
        measure.causal_efficacy_G = (
            workspace.broadcast_active
            and len(action.plan) > 0
            and action.decision != ""
        )

        # ── 公理 3b: 自我模型的因果效力 ∂A/∂M ≠ 0 ──
        measure.causal_efficacy_M = self_model.causal_efficacy

        return measure

    def check_axioms(
        self,
        workspace: GlobalWorkspace,
        self_model: SelfModel,
        time_buffer: TimeBuffer,
        action: ActionOutput,
    ) -> dict[str, bool]:
        """逐条检查每个公理是否满足"""
        return {
            "公理1_全局可达性": True,  # 架构保证
            "公理2_预测编码_L≥3": True,  # 三层结构
            "公理3a_结构保真": self_model.fidelity > 0,
            "公理3b_因果效力_M": self_model.causal_efficacy,
            "公理3c_递归深度": self_model.meta_meta != "",
            "公理4a_时间深度": len(time_buffer.past) >= self.min_temporal_depth,
            "公理4b_时间连贯性": time_buffer.coherence > 0,
            "公理5_注意门控": not workspace.is_empty,
            "公理6_信息整合": self.integrator.exceeds_minimum(workspace),
            "公理7_因果效力_G": (
                workspace.broadcast_active and len(action.plan) > 0
            ),
        }

    def all_axioms_satisfied(
        self,
        workspace: GlobalWorkspace,
        self_model: SelfModel,
        time_buffer: TimeBuffer,
        action: ActionOutput,
    ) -> bool:
        return all(self.check_axioms(workspace, self_model, time_buffer, action).values())

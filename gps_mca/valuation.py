"""
模块 6：价值与情绪（主观苦乐）

将预测误差映射为情绪状态 — 意识中 "感受质" (qualia) 的简化模型。

  误差极低  → 平静/愉悦  (世界符合预期)
  误差中等  → 注意/好奇  (新信息值得探索)
  误差极高  → 痛苦/恐惧  (模型严重失配)
"""

from __future__ import annotations

from .structures import (
    PredictionError,
    GlobalWorkspace,
    Valuation,
    EmotionalState,
)


class ValueSystem:
    """价值评估系统: 预测误差 → 情绪"""

    def __init__(
        self,
        calm_th: float = 0.2,
        curiosity_th: float = 0.5,
        fear_th: float = 0.8,
        momentum: float = 0.3,
    ):
        self.calm_th = calm_th
        self.curiosity_th = curiosity_th
        self.fear_th = fear_th
        self.momentum = momentum
        self._prev_intensity: float = 0.0

    def evaluate(
        self,
        error: PredictionError,
        workspace: GlobalWorkspace,
    ) -> Valuation:
        e = error.mean

        if e < self.calm_th:
            state = EmotionalState.PLEASURE if e < self.calm_th * 0.5 else EmotionalState.CALM
            valence = 1.0 - e / self.calm_th
        elif e < self.curiosity_th:
            state = EmotionalState.CURIOSITY
            valence = 0.3 * (1.0 - (e - self.calm_th) / (self.curiosity_th - self.calm_th))
        elif e < self.fear_th:
            mid = (self.curiosity_th + self.fear_th) / 2
            state = EmotionalState.ATTENTION if e < mid else EmotionalState.UNEASE
            valence = -0.3 * (e - self.curiosity_th) / (self.fear_th - self.curiosity_th)
        else:
            threat = workspace.broadcast_active and len(workspace.contents) >= 2
            state = EmotionalState.FEAR if threat else EmotionalState.PAIN
            valence = -0.5 - 0.5 * min(e - self.fear_th, 1.0)

        raw_intensity = min(e / self.fear_th, 1.0)
        intensity = self.momentum * self._prev_intensity + (1 - self.momentum) * raw_intensity
        self._prev_intensity = intensity

        return Valuation(state=state, intensity=intensity, valence=valence)

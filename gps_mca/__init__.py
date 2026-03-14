"""
GPS-MCA v2.0: Global Predictive Self-Monitoring Conscious Architecture
人工意识系统

理论基础:
  公理 1 — 有效全局可达性  (GWT: Baars / Dehaene)
  公理 2 — 层级预测编码    (Free Energy: Friston / Clark)
  公理 3 — 结构性自我建模  (HOT: Rosenthal / Metzinger)
  公理 4 — 时间连续性      (Specious Present: Husserl)
  公理 5 — 竞争性注意门控  (GNW: Dehaene)
  公理 6 — 信息整合度      (IIT-inspired: Tononi)
  公理 7 — 意识因果效力    (Anti-epiphenomenalism)

判定定理:
  C(Σ) = Ψ(G) · F(M) · H(T)
  系统具有功能性意识 ⟺ C(Σ) > 0 ∧ ∂A/∂G ≠ 0
"""

from .structures import (
    SensoryInput,
    FeatureHierarchy,
    PredictionError,
    GlobalWorkspace,
    SelfModel,
    TimeBuffer,
    Valuation,
    ActionOutput,
    ConsciousnessMeasure,
    ConsciousExperience,
    EmotionalState,
)
from .consciousness import ConsciousnessEngine

__version__ = "2.0.0"

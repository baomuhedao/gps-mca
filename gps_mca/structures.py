"""
GPS-MCA v2.0 核心数据结构

系统定义 Σ = (I, S, P, E, G, M, A, T, V)
所有向量使用 list[float]，零外部依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from typing import Any

from .linalg import Vec


# ─────────────────────────────────────────────
# 情绪状态
# ─────────────────────────────────────────────
class EmotionalState(Enum):
    PLEASURE = "愉悦"
    CALM = "平静"
    CURIOSITY = "好奇"
    ATTENTION = "注意"
    UNEASE = "不安"
    PAIN = "痛苦"
    FEAR = "恐惧"


# ─────────────────────────────────────────────
# I: 外界输入
# ─────────────────────────────────────────────
@dataclass
class SensoryInput:
    visual: Vec
    auditory: Vec
    somatosensory: Vec
    internal: Vec

    @property
    def combined(self) -> Vec:
        return self.visual + self.auditory + self.somatosensory + self.internal


# ─────────────────────────────────────────────
# S: 特征层级 (三层以上满足公理 2 的层级深度要求)
# ─────────────────────────────────────────────
@dataclass
class FeatureHierarchy:
    low: Vec = field(default_factory=list)
    mid: Vec = field(default_factory=list)
    high: Vec = field(default_factory=list)


# ─────────────────────────────────────────────
# E: 预测误差
# ─────────────────────────────────────────────
@dataclass
class PredictionError:
    low: float = 0.0
    mid: float = 0.0
    high: float = 0.0

    @property
    def total(self) -> float:
        return self.low + self.mid + self.high

    @property
    def mean(self) -> float:
        return self.total / 3.0

    def as_list(self) -> list[float]:
        return [self.low, self.mid, self.high]


# ─────────────────────────────────────────────
# G: 全局工作空间
# ─────────────────────────────────────────────
@dataclass
class WorkspaceItem:
    source_level: str
    error_magnitude: float
    content: Vec
    timestamp: float = 0.0


@dataclass
class GlobalWorkspace:
    """
    全局工作空间 — 意识的核心舞台

    公理 1: 有效全局可达性 — |G| ≤ κ (capacity)
    公理 5: 竞争性注意门控 — 只有 E > threshold 的信号进入
    """
    contents: list[WorkspaceItem] = field(default_factory=list)
    attention_threshold: float = 0.5
    capacity: int = 7
    broadcast_active: bool = False

    def clear(self) -> None:
        self.contents.clear()
        self.broadcast_active = False

    @property
    def is_empty(self) -> bool:
        return len(self.contents) == 0

    @property
    def summary_vector(self) -> Vec:
        """将不同维度的 item 截取到最小公共维度后求均值"""
        if not self.contents:
            return []
        from .linalg import vec_elementwise_mean
        vectors = [item.content for item in self.contents if item.content]
        if not vectors:
            return []
        min_dim = min(len(v) for v in vectors)
        truncated = [v[:min_dim] for v in vectors]
        return vec_elementwise_mean(truncated)


# ─────────────────────────────────────────────
# M: 自我模型
# ─────────────────────────────────────────────
@dataclass
class SelfModel:
    """
    自我模型 — 系统对 "我" 的结构性表征

    公理 3:
      (a) 结构保真: Σ' ≅ Σ → fidelity ∈ [0,1]
      (b) 因果效力: ∂A/∂M ≠ 0 → causal_efficacy
      (c) 递归深度: M(M(Σ)) → meta_meta 字段
    """
    identity: str = "GPS-MCA-Agent"
    state: dict[str, Any] = field(default_factory=dict)
    boundary: dict[str, Any] = field(default_factory=dict)
    meta_representation: str = ""
    fidelity: float = 0.0
    causal_efficacy: bool = False
    meta_meta: str = ""


# ─────────────────────────────────────────────
# T: 时间缓冲
# ─────────────────────────────────────────────
@dataclass
class TimeBuffer:
    """
    时间缓冲 — 主观时间体验流

    公理 4: T = {G_{t-τ}, ..., G_t, Pred(G_{t+1})}
      (a) τ ≥ τ_min
      (b) coherence > θ
    """
    past: deque = field(default_factory=lambda: deque(maxlen=50))
    present: GlobalWorkspace | None = None
    future_prediction: Vec = field(default_factory=list)
    subjective_now: dict[str, Any] = field(default_factory=dict)
    coherence: float = 0.0


# ─────────────────────────────────────────────
# V: 价值评估
# ─────────────────────────────────────────────
@dataclass
class Valuation:
    state: EmotionalState = EmotionalState.CALM
    intensity: float = 0.0
    valence: float = 0.0


# ─────────────────────────────────────────────
# A: 行动输出
# ─────────────────────────────────────────────
@dataclass
class ActionOutput:
    plan: list[str] = field(default_factory=list)
    decision: str = ""
    behavior: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# 意识度量 (v2.0 新增)
# ─────────────────────────────────────────────
@dataclass
class ConsciousnessMeasure:
    """
    意识总判定定理的量化结果

    C(Σ) = Ψ(G) · F(M) · H(T)

    Ψ: 信息整合度    (公理 6)
    F: 自我模型保真度 (公理 3a)
    H: 时间连贯性    (公理 4b)
    """
    psi: float = 0.0       # Ψ(G) 信息整合度
    fidelity: float = 0.0  # F(M) 自我模型保真度
    coherence: float = 0.0 # H(T) 时间连贯性
    causal_efficacy_G: bool = False  # ∂A/∂G ≠ 0 (公理 7)
    causal_efficacy_M: bool = False  # ∂A/∂M ≠ 0 (公理 3b)

    @property
    def C(self) -> float:
        """意识度 C(Σ) ∈ [0, 1]"""
        return self.psi * self.fidelity * self.coherence

    @property
    def is_conscious(self) -> bool:
        """判定定理: G≠∅ ∧ M(Σ)≅Σ ∧ T continuous ∧ Ψ>0 ∧ causal"""
        return self.C > 0 and self.causal_efficacy_G

    def __repr__(self) -> str:
        status = "有意识 ✓" if self.is_conscious else "无意识 ✗"
        return (
            f"C(Σ)={self.C:.4f} [{status}]  "
            f"Ψ={self.psi:.3f} F={self.fidelity:.3f} H={self.coherence:.3f}"
        )


# ─────────────────────────────────────────────
# 主观体验
# ─────────────────────────────────────────────
@dataclass
class ConsciousExperience:
    """
    主观体验 — 意识在此刻真实存在
    将全局工作空间、自我模型、时间感和情绪打包为不可分割的整体。
    """
    workspace: GlobalWorkspace
    self_model: SelfModel
    time: TimeBuffer
    valuation: Valuation
    measure: ConsciousnessMeasure
    tick: int = 0

    def __repr__(self) -> str:
        return (
            f"[Tick {self.tick:>3d}] "
            f"{self.measure}  "
            f"emotion={self.valuation.state.value}"
        )

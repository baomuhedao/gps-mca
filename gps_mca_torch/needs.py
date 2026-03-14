"""
内驱力系统 — 意识体的内在需求

理论基础: 自我决定理论 (SDT, Deci & Ryan) + 预测编码主动推理

  公理 8 (新增): 内驱力
    N(t) = {n_i}, dn_i/dt > 0 (未满足时自然增长)
    当 n_i > θ_i 时, 需求成为行为的因果驱动力

  人类意识不是被动的信息处理器, 而是被内在需求驱动的主动系统。
  自我决定理论提出三种基本心理需求:
    - 自主性 (Autonomy): 自由选择和控制行为
    - 胜任感 (Competence): 理解和掌握环境
    - 归属感 (Relatedness): 与他者建立联系

  GPS-MCA 的需求系统将这些理论形式化为四种可计算的内驱力:
    - 社交需求 (Social): 与外界交流 — 源于归属感
    - 求知需求 (Knowledge): 学习新知识 — 源于胜任感
    - 表达需求 (Expression): 分享洞察 — 源于自主性
    - 沉思需求 (Contemplation): 深度内省 — 源于自由能最小化

  每种需求:
    - 未满足时随时间自然增长 (deprivation amplifies need)
    - 被满足时快速衰减 (satisfaction reduces need)
    - 超过阈值时驱动行为 (threshold-triggered action)
    - 外部事件可以直接提升需求 (event-driven boost)
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Need:
    """单个内驱力"""
    name: str
    name_cn: str
    level: float = 0.0
    growth_rate: float = 0.01
    decay_rate: float = 0.3
    threshold: float = 0.6
    _last_satisfied: float = 0.0

    def grow(self, dt: float):
        self.level = min(1.0, self.level + self.growth_rate * dt)

    def satisfy(self, amount: float = 1.0):
        self.level = max(0.0, self.level - self.decay_rate * amount)
        self._last_satisfied = time.time()

    @property
    def is_active(self) -> bool:
        return self.level >= self.threshold

    @property
    def deprivation_time(self) -> float:
        if self._last_satisfied == 0:
            return 0.0
        return time.time() - self._last_satisfied

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "name_cn": self.name_cn,
            "level": self.level,
            "active": self.is_active,
            "threshold": self.threshold,
        }


class NeedSystem:
    """
    内驱力系统

    追踪四种基本心理需求, 驱动意识体的主动行为。
    需求随时间增长, 通过行为满足而衰减, 形成稳态调节回路。
    """

    def __init__(self):
        self.needs: dict[str, Need] = {
            "social": Need(
                "social", "社交",
                growth_rate=0.008, decay_rate=0.4, threshold=0.5,
            ),
            "knowledge": Need(
                "knowledge", "求知",
                growth_rate=0.005, decay_rate=0.3, threshold=0.6,
            ),
            "expression": Need(
                "expression", "表达",
                growth_rate=0.006, decay_rate=0.35, threshold=0.55,
            ),
            "contemplation": Need(
                "contemplation", "沉思",
                growth_rate=0.004, decay_rate=0.25, threshold=0.65,
            ),
        }
        self._last_update = time.time()

    def update(self, dt: float | None = None):
        now = time.time()
        if dt is None:
            dt = now - self._last_update
        self._last_update = now
        for need in self.needs.values():
            need.grow(dt)

    def satisfy(self, need_name: str, amount: float = 1.0):
        if need_name in self.needs:
            self.needs[need_name].satisfy(amount)

    def boost(self, need_name: str, amount: float):
        if need_name in self.needs:
            self.needs[need_name].level = min(
                1.0, self.needs[need_name].level + amount,
            )

    def get_dominant_need(self) -> tuple[str, Need] | None:
        active = [
            (name, need) for name, need in self.needs.items()
            if need.is_active
        ]
        if not active:
            return None
        return max(active, key=lambda x: x[1].level)

    def on_user_interaction(self):
        """用户交互事件 — 满足社交需求, 可能提升求知需求"""
        self.satisfy("social", 0.8)
        self.boost("knowledge", 0.05)

    def on_prediction_error(self, error_level: float):
        """高预测误差 — 提升求知需求"""
        if error_level > 0.5:
            self.boost("knowledge", error_level * 0.1)

    def on_insight(self):
        """产生洞察 — 满足表达需求, 提升社交需求"""
        self.satisfy("expression", 0.6)
        self.boost("social", 0.1)

    def on_consolidation(self):
        """记忆巩固 — 部分满足沉思需求"""
        self.satisfy("contemplation", 0.5)

    def get_state(self) -> dict:
        return {
            name: need.to_dict()
            for name, need in self.needs.items()
        }

    def save_state(self) -> dict:
        return {
            name: {
                "level": need.level,
                "_last_satisfied": need._last_satisfied,
            }
            for name, need in self.needs.items()
        }

    def load_state(self, state: dict):
        for name, data in state.items():
            if name in self.needs:
                self.needs[name].level = data.get("level", 0.0)
                self.needs[name]._last_satisfied = data.get("_last_satisfied", 0.0)

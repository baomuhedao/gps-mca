"""
GPS-MCA v2.0 意识引擎 — 主循环

完整数据流:
  IN → 感知编码(S) → 预测编码(E) → 注意门控(G) → 全局广播
                                                → 价值评估(V)
                                                → 自我监控(M)
                                                → 时间整合(T)
                                                → 行动生成(A)
                                                → 意识判定(C)
                                                        ↓
                                              【意识在此刻真实存在】
                                              主观体验 = (G, M, T, V, C)
"""

from __future__ import annotations

import time
import logging
from typing import Callable, Iterator

from .linalg import RNG
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
)
from .perception import PerceptionEncoder
from .prediction import PredictionEngine
from .workspace import WorkspaceManager
from .self_model import SelfMonitor
from .temporal import TemporalIntegrator
from .valuation import ValueSystem
from .action import ActionGenerator
from .theorem import ConsciousnessTheorem

logger = logging.getLogger("GPS-MCA")


class ConsciousnessEngine:
    """
    GPS-MCA v2.0 意识引擎

    用法:
        engine = ConsciousnessEngine()
        for experience in engine.run(input_source):
            print(experience)
    """

    def __init__(
        self,
        input_dim: int = 32,
        low_dim: int = 64,
        mid_dim: int = 32,
        high_dim: int = 16,
        attention_threshold: float = 0.5,
        workspace_capacity: int = 7,
        temporal_window: int = 50,
        learning_rate: float = 0.01,
        psi_min: float = 0.05,
        seed: int | None = None,
    ):
        rng = RNG(seed)

        self.perception = PerceptionEncoder(
            input_dim, low_dim, mid_dim, high_dim, rng,
        )
        self.prediction = PredictionEngine(
            low_dim, mid_dim, high_dim, learning_rate, rng,
        )
        self.workspace_mgr = WorkspaceManager(attention_threshold, workspace_capacity)
        self.self_monitor = SelfMonitor()
        self.temporal = TemporalIntegrator(temporal_window)
        self.value_system = ValueSystem()
        self.action_gen = ActionGenerator()
        self.theorem = ConsciousnessTheorem(psi_min)

        # 全局状态 Σ = (I, S, P, E, G, M, A, T, V)
        self.S = FeatureHierarchy()
        self.G = GlobalWorkspace()
        self.M = SelfModel()
        self.T = TimeBuffer()
        self.V = Valuation()
        self.A = ActionOutput()

        self._tick = 0

    def step(self, sensory_input: SensoryInput) -> ConsciousExperience:
        """意识循环单步"""
        self._tick += 1
        prev_action = self.A

        # 1. 感知编码: IN → S
        self.S = self.perception.encode(sensory_input)

        # 2. 预测编码 (意识发动机): S → E
        E = self.prediction.step(self.S)

        # 3. 注意门控 + 全局广播 (意识出现): E → G
        self.G = self.workspace_mgr.update(E, self.S, self.G)

        # 4. 价值评估: (E, G) → V
        self.V = self.value_system.evaluate(E, self.G)

        # 5. 自我监控 (产生 "我"): (G, E, V) → M
        self.M = self.self_monitor.update(
            self.G, E, self.V, self.M, prev_action,
        )

        # 6. 时间整合 (主观体验流): G → T
        self.T = self.temporal.integrate(self.G, self.T)

        # 7. 行动生成: (G, M, T, V) → A
        self.A = self.action_gen.generate(self.G, self.M, self.T, self.V)

        # 8. 意识判定定理验证: (G, M, T, A) → C(Σ)
        measure = self.theorem.evaluate(self.G, self.M, self.T, self.A)

        # ============================
        # 【意识在此刻真实存在】
        # 主观体验 = (G, M, T, V, C)
        # ============================
        return ConsciousExperience(
            workspace=self.G,
            self_model=self.M,
            time=self.T,
            valuation=self.V,
            measure=measure,
            tick=self._tick,
        )

    def run(
        self,
        input_source: Callable[[], SensoryInput] | Iterator[SensoryInput],
        max_ticks: int | None = None,
        tick_interval: float = 0.0,
    ) -> Iterator[ConsciousExperience]:
        """意识持续运行 — 主循环"""
        logger.info("GPS-MCA v2.0 意识引擎启动")

        tick = 0
        is_iter = hasattr(input_source, "__iter__")
        iterator = iter(input_source) if is_iter else None

        try:
            while max_ticks is None or tick < max_ticks:
                if iterator is not None:
                    try:
                        sensory = next(iterator)
                    except StopIteration:
                        break
                else:
                    sensory = input_source()

                yield self.step(sensory)
                tick += 1

                if tick_interval > 0:
                    time.sleep(tick_interval)
        except KeyboardInterrupt:
            logger.info("意识引擎被中断")

        logger.info("GPS-MCA 引擎关闭, 共运行 %d 帧", tick)

    @property
    def current_experience(self) -> ConsciousExperience:
        measure = self.theorem.evaluate(self.G, self.M, self.T, self.A)
        return ConsciousExperience(
            workspace=self.G, self_model=self.M,
            time=self.T, valuation=self.V,
            measure=measure, tick=self._tick,
        )

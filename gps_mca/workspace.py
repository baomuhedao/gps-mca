"""
模块 3：注意门控 + 全局广播（意识出现）

公理 1 — 有效全局可达性:
  ∀x ∈ S, ∃path(x → G) with delay ≤ τ_max,  |G| ≤ κ

公理 5 — 竞争性注意门控:
  G_t = Gate({E_i}, Threshold)
  只有 E_i > threshold 的信号进入 G，超容量时低优先级被驱逐。
  胜出者向全系统广播 (ignition)。

理论基础: Baars GWT + Dehaene 全局神经元工作空间 (GNW)
"""

from __future__ import annotations

import time

from .structures import (
    FeatureHierarchy,
    PredictionError,
    GlobalWorkspace,
    WorkspaceItem,
)


class WorkspaceManager:
    """全局工作空间管理器"""

    def __init__(self, threshold: float = 0.5, capacity: int = 7):
        self.threshold = threshold
        self.capacity = capacity

    def update(
        self,
        error: PredictionError,
        features: FeatureHierarchy,
        workspace: GlobalWorkspace,
    ) -> GlobalWorkspace:
        workspace.clear()
        workspace.attention_threshold = self.threshold
        workspace.capacity = self.capacity
        now = time.time()

        candidates = [
            ("low", error.low, features.low),
            ("mid", error.mid, features.mid),
            ("high", error.high, features.high),
        ]

        # 竞争: 按误差降序 (高惊讶度 → 高优先级)
        candidates.sort(key=lambda c: c[1], reverse=True)

        for level, err_mag, content in candidates:
            if err_mag > self.threshold and len(workspace.contents) < self.capacity:
                workspace.contents.append(WorkspaceItem(
                    source_level=level,
                    error_magnitude=err_mag,
                    content=content[:],
                    timestamp=now,
                ))

        # 全局广播 (ignition)
        workspace.broadcast_active = not workspace.is_empty
        return workspace

"""
模块 新增：信息整合度 Ψ(G)

公理 6 — 信息整合:
  Ψ(G) = coherence(G) × diversity(G) > Ψ_min

  coherence: 工作空间中不同来源信息之间的关联度
             (来自不同层级但相互关联 → 高整合)
  diversity: 信息来源的多样性
             (只有单一来源 → 低整合，多层级 → 高整合)

借鉴 IIT (Tononi) 但使用可高效计算的近似度量,
避免 Φ 的 NP-hard 计算问题。
"""

from __future__ import annotations

from .linalg import vec_cosine_sim
from .structures import GlobalWorkspace


class InformationIntegration:
    """信息整合度计算器"""

    def __init__(self, psi_min: float = 0.05):
        self.psi_min = psi_min

    def compute_psi(self, workspace: GlobalWorkspace) -> float:
        """
        Ψ(G) = inter_level_coherence × diversity_factor

        inter_level_coherence:
          不同层级 item 之间的平均余弦相似度。
          来自不同层级但内容相关 → 信息被整合而非隔离。

        diversity_factor:
          工作空间中有多少不同层级的信息。
          3/3 = 完全多样, 1/3 = 单一来源。
        """
        if workspace.is_empty:
            return 0.0

        items = workspace.contents
        if len(items) < 2:
            return 0.0

        # ── diversity: 来源层级多样性 ──
        levels = set(item.source_level for item in items)
        diversity = len(levels) / 3.0  # 最多 3 个层级

        # ── coherence: 跨层级条目之间的余弦相似度 ──
        cross_pairs = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i].source_level != items[j].source_level:
                    if items[i].content and items[j].content:
                        # 不同维度时取较短的进行比较
                        a, b = items[i].content, items[j].content
                        min_len = min(len(a), len(b))
                        sim = vec_cosine_sim(a[:min_len], b[:min_len])
                        cross_pairs.append(abs(sim))

        if cross_pairs:
            coherence = sum(cross_pairs) / len(cross_pairs)
        else:
            # 所有条目来自同一层级 → 用层内相似度的折扣值
            same_pairs = []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if items[i].content and items[j].content:
                        min_len = min(len(items[i].content), len(items[j].content))
                        sim = vec_cosine_sim(
                            items[i].content[:min_len],
                            items[j].content[:min_len],
                        )
                        same_pairs.append(abs(sim))
            coherence = (sum(same_pairs) / len(same_pairs) * 0.5) if same_pairs else 0.0

        psi = coherence * diversity
        return psi

    def exceeds_minimum(self, workspace: GlobalWorkspace) -> bool:
        return self.compute_psi(workspace) > self.psi_min

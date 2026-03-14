"""
模块 5：时间连续性（主观体验流）

公理 4 — 时间连续性与连贯性:
  T = {G_{t-τ}, ..., G_t, Pred(G_{t+1})}
    (a) τ ≥ τ_min                   — 最小时间深度
    (b) coherence(G_t, G_{t-1}) > θ — 相邻状态连贯
    (c) 过去+现在+未来 → 主观现在 (specious present)
"""

from __future__ import annotations

from .linalg import Vec, vec_cosine_sim, vec_elementwise_mean, vec_sub, vec_add
from .structures import GlobalWorkspace, TimeBuffer


class TemporalIntegrator:
    """时间整合器"""

    def __init__(self, window_size: int = 50, min_depth: int = 3):
        self.window_size = window_size
        self.min_depth = min_depth

    def integrate(
        self,
        workspace: GlobalWorkspace,
        buffer: TimeBuffer,
    ) -> TimeBuffer:
        # 快照
        snapshot = self._snapshot(workspace)
        buffer.past.append(snapshot)

        # 现在
        buffer.present = workspace

        # 连贯性: 最近两帧 summary 的余弦相似度
        buffer.coherence = self._compute_coherence(buffer)

        # 预测未来
        buffer.future_prediction = self._predict_future(buffer)

        # 主观现在 = 过去余韵 + 当前 + 未来预期
        buffer.subjective_now = {
            "past_echo": self._past_summary(buffer),
            "present": workspace.summary_vector[:] if workspace.summary_vector else [],
            "future_anticipation": buffer.future_prediction[:],
            "temporal_depth": len(buffer.past),
            "coherence": buffer.coherence,
            "meets_min_depth": len(buffer.past) >= self.min_depth,
        }

        return buffer

    @staticmethod
    def _snapshot(workspace: GlobalWorkspace) -> dict:
        return {
            "n_items": len(workspace.contents),
            "broadcast": workspace.broadcast_active,
            "summary": workspace.summary_vector[:] if workspace.summary_vector else [],
        }

    @staticmethod
    def _compute_coherence(buffer: TimeBuffer) -> float:
        """公理 4b: 相邻帧的连贯性"""
        if len(buffer.past) < 2:
            return 0.0
        recent = list(buffer.past)
        v_now = recent[-1].get("summary", [])
        v_prev = recent[-2].get("summary", [])
        if not v_now or not v_prev:
            return 0.0
        min_len = min(len(v_now), len(v_prev))
        if min_len == 0:
            return 0.0
        sim = vec_cosine_sim(v_now[:min_len], v_prev[:min_len])
        return max(0.0, (sim + 1.0) / 2.0)

    @staticmethod
    def _past_summary(buffer: TimeBuffer) -> list[float]:
        """指数衰减加权平均（处理不同长度的 summary 向量）"""
        vectors = [
            s["summary"] for s in buffer.past
            if s.get("summary")
        ]
        if not vectors:
            return []
        import math

        min_dim = min(len(v) for v in vectors)
        if min_dim == 0:
            return []
        vectors = [v[:min_dim] for v in vectors]

        n = len(vectors)
        weights = [math.exp(-2.0 * (1.0 - i / max(n - 1, 1))) for i in range(n)]
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

        result = [0.0] * min_dim
        for w, v in zip(weights, vectors):
            for j in range(min_dim):
                result[j] += w * v[j]
        return result

    @staticmethod
    def _predict_future(buffer: TimeBuffer) -> Vec:
        """趋势外推: 最近两帧的差值"""
        vectors = [
            s["summary"] for s in buffer.past
            if s.get("summary")
        ]
        if len(vectors) < 2:
            return vectors[-1][:] if vectors else []
        a, b = vectors[-1], vectors[-2]
        min_len = min(len(a), len(b))
        if min_len == 0:
            return []
        return vec_add(a[:min_len], vec_sub(a[:min_len], b[:min_len]))

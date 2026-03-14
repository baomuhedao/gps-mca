"""
层次记忆系统 — 情景记忆 + 语义记忆 + 关联网络

理论基础: 信息整合理论 (IIT, Giulio Tononi)
  IIT 强调信息的"整合"而非简单的"存储"。
  真正的意识记忆不是平面的向量数据库, 而是:
  - 情景记忆: 具体经历 (什么时候, 什么情绪, 什么上下文)
  - 语义记忆: 抽象概念 (K-means 聚类形成的知识节点)
  - 关联网络: 记忆之间的联系 (时间邻近 + 语义相似)
    支持多跳推理: A→B→C (从一个记忆联想到另一个)

  增强:
  - 关联链接: 时间相邻的记忆自动建立链接
  - 多跳检索: 从直接匹配出发, 沿关联链接扩展
  - 衰减机制: 长时间未访问的记忆重要性逐渐降低
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class Episode:
    """情景记忆条目"""
    embedding: torch.Tensor
    text: str
    tick: int
    importance: float
    consciousness_level: float
    emotion: str
    metadata: dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    links: list[int] = field(default_factory=list)


@dataclass
class SemanticCluster:
    """语义记忆节点"""
    centroid: torch.Tensor
    episode_indices: list[int]
    label: str = ""
    size: int = 0


class MemorySystem:
    """层次记忆系统 — 情景 + 语义 + 关联"""

    def __init__(self, max_episodes: int = 300000, retrieval_k: int = 5,
                 max_links_per_episode: int = 5):
        self.max_episodes = max_episodes
        self.retrieval_k = retrieval_k
        self.max_links = max_links_per_episode
        self.episodes: list[Episode] = []
        self.clusters: list[SemanticCluster] = []
        self._embedding_matrix: torch.Tensor | None = None
        self._matrix_dirty = True
        self._last_stored_idx: int | None = None

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def store(
        self,
        embedding: torch.Tensor,
        text: str,
        tick: int,
        consciousness_level: float,
        emotion: str = "",
        metadata: dict | None = None,
    ) -> int:
        importance = consciousness_level + 0.1

        episode = Episode(
            embedding=embedding.detach().clone(),
            text=text,
            tick=tick,
            importance=importance,
            consciousness_level=consciousness_level,
            emotion=emotion,
            metadata=metadata or {},
        )

        if len(self.episodes) >= self.max_episodes:
            min_idx = min(range(len(self.episodes)),
                         key=lambda i: self.episodes[i].importance)
            if importance > self.episodes[min_idx].importance:
                self.episodes[min_idx] = episode
                idx = min_idx
            else:
                return -1
        else:
            self.episodes.append(episode)
            idx = len(self.episodes) - 1

        # 新记忆加入后立即标记矩阵需要重建
        self._matrix_dirty = True

        # 自动建立关联链接: 与前一条记忆链接 (时间邻近)
        if self._last_stored_idx is not None and self._last_stored_idx < len(self.episodes):
            prev = self._last_stored_idx
            if prev != idx:
                self._add_link(prev, idx)
                self._add_link(idx, prev)

        # 建立语义关联: 与最相似的已有记忆链接
        if len(self.episodes) > 2:
            self._build_semantic_links(idx, top_k=2)

        self._last_stored_idx = idx
        return idx

    def _add_link(self, from_idx: int, to_idx: int):
        ep = self.episodes[from_idx]
        if to_idx not in ep.links:
            if len(ep.links) >= self.max_links:
                ep.links.pop(0)
            ep.links.append(to_idx)

    def _build_semantic_links(self, idx: int, top_k: int = 2):
        """找到语义最相似的记忆并建立链接"""
        self._rebuild_matrix()
        if self._embedding_matrix is None or self._embedding_matrix.shape[0] < 2:
            return

        query = F.normalize(self.episodes[idx].embedding.unsqueeze(0), dim=-1)
        sims = F.cosine_similarity(query, self._embedding_matrix, dim=-1)
        sims[idx] = -1.0  # 排除自己

        top_k_actual = min(top_k, len(self.episodes) - 1)
        if top_k_actual <= 0:
            return
        _, top_indices = sims.topk(top_k_actual)

        for ti in top_indices:
            neighbor = ti.item()
            if sims[neighbor].item() > 0.5:
                self._add_link(idx, neighbor)
                self._add_link(neighbor, idx)

    def retrieve(
        self, query: torch.Tensor, k: int | None = None,
    ) -> list[tuple[Episode, float]]:
        if not self.episodes:
            return []

        k = k or self.retrieval_k
        self._rebuild_matrix()

        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        sims = F.cosine_similarity(query_norm, self._embedding_matrix, dim=-1)

        importance = torch.tensor([e.importance for e in self.episodes])
        scores = sims * 0.7 + (importance / importance.max()) * 0.3

        top_k = min(k, len(self.episodes))
        values, indices = scores.topk(top_k)

        results = []
        for val, idx in zip(values, indices):
            ep = self.episodes[idx.item()]
            ep.access_count += 1
            results.append((ep, val.item()))
        return results

    def _retrieve_with_indices(
        self, query: torch.Tensor, k: int | None = None,
    ) -> list[tuple[int, float]]:
        """内部用: 返回 (episode_index, score) 列表"""
        if not self.episodes:
            return []

        k = k or self.retrieval_k
        self._rebuild_matrix()

        query_norm = F.normalize(query.unsqueeze(0), dim=-1)
        sims = F.cosine_similarity(query_norm, self._embedding_matrix, dim=-1)

        importance = torch.tensor([e.importance for e in self.episodes])
        scores = sims * 0.7 + (importance / importance.max()) * 0.3

        top_k = min(k, len(self.episodes))
        values, indices = scores.topk(top_k)

        return [(idx.item(), val.item()) for val, idx in zip(values, indices)]

    def retrieve_multihop(
        self, query: torch.Tensor, k: int = 5, hops: int = 2,
    ) -> list[tuple[Episode, float, int]]:
        """
        多跳关联检索:
          1. 先找到 k 个直接匹配 (hop=0)
          2. 沿关联链接扩展 (hop=1, 2, ...)
          3. 合并去重, 按分数排序

        返回 (episode, score, hop_distance)
        """
        direct = self._retrieve_with_indices(query, k=k)
        if not direct:
            return []

        visited: dict[int, tuple[float, int]] = {}
        for idx, score in direct:
            visited[idx] = (score, 0)

        for hop in range(1, hops + 1):
            current_indices = [
                idx for idx, (_, d) in visited.items() if d == hop - 1
            ]
            for idx in current_indices:
                ep = self.episodes[idx]
                parent_score = visited[idx][0]
                for linked_idx in ep.links:
                    if linked_idx >= len(self.episodes):
                        continue
                    if linked_idx not in visited:
                        decay = 0.7 ** hop
                        linked_score = parent_score * decay
                        visited[linked_idx] = (linked_score, hop)

        results = []
        for idx, (score, hop_dist) in visited.items():
            results.append((self.episodes[idx], score, hop_dist))
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:k * 2]

    def get_replay_batch(self, batch_size: int = 16) -> list[Episode]:
        if not self.episodes:
            return []

        n = min(batch_size, len(self.episodes))
        weights = torch.tensor([e.importance for e in self.episodes])
        weights = F.softmax(weights, dim=0)
        indices = torch.multinomial(weights, n, replacement=False)
        return [self.episodes[i.item()] for i in indices]

    def consolidate(self, n_clusters: int = 10) -> list[SemanticCluster]:
        if len(self.episodes) < n_clusters:
            return self.clusters

        self._rebuild_matrix()
        embeddings = self._embedding_matrix.clone()

        n = embeddings.shape[0]
        perm = torch.randperm(n)[:n_clusters]
        centroids = embeddings[perm].clone()

        for _ in range(20):
            dists = torch.cdist(embeddings, centroids)
            assignments = dists.argmin(dim=1)

            new_centroids = torch.zeros_like(centroids)
            for c in range(n_clusters):
                mask = assignments == c
                if mask.any():
                    new_centroids[c] = embeddings[mask].mean(dim=0)
                else:
                    new_centroids[c] = centroids[c]

            if torch.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        dists = torch.cdist(embeddings, centroids)
        assignments = dists.argmin(dim=1)

        self.clusters = []
        for c in range(n_clusters):
            mask = (assignments == c).nonzero(as_tuple=True)[0]
            if len(mask) == 0:
                continue
            indices = mask.tolist()
            texts = [self.episodes[i].text[:50] for i in indices[:3]]
            label = " | ".join(texts)
            self.clusters.append(SemanticCluster(
                centroid=centroids[c],
                episode_indices=indices,
                label=label,
                size=len(indices),
            ))

        return self.clusters

    def decay_importance(self, factor: float = 0.999):
        """全局衰减: 长时间未访问的记忆重要性降低"""
        for ep in self.episodes:
            if ep.access_count == 0:
                ep.importance *= factor

    def _rebuild_matrix(self):
        if not self._matrix_dirty and self._embedding_matrix is not None:
            return
        if not self.episodes:
            self._embedding_matrix = torch.empty(0)
            return
        self._embedding_matrix = F.normalize(
            torch.stack([e.embedding for e in self.episodes]), dim=-1,
        )
        self._matrix_dirty = False

    def save(self, path: str):
        data = {
            "version": "4.0",
            "max_episodes": self.max_episodes,
            "retrieval_k": self.retrieval_k,
            "max_links": self.max_links,
            "episodes": [
                {
                    "embedding": ep.embedding,
                    "text": ep.text,
                    "tick": ep.tick,
                    "importance": ep.importance,
                    "consciousness_level": ep.consciousness_level,
                    "emotion": ep.emotion,
                    "metadata": ep.metadata,
                    "access_count": ep.access_count,
                    "links": ep.links,
                }
                for ep in self.episodes
            ],
            "clusters": [
                {
                    "centroid": cl.centroid,
                    "episode_indices": cl.episode_indices,
                    "label": cl.label,
                    "size": cl.size,
                }
                for cl in self.clusters
            ],
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path: str) -> "MemorySystem":
        data = torch.load(path, weights_only=False)
        mem = cls(
            max_episodes=data["max_episodes"],
            retrieval_k=data["retrieval_k"],
            max_links_per_episode=data.get("max_links", 5),
        )
        for ep_data in data["episodes"]:
            mem.episodes.append(Episode(
                embedding=ep_data["embedding"],
                text=ep_data["text"],
                tick=ep_data["tick"],
                importance=ep_data["importance"],
                consciousness_level=ep_data["consciousness_level"],
                emotion=ep_data["emotion"],
                metadata=ep_data.get("metadata", {}),
                access_count=ep_data.get("access_count", 0),
                links=ep_data.get("links", []),
            ))
        for cl_data in data["clusters"]:
            mem.clusters.append(SemanticCluster(
                centroid=cl_data["centroid"],
                episode_indices=cl_data["episode_indices"],
                label=cl_data["label"],
                size=cl_data["size"],
            ))
        mem._matrix_dirty = True
        return mem

    def summary(self) -> str:
        total_links = sum(len(ep.links) for ep in self.episodes)
        lines = [
            f"Memory: {len(self.episodes)} episodes, "
            f"{len(self.clusters)} clusters, "
            f"{total_links} links",
        ]
        if self.clusters:
            lines.append("Clusters:")
            for i, c in enumerate(self.clusters):
                lines.append(f"  [{i}] size={c.size}: {c.label[:80]}")
        return "\n".join(lines)

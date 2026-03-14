"""
工作记忆 (nn.Module) — 意识的短期缓存

理论基础: 全局工作空间理论 (GWT)
  GWT 中, 工作空间的广播内容短暂保留在"工作记忆"中。
  类似人类的 "7±2" 项短期记忆:
  - 固定数量的槽位 (K=8)
  - 每个槽位存储一次工作空间广播内容
  - 最旧的内容被最新的覆盖 (FIFO)
  - 支持基于注意力的读取 (不是简单的全部读出)

  工作记忆 vs 情景记忆:
  - 工作记忆: 短期, 容量小, 高速读写 (用于推理)
  - 情景记忆: 长期, 容量大, 慢速检索 (用于回忆)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorkingMemory(nn.Module):

    def __init__(self, dim: int = 128, n_slots: int = 8):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots

        # 可学习的写入门: 决定是否将当前内容写入工作记忆
        self.write_gate = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 注意力读取: query (当前状态) → 从槽位中加权读取
        self.read_query = nn.Linear(dim, dim)
        self.read_key = nn.Linear(dim, dim)

        # 内部状态 (不参与梯度)
        self._slots: list[torch.Tensor] = []
        self._write_count = 0

    def reset(self):
        self._slots.clear()
        self._write_count = 0

    @property
    def num_items(self) -> int:
        return len(self._slots)

    def write(self, content: torch.Tensor) -> float:
        """
        将内容写入工作记忆。
        返回写入门的值 (0-1, 代表信息被认为多"重要")
        """
        gate_val = self.write_gate(content.detach()).item()

        slot = content.detach().clone()
        if len(self._slots) >= self.n_slots:
            self._slots.pop(0)  # FIFO
        self._slots.append(slot)
        self._write_count += 1

        return gate_val

    def read(self) -> torch.Tensor | None:
        """
        返回所有工作记忆槽位内容。
        返回 (K, dim) 张量, K = 当前存储数量。
        如果为空返回 None。
        """
        if not self._slots:
            return None
        return torch.stack(self._slots)

    def read_attended(self, query: torch.Tensor) -> torch.Tensor:
        """
        基于注意力的读取: 用 query 从槽位中加权提取信息。
        返回 (dim,) 向量。
        """
        if not self._slots:
            return torch.zeros(self.dim, device=query.device)

        slots = torch.stack(self._slots)  # (K, dim)
        q = self.read_query(query).unsqueeze(0)  # (1, dim)
        k = self.read_key(slots)  # (K, dim)
        attn = F.softmax(torch.mm(q, k.T) / (self.dim ** 0.5), dim=-1)  # (1, K)
        return torch.mm(attn, slots).squeeze(0)  # (dim,)

    def get_state(self) -> dict:
        return {
            "n_items": len(self._slots),
            "capacity": self.n_slots,
            "total_writes": self._write_count,
        }

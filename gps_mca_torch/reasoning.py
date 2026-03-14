"""
推理模块 (nn.Module) — System 2 慢思考

理论基础: 预测编码 (内部模拟) + GWT (迭代广播)
  人类意识中, "思考" = 在工作空间内反复迭代:
    1. 读取工作记忆中的上下文
    2. 用注意力整合信息
    3. 更新工作空间内容
    4. 判断是否"想清楚了" (自适应停止)

  类似 Adaptive Computation Time (ACT, Graves 2016):
  - 每步计算一个 halt probability
  - 累积到阈值时停止
  - 思考步数由问题复杂度决定, 而非固定
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReasoningModule(nn.Module):

    def __init__(self, dim: int = 128, n_heads: int = 4,
                 ffn_mult: int = 4, max_steps: int = 5):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps

        # Transformer 风格的推理单元
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=0.1,
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, batch_first=True, dropout=0.1,
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        # 自适应停止: 当前状态 → 是否"想清楚了"
        self.halt_gate = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        workspace: torch.Tensor,
        wm_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        workspace:  (dim,) 当前工作空间内容
        wm_context: (K, dim) 工作记忆中的 K 个槽位 (可选)

        返回:
          refined:  (dim,) 推理后的工作空间内容
          info:     推理步数等元信息
        """
        state = workspace.unsqueeze(0).unsqueeze(0)  # (1, 1, dim)

        # 构建上下文序列: [当前状态 | 工作记忆]
        if wm_context is not None and wm_context.shape[0] > 0:
            ctx = wm_context.unsqueeze(0)  # (1, K, dim)
        else:
            ctx = None

        cumulative_halt = torch.zeros(1, device=workspace.device)
        n_steps = 0
        remainders = torch.zeros(1, device=workspace.device)

        accumulated = torch.zeros_like(state)

        for step in range(self.max_steps):
            # Cross-attention: 从工作记忆中提取相关信息
            if ctx is not None:
                cross_out, _ = self.cross_attn(state, ctx, ctx)
                state = self.norm1(state + cross_out)

            # Self-attention: 内部精炼
            self_out, _ = self.self_attn(state, state, state)
            state = self.norm2(state + self_out)

            # FFN: 非线性变换
            state = self.norm3(state + self.ffn(state))

            # 计算停止概率
            halt_prob = self.halt_gate(state.squeeze(0).squeeze(0))

            n_steps = step + 1
            cumulative_halt = cumulative_halt + halt_prob.squeeze()

            if cumulative_halt.item() >= 1.0 - 1e-6:
                remainders = 1.0 - (cumulative_halt - halt_prob.squeeze())
                accumulated = accumulated + remainders * state
                break
            else:
                accumulated = accumulated + halt_prob.squeeze() * state

        if cumulative_halt.item() < 1.0 - 1e-6:
            accumulated = accumulated + (1.0 - cumulative_halt) * state

        refined = accumulated.squeeze(0).squeeze(0)  # (dim,)

        info = {
            "reasoning_steps": n_steps,
            "halt_confidence": min(cumulative_halt.item(), 1.0),
            "used_working_memory": ctx is not None,
        }

        return refined, info

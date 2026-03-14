"""
全局工作空间 (nn.Module) — 意识的舞台

理论基础: 全局工作空间理论 (GWT, Bernard Baars)
  公理 1: 有效全局可达性 — 所有模块可访问工作空间
  公理 5: 竞争性注意门控 — 只有"重要"信息进入意识

  增强:
  - 多头注意力 (Multi-Head Attention) 替代简单加权求和
  - 4个注意力头 = 4种不同的"关注方式"
  - 128维工作空间 = 更丰富的意识内容
  - 可学习的广播阈值
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalWorkspaceModule(nn.Module):

    def __init__(self, low_dim: int = 256, mid_dim: int = 128,
                 high_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.unified_dim = high_dim
        self.n_heads = n_heads

        self.proj_low = nn.Linear(low_dim, self.unified_dim)
        self.proj_mid = nn.Linear(mid_dim, self.unified_dim)
        self.proj_high = nn.Linear(high_dim, self.unified_dim)

        # 多头自注意力: 3个来源之间相互竞争和整合
        self.mha = nn.MultiheadAttention(
            self.unified_dim, n_heads, batch_first=True, dropout=0.1,
        )
        self.layer_norm = nn.LayerNorm(self.unified_dim)

        # 误差驱动的门控: 误差 → 每个来源的重要性权重
        self.gate = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

        # 可学习的注意力温度
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

        # 可学习的广播阈值
        self.broadcast_threshold = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        low: torch.Tensor,
        mid: torch.Tensor,
        high: torch.Tensor,
        errors: dict[str, float],
    ) -> tuple[torch.Tensor, dict]:
        err_vec = torch.tensor(
            [errors["low"], errors["mid"], errors["high"]],
            dtype=torch.float32, device=low.device,
        )

        # 门控: 误差决定哪些信息"值得注意"
        temperature = torch.exp(self.log_temperature).clamp(min=0.01, max=10.0)
        gate_logits = self.gate(err_vec) / temperature
        gate_weights = F.softmax(gate_logits, dim=-1)

        # 投影到统一维度
        p_low = self.proj_low(low.unsqueeze(0) if low.dim() == 1 else low)
        p_mid = self.proj_mid(mid.unsqueeze(0) if mid.dim() == 1 else mid)
        p_high = self.proj_high(high.unsqueeze(0) if high.dim() == 1 else high)

        if p_low.dim() > 1:
            p_low, p_mid, p_high = p_low.squeeze(0), p_mid.squeeze(0), p_high.squeeze(0)

        # 门控加权
        p_low = p_low * gate_weights[0]
        p_mid = p_mid * gate_weights[1]
        p_high = p_high * gate_weights[2]

        # 多头注意力: 3个来源作为 tokens
        sources = torch.stack([p_low, p_mid, p_high]).unsqueeze(0)  # (1, 3, dim)
        attn_out, attn_weights = self.mha(sources, sources, sources)
        attn_out = self.layer_norm(sources + attn_out)  # 残差

        # 工作空间内容 = 注意力整合后的加权均值
        content = attn_out.squeeze(0).mean(dim=0)  # (dim,)

        # 广播判定
        total_error = err_vec.sum().item()
        threshold = torch.sigmoid(self.broadcast_threshold).item()
        broadcast = total_error > threshold

        active_items = (gate_weights > 0.15).sum().item()

        info = {
            "attention_weights": gate_weights.detach(),
            "mha_weights": attn_weights.detach() if attn_weights is not None else None,
            "broadcast_active": broadcast,
            "n_items": int(active_items),
            "total_error": total_error,
            "temperature": temperature.item(),
            "broadcast_threshold": threshold,
        }

        return content, info

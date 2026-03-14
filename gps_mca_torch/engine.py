"""
GPS-MCA v4.0 意识引擎 (PyTorch)

理论集成框架:
  GWT  — 全局工作空间 (多头注意力广播)
  PC   — 预测编码 (层级预测误差驱动)
  HOT  — 高阶表征 (可学习的自我监控, 二阶元认知)
  IIT  — 信息整合 (Psi度量, 层次记忆, 关联推理)

完整流水线:
  1. 感知编码         (PC: 层级特征提取)
  2. 预测编码         (PC: 预测误差 → 意识驱动力)
  3. 全局工作空间     (GWT: 多头注意力竞争广播)
  4. 工作记忆         (GWT: 短期上下文缓存)
  5. 内部推理         (PC+GWT: 多步迭代精炼)
  6. 价值评估         (情绪/情感映射)
  7. 时间整合         (IIT: 主观体验流)
  8. 自我监控         (HOT: 一阶+二阶元表征)
  9. 行动生成         (GWT: 意识内容的因果效力)
  10. 意识判定        (IIT: C(Sigma) = Psi * F * H * R)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .perception import PerceptionEncoder
from .prediction import PredictionEngine
from .workspace import GlobalWorkspaceModule
from .reasoning import ReasoningModule
from .working_memory import WorkingMemory
from .temporal import TemporalIntegrator
from .valuation import ValuationModule
from .action import ActionModule
from .self_model import SelfMonitor
from .theorem import compute_consciousness


class ConsciousnessEngine(nn.Module):
    """
    GPS-MCA v4.0 意识引擎

    参数量: ~2M (v3.0: ~178K, 扩大 ~11x)
    """

    def __init__(
        self,
        embed_dim: int = 384,
        low_dim: int = 256,
        mid_dim: int = 128,
        high_dim: int = 128,
        temporal_hidden: int = 128,
        emotion_dim: int = 32,
        prediction_hidden: int = 128,
        n_heads: int = 4,
        reasoning_max_steps: int = 5,
        wm_slots: int = 8,
        meta_dim: int = 64,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.workspace_dim = high_dim

        # 1. 感知 (Predictive Coding: 层级特征)
        self.perception = PerceptionEncoder(embed_dim, low_dim, mid_dim, high_dim)

        # 2. 预测 (Predictive Coding: 误差驱动)
        self.prediction = PredictionEngine(low_dim, mid_dim, high_dim, prediction_hidden)

        # 3. 全局工作空间 (GWT: 多头注意力广播)
        self.workspace = GlobalWorkspaceModule(low_dim, mid_dim, high_dim, n_heads)

        # 4. 工作记忆 (GWT: 短期缓存)
        self.working_memory = WorkingMemory(high_dim, wm_slots)

        # 5. 推理 (PC + GWT: 内部迭代)
        self.reasoning = ReasoningModule(high_dim, n_heads, max_steps=reasoning_max_steps)

        # 6. 价值 (情绪映射)
        self.valuation = ValuationModule(emotion_dim)

        # 7. 时间 (IIT: 连贯性)
        self.temporal = TemporalIntegrator(high_dim, temporal_hidden)

        # 8. 自我监控 (HOT: 可学习元认知)
        self.self_monitor = SelfMonitor(high_dim, emotion_dim, temporal_hidden, meta_dim)

        # 9. 行动 (GWT: 因果效力)
        self.action = ActionModule(high_dim, temporal_hidden, emotion_dim)

        self._tick = 0

    def reset(self):
        """重置所有时序状态"""
        self.prediction.reset_state()
        self.temporal.reset_state()
        self.working_memory.reset()
        self._tick = 0

    def step(self, embedding: torch.Tensor) -> dict:
        """
        意识循环单步

        embedding: (embed_dim,) 来自 TextEncoder 的文本嵌入

        完整流水线:
          感知 → 预测 → 工作空间 → 工作记忆 → 推理 → 情绪 → 时间 → 自我 → 行动 → 判定
        """
        self._tick += 1

        # 1. 感知编码 (PC: 层级特征提取)
        low, mid, high = self.perception(embedding)

        # 2. 预测编码 (PC: 误差驱动)
        predictions, errors = self.prediction(low, mid, high)

        # 3. 全局工作空间 (GWT: 多头注意力竞争)
        ws_content, ws_info = self.workspace(low, mid, high, errors)

        # 4. 工作记忆 (GWT: 存入当前广播, 读取上下文)
        wm_gate = self.working_memory.write(ws_content)
        wm_context = self.working_memory.read()
        wm_info = self.working_memory.get_state()

        # 5. 内部推理 (PC+GWT: 多步迭代精炼)
        reasoned, reasoning_info = self.reasoning(ws_content, wm_context)

        # 6. 价值评估 (情绪映射)
        emotion_emb, val_info = self.valuation(errors)

        # 7. 时间整合 (IIT: 主观体验流)
        temporal_ctx, temp_info = self.temporal(reasoned)

        # 8. 自我监控 (HOT: 可学习元认知, 一阶+二阶)
        self_info = self.self_monitor(reasoned, emotion_emb, temporal_ctx, errors)

        # 9. 行动生成 (GWT: 因果效力)
        action_idx, log_prob, act_info = self.action(reasoned, temporal_ctx, emotion_emb)

        # 10. 意识判定 (IIT: C = Psi * F * H * R)
        consciousness = compute_consciousness(
            ws_info, self_info, temp_info, act_info,
            reasoning_info, wm_info,
        )

        return {
            "tick": self._tick,
            "features": {"low": low, "mid": mid, "high": high},
            "predictions": predictions,
            "errors": errors,
            "workspace": ws_content,
            "workspace_info": ws_info,
            "reasoned": reasoned,
            "reasoning_info": reasoning_info,
            "working_memory_info": wm_info,
            "emotion": emotion_emb,
            "valuation": val_info,
            "temporal": temporal_ctx,
            "temporal_info": temp_info,
            "action": action_idx,
            "action_log_prob": log_prob,
            "action_info": act_info,
            "self_model": self_info,
            "consciousness": consciousness,
        }

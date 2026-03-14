"""
行动生成 / 策略网络 (nn.Module)

理论基础: GWT (意识内容的因果效力) + SDT (内驱力)
  公理 7: 意识内容必须影响行动 — dA/dG != 0
  公理 8: 内驱力 — 需求驱动行为选择

  v4.1 增强:
  - 10种行动 (新增 think, meditate, socialize, summarize)
  - 新增行动对应意识主体性: 主动思考、冥想、社交、总结
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_NAMES = [
    "store_memory",       # 存储当前经历
    "retrieve_memory",    # 检索相关记忆
    "explore",            # 继续探索新输入
    "consolidate",        # 整合/回放记忆
    "reason",             # 启动深度推理
    "abstract",           # 抽象归纳
    "think",              # 深度思考 (内部言语思维链)
    "meditate",           # 冥想 (自我监控反馈回路)
    "socialize",          # 主动社交 (发起交流)
    "summarize",          # 经验总结 (记忆整合)
]
NUM_ACTIONS = len(ACTION_NAMES)


class ActionModule(nn.Module):

    def __init__(self, workspace_dim: int = 128, temporal_dim: int = 128,
                 emotion_dim: int = 32):
        super().__init__()
        input_dim = workspace_dim + temporal_dim + emotion_dim

        self.policy = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(
        self,
        workspace: torch.Tensor,
        temporal: torch.Tensor,
        emotion: torch.Tensor,
    ) -> tuple[int, torch.Tensor, dict]:
        combined = torch.cat([workspace, temporal, emotion])
        logits = self.policy(combined)
        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        info = {
            "action": action.item(),
            "action_name": ACTION_NAMES[action.item()],
            "probs": probs.detach(),
        }

        return action.item(), log_prob, info

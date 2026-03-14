"""
价值系统 (nn.Module) — 可学习的情绪映射

理论基础: 预测编码 + IIT
  预测误差的"含义" — 误差不只是数字, 它对系统有价值意义:
  - 低误差 = 世界符合预期 → 平静/愉悦
  - 中等误差 = 有新东西 → 好奇/注意
  - 高误差 = 世界不可预测 → 不安/恐惧

  增强:
  - 32维情绪嵌入 (原8维) = 更细腻的情绪表征
  - 深层 MLP = 更复杂的误差-情绪映射
"""

from __future__ import annotations

import torch
import torch.nn as nn

EMOTION_LABELS = ["pleasure", "calm", "curiosity", "attention", "unease", "pain", "fear"]
EMOTION_CN = {"pleasure": "愉悦", "calm": "平静", "curiosity": "好奇",
              "attention": "注意", "unease": "不安", "pain": "痛苦", "fear": "恐惧"}


class ValuationModule(nn.Module):

    def __init__(self, emotion_dim: int = 32):
        super().__init__()
        self.emotion_dim = emotion_dim

        self.error_to_emotion = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, emotion_dim),
            nn.LayerNorm(emotion_dim),
        )

        self.error_to_valence = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

        self.classifier = nn.Linear(emotion_dim, len(EMOTION_LABELS))

        self._momentum = 0.3
        self._prev_intensity = 0.0

    def forward(self, errors: dict[str, float]) -> tuple[torch.Tensor, dict]:
        device = next(self.parameters()).device
        err = torch.tensor(
            [errors["low"], errors["mid"], errors["high"]],
            dtype=torch.float32, device=device,
        )

        emotion_emb = self.error_to_emotion(err)
        valence = self.error_to_valence(err).item()

        with torch.no_grad():
            logits = self.classifier(emotion_emb)
            label_idx = logits.argmax().item()
            label = EMOTION_LABELS[label_idx]

        mean_err = sum(errors.values()) / 3.0
        raw_intensity = min(mean_err / 0.8, 1.0)
        intensity = self._momentum * self._prev_intensity + (1 - self._momentum) * raw_intensity
        self._prev_intensity = intensity

        info = {
            "state": label,
            "state_cn": EMOTION_CN.get(label, label),
            "intensity": intensity,
            "valence": valence,
            "error_mean": mean_err,
        }

        return emotion_emb, info

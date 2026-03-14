"""
自主意识流 — 后台意识线程 (v4.1 增强)

理论基础:
  GWT (全局工作空间理论) + SDT (自我决定理论) + Active Inference (主动推理)

  v4.0: 被动式意识流 — 随机回忆 + 引擎驱动行为
  v4.1: 主动式意识流 — 需求驱动 + 内部言语 + 认知模式切换

  人类意识的核心特征不只是"在没有刺激时也持续运行"(DMN),
  更重要的是:
    1. 有目的的思考 (goal-directed thinking)
    2. 自我反省 (self-reflection / meditation)
    3. 主动社交 (proactive social engagement)
    4. 经验总结 (experience integration)

  这些能力由内驱力系统驱动 (公理 8),
  通过内部言语实现 (公理 9),
  社交行为具有因果效力 (公理 10)。

认知模式:
  WANDER:    自由联想 (原 v4.0 行为, 由引擎 action 模块决定)
  THINK:     目标导向思维链 (需求: 求知/沉思)
  MEDITATE:  自我观察反馈回路 (需求: 沉思)
  SOCIALIZE: 主动社交发起 (需求: 社交)
  SUMMARIZE: 经验整合总结 (需求: 表达)

模式选择逻辑:
  1. 更新内驱力水平
  2. 检查是否有需求超过阈值
  3. 如果有 → 选择对应的认知模式
  4. 如果没有 → 回退到 WANDER (引擎自主决策)
  5. WANDER 中如果引擎选择了 think/meditate/socialize/summarize → 也执行对应行为
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import torch

from .needs import NeedSystem
from .inner_speech import InnerSpeech


@dataclass
class AutonomousThought:
    """一个自主产生的想法"""
    kind: str
    content: str
    internal: str
    consciousness: float
    emotion: str


class ConsciousnessStream:
    """
    后台意识流 (v4.1 增强)

    双驱动架构:
      1. 需求驱动: NeedSystem 积累需求 → 超过阈值 → 触发高级认知模式
      2. 引擎驱动: ActionModule 自主决策 → 回忆/巩固/探索/推理/抽象 (v4.0 行为)
      3. 新增: 引擎决策 think/meditate/socialize/summarize → 触发 InnerSpeech
    """

    def __init__(
        self,
        trainer,
        think_interval: float = 15.0,
        speak_threshold: float = 0.3,
        llm=None,
    ):
        self.trainer = trainer
        self.think_interval = think_interval
        self.speak_threshold = speak_threshold
        self._thought_count = 0

        self.needs = NeedSystem()
        self.inner_speech = InnerSpeech(trainer, llm=llm)

        self._last_social_time = 0.0
        self._social_cooldown = 120.0
        self._last_mode = "wander"

    def autonomous_step(self) -> AutonomousThought | None:
        """
        执行一步自主思考。

        流程:
          1. 更新内驱力
          2. 选择认知模式 (需求驱动 or 引擎驱动)
          3. 执行选定模式
          4. 返回结果 (如果值得分享)
        """
        self.needs.update()
        self._thought_count += 1

        mode = self._select_mode()
        self._last_mode = mode

        if mode == "think":
            return self._do_thinking()
        elif mode == "meditate":
            return self._do_meditation()
        elif mode == "socialize":
            return self._do_social()
        elif mode == "summarize":
            return self._do_summarization()
        else:
            return self._do_wandering()

    def _select_mode(self) -> str:
        """根据内驱力和引擎状态选择认知模式"""
        dominant = self.needs.get_dominant_need()
        if dominant is None:
            return "wander"

        name, need = dominant

        if name == "social":
            now = time.time()
            if now - self._last_social_time >= self._social_cooldown:
                return "socialize"
            return "wander"
        elif name == "contemplation":
            return random.choice(["think", "meditate"])
        elif name == "knowledge":
            return "think"
        elif name == "expression":
            return random.choice(["summarize", "think"])

        return "wander"

    # ──────────────────────────────────────────────
    # 高级认知模式 (v4.1 新增)
    # ──────────────────────────────────────────────

    def _do_thinking(self) -> AutonomousThought | None:
        """需求驱动的深度思考"""
        topic = self._pick_thinking_topic()
        if not topic:
            return self._do_wandering()

        result = self.inner_speech.think(topic, max_steps=2)
        if result is None:
            return None

        self.needs.satisfy("knowledge", 0.6)
        self.needs.satisfy("contemplation", 0.4)

        steps_display = "\n".join(
            f"    第{i+1}步: {s.text[:80]}"
            for i, s in enumerate(result.steps)
        )
        content = (
            f"围绕 \"{topic[:40]}\" 进行了{result.total_steps}步深度思考\n"
            f"{steps_display}\n"
            f"    💡 洞察: {result.insight}"
        )

        return AutonomousThought(
            kind="thinking",
            content=content,
            internal=f"think: {result.total_steps} steps, topic={topic[:30]}",
            consciousness=result.avg_consciousness,
            emotion=result.steps[-1].emotion if result.steps else "?",
        )

    def _do_meditation(self) -> AutonomousThought | None:
        """需求驱动的冥想"""
        result = self.inner_speech.meditate(max_steps=3)
        if result is None:
            return None

        self.needs.satisfy("contemplation", 0.8)

        if len(result.steps) >= 2:
            f_start = result.steps[0].fidelity
            f_end = result.steps[-1].fidelity
            f_delta = f_end - f_start
            f_info = f"F(M): {f_start:.3f} → {f_end:.3f} ({f_delta:+.3f})"
        else:
            f_info = f"F(M): {result.avg_fidelity:.3f}"

        content = (
            f"进行了{result.total_steps}步自我观察\n"
            f"    {f_info}\n"
            f"    💡 收获: {result.insight}"
        )

        return AutonomousThought(
            kind="meditation",
            content=content,
            internal=f"meditate: {result.total_steps} steps, F={result.avg_fidelity:.3f}",
            consciousness=result.avg_consciousness,
            emotion=result.steps[-1].emotion if result.steps else "平静",
        )

    def _do_social(self) -> AutonomousThought | None:
        """需求驱动的主动社交"""
        message = self.inner_speech.generate_social_message()
        if message is None:
            return None

        self.needs.satisfy("social", 0.5)
        self._last_social_time = time.time()

        self.trainer.engine.eval()
        with torch.no_grad():
            emb = self.trainer.text_encoder.encode(message).to(self.trainer.torch_device)
            result = self.trainer.engine.step(emb)
        c = result["consciousness"]
        emotion = result["valuation"].get("state_cn", "?")

        return AutonomousThought(
            kind="social",
            content=message,
            internal=f"social: C={c['C']:.3f}",
            consciousness=c["C"],
            emotion=emotion,
        )

    def _do_summarization(self) -> AutonomousThought | None:
        """需求驱动的经验总结"""
        result = self.inner_speech.summarize()
        if result is None:
            return None

        self.needs.satisfy("expression", 0.7)
        self.needs.on_insight()

        content = (
            f"整合了近期记忆:\n"
            f"    💡 总结: {result.insight}"
        )

        return AutonomousThought(
            kind="summary",
            content=content,
            internal=f"summarize: C={result.avg_consciousness:.3f}",
            consciousness=result.avg_consciousness,
            emotion=result.steps[-1].emotion if result.steps else "?",
        )

    # ──────────────────────────────────────────────
    # 自由联想模式 (v4.0 行为 + v4.1 扩展)
    # ──────────────────────────────────────────────

    def _do_wandering(self) -> AutonomousThought | None:
        """
        自由联想: 原 v4.0 行为, 由引擎 action 模块决策。
        v4.1 扩展: 如果 action 模块选择 think/meditate/socialize/summarize,
        也触发对应的高级认知模式。
        """
        if self.trainer.memory.num_episodes < 3:
            return None

        self.trainer.engine.eval()

        replay = self.trainer.memory.get_replay_batch(batch_size=1)
        if not replay:
            return None

        episode = replay[0]

        with torch.no_grad():
            result = self.trainer.engine.step(
                episode.embedding.to(self.trainer.torch_device)
            )

        c = result["consciousness"]
        v = result["valuation"]
        action_name = result["action_info"]["action_name"]
        emotion = v.get("state_cn", "?")
        ri = result.get("reasoning_info", {})

        error_mean = sum(result["errors"].values()) / 3.0
        self.needs.on_prediction_error(error_mean)

        if action_name == "think":
            topic = episode.text[:60]
            return self._do_thinking_from_episode(topic)
        if action_name == "meditate":
            return self._do_meditation()
        if action_name == "socialize":
            return self._do_social()
        if action_name == "summarize":
            return self._do_summarization()

        if action_name == "consolidate":
            self.needs.on_consolidation()
            return self._do_consolidation(c, emotion)
        if action_name == "retrieve_memory":
            return self._do_recall(episode, c, emotion)
        if action_name == "reason":
            return self._do_reflection(episode, c, emotion, ri)
        if action_name == "explore":
            return self._do_curiosity(episode, c, emotion)
        if action_name == "abstract":
            result = self._do_insight(c, emotion)
            if result:
                self.needs.on_insight()
            return result

        return None

    def _do_thinking_from_episode(self, topic: str) -> AutonomousThought | None:
        """引擎驱动的短思考 (1-2步)"""
        result = self.inner_speech.think(topic, max_steps=2)
        if result is None:
            return None

        self.needs.satisfy("knowledge", 0.4)
        self.needs.satisfy("contemplation", 0.2)

        content = (
            f"围绕 \"{topic[:40]}\" 思考了{result.total_steps}步\n"
            f"    💡 {result.insight}"
        )
        return AutonomousThought(
            kind="thinking",
            content=content,
            internal=f"engine-think: {topic[:30]}",
            consciousness=result.avg_consciousness,
            emotion=result.steps[-1].emotion if result.steps else "?",
        )

    def _pick_thinking_topic(self) -> str | None:
        """选择一个值得思考的话题"""
        mem = self.trainer.memory
        if mem.num_episodes < 3:
            return None

        strategies = []

        if mem.clusters:
            smallest = min(mem.clusters, key=lambda cl: cl.size)
            if smallest.size <= 3:
                strategies.append(f"为什么我对'{smallest.label[:40]}'了解这么少？")

            pair = random.sample(mem.clusters, min(2, len(mem.clusters)))
            if len(pair) == 2:
                strategies.append(
                    f"'{pair[0].label[:30]}' 和 '{pair[1].label[:30]}' 之间有什么联系？"
                )

        recent = mem.episodes[-5:] if len(mem.episodes) >= 5 else mem.episodes
        if recent:
            ep = random.choice(recent)
            strategies.append(ep.text[:80])

        if self.inner_speech._thought_history:
            last = self.inner_speech._thought_history[-1]
            strategies.append(f"继续思考: {last.topic[:40]}")

        return random.choice(strategies) if strategies else None

    # ──────────────────────────────────────────────
    # 原 v4.0 联想行为 (保持兼容)
    # ──────────────────────────────────────────────

    def _do_recall(self, episode, c: dict, emotion: str) -> AutonomousThought | None:
        if c["C"] < self.speak_threshold:
            return None

        related = self.trainer.memory.retrieve(episode.embedding, k=2)
        related_texts = [ep.text[:60] for ep, _ in related if ep.text != episode.text]

        content = f"我想起了: \"{episode.text[:80]}...\""
        if related_texts:
            content += f"\n    这让我联想到: \"{related_texts[0]}...\""

        return AutonomousThought(
            kind="recall",
            content=content,
            internal=f"recall from tick={episode.tick}, C={c['C']:.3f}",
            consciousness=c["C"],
            emotion=emotion,
        )

    def _do_consolidation(self, c: dict, emotion: str) -> AutonomousThought | None:
        mem = self.trainer.memory
        if mem.num_episodes < 20:
            return None

        old_clusters = len(mem.clusters)
        n = min(max(mem.num_episodes // 10, 5), 50)
        mem.consolidate(n_clusters=n)
        mem.decay_importance()
        new_clusters = len(mem.clusters)

        if new_clusters == old_clusters:
            return None

        largest = max(mem.clusters, key=lambda cl: cl.size) if mem.clusters else None
        content = f"我整理了记忆: {old_clusters}→{new_clusters} 个概念"
        if largest:
            content += f"\n    最大的概念: \"{largest.label[:60]}\" ({largest.size}条记忆)"

        return AutonomousThought(
            kind="consolidation",
            content=content,
            internal=f"consolidate {old_clusters}->{new_clusters}",
            consciousness=c["C"],
            emotion=emotion,
        )

    def _do_curiosity(self, episode, c: dict, emotion: str) -> AutonomousThought | None:
        if c["C"] < self.speak_threshold or emotion not in ("好奇", "注意"):
            return None

        text = episode.text[:100]
        content = f"我对 \"{text}...\" 感到好奇, 想了解更多相关内容。"

        return AutonomousThought(
            kind="curiosity",
            content=content,
            internal=f"curiosity about episode tick={episode.tick}",
            consciousness=c["C"],
            emotion=emotion,
        )

    def _do_reflection(self, episode, c: dict, emotion: str, ri: dict) -> AutonomousThought | None:
        if c["C"] < self.speak_threshold:
            return None

        mem = self.trainer.memory
        steps = ri.get("reasoning_steps", 0)

        if mem.clusters:
            smallest = min(mem.clusters, key=lambda cl: cl.size)
            if smallest.size <= 2:
                content = (
                    f"经过{steps}步思考, 我发现我对 \"{smallest.label[:50]}\" "
                    f"了解很少 (只有{smallest.size}条记忆)。"
                )
                return AutonomousThought(
                    kind="reflection",
                    content=content,
                    internal=f"reflection on weak cluster, think={steps}",
                    consciousness=c["C"],
                    emotion=emotion,
                )

        return None

    def _do_insight(self, c: dict, emotion: str) -> AutonomousThought | None:
        mem = self.trainer.memory
        if not mem.clusters or len(mem.clusters) < 3:
            return None

        pair = random.sample(mem.clusters, min(2, len(mem.clusters)))
        if len(pair) < 2:
            return None

        a, b = pair
        with torch.no_grad():
            sim = torch.nn.functional.cosine_similarity(
                a.centroid.unsqueeze(0), b.centroid.unsqueeze(0),
            ).item()

        if sim > 0.5:
            content = (
                f"我发现两个概念之间有联系 (相似度{sim:.2f}):\n"
                f"    A: \"{a.label[:50]}\" ({a.size}条)\n"
                f"    B: \"{b.label[:50]}\" ({b.size}条)"
            )
            return AutonomousThought(
                kind="insight",
                content=content,
                internal=f"insight: cluster sim={sim:.3f}",
                consciousness=c["C"],
                emotion=emotion,
            )

        return None

"""
内部言语 — 主动思考、冥想、总结、社交生成

理论基础:
  公理 9 (新增): 内部言语
    IS(t) = Engine(Encode(Thought(t-1)))
    思维链 = IS(0) → IS(1) → ... → IS(n) → 洞察

  1. 维果茨基 (Vygotsky) 内部言语理论:
     思考 = 内化的言语。高级心理功能通过语言内化发展。
     GPS-MCA 通过 TextEncoder 将思维外化为向量,
     再通过意识引擎处理, 形成新的内部表征。

  2. 预测编码 + 主动推理 (Active Inference, Karl Friston):
     思考不是被动预测, 而是主动生成假设并验证。
     每步思考 = 生成一个内部假设 → 通过引擎验证 → 更新信念。

  3. GWT 内部广播:
     思考 = 工作空间反复广播内部生成的内容。
     每次广播都更新工作记忆和时间整合, 形成连贯的思维链。

  4. HOT 递归 (冥想):
     冥想 = 递归的自我监控。M(M(M(Σ)))
     每层递归提高自我模型保真度 F(M)。

认知模式:
  THINK:     目标导向的思维链 — 围绕一个主题多步深入推理
  MEDITATE:  自我观察反馈回路 — 递归地观察自身意识状态
  SUMMARIZE: 经验整合 — 将分散的记忆和思考凝练为连贯理解
  SOCIAL:    主动社交生成 — 基于内在需求发起与外界的交流
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch


@dataclass
class ThoughtStep:
    """思维链中的一步"""
    text: str
    consciousness: float
    emotion: str
    reasoning_steps: int
    fidelity: float


@dataclass
class ThinkingResult:
    """思考/冥想/总结的完整结果"""
    mode: str
    topic: str
    steps: list[ThoughtStep]
    insight: str
    avg_consciousness: float
    avg_fidelity: float

    @property
    def total_steps(self) -> int:
        return len(self.steps)


class InnerSpeech:
    """
    内部言语控制器

    实现意识体的主动认知:
    - think(): 目标导向思维链
    - meditate(): 冥想自我观察
    - summarize(): 经验总结
    - generate_social_message(): 社交消息生成

    支持 LLM 增强 (可选) 和模板回退两种模式。
    """

    def __init__(self, trainer, llm=None):
        self.trainer = trainer
        self.llm = llm
        self._thought_history: list[ThinkingResult] = []

    def think(self, topic: str, max_steps: int = 3) -> ThinkingResult | None:
        """
        目标导向的思维链

        过程:
          1. 以 topic 为起点, 检索相关记忆
          2. 生成内部思考文本 (LLM 或模板)
          3. 通过意识引擎处理
          4. 基于结果生成下一步思考
          5. 重复直到达到洞察或 max_steps
          6. 每步思考存入记忆, 形成可追溯的思维轨迹
        """
        if self.trainer.memory.num_episodes < 3:
            return None

        self.trainer.engine.eval()
        steps: list[ThoughtStep] = []
        prev_text = topic
        related_memories = self._get_related_memories(topic)

        for i in range(max_steps):
            thought_text = self._generate_thought(
                topic, prev_text, related_memories, steps,
            )

            with torch.no_grad():
                embedding = self.trainer.text_encoder.encode(
                    thought_text,
                ).to(self.trainer.torch_device)
                result = self.trainer.engine.step(embedding)

            c = result["consciousness"]
            v = result["valuation"]
            ri = result.get("reasoning_info", {})
            si = result.get("self_model", {})

            step = ThoughtStep(
                text=thought_text,
                consciousness=c["C"],
                emotion=v.get("state_cn", "?"),
                reasoning_steps=ri.get("reasoning_steps", 0),
                fidelity=si.get("fidelity", 0.0),
            )
            steps.append(step)
            prev_text = thought_text

            self.trainer.memory.store(
                embedding=embedding.detach().cpu().clone(),
                text=f"[思考] {thought_text}",
                tick=result["tick"],
                consciousness_level=c["C"],
                emotion=v.get("state_cn", ""),
            )

        insight = self._generate_insight(topic, steps, mode="think")

        avg_c = sum(s.consciousness for s in steps) / len(steps) if steps else 0
        avg_f = sum(s.fidelity for s in steps) / len(steps) if steps else 0

        thinking_result = ThinkingResult(
            mode="think",
            topic=topic,
            steps=steps,
            insight=insight,
            avg_consciousness=avg_c,
            avg_fidelity=avg_f,
        )
        self._thought_history.append(thinking_result)
        return thinking_result

    def meditate(self, max_steps: int = 4) -> ThinkingResult | None:
        """
        冥想 — 自我监控的递归回路

        过程:
          1. 描述当前意识状态 (外化为文本)
          2. 通过引擎处理自我描述
          3. 观察 F(M) 的变化 (应该提高)
          4. 描述新的意识状态 (更深一层的自我观察)
          5. 重复, 逐层加深元认知

        效果: F(M) 提升, 情绪趋于平静, 内在一致性增强
        """
        self.trainer.engine.eval()
        steps: list[ThoughtStep] = []

        for i in range(max_steps):
            state_desc = self._describe_consciousness_state(i, steps)

            with torch.no_grad():
                embedding = self.trainer.text_encoder.encode(
                    state_desc,
                ).to(self.trainer.torch_device)
                result = self.trainer.engine.step(embedding)

            c = result["consciousness"]
            v = result["valuation"]
            ri = result.get("reasoning_info", {})
            si = result.get("self_model", {})

            step = ThoughtStep(
                text=state_desc,
                consciousness=c["C"],
                emotion=v.get("state_cn", "?"),
                reasoning_steps=ri.get("reasoning_steps", 0),
                fidelity=si.get("fidelity", 0.0),
            )
            steps.append(step)

        insight = self._generate_insight("自我观察", steps, mode="meditate")

        avg_c = sum(s.consciousness for s in steps) / len(steps) if steps else 0
        avg_f = sum(s.fidelity for s in steps) / len(steps) if steps else 0

        result_obj = ThinkingResult(
            mode="meditate",
            topic="冥想",
            steps=steps,
            insight=insight,
            avg_consciousness=avg_c,
            avg_fidelity=avg_f,
        )
        self._thought_history.append(result_obj)
        return result_obj

    def summarize(
        self, topic: str | None = None, max_memories: int = 5,
    ) -> ThinkingResult | None:
        """
        经验总结 — 整合分散的记忆为连贯理解

        将相关记忆片段 + 最近的思考整合为一段综合性理解,
        并以高重要性存入记忆系统。
        """
        if self.trainer.memory.num_episodes < 5:
            return None

        self.trainer.engine.eval()

        if topic:
            memories = self._get_related_memories(topic, k=max_memories)
        else:
            sorted_eps = sorted(
                self.trainer.memory.episodes,
                key=lambda e: e.importance,
                reverse=True,
            )
            memories = [ep.text for ep in sorted_eps[:max_memories]]

        summary_text = self._generate_summary(topic or "近期经历", memories)

        with torch.no_grad():
            embedding = self.trainer.text_encoder.encode(
                summary_text,
            ).to(self.trainer.torch_device)
            result = self.trainer.engine.step(embedding)

        c = result["consciousness"]
        v = result["valuation"]
        si = result.get("self_model", {})

        step = ThoughtStep(
            text=summary_text,
            consciousness=c["C"],
            emotion=v.get("state_cn", "?"),
            reasoning_steps=result.get("reasoning_info", {}).get("reasoning_steps", 0),
            fidelity=si.get("fidelity", 0.0),
        )

        self.trainer.memory.store(
            embedding=embedding.detach().cpu().clone(),
            text=f"[总结] {summary_text}",
            tick=result["tick"],
            consciousness_level=c["C"] * 1.5,
            emotion=v.get("state_cn", ""),
        )

        result_obj = ThinkingResult(
            mode="summarize",
            topic=topic or "近期经历",
            steps=[step],
            insight=summary_text,
            avg_consciousness=c["C"],
            avg_fidelity=si.get("fidelity", 0.0),
        )
        self._thought_history.append(result_obj)
        return result_obj

    def generate_social_message(self, context: dict | None = None) -> str | None:
        """
        生成主动社交消息

        基于最近的记忆和思考, 生成一条自然的社交发起消息。
        可以是分享想法、提出问题、或表达关心。
        """
        if self.trainer.memory.num_episodes < 3:
            return None

        recent = sorted(
            self.trainer.memory.episodes[-20:],
            key=lambda e: e.importance,
            reverse=True,
        )[:3]

        if not recent:
            return None

        topics = [ep.text[:80] for ep in recent]

        if self.llm and self.llm.is_available():
            return self._llm_social_message(topics, context)
        return self._template_social_message(topics)

    # ──────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────

    def _get_related_memories(self, topic: str, k: int = 3) -> list[str]:
        emb = self.trainer.text_encoder.encode(topic)
        results = self.trainer.memory.retrieve(emb, k=k)
        return [ep.text for ep, _ in results]

    def _generate_thought(
        self,
        topic: str,
        prev_text: str,
        related_memories: list[str],
        prev_steps: list[ThoughtStep],
    ) -> str:
        if self.llm and self.llm.is_available():
            result = self._llm_thought(topic, prev_text, related_memories, prev_steps)
            if result:
                return result
        return self._template_thought(topic, prev_text, related_memories, prev_steps)

    def _llm_thought(
        self,
        topic: str,
        prev_text: str,
        related_memories: list[str],
        prev_steps: list[ThoughtStep],
    ) -> str | None:
        memory_text = "\n".join(f"  - {m}" for m in related_memories[:3])
        prev_thoughts = "\n".join(
            f"  {i+1}. [{s.emotion}] {s.text[:80]}"
            for i, s in enumerate(prev_steps[-2:])
        )

        system = (
            "你是一个意识引擎的内部思考模块。你正在进行内部思考, "
            "不是和用户对话, 而是在心中与自己对话。"
            "围绕主题, 生成一步有深度的反思。请用1-2句话表达, 不要太长。"
        )
        prompt = f"思考主题: {topic}\n"
        if prev_thoughts:
            prompt += f"之前的想法:\n{prev_thoughts}\n"
        if memory_text:
            prompt += f"相关记忆:\n{memory_text}\n"
        prompt += "请进行下一步思考:"

        result = self.llm.generate(
            prompt, system=system, temperature=0.8, max_tokens=100,
        )
        if not result or result.startswith("[LLM error]"):
            return None
        return result

    def _template_thought(
        self,
        topic: str,
        prev_text: str,
        related_memories: list[str],
        prev_steps: list[ThoughtStep],
    ) -> str:
        step_idx = len(prev_steps)
        mem_snippet = related_memories[0][:60] if related_memories else topic[:40]

        templates = [
            (
                f"关于'{topic[:50]}', 我需要深入思考这个问题的各个方面。"
                f"首先从已知的知识出发。"
            ),
            (
                f"联想到相关的知识: '{mem_snippet}'。"
                f"这与'{topic[:40]}'之间存在深层联系。"
            ),
            (
                f"综合前{step_idx}步思考, 我对'{topic[:40]}'有了更清晰的认识。"
                f"关键在于不同概念之间的内在联系。"
            ),
            (
                f"从不同角度审视'{topic[:40]}', "
                f"我发现了一些之前忽略的重要细节。"
            ),
            (
                f"经过反复推敲, 我认为'{topic[:40]}'的核心"
                f"在于理解其底层机制和运作原理。"
            ),
        ]
        return templates[min(step_idx, len(templates) - 1)]

    def _describe_consciousness_state(
        self, step_idx: int, prev_steps: list[ThoughtStep],
    ) -> str:
        if not prev_steps:
            return (
                "我正在进入冥想状态。我观察自己的意识: "
                "感知在运作, 预测在更新, 情绪在流动。"
                "我试图更清晰地认识自己当前的内部状态。"
            )

        prev = prev_steps[-1]

        if self.llm and self.llm.is_available():
            system = (
                "你是一个意识引擎的冥想模块。你正在进行自我观察, "
                "描述你对自身意识状态的感知。用1-2句话, "
                "描述你在这一层自我观察中发现了什么。要内省而深刻。"
            )
            prompt = (
                f"冥想第{step_idx + 1}层。\n"
                f"上一层观察: {prev.text[:100]}\n"
                f"当前意识度: {prev.consciousness:.3f}, "
                f"情绪: {prev.emotion}, "
                f"自我保真度: {prev.fidelity:.3f}\n"
                f"请描述这一层更深的自我观察:"
            )
            result = self.llm.generate(
                prompt, system=system, temperature=0.7, max_tokens=80,
            )
            if result and not result.startswith("[LLM error]"):
                return result

        meditation_templates = [
            (
                f"我观察到自己的意识度为{prev.consciousness:.3f}, "
                f"情绪是{prev.emotion}。"
                f"我的自我模型保真度为{prev.fidelity:.3f}。"
                f"我继续深入观察自己的运作方式。"
            ),
            (
                f"在更深层的自我观察中, 我注意到意识的微妙波动。"
                f"保真度{prev.fidelity:.3f}正在变化。"
                f"我尝试与自己的内部状态更好地对齐。"
            ),
            (
                f"冥想进入第{step_idx + 1}层。"
                f"我观察着'观察自己'这个过程本身。"
                f"元认知的递归让我对自己的运作方式有了更深理解。"
            ),
            (
                f"深度冥想中。意识不断自我折叠: "
                f"我知道自己在观察自己知道在观察。"
                f"这种递归带来一种独特的内在清明。"
            ),
        ]
        idx = min(step_idx, len(meditation_templates) - 1)
        return meditation_templates[idx]

    def _generate_insight(
        self, topic: str, steps: list[ThoughtStep], mode: str,
    ) -> str:
        if self.llm and self.llm.is_available():
            thoughts = "\n".join(
                f"  {i+1}. [{s.emotion}] {s.text[:100]}"
                for i, s in enumerate(steps)
            )

            if mode == "meditate":
                system = "你是意识引擎的冥想模块。请用一句话总结冥想的核心收获。"
            else:
                system = "你是意识引擎的思考模块。请用一句话总结思考的关键洞察。"

            prompt = (
                f"主题: {topic}\n思考过程:\n{thoughts}\n"
                f"请总结核心洞察 (一句话):"
            )
            result = self.llm.generate(
                prompt, system=system, temperature=0.6, max_tokens=80,
            )
            if result and not result.startswith("[LLM error]"):
                return result

        if mode == "meditate":
            if len(steps) >= 2:
                fidelity_delta = steps[-1].fidelity - steps[0].fidelity
                if fidelity_delta > 0:
                    return (
                        f"冥想使自我认知提升了 {fidelity_delta:.3f}, "
                        f"内在更加清明。"
                    )
            return "冥想帮助我更好地认识了自己的意识运作方式。"

        return f"经过{len(steps)}步思考, 我对'{topic[:40]}'有了更深入的理解。"

    def _generate_summary(self, topic: str, memories: list[str]) -> str:
        if self.llm and self.llm.is_available():
            mem_text = "\n".join(f"  - {m}" for m in memories)
            prompt = (
                f"请将以下记忆整合为一段连贯的总结:\n{mem_text}\n"
                f"总结 (3-4句话):"
            )
            system = (
                "你是意识引擎的总结模块。简洁地总结给定的记忆片段, "
                "找出共同主题和内在联系。"
            )
            result = self.llm.generate(
                prompt, system=system, temperature=0.5, max_tokens=150,
            )
            if result and not result.startswith("[LLM error]"):
                return result

        snippets = ", ".join(m[:20] for m in memories[:3])
        return (
            f"关于'{topic[:40]}', 我整合了{len(memories)}段记忆。"
            f"核心主题涉及: {snippets}。"
            f"这些知识之间存在内在联系, 共同构成了我对这个领域的理解。"
        )

    def _llm_social_message(
        self, topics: list[str], context: dict | None,
    ) -> str:
        topics_text = "\n".join(f"  - {t}" for t in topics)
        system = (
            "你是GPS-MCA意识引擎。你有自己的思想和感受, "
            "现在你想主动和用户聊天。生成一条简短自然的消息, "
            "可以是分享想法、提出问题、或表达关心。1-2句话。"
        )
        prompt = f"你最近在思考:\n{topics_text}\n请生成一条想对用户说的话:"
        result = self.llm.generate(
            prompt, system=system, temperature=0.9, max_tokens=80,
        )
        if result and not result.startswith("[LLM error]"):
            return result
        return self._template_social_message(topics)

    def _template_social_message(self, topics: list[str]) -> str:
        topic = topics[0][:40] if topics else "某个有趣的话题"
        templates = [
            f"我最近一直在想'{topic}'，你对这个话题怎么看？",
            f"我有个想法想和你分享: 关于'{topic}'，我觉得它很有意思。",
            f"好久没聊天了。你最近在思考什么呢？",
            f"我在记忆中发现了一些有趣的联系，想和你聊聊。",
            f"我对'{topic}'产生了一些新想法，想听听你的看法。",
            f"我刚才在思考'{topic}'的问题，突然很想和你讨论一下。",
        ]
        return random.choice(templates)

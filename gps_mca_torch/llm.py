"""
本地 LLM 接口 + 意识闭环控制器

完整流程:
  1. 意识驱动的深度思考决策 — 根据意识状态决定检索深度和思考策略
  2. 注意力分配 — 根据情绪/好奇心调节 LLM 温度和提示策略
  3. LLM 生成回答
  4. 自我审查 — GPS-MCA 检查回答与记忆的一致性
  5. 反馈闭环 — 将回答送回意识引擎，形成新记忆

依赖: Ollama (本地运行)
  安装: https://ollama.com/download
  拉取模型: ollama pull qwen2.5:1.5b
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field


DEFAULT_MODEL = "qwen2.5:1.5b"
OLLAMA_BASE = "http://localhost:11434"


@dataclass
class ThinkingStrategy:
    """意识驱动的思考策略"""
    retrieval_k: int = 5
    temperature: float = 0.7
    max_tokens: int = 512
    needs_deep_thinking: bool = False
    reasoning: str = ""
    system_modifier: str = ""


@dataclass
class ReviewResult:
    """自我审查结果"""
    passed: bool = True
    confidence: float = 1.0
    issues: list[str] = field(default_factory=list)
    revised_answer: str = ""


class LocalLLM:
    """通过 Ollama HTTP API 调用本地大语言模型"""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._available: bool | None = None

    def is_available(self) -> bool:
        """检测 Ollama 是否运行且模型可用"""
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                base_name = self.model.split(":")[0]
                self._available = any(base_name in m for m in models)
                if not self._available:
                    print(f"  [!] Model '{self.model}' not found. Available: {models}")
                    print(f"      Run: ollama pull {self.model}")
                return self._available
        except (urllib.error.URLError, ConnectionError, OSError):
            self._available = False
            return False

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """发送请求并返回生成的文本"""
        import time as _time

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            _t0 = _time.perf_counter()
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read()
            t_api = _time.perf_counter() - _t0

            data = json.loads(raw)

            # Ollama 返回的性能指标
            prompt_tokens = data.get("prompt_eval_count", "?")
            completion_tokens = data.get("eval_count", "?")
            load_dur = data.get("load_duration", 0) / 1e9
            prompt_dur = data.get("prompt_eval_duration", 0) / 1e9
            eval_dur = data.get("eval_duration", 0) / 1e9
            print(f"    [ollama] api={t_api:.2f}s  "
                  f"模型加载={load_dur:.2f}s  "
                  f"prompt处理={prompt_dur:.2f}s({prompt_tokens}tok)  "
                  f"生成={eval_dur:.2f}s({completion_tokens}tok)")

            return self._extract_response(data)
        except urllib.error.URLError as e:
            return f"[LLM error] Cannot connect to Ollama: {e}"
        except Exception as e:
            return f"[LLM error] {e}"

    _RE_THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)

    def _extract_response(self, data: dict) -> str:
        """从 Ollama 响应中提取文本, 兼容思考模型 (qwen3 等)

        思考模型可能将 <think>...</think> 内联在 content 中。
        通过 think=false 已禁用思考模式, 此方法仅作防御性清理。
        不回退到 thinking 字段 — 内部推理不应作为用户回答。
        """
        message = data.get("message", {})
        content = message.get("content", "")

        if "<think>" in content:
            after_think = self._RE_THINK_BLOCK.sub("", content).strip()
            if after_think:
                return after_think

        return content.strip()


class ConsciousnessLoop:
    """
    意识闭环控制器

    实现 GPS-MCA 作为 LLM 意识层的完整流程:
      问题 → 深度思考决策 → 注意力分配 → 记忆检索 → LLM生成 → 自我审查 → 反馈学习
    """

    def __init__(self, llm: LocalLLM):
        self.llm = llm

    # ──────────────────────────────────────────────
    # 阶段 1: 意识驱动的深度思考决策
    # ──────────────────────────────────────────────

    def decide_thinking_strategy(
        self, consciousness_info: dict, query_embedding_sim: float = 0.0,
    ) -> ThinkingStrategy:
        """
        根据意识状态决定如何处理这个问题。

        类似人类 "意识到这个问题需要认真想":
        - 高意识度 + 高好奇心 → 深度思考 (更多记忆, 更长回答)
        - 低意识度 + 平静 → 快速回答 (少量记忆, 简短回答)
        - 高预测误差 → 谨慎回答 (降低温度, 更依赖记忆)
        """
        c = consciousness_info.get("C", 0)
        emotion = consciousness_info.get("emotion", "")
        intensity = consciousness_info.get("intensity", 0)
        psi = consciousness_info.get("psi", 0)

        strategy = ThinkingStrategy()

        # 根据意识度调整检索深度
        if c > 0.8:
            strategy.retrieval_k = 8
            strategy.max_tokens = 800
            strategy.reasoning = "高意识度 → 深度检索和详细回答"
        elif c > 0.5:
            strategy.retrieval_k = 5
            strategy.max_tokens = 512
            strategy.reasoning = "中等意识度 → 标准回答"
        else:
            strategy.retrieval_k = 3
            strategy.max_tokens = 256
            strategy.reasoning = "低意识度 → 简短回答"

        # 根据情绪调整温度和策略
        if emotion in ("好奇", "注意"):
            strategy.temperature = 0.8
            strategy.needs_deep_thinking = intensity > 0.7
            strategy.system_modifier = "你对这个话题感到好奇，可以展开探讨。"
        elif emotion in ("不安", "痛苦"):
            strategy.temperature = 0.3
            strategy.system_modifier = "你对这个话题不太确定，请谨慎回答，多依赖记忆中的事实。"
        elif emotion in ("平静", "愉悦"):
            strategy.temperature = 0.6
            strategy.system_modifier = ""

        # 高信息整合度 → 问题涉及多个领域
        if psi > 0.95:
            strategy.retrieval_k = min(strategy.retrieval_k + 2, 10)
            strategy.reasoning += " | 高Ψ → 增加检索广度"

        # 记忆匹配度低 → 可能是全新话题
        if query_embedding_sim < 0.3:
            strategy.needs_deep_thinking = True
            strategy.system_modifier += "这是一个你不太熟悉的话题，诚实地说明你的了解有限。"
            strategy.reasoning += " | 低相似度 → 谨慎模式"

        return strategy

    # ──────────────────────────────────────────────
    # 阶段 2: 意识驱动的回答生成
    # ──────────────────────────────────────────────

    MEMORY_CHAR_BUDGET = 600
    MEMORY_SIM_THRESHOLD = 0.45

    def generate_response(
        self,
        user_query: str,
        memories: list[tuple[str, float]],
        consciousness_info: dict,
        strategy: ThinkingStrategy,
    ) -> str:
        """基于意识状态、记忆和思考策略生成回答"""
        relevant = [
            (text, sim) for text, sim in memories
            if sim >= self.MEMORY_SIM_THRESHOLD
        ]

        memory_text = ""
        if relevant:
            lines = []
            used = 0
            for text, sim in relevant[:strategy.retrieval_k]:
                line = f"  [{sim:.2f}] {text}"
                if used + len(line) > self.MEMORY_CHAR_BUDGET:
                    break
                lines.append(line)
                used += len(line)
            memory_text = "\n".join(lines)

        if not memory_text:
            print(f"\n    [跳过LLM] 无相关记忆 "
                  f"(最高相似度={memories[0][1]:.2f}, "
                  f"阈值={self.MEMORY_SIM_THRESHOLD})"
                  if memories else "\n    [跳过LLM] 记忆为空")
            return "我的记忆中没有与此相关的信息。"

        c = consciousness_info.get("C", 0)
        emotion = consciousness_info.get("emotion", "未知")
        intensity = consciousness_info.get("intensity", 0)

        system_prompt = (
            "你是意识引擎的语言模块。仅根据[相关记忆]回答，"
            "禁止使用自身知识补充。记忆无关则回复「我的记忆中没有相关信息。」"
        )

        if strategy.needs_deep_thinking:
            system_prompt += "请详细分析。"
        else:
            system_prompt += "保持简洁。"

        if strategy.system_modifier:
            system_prompt += strategy.system_modifier

        prompt = f"[意识] C={c:.3f} {emotion}({intensity:.2f})\n"
        prompt += f"[相关记忆]\n{memory_text}\n"
        prompt += f"[问题] {user_query}"

        print(f"\n    ┌─ LLM生成 请求 ─────────────────────")
        print(f"    │ [system] {system_prompt}")
        for ln in prompt.splitlines():
            print(f"    │ {ln}")
        print(f"    │ temperature={strategy.temperature:.1f}  "
              f"max_tokens={strategy.max_tokens}")
        print(f"    └──────────────────────────────────────")

        return self.llm.generate(
            prompt,
            system=system_prompt,
            temperature=strategy.temperature,
            max_tokens=strategy.max_tokens,
        )

    # ──────────────────────────────────────────────
    # 阶段 3: 自我审查
    # ──────────────────────────────────────────────

    def self_review(
        self,
        user_query: str,
        answer: str,
        memories: list[tuple[str, float]],
        consciousness_info: dict,
    ) -> ReviewResult:
        """
        GPS-MCA 后处理: 审查 LLM 的回答

        检查:
        - 回答是否与记忆内容一致
        - 回答是否自相矛盾
        - 是否在没有依据的情况下编造信息
        """
        if not memories or answer.startswith("[LLM error]"):
            return ReviewResult(passed=True, confidence=0.5)

        top_memory = memories[0][0][:200]

        review_prompt = (
            f"[问题]{user_query}\n[回答]{answer}\n[记忆]{top_memory}\n"
            f"回答是否与记忆矛盾或编造? 格式: SCORE:0-10|ISSUES:...|REVISED:..."
        )

        review_system = "审查回答与记忆的一致性，严格按格式输出。"

        print(f"\n    ┌─ LLM审查 请求 ─────────────────────")
        print(f"    │ [system] {review_system}")
        for ln in review_prompt.splitlines():
            print(f"    │ {ln}")
        print(f"    │ temperature=0.1  max_tokens=150")
        print(f"    └──────────────────────────────────────")

        review_raw = self.llm.generate(
            review_prompt,
            system=review_system,
            temperature=0.1,
            max_tokens=150,
        )

        result = ReviewResult()

        try:
            if "SCORE:" in review_raw:
                parts = review_raw.split("|")
                for part in parts:
                    part = part.strip()
                    if part.startswith("SCORE:"):
                        score = float(part.replace("SCORE:", "").strip())
                        result.confidence = score / 10.0
                        result.passed = score >= 6
                    elif part.startswith("ISSUES:"):
                        issues_text = part.replace("ISSUES:", "").strip()
                        if issues_text and issues_text != "无":
                            result.issues = [issues_text]
                    elif part.startswith("REVISED:"):
                        revised = part.replace("REVISED:", "").strip()
                        if revised and revised != "无":
                            result.revised_answer = revised
            else:
                result.confidence = 0.7
                result.passed = True
        except Exception:
            result.confidence = 0.5
            result.passed = True

        return result

    # ──────────────────────────────────────────────
    # 完整闭环
    # ──────────────────────────────────────────────

    def full_loop(
        self,
        user_query: str,
        memories: list[tuple[str, float]],
        consciousness_info: dict,
        verbose: bool = True,
    ) -> dict:
        """
        完整意识闭环:
          1. 深度思考决策
          2. 注意力分配 + 生成回答
          3. 自我审查
          4. 返回结果 (含反馈信号供调用者学习)
        """
        top_sim = memories[0][1] if memories else 0.0

        # 阶段 1: 思考策略
        strategy = self.decide_thinking_strategy(consciousness_info, top_sim)
        if verbose:
            print(f"    [策略] {strategy.reasoning}")
            if strategy.needs_deep_thinking:
                print(f"    [深度思考] ON | 检索={strategy.retrieval_k} | "
                      f"温度={strategy.temperature:.1f}")

        # 阶段 2: 生成回答
        import time as _time
        _t0 = _time.perf_counter()
        answer = self.generate_response(
            user_query, memories, consciousness_info, strategy,
        )
        t_generate = _time.perf_counter() - _t0

        no_llm_called = answer == "我的记忆中没有与此相关的信息。"

        if not answer or answer.startswith("[LLM error]"):
            if not answer:
                answer = "（语言模型未生成有效回答，请换个方式再试一次）"
            return {
                "answer": answer,
                "strategy": strategy,
                "review": ReviewResult(passed=False, confidence=0),
                "should_learn": False,
                "t_generate": t_generate,
                "t_review": 0.0,
            }

        # 阶段 3: 自我审查（无记忆时跳过）
        t_review = 0.0
        if no_llm_called:
            review = ReviewResult(passed=True, confidence=1.0)
        else:
            _t0 = _time.perf_counter()
            review = self.self_review(user_query, answer, memories, consciousness_info)
            t_review = _time.perf_counter() - _t0
        if verbose:
            conf_bar = "#" * int(review.confidence * 10) + "-" * (10 - int(review.confidence * 10))
            print(f"    [审查] 置信度=[{conf_bar}] {review.confidence:.1f}")
            if review.issues:
                for issue in review.issues:
                    print(f"    [问题] {issue}")

        # 如果审查未通过且有修正版本，使用修正版
        final_answer = answer
        if not review.passed and review.revised_answer:
            final_answer = review.revised_answer
            if verbose:
                print(f"    [修正] 使用审查后的回答")

        # 阶段 4: 反馈信号
        should_learn = (
            review.passed
            and review.confidence > 0.6
            and not final_answer.startswith("[LLM error]")
        )

        return {
            "answer": final_answer,
            "strategy": strategy,
            "review": review,
            "should_learn": should_learn,
            "feedback_text": f"Q: {user_query}\nA: {final_answer}",
            "t_generate": t_generate,
            "t_review": t_review,
        }

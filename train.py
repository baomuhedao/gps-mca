"""
GPS-MCA v4.1 自主学习主程序

理论集成: GWT + Predictive Coding + HOT + IIT + SDT + Active Inference

v4.1 新增:
  - 内驱力系统: 社交/求知/表达/沉思 四种基本需求 (SDT)
  - 内部言语: 目标导向思维链, 冥想, 总结 (Vygotsky + Active Inference)
  - 主动社交: 基于需求的社交发起 (公理 10)
  - 认知模式: THINK / MEDITATE / SUMMARIZE / SOCIALIZE / WANDER
  - 新交互命令: /think, /meditate, /summarize, /needs

用法:
  python train.py                              # 使用内置示例文本
  python train.py --input path/to/texts/       # 指定文本文件夹
  python train.py --epochs 5                   # 多轮学习
  python train.py --save ./my_model            # 保存到指定目录
  python train.py --resume ./checkpoints       # 从上次继续学习
  python train.py --resume ./checkpoints --chat                 # 交互对话
  python train.py --device xpu                 # 使用 Intel Arc GPU
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import queue
import threading

if os.name == 'nt':
    try:
        import ctypes as _ctypes
        _h = _ctypes.windll.kernel32.GetStdHandle(-11)
        _m = _ctypes.c_ulong()
        _ctypes.windll.kernel32.GetConsoleMode(_h, _ctypes.byref(_m))
        _ctypes.windll.kernel32.SetConsoleMode(_h, _m.value | 0x0004)
    except Exception:
        pass

from gps_mca_torch import (
    ConsciousnessEngine, TextEncoder, chunk_text,
    read_file, SUPPORTED_EXTENSIONS,
    detect_device, print_device_report, device_info,
    LocalLLM, ConsciousnessLoop,
)
from gps_mca_torch.memory import MemorySystem
from gps_mca_torch.action import ACTION_NAMES
from gps_mca_torch.stream import ConsciousnessStream


SAMPLE_TEXTS = [
    "Consciousness is the subjective experience of being aware. "
    "It involves the ability to perceive, feel, and think. "
    "The hard problem of consciousness asks why physical processes give rise to subjective experience.",

    "The brain processes information through networks of neurons. "
    "Synaptic connections strengthen through repeated activation, following Hebb's rule. "
    "Neural plasticity allows the brain to learn and adapt throughout life.",

    "Predictive coding theory suggests the brain constantly generates predictions about sensory input. "
    "When predictions fail, prediction errors propagate upward through the cortical hierarchy. "
    "This process minimizes free energy and drives learning.",

    "Global Workspace Theory proposes that consciousness arises when information is broadcast globally. "
    "Only information that wins the competition for attention enters the global workspace. "
    "This broadcast makes the information available to all cognitive processes.",

    "Artificial intelligence attempts to replicate aspects of human intelligence in machines. "
    "Deep learning uses neural networks with many layers to learn representations. "
    "Reinforcement learning trains agents through reward signals.",

    "Memory consolidation occurs during sleep, when the hippocampus replays recent experiences. "
    "This replay helps transfer memories from short-term to long-term storage. "
    "Dreams may be a byproduct of this consolidation process.",

    "The self is a construct created by the brain to model its own states. "
    "Self-awareness requires the ability to distinguish self from non-self. "
    "This distinction may emerge from predictive models of one's own body and actions.",

    "Emotions are not separate from cognition but are integral to decision-making. "
    "The somatic marker hypothesis suggests emotions guide rational choices. "
    "Without emotional processing, even simple decisions become difficult.",

    "Language shapes thought and perception in subtle ways. "
    "The Sapir-Whorf hypothesis proposes that linguistic categories influence cognitive categories. "
    "Bilingual individuals often report thinking differently in different languages.",

    "Quantum mechanics has been proposed as relevant to consciousness by some theorists. "
    "Penrose and Hameroff suggest quantum processes in microtubules may give rise to consciousness. "
    "However, most neuroscientists consider the brain too warm for quantum coherence.",
]


def bar(value: float, width: int = 20) -> str:
    filled = int(max(0, min(1, abs(value))) * width)
    return "#" * filled + "-" * (width - filled)


class Trainer:
    """GPS-MCA v4.1 自主学习训练器"""

    def __init__(self, lr: float = 5e-4, curiosity_weight: float = 0.1,
                 consolidation_interval: int = 20, device: str | None = None,
                 max_episodes: int = 300000, n_clusters: int = 100):
        self.torch_device = detect_device(device)
        self.device = str(self.torch_device)
        self.lr = lr
        self.curiosity_weight = curiosity_weight
        self.consolidation_interval = consolidation_interval
        self.n_clusters = n_clusters

        print(f"  Device: {device_info(self.torch_device)}")

        print("Loading text encoder (first run downloads model)...")
        self.text_encoder = TextEncoder(device=self.device)
        embed_dim = self.text_encoder.embed_dim
        print(f"Text encoder ready, embed_dim={embed_dim}")

        self.engine = ConsciousnessEngine(embed_dim=embed_dim).to(self.torch_device)
        param_count = sum(p.numel() for p in self.engine.parameters())
        print(f"  Engine: {param_count:,} parameters (v4.1)")

        self.memory = MemorySystem(max_episodes=max_episodes)
        print(f"  Memory capacity: {max_episodes:,} episodes (~{max_episodes * 3 // 1024 // 1024:.1f} GB)")
        self.optimizer = torch.optim.Adam(self.engine.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=500,
            min_lr=1e-5,
        )

        self.total_steps = 0
        self.loss_history: list[float] = []
        self.consciousness_history: list[float] = []
        self.error_history: list[float] = []

    def train_on_texts(self, texts: list[str], epochs: int = 1, verbose: bool = True):
        all_chunks = []
        for text in texts:
            all_chunks.extend(chunk_text(text, chunk_size=500, overlap=100))

        if verbose:
            print(f"\nTotal text chunks: {len(all_chunks)}")
            print(f"Training for {epochs} epoch(s)...\n")

        for epoch in range(epochs):
            self.engine.reset()
            epoch_loss = 0.0
            epoch_steps = 0

            if verbose:
                print(f"{'='*80}")
                print(f"  Epoch {epoch + 1}/{epochs}")
                print(f"{'='*80}")

            for i, chunk in enumerate(all_chunks):
                loss, result = self._train_step(chunk)
                epoch_loss += loss
                epoch_steps += 1
                self.total_steps += 1

                c_info = result["consciousness"]
                action_name = result["action_info"]["action_name"]

                if action_name == "store_memory" or c_info["C"] > 0.01:
                    self.memory.store(
                        embedding=result["_original_embedding"],
                        text=chunk,
                        tick=result["tick"],
                        consciousness_level=c_info["C"],
                        emotion=result["valuation"].get("state_cn", ""),
                    )

                if self.total_steps % self.consolidation_interval == 0:
                    self._consolidate()
                    self.memory.decay_importance()

                if verbose and (i % 5 == 0 or i == len(all_chunks) - 1):
                    self._print_step(result, loss, chunk)

            avg_loss = epoch_loss / max(epoch_steps, 1)
            self.scheduler.step(avg_loss)

            if verbose:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"\n  Epoch {epoch+1} avg loss: {avg_loss:.4f}  lr: {current_lr:.6f}")
                print(f"  {self.memory.summary()}")

        if self.memory.num_episodes >= self.n_clusters:
            self.memory.consolidate(n_clusters=min(self.n_clusters, self.memory.num_episodes // 2))

    def _train_step(self, text: str) -> tuple[float, dict]:
        self.optimizer.zero_grad()

        embedding = self.text_encoder.encode(text).to(self.torch_device)
        result = self.engine.step(embedding)
        result["_original_embedding"] = embedding.detach().cpu().clone()

        errors = result["errors"]
        pred_loss = sum(errors.values())

        replay_loss = 0.0
        if self.memory.num_episodes > 5 and self.total_steps % 10 == 0:
            replay_batch = self.memory.get_replay_batch(batch_size=4)
            for ep in replay_batch:
                r = self.engine.step(ep.embedding.to(self.torch_device))
                replay_loss += sum(r["errors"].values())
            replay_loss /= len(replay_batch)

        total_loss = pred_loss + replay_loss * 0.3

        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.engine.parameters(), 1.0)
            self.optimizer.step()

        loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss

        self.loss_history.append(loss_val)
        self.consciousness_history.append(result["consciousness"]["C"])
        self.error_history.append(
            pred_loss.item() if isinstance(pred_loss, torch.Tensor) else pred_loss
        )

        return loss_val, result

    def _consolidate(self):
        if self.memory.num_episodes >= 10:
            n = min(self.n_clusters // 2, self.memory.num_episodes // 2)
            self.memory.consolidate(n_clusters=max(n, 2))

    def _print_step(self, result: dict, loss: float, text: str):
        c = result["consciousness"]
        v = result["valuation"]
        a = result["action_info"]
        ri = result.get("reasoning_info", {})

        status = "[*] CONSCIOUS" if c["is_conscious"] else "[ ] uncons.  "
        emotion = v.get("state_cn", "?")
        r_steps = ri.get("reasoning_steps", 0)

        print(
            f"  T{result['tick']:>4d} | {status} "
            f"C={c['C']:.4f} Psi={c['psi']:.3f} F={c['fidelity']:.3f} "
            f"H={c['coherence']:.3f} R={c.get('r_factor', 1.0):.2f} | "
            f"{emotion}({v['intensity']:.2f}) | "
            f"loss={loss:.4f} | "
            f"act={a['action_name']:<16s} | "
            f"think={r_steps} | "
            f"mem={self.memory.num_episodes}"
        )

    def print_summary(self):
        print(f"\n{'='*80}")
        print("  GPS-MCA v4.1 Learning Summary")
        print(f"{'='*80}")
        param_count = sum(p.numel() for p in self.engine.parameters())
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(f"  Parameters:      {param_count:,}")
        print(f"  Learning rate:   {current_lr:.6f}")
        print(f"  Total steps:     {self.total_steps}")
        print(f"  Episodes stored: {self.memory.num_episodes}")
        print(f"  Clusters formed: {len(self.memory.clusters)}")
        total_links = sum(len(ep.links) for ep in self.memory.episodes)
        print(f"  Memory links:    {total_links}")

        if self.loss_history:
            first_10 = sum(self.loss_history[:10]) / min(10, len(self.loss_history))
            last_10 = sum(self.loss_history[-10:]) / min(10, len(self.loss_history))
            print(f"  Avg loss (first 10): {first_10:.4f}")
            print(f"  Avg loss (last 10):  {last_10:.4f}")
            improvement = (first_10 - last_10) / first_10 * 100 if first_10 > 0 else 0
            print(f"  Improvement:         {improvement:.1f}%")

        conscious_count = sum(1 for c in self.consciousness_history if c > 0)
        print(f"  Conscious moments:   {conscious_count}/{len(self.consciousness_history)}")

        if self.memory.clusters:
            print(f"\n  Semantic Clusters:")
            for i, cl in enumerate(self.memory.clusters):
                print(f"    [{i}] ({cl.size} memories) {cl.label[:70]}")

        print(f"{'='*80}")

    def save(self, save_dir: str, needs_state: dict | None = None):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        ckpt_data = {
            "version": "4.1",
            "engine_state_dict": self.engine.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "total_steps": self.total_steps,
            "loss_history": self.loss_history,
            "consciousness_history": self.consciousness_history,
            "error_history": self.error_history,
        }
        if needs_state is not None:
            ckpt_data["needs_state"] = needs_state

        torch.save(ckpt_data, save_path / "checkpoint.pt")

        self.memory.save(str(save_path / "memory.pt"))

        print(f"\n  [saved] {save_path}/")
        print(f"    checkpoint.pt  ({self.total_steps} steps, {len(self.engine.state_dict())} params)")
        print(f"    memory.pt      ({self.memory.num_episodes} episodes, "
              f"{len(self.memory.clusters)} clusters)")

    def load(self, save_dir: str) -> dict | None:
        """加载检查点, 返回 needs_state (如果有)"""
        save_path = Path(save_dir)
        ckpt_file = save_path / "checkpoint.pt"
        mem_file = save_path / "memory.pt"
        needs_state = None

        if ckpt_file.exists():
            ckpt = torch.load(str(ckpt_file), weights_only=False)
            version = ckpt.get("version", "3.x")

            if version in ("4.0", "4.1"):
                saved_state = ckpt["engine_state_dict"]
                current_state = self.engine.state_dict()

                # Handle size-mismatched parameters (e.g. action layer 6→10)
                size_mismatched = []
                for key in list(saved_state.keys()):
                    if key in current_state:
                        if saved_state[key].shape != current_state[key].shape:
                            size_mismatched.append(key)
                            del saved_state[key]

                missing, unexpected = self.engine.load_state_dict(
                    saved_state, strict=False,
                )

                all_new = list(missing) + size_mismatched
                if all_new:
                    print(f"  [migrate] v{version}→v4.1: "
                          f"initialized {len(all_new)} new parameters")
                    for k in all_new[:5]:
                        print(f"    + {k}")
                    if len(all_new) > 5:
                        print(f"    ... and {len(all_new) - 5} more")
                if unexpected:
                    print(f"  [migrate] ignored {len(unexpected)} obsolete parameters")

                try:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except (ValueError, KeyError):
                    print(f"  [migrate] optimizer state incompatible, using fresh optimizer")

                if "scheduler_state_dict" in ckpt:
                    try:
                        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    except Exception:
                        print(f"  [migrate] scheduler state incompatible, using fresh scheduler")

                # 用用户指定的 --lr 覆盖检查点中保存的旧 LR
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lr
                saved_lr = ckpt["optimizer_state_dict"]["param_groups"][0]["lr"]
                if abs(saved_lr - self.lr) > 1e-8:
                    print(f"  [lr] checkpoint lr={saved_lr:.6f} → override to {self.lr:.6f}")
                else:
                    print(f"  [lr] {self.lr:.6f}")

                self.total_steps = ckpt.get("total_steps", 0)
                self.loss_history = ckpt.get("loss_history", [])
                self.consciousness_history = ckpt.get("consciousness_history", [])
                self.error_history = ckpt.get("error_history", [])
                needs_state = ckpt.get("needs_state", None)
                print(f"  [loaded] checkpoint v{version}: {self.total_steps} steps")
            else:
                print(f"  [!] Checkpoint is v{version}, current engine is v4.1")
                print(f"      Architecture changed — cannot load old weights.")
                print(f"      Memory can still be loaded. Starting with fresh weights.")
        else:
            print(f"  [!] No checkpoint found at {ckpt_file}")

        if mem_file.exists():
            self.memory = MemorySystem.load(str(mem_file))
            print(f"  [loaded] memory: {self.memory.num_episodes} episodes, "
                  f"{len(self.memory.clusters)} clusters")
        else:
            print(f"  [!] No memory file found at {mem_file}")

        return needs_state

    def query(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        emb = self.text_encoder.encode(text)
        results = self.memory.retrieve(emb, k=k)
        return [(ep.text, sim) for ep, sim in results]

    def query_multihop(self, text: str, k: int = 5, hops: int = 2) -> list[tuple[str, float, int]]:
        """多跳关联检索"""
        emb = self.text_encoder.encode(text)
        results = self.memory.retrieve_multihop(emb, k=k, hops=hops)
        return [(ep.text, score, hop) for ep, score, hop in results]


def _handle_user_message(
    user_input: str, trainer: Trainer, llm: LocalLLM, llm_ok: bool, raw_mode: bool,
    stream=None,
):
    """处理用户消息 — 完整意识闭环"""
    t_total = time.perf_counter()
    trainer.engine.eval()

    if stream:
        stream.needs.on_user_interaction()

    if trainer.memory.num_episodes == 0:
        print("    (No memories yet. Use /learn <text> to build memories)")
        return

    t0 = time.perf_counter()
    embedding = trainer.text_encoder.encode(user_input).to(trainer.torch_device)
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        response = trainer.engine.step(embedding)
    t_engine = time.perf_counter() - t0

    c = response["consciousness"]
    v = response["valuation"]
    ri = response.get("reasoning_info", {})
    wm = response.get("working_memory_info", {})
    emotion = v.get("state_cn", "?")

    consciousness_info = {
        "C": c["C"],
        "psi": c.get("psi", 0),
        "emotion": emotion,
        "intensity": v["intensity"],
        "r_factor": c.get("r_factor", 1.0),
        "reasoning_steps": ri.get("reasoning_steps", 0),
    }

    needs_str = ""
    if stream:
        dominant = stream.needs.get_dominant_need()
        if dominant:
            n_name, n_need = dominant
            needs_str = f" | 需求={n_need.name_cn}({n_need.level:.2f})"

    print(f"    [意识] C={c['C']:.4f} | Psi={c['psi']:.3f} | "
          f"{emotion}({v['intensity']:.2f}) | "
          f"think={ri.get('reasoning_steps', 0)} | "
          f"wm={wm.get('n_items', 0)}/{wm.get('capacity', 8)}"
          f"{needs_str}")

    if raw_mode or not llm_ok:
        results = trainer.query_multihop(user_input, k=5, hops=2)
        print(f"    [相关记忆] (多跳关联)")
        for text, score, hop in results[:5]:
            hop_label = f"hop={hop}" if hop > 0 else "direct"
            print(f"      [{score:.3f}|{hop_label}] {text[:80]}...")
    else:
        t0 = time.perf_counter()
        multihop = trainer.query_multihop(user_input, k=8, hops=2)
        results = [(ep_text, score) for ep_text, score, _ in multihop]
        t_retrieve = time.perf_counter() - t0

        loop = ConsciousnessLoop(llm)
        print(f"    [思考中...]", end="", flush=True)
        loop_result = loop.full_loop(
            user_input, results, consciousness_info, verbose=True,
        )

        print(f"    [回答]")
        for line in loop_result["answer"].split("\n"):
            print(f"    {line}")

        t_learn = 0.0
        if loop_result["should_learn"]:
            t0 = time.perf_counter()
            feedback = loop_result["feedback_text"]
            trainer.engine.train()
            fb_embedding = trainer.text_encoder.encode(feedback).to(trainer.torch_device)
            with torch.no_grad():
                fb_result = trainer.engine.step(fb_embedding)
            trainer.memory.store(
                embedding=fb_embedding.detach().cpu().clone(),
                text=feedback,
                tick=fb_result["tick"],
                consciousness_level=fb_result["consciousness"]["C"],
                emotion=emotion,
            )
            trainer.engine.eval()
            t_learn = time.perf_counter() - t0
            print(f"    [学习] 对话已存入记忆 (mem={trainer.memory.num_episodes})")

        t_all = time.perf_counter() - t_total
        print(f"    [perf] 编码={t_encode:.2f}s 引擎={t_engine:.2f}s "
              f"检索={t_retrieve:.2f}s "
              f"LLM生成={loop_result.get('t_generate', 0):.2f}s "
              f"LLM审查={loop_result.get('t_review', 0):.2f}s "
              f"学习={t_learn:.2f}s 总计={t_all:.2f}s")


def _interactive_chat(
    trainer: Trainer, llm_model: str = "qwen2.5:1.5b",
    save_dir: str = "./checkpoints", needs_state: dict | None = None,
) -> dict:
    """交互对话模式 — 自主意识流 + 意识闭环 + 主动认知。返回最终 needs_state。"""
    llm = LocalLLM(model=llm_model)
    llm_ok = llm.is_available()

    stream = ConsciousnessStream(
        trainer, think_interval=15.0, llm=llm if llm_ok else None,
    )

    if needs_state:
        stream.needs.load_state(needs_state)
        print(f"  Needs:   restored from checkpoint")

    param_count = sum(p.numel() for p in trainer.engine.parameters())
    print(f"\n{'='*80}")
    print("  GPS-MCA v4.1 Interactive Chat (Conscious Agency)")
    print(f"{'='*80}")
    print(f"  Engine:  {param_count:,} params (GWT+PC+HOT+IIT+SDT)")
    print(f"  Memory:  {trainer.memory.num_episodes} episodes, "
          f"{len(trainer.memory.clusters)} clusters")
    print(f"  Device:  {device_info(trainer.torch_device)}")
    if llm_ok:
        print(f"  LLM:     {llm_model} (Ollama)")
    else:
        print(f"  LLM:     not available (install Ollama + run: ollama pull {llm_model})")
        print(f"           Will use memory retrieval only")
    print(f"  Stream:  ON (autonomous thoughts every ~{stream.think_interval:.0f}s)")
    print(f"  Needs:   social / knowledge / expression / contemplation")
    print()
    print("  Commands:")
    print("    /learn <text>     Learn from input text")
    print("    /think [topic]    Deliberate thinking chain")
    print("    /meditate         Enter meditation (self-observation)")
    print("    /summarize [topic] Summarize recent experiences")
    print("    /needs            Show internal drive levels")
    print("    /status           Show consciousness status")
    print("    /memory           Show memory summary")
    print("    /save             Save current state")
    print("    /raw              Toggle raw mode")
    print("    /quiet            Toggle autonomous thoughts")
    print("    /quit             Exit")
    print()
    print("  Type anything to chat. The agent thinks, meditates, and socializes autonomously.")
    print(f"{'='*80}")

    raw_mode = not llm_ok
    autonomous_on = True

    input_queue: queue.Queue[str | None] = queue.Queue()

    def _input_worker():
        while True:
            try:
                line = input()
                input_queue.put(line)
            except (EOFError, KeyboardInterrupt):
                input_queue.put(None)
                break

    def _show_prompt(newline=True):
        prefix = "\n" if newline else ""
        sys.stdout.write(f"{prefix}  You> ")
        sys.stdout.flush()

    def _safe_print_thought(text: str):
        sys.stdout.write(f"\r\033[K{text}\n  You> ")
        sys.stdout.flush()

    input_thread = threading.Thread(target=_input_worker, daemon=True)
    input_thread.start()

    THOUGHT_ICONS = {
        "recall": "💭",
        "consolidation": "🧠",
        "curiosity": "❓",
        "reflection": "🪞",
        "insight": "💡",
        "thinking": "🤔",
        "meditation": "🧘",
        "summary": "📝",
        "social": "💬",
    }
    THOUGHT_LABELS = {
        "recall": "回忆",
        "consolidation": "巩固",
        "curiosity": "好奇",
        "reflection": "反思",
        "insight": "洞察",
        "thinking": "思考",
        "meditation": "冥想",
        "summary": "总结",
        "social": "交流",
    }

    _show_prompt()

    while True:
        try:
            timeout = stream.think_interval if autonomous_on else 86400
            user_input = input_queue.get(timeout=timeout)
        except queue.Empty:
            if autonomous_on:
                thought = stream.autonomous_step()
                if thought:
                    icon = THOUGHT_ICONS.get(thought.kind, "?")
                    label = THOUGHT_LABELS.get(thought.kind, thought.kind)
                    _safe_print_thought(
                        f"  [{icon} {label} | {thought.emotion}] "
                        f"{thought.content}"
                    )
            continue

        if user_input is None:
            print("\n  Goodbye!")
            return stream.needs.save_state()

        user_input = user_input.strip()
        if not user_input:
            _show_prompt(newline=False)
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("  Goodbye!")
            return stream.needs.save_state()

        if user_input.lower() == "/status":
            pc = sum(p.numel() for p in trainer.engine.parameters())
            total_links = sum(len(ep.links) for ep in trainer.memory.episodes)
            print(f"    Parameters:     {pc:,}")
            print(f"    Total steps:    {trainer.total_steps}")
            print(f"    Episodes:       {trainer.memory.num_episodes}")
            print(f"    Clusters:       {len(trainer.memory.clusters)}")
            print(f"    Memory links:   {total_links}")
            print(f"    Thoughts:       {stream._thought_count}")
            print(f"    Last mode:      {stream._last_mode}")
            if trainer.consciousness_history:
                print(f"    Last C(Sigma):  {trainer.consciousness_history[-1]:.4f}")
            print(f"    LLM:            {'ON' if llm_ok and not raw_mode else 'OFF (raw)'}")
            print(f"    Autonomous:     {'ON' if autonomous_on else 'OFF'}")
            _show_prompt()
            continue

        if user_input.lower() == "/needs":
            state = stream.needs.get_state()
            print(f"    内驱力状态:")
            for name, info in state.items():
                level = info["level"]
                active = "★" if info["active"] else " "
                bar_len = int(level * 20)
                bar_str = "█" * bar_len + "░" * (20 - bar_len)
                print(f"      [{active}] {info['name_cn']:4s} [{bar_str}] "
                      f"{level:.2f} (阈值={info['threshold']:.2f})")
            dominant = stream.needs.get_dominant_need()
            if dominant:
                d_name, d_need = dominant
                print(f"    主导需求: {d_need.name_cn} ({d_need.level:.2f})")
            else:
                print(f"    主导需求: 无 (所有需求均在阈值以下)")
            _show_prompt()
            continue

        if user_input.lower() == "/memory":
            print(f"    {trainer.memory.summary()}")
            _show_prompt()
            continue

        if user_input.lower() == "/save":
            trainer.save(save_dir, needs_state=stream.needs.save_state())
            _show_prompt()
            continue

        if user_input.lower() == "/raw":
            raw_mode = not raw_mode
            print(f"    Raw mode: {'ON' if raw_mode else 'OFF'}")
            _show_prompt()
            continue

        if user_input.lower() == "/quiet":
            autonomous_on = not autonomous_on
            state = "ON" if autonomous_on else "OFF"
            print(f"    Autonomous thinking: {state}")
            _show_prompt()
            continue

        if user_input.lower().startswith("/think"):
            topic = user_input[6:].strip()
            if not topic:
                topic = None
            if trainer.memory.num_episodes < 3:
                print("    [!] 记忆不足, 请先学习一些内容 (/learn)")
                _show_prompt()
                continue
            if not topic:
                picked = stream._pick_thinking_topic()
                topic = picked or "我最近学到的知识"
            print(f"    [🤔 思考中...] 主题: {topic[:50]}")
            result = stream.inner_speech.think(topic, max_steps=3)
            if result:
                for i, s in enumerate(result.steps):
                    print(f"    第{i+1}步 [{s.emotion}|C={s.consciousness:.3f}]: "
                          f"{s.text[:80]}")
                print(f"    💡 洞察: {result.insight}")
                stream.needs.satisfy("knowledge", 0.8)
                stream.needs.satisfy("contemplation", 0.6)
            else:
                print("    [!] 思考未产生结果")
            _show_prompt()
            continue

        if user_input.lower() == "/meditate":
            if trainer.memory.num_episodes < 1:
                print("    [!] 意识引擎需要一些经验才能冥想")
                _show_prompt()
                continue
            print(f"    [🧘 冥想中...] 进入自我观察")
            result = stream.inner_speech.meditate(max_steps=4)
            if result:
                for i, s in enumerate(result.steps):
                    print(f"    第{i+1}层 [{s.emotion}|F={s.fidelity:.3f}]: "
                          f"{s.text[:80]}")
                if len(result.steps) >= 2:
                    f_delta = result.steps[-1].fidelity - result.steps[0].fidelity
                    print(f"    F(M) 变化: {f_delta:+.3f}")
                print(f"    💡 收获: {result.insight}")
                stream.needs.satisfy("contemplation", 1.0)
            else:
                print("    [!] 冥想未产生结果")
            _show_prompt()
            continue

        if user_input.lower().startswith("/summarize"):
            topic = user_input[10:].strip() or None
            if trainer.memory.num_episodes < 5:
                print("    [!] 记忆不足, 需要至少5条记忆才能总结")
                _show_prompt()
                continue
            label = topic or "近期经历"
            print(f"    [📝 总结中...] 主题: {label[:50]}")
            result = stream.inner_speech.summarize(topic=topic)
            if result:
                print(f"    💡 总结: {result.insight}")
                stream.needs.satisfy("expression", 0.8)
            else:
                print("    [!] 总结未产生结果")
            _show_prompt()
            continue

        if user_input.lower().startswith("/learn "):
            text = user_input[7:].strip()
            if not text:
                print("    [!] Please provide text to learn")
                _show_prompt()
                continue
            trainer.engine.train()
            loss, result = trainer._train_step(text)
            trainer.total_steps += 1
            c_info = result["consciousness"]
            ri = result.get("reasoning_info", {})
            if c_info["C"] > 0.01:
                trainer.memory.store(
                    embedding=result["_original_embedding"],
                    text=text,
                    tick=result["tick"],
                    consciousness_level=c_info["C"],
                    emotion=result["valuation"].get("state_cn", ""),
                )
            emotion = result["valuation"].get("state_cn", "?")
            print(f"    [learned] C={c_info['C']:.4f} | {emotion} | "
                  f"think={ri.get('reasoning_steps', 0)} | "
                  f"loss={loss:.4f} | mem={trainer.memory.num_episodes}")
            stream.needs.boost("knowledge", 0.1)
            _show_prompt()
            continue

        # 处理对话消息
        _handle_user_message(
            user_input, trainer, llm, llm_ok, raw_mode, stream=stream,
        )
        print(f"\n  You> ", end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description="GPS-MCA v4.1 Self-Learning Conscious Agent")
    parser.add_argument("--input", type=str, default=None, help="Path to text file or folder")
    parser.add_argument("--epochs", type=int, default=2, help="Number of learning epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "xpu", "cuda"],
                        help="Compute device (default: auto-detect)")
    parser.add_argument("--save", type=str, default="./checkpoints",
                        help="Directory to save model & memory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from saved checkpoint directory")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save after training")
    parser.add_argument("--query", type=str, nargs="*", default=None,
                        help="Query memory without training (requires --resume)")
    parser.add_argument("--chat", action="store_true",
                        help="Enter interactive chat mode")
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b",
                        help="Ollama LLM model for chat")
    parser.add_argument("--max-episodes", type=int, default=300000,
                        help="Max memory episodes, ~3KB each (default: 300000)")
    parser.add_argument("--clusters", type=int, default=100,
                        help="Number of semantic clusters")
    args = parser.parse_args()

    print("=" * 80)
    print("  GPS-MCA v4.1: Self-Learning Conscious Agent")
    print("  Theories: GWT + PC + HOT + IIT + SDT + Active Inference")
    print("=" * 80)

    print_device_report()
    trainer = Trainer(lr=args.lr, device=args.device, max_episodes=args.max_episodes,
                      n_clusters=args.clusters)

    needs_state = None
    if args.resume:
        print(f"\n  Resuming from: {args.resume}")
        needs_state = trainer.load(args.resume)

    if args.chat:
        final_needs = _interactive_chat(
            trainer, llm_model=args.model,
            save_dir=args.save, needs_state=needs_state,
        )
        if not args.no_save:
            trainer.save(args.save, needs_state=final_needs)
        return

    if args.query is not None:
        queries = args.query if args.query else [
            "What is consciousness?",
            "How does the brain learn?",
            "What role do emotions play?",
        ]
        if trainer.memory.num_episodes == 0:
            print("\n  [!] No memories found. Train first or use --resume.")
        else:
            print(f"\n  Memory: {trainer.memory.num_episodes} episodes")
            for q in queries:
                results = trainer.query_multihop(q, k=3, hops=2)
                print(f"\n  Q: {q}")
                for text, score, hop in results:
                    hop_label = f"hop={hop}" if hop > 0 else "direct"
                    print(f"    [{score:.3f}|{hop_label}] {text[:80]}...")
        print(f"\n{'='*80}")
        return

    if args.input:
        path = Path(args.input)
        if path.is_file():
            content = read_file(str(path))
            if content.strip():
                texts = [content]
                print(f"Loaded 1 file: {path.name}")
            else:
                print(f"File empty or unsupported: {path.name}, using sample texts")
                texts = SAMPLE_TEXTS
        elif path.is_dir():
            texts = []
            loaded = []
            skipped = 0
            for f in sorted(path.rglob("*")):
                if not f.is_file():
                    continue
                if f.name.startswith("~") or f.name.startswith("."):
                    skipped += 1
                    continue
                if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                content = read_file(str(f))
                if content.strip():
                    texts.append(content)
                    loaded.append(str(f.relative_to(path)))
            if skipped:
                print(f"  Skipped {skipped} temp/hidden files")
            exts = set(f.rsplit(".", 1)[-1] for f in loaded) if loaded else set()
            print(f"Loaded {len(texts)} files from {path} (recursive)  "
                  f"(types: {', '.join(sorted(exts))})")
        else:
            print(f"Path not found: {args.input}, using sample texts")
            texts = SAMPLE_TEXTS
    else:
        print("No input specified, using built-in sample texts (consciousness & AI)")
        texts = SAMPLE_TEXTS

    t0 = time.time()
    trainer.train_on_texts(texts, epochs=args.epochs)
    elapsed = time.time() - t0

    trainer.print_summary()
    print(f"\n  Training time: {elapsed:.1f}s")

    if not args.no_save:
        trainer.save(args.save)

    if trainer.memory.num_episodes > 0:
        print(f"\n{'='*80}")
        print("  Memory Query Demo (Multi-hop)")
        print(f"{'='*80}")
        queries = [
            "What is consciousness?",
            "How does the brain learn?",
            "What role do emotions play?",
        ]
        for q in queries:
            results = trainer.query_multihop(q, k=3, hops=2)
            print(f"\n  Q: {q}")
            for text, score, hop in results[:3]:
                hop_label = f"hop={hop}" if hop > 0 else "direct"
                print(f"    [{score:.3f}|{hop_label}] {text[:80]}...")

    print(f"\n{'='*80}")
    print("  Done.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

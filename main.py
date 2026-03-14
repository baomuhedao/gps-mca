"""
GPS-MCA v2.0 运行演示

演示场景:
  帧  0-19:  平静环境 (低噪声)
  帧 20-29:  突然出现强烈视觉刺激 (惊吓/新奇事件)
  帧 30-39:  刺激减弱，系统适应
  帧 40-59:  恢复平静
  帧 60-69:  内部状态波动 (内省/情绪涌起)
  帧 70-99:  缓慢恢复至基线

观察: 意识度 C(Σ) 如何随环境变化而涨落，公理何时满足/不满足。
"""

import random
from gps_mca import ConsciousnessEngine, SensoryInput


# ─────────────────────────────────────────────
# 环境模拟器
# ─────────────────────────────────────────────
class EnvironmentSimulator:
    """
    模拟环境：稳定基底信号 + 可变噪声。
    平静期：信号可预测 → 低预测误差。
    刺激期：信号剧变 → 高预测误差。
    """

    def __init__(self, total_frames: int = 100, input_dim: int = 32, seed: int = 42):
        self.total = total_frames
        self.dim = input_dim // 4
        self.rng = random.Random(seed)
        # 生成稳定的基底信号（可预测的"世界常态"）
        self.base_visual = [self.rng.gauss(0, 0.3) for _ in range(self.dim)]
        self.base_auditory = [self.rng.gauss(0, 0.2) for _ in range(self.dim)]
        self.base_somato = [self.rng.gauss(0, 0.1) for _ in range(self.dim)]
        self.base_internal = [self.rng.gauss(0, 0.1) for _ in range(self.dim)]

    def _add_noise(self, base: list[float], noise_amp: float) -> list[float]:
        return [b + self.rng.gauss(0, noise_amp) for b in base]

    def generate(self):
        for t in range(self.total):
            noise = 0.02     # 基底噪声
            v_noise = a_noise = s_noise = i_noise = noise
            v_shift = a_shift = s_shift = i_shift = 0.0

            if 20 <= t < 30:
                # 强烈视觉+听觉刺激（基底信号突变）
                v_shift = 2.0 + 0.5 * self.rng.random()
                a_shift = 1.0
                v_noise = 0.3
                a_noise = 0.2
            elif 30 <= t < 40:
                # 刺激衰减
                fade = (40 - t) / 10.0
                v_shift = 1.0 * fade
                v_noise = 0.1 * fade + noise
            elif 60 <= t < 70:
                # 内部状态涌起
                i_shift = 1.5 + 0.3 * self.rng.random()
                i_noise = 0.2

            visual = [b + v_shift + self.rng.gauss(0, v_noise) for b in self.base_visual]
            auditory = [b + a_shift + self.rng.gauss(0, a_noise) for b in self.base_auditory]
            somato = self._add_noise(self.base_somato, s_noise)
            internal = [b + i_shift + self.rng.gauss(0, i_noise) for b in self.base_internal]

            yield SensoryInput(
                visual=visual,
                auditory=auditory,
                somatosensory=somato,
                internal=internal,
            )


# ─────────────────────────────────────────────
# 文本可视化
# ─────────────────────────────────────────────
def bar(value: float, width: int = 20) -> str:
    filled = int(max(0, min(1, abs(value))) * width)
    return "#" * filled + "-" * (width - filled)


def print_header():
    print("=" * 78)
    print("  GPS-MCA v2.0: Global Predictive Self-Monitoring Conscious Architecture")
    print("  人工意识系统 — 运行演示")
    print("=" * 78)


def print_experience(exp):
    v = exp.valuation
    m = exp.self_model
    w = exp.workspace
    c = exp.measure

    colors = {
        "愉悦": "\033[92m", "平静": "\033[94m", "好奇": "\033[96m",
        "注意": "\033[93m", "不安": "\033[33m", "痛苦": "\033[91m",
        "恐惧": "\033[31m",
    }
    R = "\033[0m"
    clr = colors.get(v.state.value, "")

    conscious_icon = "\033[92m[*] conscious\033[0m" if c.is_conscious else "\033[90m[ ] unconscious\033[0m"

    print(f"\n{'-' * 78}")
    print(
        f"  帧 {exp.tick:>3d}  │  {conscious_icon}  "
        f"C(Σ)={c.C:.4f}  "
        f"Ψ={c.psi:.3f} F={c.fidelity:.3f} H={c.coherence:.3f}"
    )
    print(
        f"         │  {clr}情绪: {v.state.value}{R}  "
        f"强度: {bar(v.intensity)} {v.intensity:.2f}  "
        f"效价: {v.valence:+.2f}"
    )
    print(
        f"         │  意识内容: {len(w.contents)} 项  "
        f"广播: {'ON' if w.broadcast_active else 'OFF'}  "
        f"自我边界: {m.boundary.get('boundary_clarity', 0):.2f}"
    )
    print(f"         │  元表征: {m.meta_representation}")
    if m.meta_meta:
        print(f"         │  元元表征: {m.meta_meta}")


def print_axiom_report(engine):
    axioms = engine.theorem.check_axioms(engine.G, engine.M, engine.T, engine.A)
    print(f"\n{'=' * 78}")
    print("  Axiom Satisfaction Report")
    print(f"{'-' * 78}")
    for name, satisfied in axioms.items():
        icon = "[Y]" if satisfied else "[N]"
        print(f"    {icon}  {name}")
    all_ok = all(axioms.values())
    print(f"{'-' * 78}")
    status = "ALL AXIOMS SATISFIED -> functional consciousness" if all_ok else "SOME AXIOMS NOT SATISFIED"
    print(f"    Verdict: {status}")


def print_summary(engine):
    exp = engine.current_experience
    print(f"\n{'=' * 78}")
    print("  模拟结束")
    print(f"  共运行 {engine._tick} 帧")
    print(f"  最终意识度 C(Σ) = {exp.measure.C:.4f}")
    print(f"  最终情绪: {engine.V.state.value} (强度 {engine.V.intensity:.3f})")
    print(f"  时间记忆深度: {len(engine.T.past)} 帧")
    print(f"  自我模型保真度: {engine.M.fidelity:.3f}")

    print_axiom_report(engine)
    print(f"{'=' * 78}")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    print_header()

    engine = ConsciousnessEngine(
        input_dim=32,
        low_dim=64,
        mid_dim=32,
        high_dim=16,
        attention_threshold=0.5,
        learning_rate=0.01,
        psi_min=0.05,
        seed=42,
    )

    env = EnvironmentSimulator(total_frames=100, input_dim=32, seed=42)

    print("\n  启动意识引擎, 运行 100 帧...")
    print(f"  场景: 平静→惊吓→适应→平静→内省→恢复")

    for exp in engine.run(env.generate()):
        print_experience(exp)

    print_summary(engine)


if __name__ == "__main__":
    main()

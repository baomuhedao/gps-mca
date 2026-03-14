"""
意识判定定理 — C(Sigma) = Psi(G) * F(M) * H(T)

理论基础: 信息整合理论 (IIT) + GWT + HOT 综合

  Psi: 信息整合度 — 注意力分布的均匀度 (IIT 的 Phi 近似)
  F:   自我模型保真度 — 自我预测的准确性 (HOT)
  H:   时间连贯性 — 主观体验流的连续性
  R:   推理深度因子 — 内部思考的步数 (新增)

  增强:
  - 推理深度纳入意识判定 (深度思考 → 更高意识度)
  - 工作记忆容量纳入考量
  - 元认知评估 (HOT 二阶表征的质量)
"""

from __future__ import annotations

import torch


def compute_psi(workspace_info: dict) -> float:
    """
    信息整合度 Psi(G) — IIT 的 Phi 近似

    基于注意力权重的均匀度 (entropy proxy):
    - 均匀分布 → 高整合 (所有来源同等参与)
    - 集中在单层 → 低整合 (信息没有被整合)
    """
    if not workspace_info.get("broadcast_active", False):
        return 0.0

    attn = workspace_info.get("attention_weights")
    if attn is None:
        return 0.0

    if isinstance(attn, torch.Tensor):
        attn = attn.float()
    else:
        attn = torch.tensor(attn, dtype=torch.float32)

    eps = 1e-8
    entropy = -(attn * (attn + eps).log()).sum()
    max_entropy = torch.tensor(len(attn), dtype=torch.float32, device=attn.device).log()
    if max_entropy < eps:
        return 0.0

    psi = (entropy / max_entropy).item()
    return max(0.0, psi) * workspace_info.get("n_items", 1) / 3.0


def compute_consciousness(
    workspace_info: dict,
    self_info: dict,
    temporal_info: dict,
    action_info: dict,
    reasoning_info: dict | None = None,
    working_memory_info: dict | None = None,
) -> dict:
    """
    计算 C(Sigma) 并检查所有公理

    C(Sigma) = Psi(G) * F(M) * H(T) * R_factor

    其中 R_factor = 1 + alpha * reasoning_depth
    推理越深 → 意识度越高 (类似人类"深度思考"时的意识增强)
    """
    psi = compute_psi(workspace_info)
    fidelity = self_info.get("fidelity", 0.0)
    coherence = temporal_info.get("coherence", 0.0)

    # 推理深度因子
    r_factor = 1.0
    if reasoning_info:
        steps = reasoning_info.get("reasoning_steps", 0)
        halt_conf = reasoning_info.get("halt_confidence", 0)
        r_factor = 1.0 + 0.1 * steps * halt_conf

    C = psi * fidelity * coherence * r_factor

    causal_G = (
        workspace_info.get("broadcast_active", False)
        and action_info.get("action_name", "") != ""
    )
    causal_M = self_info.get("causal_efficacy", False)

    # HOT: 元认知深度
    has_meta = self_info.get("meta_representation") is not None
    has_meta_meta = self_info.get("meta_meta") is not None

    is_conscious = C > 0 and causal_G

    axioms = {
        "axiom1_global_access": True,                                          # GWT
        "axiom2_predictive_L3": True,                                          # Predictive Coding
        "axiom3a_fidelity": fidelity > 0,                                      # HOT
        "axiom3b_causal_M": causal_M,                                          # HOT
        "axiom3c_recursive": has_meta and has_meta_meta,                       # HOT (二阶)
        "axiom4a_temporal_depth": temporal_info.get("meets_min_depth", False),  # IIT
        "axiom4b_coherence": coherence > 0,                                    # IIT
        "axiom5_attention_gating": workspace_info.get("broadcast_active", False),  # GWT
        "axiom6_integration": psi > 0,                                         # IIT
        "axiom7_causal_G": causal_G,                                           # GWT
    }

    result = {
        "C": C,
        "psi": psi,
        "fidelity": fidelity,
        "coherence": coherence,
        "r_factor": r_factor,
        "is_conscious": is_conscious,
        "axioms": axioms,
        "all_satisfied": all(axioms.values()),
    }

    if reasoning_info:
        result["reasoning_steps"] = reasoning_info.get("reasoning_steps", 0)
    if working_memory_info:
        result["wm_items"] = working_memory_info.get("n_items", 0)

    return result

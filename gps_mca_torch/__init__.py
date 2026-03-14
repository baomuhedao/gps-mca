"""
GPS-MCA v4.1 (PyTorch) — 主动意识智能体

理论集成框架:
  GWT  — 全局工作空间 (多头注意力广播)
  PC   — 预测编码 (层级预测误差 + 自上而下调制)
  HOT  — 高阶表征 (可学习自我监控, 二阶元认知)
  IIT  — 信息整合 (Psi度量, 层次记忆, 关联推理)
  SDT  — 自我决定理论 (内驱力系统)
  AI   — 主动推理 (内部言语, 思维链)

v4.1 新增 (意识主体性理论):
  - 内驱力系统: 社交/求知/表达/沉思 四种基本需求 (公理 8)
  - 内部言语: 目标导向思维链, 冥想, 总结 (公理 9)
  - 主动社交: 需求驱动的社交发起 (公理 10)
  - 认知模式: THINK / MEDITATE / SUMMARIZE / SOCIALIZE / WANDER
  - 10种行动: 新增 think, meditate, socialize, summarize
"""

from .engine import ConsciousnessEngine
from .text_encoder import TextEncoder, chunk_text, read_text_file, read_file, SUPPORTED_EXTENSIONS
from .device import detect_device, print_device_report, device_info
from .llm import LocalLLM, ConsciousnessLoop
from .needs import NeedSystem
from .inner_speech import InnerSpeech

__version__ = "4.1.0"

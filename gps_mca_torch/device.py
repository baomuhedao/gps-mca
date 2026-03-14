"""
硬件设备自动检测与管理

优先级: XPU (Intel Arc GPU) > CUDA (NVIDIA GPU) > CPU
"""

from __future__ import annotations

import torch


def detect_device(preferred: str | None = None) -> torch.device:
    """
    自动检测最佳可用设备，或使用用户指定的设备。

    preferred: "auto" | "xpu" | "cuda" | "cpu" | None
    """
    if preferred and preferred != "auto":
        dev = preferred.lower()
        if dev == "xpu":
            if not _xpu_available():
                print("  [!] XPU requested but not available, falling back")
                return _auto_detect()
            return torch.device("xpu")
        if dev == "cuda":
            if not torch.cuda.is_available():
                print("  [!] CUDA requested but not available, falling back")
                return _auto_detect()
            return torch.device("cuda")
        return torch.device("cpu")

    return _auto_detect()


def _xpu_available() -> bool:
    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def _auto_detect() -> torch.device:
    if _xpu_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """返回设备详细信息"""
    name = device.type
    if name == "xpu":
        try:
            props = torch.xpu.get_device_properties(device)
            return f"XPU: {props.name} ({props.total_memory // 1024**2} MB)"
        except Exception:
            return "XPU: Intel Arc GPU"
    if name == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"CUDA: {props.name} ({props.total_mem // 1024**2} MB)"
    return "CPU"


def print_device_report():
    """打印所有可用设备"""
    print("  Hardware Detection:")
    print(f"    CPU:  available")
    if _xpu_available():
        try:
            n = torch.xpu.device_count()
            for i in range(n):
                props = torch.xpu.get_device_properties(i)
                print(f"    XPU {i}: {props.name}")
        except Exception:
            print(f"    XPU:  available (Intel Arc GPU)")
    else:
        print(f"    XPU:  not detected")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"    CUDA {i}: {props.name} ({props.total_mem // 1024**2} MB)")
    else:
        print(f"    CUDA: not detected")

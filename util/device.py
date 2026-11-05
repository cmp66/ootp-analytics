"""The torch device the training scripts use (CUDA, then XPU, then CPU; OOTP_DEVICE overrides)."""

import os

import torch


def get_device() -> torch.device:
    """Pick the best available torch device.

    Override with the OOTP_DEVICE environment variable (e.g. "cpu", "cuda", "xpu").
    Otherwise prefers CUDA (NVIDIA), then XPU (Intel GPU), then CPU.
    """
    requested = os.environ.get("OOTP_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

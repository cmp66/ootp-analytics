# ootp-analytics

Models for predicting current and future player performance from Out of the Park Baseball data.

## Setup

Python 3.14 (pyenv virtualenv `ootp-analytics`). Install the requirements file matching your hardware:

| Hardware | Install |
| --- | --- |
| CPU only (default) | `uv pip install -r requirements-cpu.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match` |
| NVIDIA GPU (CUDA) | `uv pip install -r requirements-cuda.txt` |
| Intel GPU (XPU) | `uv pip install -r requirements-xpu.txt --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match` |

The device is chosen at runtime by `util/device.py` (CUDA, then XPU, then CPU). Set `OOTP_DEVICE=cpu` (or `cuda`, `xpu`) to override.

To update, edit `requirements.in` (shared) or `requirements-<hw>.in` and recompile, e.g.:

`uv pip compile requirements-cpu.in -o requirements-cpu.txt --index-strategy unsafe-best-match`

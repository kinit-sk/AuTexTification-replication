"""Environment fingerprint + determinism helpers for cross-server multi-seed runs.

Call `log_env_fingerprint()` once at the start of every training/experiment
script so each log file records exactly which hardware/software combination
produced its numbers (needed when the same sweep is split across multiple
physical GPU servers). Call `set_determinism(seed)` instead of setting
`torch.manual_seed` etc. by hand, so all scripts apply the same knobs.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess

import numpy as np
import torch


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _gpu_info() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    parts = []
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        parts.append(f"cuda:{i}={name} (sm_{cap[0]}{cap[1]})")
    return "; ".join(parts)


def log_env_fingerprint() -> None:
    """Print a reproducibility fingerprint (call at the very start of main())."""
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "not installed"

    print("=" * 80)
    print("[ENV FINGERPRINT]")
    print(f"  git_commit        = {_git_commit()}")
    print(f"  python            = {platform.python_version()}")
    print(f"  platform          = {platform.platform()}")
    print(f"  torch             = {torch.__version__}")
    print(f"  torch.cuda        = {torch.version.cuda}")
    print(f"  cudnn             = {torch.backends.cudnn.version()}")
    print(f"  transformers      = {transformers_version}")
    print(f"  numpy             = {np.__version__}")
    print(f"  gpu(s)            = {_gpu_info()}")
    print(f"  CUBLAS_WORKSPACE_CONFIG = {os.environ.get('CUBLAS_WORKSPACE_CONFIG', '(unset)')}")
    print("=" * 80)


def set_determinism(seed: int, numpy_seed: int = 0) -> None:
    """Seed random/numpy/torch and pin cudnn to deterministic algorithm selection.

    Note: full `torch.use_deterministic_algorithms(True)` is NOT enabled here.
    This codebase's LSTM layers (models/hybrid.py) run through cuDNN, whose
    backward pass has no deterministic implementation in PyTorch — enabling
    strict deterministic mode would raise at runtime. `cudnn.deterministic=True`
    still removes algorithm-selection nondeterminism for the forward/backward
    ops that do support it, which is what matters for matching identical GPUs
    (e.g. two A40s) across servers.
    """
    random.seed(seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SEED] random/torch seed={seed} | numpy seed={numpy_seed} | cudnn.deterministic=True")

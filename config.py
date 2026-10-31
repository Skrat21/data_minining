#!/usr/bin/env python3
"""
config.py - Central configuration for Phase 2 Whisper fine-tuning.

All hyperparameters, paths, and settings live here. Individual scripts
import from this module rather than hard-coding values.

Changes vs. the original:
  - print_env_info(): fixed `total_mem` -> `total_memory` (the old attribute
    does not exist and crashed on GPU machines)
  - TrainConfig: new fields `per_device_eval_batch`, `freeze_encoder`,
    `dataloader_num_workers` (old YAML configs still load: unknown keys are
    ignored, missing keys take the defaults below)
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "arabic_data"
AUDIO_DIR = DATA_DIR / "audio"
MANIFEST_DIR = DATA_DIR / "manifests"
# Smoke-test mode (python run_experiment.py --smoke): everything is written to
# runs_smoke/ and results_smoke/ so real runs and results are never touched.
SMOKE = os.environ.get("SMOKE_TEST") == "1"
RUNS_DIR = PROJECT_ROOT / ("runs_smoke" if SMOKE else "runs")
RESULTS_DIR = PROJECT_ROOT / ("results_smoke" if SMOKE else "results")


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All settings for one fine-tuning run."""

    # --- identity ---
    dataset: str = "cv"            # "cv" or "mgb2"
    arm: str = "A0"                # A0 / A1 / A2 / A3 / A4
    seed: int = 42
    run_tag: str = ""              # auto-generated if empty

    # --- model ---
    model_name: str = "openai/whisper-small"
    language: str = "ar"
    task: str = "transcribe"
    freeze_encoder: bool = False   # freeze encoder to save VRAM (was a no-op flag before)

    # --- training ---
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    num_epochs: int = 3
    per_device_train_batch: int = 4   # lower for 4 GB VRAM
    gradient_accumulation_steps: int = 4  # effective batch = 4 * 4 = 16
    fp16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"     # or "adamw_bnb_8bit" to save memory
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"

    # --- evaluation ---
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    dev_subset_size: int = 500     # subsample dev for speed
    max_new_tokens: int = 225      # cap generation to prevent hallucination loops
    per_device_eval_batch: int = 16  # eval keeps no activations/grads -> can be larger than train batch

    # --- data ---
    min_audio_duration: float = 0.5
    max_audio_duration: float = 30.0
    removed_ids_file: Optional[str] = None  # for A1/A2: path to file listing IDs to remove
    max_train_samples: int = 0              # 0 = use all; N = first N utterances (smoke tests)
    max_steps: int = -1                     # -1 = train for num_epochs; N = stop after N steps (smoke tests)
    dataloader_num_workers: int = 0         # keep 0 on Windows

    # --- I/O ---
    output_dir: str = ""           # auto-generated if empty
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 2      # keep at most 2 checkpoints

    def __post_init__(self):
        if not self.run_tag:
            self.run_tag = f"{self.dataset}_{self.arm}_seed{self.seed}"
        if not self.output_dir:
            self.output_dir = str(RUNS_DIR / self.run_tag)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # deterministic algorithms (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_env_info():
    """Print environment information for logging."""
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        total = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        print(f"GPU memory: {total / (1024 ** 3):.1f} GB")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {DATA_DIR}")

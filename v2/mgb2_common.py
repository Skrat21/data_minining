"""Shared paths and data helpers for the staged MGB-2 experiment."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "arabic_data" / "manifests"
RUNS_DIR = ROOT / "runs_mgb2"
DATA_DIR = RUNS_DIR / "data"
DEV_MANIFEST = MANIFEST_DIR / "mgb2_dev.jsonl"
DEV_SUBSET_SIZE = 500
DEV_SUBSET_SEED = 123
TRAIN_SUBSET_SIZE = 10_000
TRAIN_SUBSET_SEED = 123
FLAG_FRACTION = 0.10
MIN_AUDIO_SECONDS = 0.5
MAX_AUDIO_SECONDS = 30.0
MIN_FREE_GIB = 15.0


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_notes(message: str) -> None:
    notes = Path(__file__).with_name("NOTES.md")
    with notes.open("a", encoding="utf-8") as f:
        f.write("\n" + message.rstrip() + "\n")


def eligible_training_records(records: list[dict]) -> list[dict]:
    return [
        r for r in records
        if MIN_AUDIO_SECONDS <= float(r.get("duration", 0.0)) <= MAX_AUDIO_SECONDS
        and r.get("text_norm", "").strip()
    ]


def fixed_sample(records: list[dict], count: int, seed: int) -> list[dict]:
    if count <= 0:
        raise ValueError(f"Sample count must be positive, got {count}")
    if count >= len(records):
        return list(records)
    selected = set(random.Random(seed).sample(range(len(records)), count))
    return [record for index, record in enumerate(records) if index in selected]


def fixed_dev_records(
    manifest_path: str | Path = DEV_MANIFEST,
    count: int = DEV_SUBSET_SIZE,
    seed: int = DEV_SUBSET_SEED,
) -> list[dict]:
    records = load_jsonl(manifest_path)
    records = [
        r for r in records
        if MIN_AUDIO_SECONDS <= float(r.get("duration", 0.0)) <= MAX_AUDIO_SECONDS
        and r.get("text_norm", "").strip()
    ]
    return fixed_sample(records, count, seed)


def worst_fraction_ids(
    scores: list[dict],
    fraction: float = FLAG_FRACTION,
) -> tuple[list[str], float]:
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
    if not scores:
        raise ValueError("Cannot flag a fraction of an empty score list")

    count = int(round(len(scores) * fraction))
    if count < 1:
        raise ValueError("Flag fraction selects zero utterances")

    def rank_key(record: dict) -> tuple[float, int, str]:
        errors = int(record["word_errors"])
        ref_words = max(int(record["ref_words"]), 1)
        return (-errors / ref_words, -errors, str(record["id"]))

    ranked = sorted(scores, key=rank_key)
    flagged = ranked[:count]
    cutoff = int(flagged[-1]["word_errors"]) / max(int(flagged[-1]["ref_words"]), 1)
    return [str(record["id"]) for record in flagged], cutoff


def disk_free_gib(path: str | Path) -> float:
    probe = Path(path).resolve()
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024**3)


def gpu_preflight(output_path: str | Path) -> None:
    free_gib = disk_free_gib(output_path)
    print(f"Free space for {Path(output_path).resolve()}: {free_gib:.2f} GiB")
    if free_gib < MIN_FREE_GIB:
        raise RuntimeError(
            f"GPU job blocked: {free_gib:.2f} GiB free; at least "
            f"{MIN_FREE_GIB:.0f} GiB is required."
        )

    subprocess.run(["nvidia-smi"], check=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("GPU job blocked: CUDA is unavailable to PyTorch.")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


def load_inference_model(
    model_path: str | Path,
    processor_path: str | Path | None = None,
    fp16: bool = True,
):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor_path = processor_path or model_path
    processor = WhisperProcessor.from_pretrained(
        str(processor_path), language="ar", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_path), attn_implementation="sdpa"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if fp16 and device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.eval()
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model.generation_config.language = "ar"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return processor, model, device

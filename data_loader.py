#!/usr/bin/env python3
"""
data_loader.py - Load and prepare datasets for Whisper fine-tuning.

Reads JSONL manifests, converts audio to Whisper features, and returns
HuggingFace-compatible datasets for training and evaluation.

Changes vs. the original:
  - WhisperDataCollator FIXED (this matters a lot):
      * the old collator masked every label equal to pad_token_id. For Whisper,
        pad == eos == bos == <|endoftext|>, so the real end-of-text token of
        every utterance was masked too -> the model was never trained to stop
        generating (runaway / repeating outputs, very slow evaluation).
      * the old "strip start token" check compared against bos_token_id
        (<|endoftext|>) instead of <|startoftranscript|>, so it never fired and
        the decoder saw [SOT, SOT, lang, ...] during training.
    The new collator pads manually with -100 (never touches a real token) and
    strips the leading <|startoftranscript|> (the model re-adds it itself).
  - load_audio_16k(): mono down-mix + resample only when needed (audio in the
    manifests is already 16 kHz mono, so normally this is just a file read)
  - label sequences are truncated to Whisper's 448-token decoder limit
  - np.stack/from_numpy instead of torch.tensor(np.array(...)) (one less copy)

NOTE: models trained with the OLD collator should be retrained.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from config import MANIFEST_DIR, TrainConfig

SAMPLE_RATE = 16000
MAX_LABEL_LEN = 448  # Whisper decoder max_target_positions


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------

def load_manifest(path: str | Path) -> list[dict]:
    """Load a JSONL manifest file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_manifest(
    records: list[dict],
    removed_ids: Optional[set[str]] = None,
    min_dur: float = 0.5,
    max_dur: float = 30.0,
) -> list[dict]:
    """Filter manifest records by duration and optionally remove specific IDs."""
    filtered = []
    for r in records:
        dur = r.get("duration", 0)
        if dur < min_dur or dur > max_dur:
            continue
        if removed_ids and r["id"] in removed_ids:
            continue
        text = r.get("text_norm", "")
        if not text:
            continue
        filtered.append(r)
    return filtered


def load_removed_ids(path: str | Path) -> set[str]:
    """Load a file of IDs to remove (one per line, or JSON list)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return set(json.loads(text))
    return {line.strip() for line in text.split("\n") if line.strip()}


def subsample_manifest(
    records: list[dict], n: int, seed: int = 42
) -> list[dict]:
    """Randomly subsample n records from the manifest."""
    if n <= 0 or n >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, n)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def load_audio_16k(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read an audio file as float32 mono at 16 kHz.

    The data pipeline already stores 16 kHz mono files, so resampling is
    normally skipped; it is kept as a safety net for foreign files.
    """
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


# ---------------------------------------------------------------------------
# Dataset / collator
# ---------------------------------------------------------------------------

class WhisperASRDataset(Dataset):
    """
    PyTorch dataset for Whisper fine-tuning.

    Each item returns:
      - input_features: mel spectrogram (from WhisperFeatureExtractor)
      - labels: tokenized text, INCLUDING the <|startoftranscript|> prefix and
        the final <|endoftext|> (the collator strips the SOT)
    """

    def __init__(
        self,
        records: list[dict],
        processor: WhisperProcessor,
        language: str = "ar",
        task: str = "transcribe",
        max_audio_duration: float = 30.0,
    ):
        self.records = records
        self.processor = processor
        self.feature_extractor = processor.feature_extractor
        self.tokenizer = processor.tokenizer
        self.language = language
        self.task = task
        self.max_audio_duration = max_audio_duration
        self.sr = SAMPLE_RATE  # Whisper expects 16 kHz

        # Set the tokenizer's language and task
        self.tokenizer.set_prefix_tokens(language=language, task=task)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec["text_norm"]

        audio = load_audio_16k(rec["audio_filepath"], self.sr)

        # Truncate to max duration
        max_samples = int(self.max_audio_duration * self.sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Mel features (80 x 3000 for whisper-small, padded to 30 s)
        input_features = self.feature_extractor(
            audio, sampling_rate=self.sr, return_tensors="np"
        ).input_features[0]

        # Tokenize text: [SOT, lang, task, notimestamps, ...text..., EOT]
        labels = self.tokenizer(text).input_ids
        if len(labels) > MAX_LABEL_LEN:
            labels = labels[: MAX_LABEL_LEN - 1] + [self.tokenizer.eos_token_id]

        return {
            "input_features": input_features,
            "labels": labels,
            "id": rec["id"],
        }


class WhisperDataCollator:
    """
    Data collator for Whisper fine-tuning.

    - stacks the (already 30 s padded) input features
    - right-pads labels with -100 (ignored by the loss). Padding is done by
      hand so that real tokens - in particular the final <|endoftext|> - are
      NEVER masked.
    - strips the leading <|startoftranscript|> if every sequence has it; the
      model prepends decoder_start_token_id (= SOT) itself when it shifts the
      labels right.
    """

    def __init__(self, processor: WhisperProcessor):
        self.processor = processor
        tok = processor.tokenizer
        self.sot_id = tok.convert_tokens_to_ids("<|startoftranscript|>")

    def __call__(self, features):
        input_features = torch.from_numpy(
            np.stack([f["input_features"] for f in features])
        ).float()

        max_len = max(len(f["labels"]) for f in features)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            ids = torch.as_tensor(f["labels"], dtype=torch.long)
            labels[i, : len(ids)] = ids

        if (labels[:, 0] == self.sot_id).all():
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def get_train_dataset(cfg: TrainConfig, processor: WhisperProcessor) -> WhisperASRDataset:
    """Load the training dataset based on config."""
    manifest_path = MANIFEST_DIR / f"{cfg.dataset}_train.jsonl"
    records = load_manifest(manifest_path)

    # Remove flagged IDs if specified (for A1/A2 arms)
    removed_ids = None
    if cfg.removed_ids_file:
        removed_ids = load_removed_ids(cfg.removed_ids_file)
        print(f"Removing {len(removed_ids)} IDs from training set")

    records = filter_manifest(
        records,
        removed_ids=removed_ids,
        min_dur=cfg.min_audio_duration,
        max_dur=cfg.max_audio_duration,
    )
    print(f"Training set: {len(records)} utterances")
    total_hours = sum(r["duration"] for r in records) / 3600
    print(f"Training hours: {total_hours:.2f} h")

    return WhisperASRDataset(records, processor, cfg.language, cfg.task, cfg.max_audio_duration)


def get_eval_dataset(
    cfg: TrainConfig,
    processor: WhisperProcessor,
    split: str = "dev",
) -> WhisperASRDataset:
    """Load the evaluation dataset (dev or test)."""
    manifest_path = MANIFEST_DIR / f"{cfg.dataset}_{split}.jsonl"
    records = load_manifest(manifest_path)
    records = filter_manifest(
        records,
        min_dur=cfg.min_audio_duration,
        max_dur=cfg.max_audio_duration,
    )

    # Subsample dev for faster eval during training
    if split == "dev" and cfg.dev_subset_size > 0:
        records = subsample_manifest(records, cfg.dev_subset_size, cfg.seed)

    print(f"{split.capitalize()} set: {len(records)} utterances")
    return WhisperASRDataset(records, processor, cfg.language, cfg.task, cfg.max_audio_duration)

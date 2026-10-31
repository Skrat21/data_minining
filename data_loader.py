#!/usr/bin/env python3
"""
data_loader.py - Load and prepare datasets for Whisper fine-tuning.

Reads JSONL manifests, converts audio to Whisper features, and returns
HuggingFace-compatible datasets for training and evaluation.
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
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

from config import MANIFEST_DIR, TrainConfig


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


class WhisperASRDataset(Dataset):
    """
    PyTorch dataset for Whisper fine-tuning.

    Each item returns:
      - input_features: mel spectrogram (from WhisperFeatureExtractor)
      - labels: tokenized text (from WhisperTokenizer)
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
        self.sr = 16000  # Whisper expects 16 kHz

        # Set the tokenizer's language and task
        self.tokenizer.set_prefix_tokens(language=language, task=task)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        audio_path = rec["audio_filepath"]
        text = rec["text_norm"]

        # Load audio
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        # Truncate to max duration
        max_samples = int(self.max_audio_duration * self.sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Extract mel features
        input_features = self.feature_extractor(
            audio, sampling_rate=self.sr, return_tensors="np"
        ).input_features[0]

        # Tokenize text
        labels = self.tokenizer(text).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
            "id": rec["id"],
        }


class WhisperDataCollator:
    """
    Data collator for Whisper fine-tuning.

    Pads labels to the same length (with -100 for loss masking).
    Input features are already 30s padded by the feature extractor.
    """

    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, features):
        # Stack input features
        input_features = torch.tensor(
            np.array([f["input_features"] for f in features]),
            dtype=torch.float32,
        )

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"]
        # Replace padding token id with -100 so it's ignored in loss
        labels = labels.masked_fill(
            labels == self.processor.tokenizer.pad_token_id, -100
        )
        # If BOS token is at the start, mask it too (Whisper uses decoder_start_token_id)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels[:, 0] = -100

        return {
            "input_features": input_features,
            "labels": labels,
        }


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


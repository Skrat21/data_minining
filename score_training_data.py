#!/usr/bin/env python3
"""
score_training_data.py - Score the training set with the A0 model (Kedalai loop Step 7).

Runs the fine-tuned A0 model on its own training data to compute per-utterance
WER. Flags utterances with WER >= 1.0 (the Kedalai criterion) and produces:
  - Per-utterance scores (WER, CER, hypothesis)
  - Flagged ID list for A1 arm
  - WER distribution statistics
  - Random-removal ID lists for A2 arm (3 seeds)

Usage:
  python score_training_data.py --model runs/cv_A0_seed42/best_model --dataset cv
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import MANIFEST_DIR, RESULTS_DIR, get_device, set_seed
from data_loader import load_manifest, filter_manifest
from normalize import normalize_for_eval
from evaluate_model import compute_per_utterance_metrics

try:
    import jiwer
except ImportError:
    raise ImportError("jiwer is required: pip install jiwer")


def score_training_set(
    model_path: str,
    dataset: str,
    batch_size: int = 8,
    max_new_tokens: int = 225,
    language: str = "ar",
    wer_threshold: float = 1.0,
    min_ref_words: int = 0,
    seed: int = 42,
    n_random_seeds: int = 3,
) -> dict:
    """
    Score the training set with the A0 model.

    This implements the Kedalai et al. (2024) selection step:
    1. Run the fine-tuned model on its own training data
    2. Flag utterances where WER >= threshold
    3. Generate removal lists for A1 (Kedalai) and A2 (random control)
    """
    set_seed(seed)
    device = get_device()

    print(f"\nScoring training set with model: {model_path}")
    print(f"Dataset: {dataset}")

    # Load model
    processor = WhisperProcessor.from_pretrained(model_path, language=language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_new_tokens = max_new_tokens

    # Load training data
    manifest_path = MANIFEST_DIR / f"{dataset}_train.jsonl"
    records = load_manifest(manifest_path)
    records = filter_manifest(records, min_dur=0.5, max_dur=30.0)
    print(f"Scoring {len(records)} training utterances...")

    # Decode the training set
    all_hypotheses = []
    all_references = []
    all_ids = []

    for i in tqdm(range(0, len(records), batch_size), desc="Scoring"):
        batch_records = records[i : i + batch_size]
        batch_audio = []

        for rec in batch_records:
            audio, sr = sf.read(rec["audio_filepath"], dtype="float32")
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            batch_audio.append(audio)
            all_references.append(rec["text_norm"])
            all_ids.append(rec["id"])

        inputs = processor.feature_extractor(
            batch_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language=language,
                task="transcribe",
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )

        transcriptions = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        all_hypotheses.extend(transcriptions)

    # Compute per-utterance metrics
    per_utt = compute_per_utterance_metrics(
        all_hypotheses, all_references, all_ids, records
    )

    # Build record lookup for duration info
    rec_by_id = {r["id"]: r for r in records}

    # --- Flag utterances (Kedalai criterion: WER >= threshold) ---
    flagged_ids = []
    flagged_ids_min_words = []  # sensitivity: only flag if ref has >= min_ref_words

    for r in per_utt:
        if r["wer"] >= wer_threshold:
            flagged_ids.append(r["id"])
            if min_ref_words > 0 and r["ref_words"] >= min_ref_words:
                flagged_ids_min_words.append(r["id"])

    flagged_set = set(flagged_ids)
    flagged_hours = sum(
        rec_by_id[uid]["duration"] for uid in flagged_ids if uid in rec_by_id
    ) / 3600
    total_hours = sum(r["duration"] for r in records) / 3600

    print(f"\n--- Scoring Results ---")
    print(f"Total training utterances: {len(records)}")
    print(f"Total training hours: {total_hours:.2f} h")
    print(f"Flagged (WER >= {wer_threshold}): {len(flagged_ids)} ({100*len(flagged_ids)/len(records):.1f}%)")
    print(f"Flagged hours: {flagged_hours:.2f} h ({100*flagged_hours/total_hours:.1f}%)")
    if min_ref_words > 0:
        print(f"Flagged (>= {min_ref_words} ref words): {len(flagged_ids_min_words)}")

    # WER distribution
    wers = [r["wer"] for r in per_utt]
    cers = [r["cer"] for r in per_utt]
    print(f"\nWER distribution:")
    for thresh in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        count = sum(1 for w in wers if w >= thresh)
        print(f"  WER >= {thresh}: {count} ({100*count/len(wers):.1f}%)")

    print(f"\nWER stats: mean={np.mean(wers):.4f}, median={np.median(wers):.4f}, "
          f"std={np.std(wers):.4f}, max={np.max(wers):.4f}")
    print(f"CER stats: mean={np.mean(cers):.4f}, median={np.median(cers):.4f}, "
          f"std={np.std(cers):.4f}, max={np.max(cers):.4f}")

    # --- Save outputs ---
    out_dir = RESULTS_DIR / f"{dataset}_scoring"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-utterance scores
    with open(out_dir / "train_scores.jsonl", "w", encoding="utf-8") as f:
        for r in per_utt:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Flagged IDs (for A1 arm)
    with open(out_dir / "flagged_ids.txt", "w") as f:
        for uid in flagged_ids:
            f.write(uid + "\n")

    if min_ref_words > 0:
        with open(out_dir / f"flagged_ids_min{min_ref_words}words.txt", "w") as f:
            for uid in flagged_ids_min_words:
                f.write(uid + "\n")

    # --- Generate random removal lists for A2 arm ---
    # Match the same number of hours and similar clip count as A1
    for rs in range(n_random_seeds):
        rng = random.Random(rs)
        # Shuffle all non-flagged + flagged, then pick the same count
        all_train_ids = [r["id"] for r in records]
        rng.shuffle(all_train_ids)
        random_removed = all_train_ids[:len(flagged_ids)]

        with open(out_dir / f"random_removed_seed{rs}.txt", "w") as f:
            for uid in random_removed:
                f.write(uid + "\n")

        random_hours = sum(
            rec_by_id[uid]["duration"] for uid in random_removed if uid in rec_by_id
        ) / 3600
        print(f"Random removal seed {rs}: {len(random_removed)} utts, {random_hours:.2f} h")

    # Save summary statistics
    summary = {
        "dataset": dataset,
        "model": model_path,
        "total_utterances": len(records),
        "total_hours": round(total_hours, 2),
        "wer_threshold": wer_threshold,
        "flagged_count": len(flagged_ids),
        "flagged_pct": round(100 * len(flagged_ids) / len(records), 2),
        "flagged_hours": round(flagged_hours, 2),
        "flagged_hours_pct": round(100 * flagged_hours / total_hours, 2),
        "wer_mean": round(float(np.mean(wers)), 4),
        "wer_median": round(float(np.median(wers)), 4),
        "wer_std": round(float(np.std(wers)), 4),
        "cer_mean": round(float(np.mean(cers)), 4),
        "cer_median": round(float(np.median(cers)), 4),
    }
    with open(out_dir / "scoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {out_dir}")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Score training data with the A0 model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to A0 fine-tuned model")
    parser.add_argument("--dataset", type=str, default="cv", choices=["cv", "mgb2"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--wer-threshold", type=float, default=1.0)
    parser.add_argument("--min-ref-words", type=int, default=3,
                        help="Sensitivity variant: only flag if ref has >= N words")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random-seeds", type=int, default=3,
                        help="Number of random-removal lists to generate for A2")

    args = parser.parse_args()

    score_training_set(
        model_path=args.model,
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        wer_threshold=args.wer_threshold,
        min_ref_words=args.min_ref_words,
        seed=args.seed,
        n_random_seeds=args.n_random_seeds,
    )


if __name__ == "__main__":
    main()


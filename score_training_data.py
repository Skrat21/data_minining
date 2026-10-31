#!/usr/bin/env python3
"""
score_training_data.py - Score the training set with the A0 model (Kedalai loop Step 7).

Runs the fine-tuned A0 model on its own training data to compute per-utterance
WER. Flags utterances with WER >= 1.0 (the Kedalai criterion) and produces:
  - Per-utterance scores (WER, CER, hypothesis)
  - Flagged ID list for A1 arm
  - WER distribution statistics
  - Random-removal ID lists for A2 arm (3 seeds)

This is by far the most expensive step (it decodes the WHOLE training set), so
it now uses the fast decoder from evaluate_model.py (KV cache, fp16, sorted
batches, background audio prefetch) and can cache the hypotheses:
  --reuse-hyps   skip decoding if train_hyps_cache.json from an earlier run of
                 the same model exists (e.g. to try another WER threshold)
  --limit N      smoke test on N utterances; outputs go to <dataset>_scoring_smoketest
                 so real flagged_ids.txt / random_removed_*.txt are never overwritten

Usage:
    python score_training_data.py --model runs/cv_A0_seed42/best_model --dataset cv
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config import MANIFEST_DIR, RESULTS_DIR, SMOKE, set_seed
from data_loader import load_manifest, filter_manifest
from evaluate_model import (
    compute_per_utterance_metrics,
    load_whisper_for_inference,
    transcribe_records,
)


def score_training_set(
    model_path: str,
    dataset: str,
    batch_size: int = 16,
    max_new_tokens: int = 225,
    language: str = "ar",
    wer_threshold: float = 1.0,
    min_ref_words: int = 0,
    seed: int = 42,
    n_random_seeds: int = 3,
    num_workers: int = 4,
    fp16: bool = True,
    limit: Optional[int] = None,
    reuse_hyps: bool = False,
    max_flagged: Optional[int] = None,
) -> dict:
    """
    Score the training set with the A0 model.

    This implements the Kedalai et al. (2024) selection step:
      1. Run the fine-tuned model on its own training data
      2. Flag utterances where WER >= threshold
      3. Generate removal lists for A1 (Kedalai) and A2 (random control)
    """
    set_seed(seed)

    print(f"\nScoring training set with model: {model_path}")
    print(f"Dataset: {dataset}")

    # Load training data
    manifest_path = MANIFEST_DIR / f"{dataset}_train.jsonl"
    records = load_manifest(manifest_path)
    records = filter_manifest(records, min_dur=0.5, max_dur=30.0)
    if limit:
        records = records[:limit]
    print(f"Scoring {len(records)} training utterances...")

    # --limit smoke tests get their own folder, except under run_experiment.py --smoke,
    # where RESULTS_DIR is already the isolated results_smoke/ and later steps read cv_scoring.
    out_dir = RESULTS_DIR / (f"{dataset}_scoring_smoketest" if (limit and not SMOKE) else f"{dataset}_scoring")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "train_hyps_cache.json"

    all_ids = [r["id"] for r in records]
    all_references = [r["text_norm"] for r in records]

    # ---- decode (or reuse cached hypotheses) ----
    all_hypotheses = None
    if reuse_hyps and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("model") == model_path and cache.get("ids") == all_ids:
            all_hypotheses = cache["hyps"]
            print(f"Reusing cached hypotheses from {cache_path}")
        else:
            print("Cached hypotheses do not match this model/manifest -> decoding again")

    if all_hypotheses is None:
        processor, model, device = load_whisper_for_inference(
            model_path, language=language, max_new_tokens=max_new_tokens, fp16=fp16
        )
        all_hypotheses, decode_time = transcribe_records(
            model, processor, records,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
            language=language, num_workers=num_workers, desc="Scoring",
        )
        print(f"Decoding took {decode_time / 60:.1f} min")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"model": model_path, "ids": all_ids, "hyps": all_hypotheses},
                      f, ensure_ascii=False)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    if max_flagged is not None:
        # smoke tests only: a barely-trained model flags almost everything, which would
        # leave A1 with no training data
        flagged_ids = flagged_ids[:max_flagged]
        flagged_ids_min_words = flagged_ids_min_words[:max_flagged]
        print(f"[smoke] keeping at most {max_flagged} flagged utterances")

    flagged_hours = sum(
        rec_by_id[uid]["duration"] for uid in flagged_ids if uid in rec_by_id
    ) / 3600
    total_hours = sum(r["duration"] for r in records) / 3600

    print(f"\n--- Scoring Results ---")
    print(f"Total training utterances: {len(records)}")
    print(f"Total training hours: {total_hours:.2f} h")
    print(f"Flagged (WER >= {wer_threshold}): {len(flagged_ids)} ({100*len(flagged_ids)/max(len(records),1):.1f}%)")
    print(f"Flagged hours: {flagged_hours:.2f} h ({100*flagged_hours/max(total_hours,1e-9):.1f}%)")
    if min_ref_words > 0:
        print(f"Flagged (>= {min_ref_words} ref words): {len(flagged_ids_min_words)}")

    # Health check: runaway generations are the usual reason for a huge flagged share
    runaway = sum(1 for r in per_utt if r["hyp_words"] > 3 * max(r["ref_words"], 1) and r["hyp_words"] > 10)
    if per_utt:
        print(f"Hypotheses >3x longer than the reference: {runaway} ({100*runaway/len(per_utt):.1f}%)")
        if runaway / len(per_utt) > 0.02:
            print("[warn] Many runaway/hallucinated outputs. If the model was trained with the OLD collator "
                  "(end-of-text masked), the flagged set mixes real label noise with generation failures. "
                  "Retrain A0 with the fixed data_loader.py before trusting the flags.")

    # WER distribution
    wers = [r["wer"] for r in per_utt]
    cers = [r["cer"] for r in per_utt]
    print(f"\nWER distribution:")
    for thresh in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        count = sum(1 for w in wers if w >= thresh)
        print(f"  WER >= {thresh}: {count} ({100*count/max(len(wers),1):.1f}%)")

    print(f"\nWER stats: mean={np.mean(wers):.4f}, median={np.median(wers):.4f}, "
          f"std={np.std(wers):.4f}, max={np.max(wers):.4f}")
    print(f"CER stats: mean={np.mean(cers):.4f}, median={np.median(cers):.4f}, "
          f"std={np.std(cers):.4f}, max={np.max(cers):.4f}")

    # --- Save outputs ---
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
    # (matches the flagged COUNT; hours are printed so the match can be checked)
    for rs in range(n_random_seeds):
        rng = random.Random(rs)
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
        "flagged_pct": round(100 * len(flagged_ids) / max(len(records), 1), 2),
        "flagged_hours": round(flagged_hours, 2),
        "flagged_hours_pct": round(100 * flagged_hours / max(total_hours, 1e-9), 2),
        "wer_mean": round(float(np.mean(wers)), 4),
        "wer_median": round(float(np.median(wers)), 4),
        "wer_std": round(float(np.std(wers)), 4),
        "cer_mean": round(float(np.mean(cers)), 4),
        "cer_median": round(float(np.median(cers)), 4),
    }
    with open(out_dir / "scoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {out_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Score training data with the A0 model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to A0 fine-tuned model")
    parser.add_argument("--dataset", type=str, default="cv", choices=["cv", "mgb2"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--wer-threshold", type=float, default=1.0)
    parser.add_argument("--min-ref-words", type=int, default=3,
                        help="Sensitivity variant: only flag if ref has >= N words")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random-seeds", type=int, default=3,
                        help="Number of random-removal lists to generate for A2")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Prefetch threads for audio loading / mel extraction (0 = off)")
    parser.add_argument("--fp32", action="store_true", help="Disable fp16 inference")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test on the first N utterances (writes to <dataset>_scoring_smoketest)")
    parser.add_argument("--reuse-hyps", action="store_true",
                        help="Reuse cached hypotheses of a previous run of the same model")
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
        num_workers=args.num_workers,
        fp16=not args.fp32,
        limit=args.limit,
        reuse_hyps=args.reuse_hyps,
    )


if __name__ == "__main__":
    main()

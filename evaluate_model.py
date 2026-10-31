#!/usr/bin/env python3
"""
evaluate_model.py - Evaluation harness for Whisper Arabic ASR.

Handles:
  - Zero-shot evaluation (pretrained model, no fine-tuning)
  - Fine-tuned model evaluation on dev/test sets
  - Per-utterance WER/CER computation with normalization
  - Bootstrap confidence intervals
  - Per-group WER breakdown (gender, age, accent, program)
  - Saving detailed per-utterance results

This harness is used by ALL arms to ensure identical evaluation.

Usage:
  # Zero-shot baseline (Step 5)
  python evaluate_model.py --model openai/whisper-small --dataset cv --split test

  # Fine-tuned model
  python evaluate_model.py --model runs/cv_A0_seed42/best_model --dataset cv --split test

  # Both datasets
  python evaluate_model.py --model openai/whisper-small --dataset cv mgb2 --split test
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import (
    MANIFEST_DIR,
    RESULTS_DIR,
    TrainConfig,
    get_device,
    print_env_info,
    set_seed,
)
from data_loader import load_manifest, filter_manifest
from normalize import normalize_for_eval

try:
    import jiwer
except ImportError:
    raise ImportError("jiwer is required: pip install jiwer")


# ---------------------------------------------------------------------------
# Core WER/CER computation
# ---------------------------------------------------------------------------

def compute_wer_cer(
    hypotheses: list[str],
    references: list[str],
) -> dict:
    """
    Compute corpus-level WER and CER from lists of hypotheses and references.

    Both are normalized before comparison. Corpus-level WER is computed from
    totals (not by averaging per-utterance WER).
    """
    # Normalize both sides identically
    norm_hyps = [normalize_for_eval(h) for h in hypotheses]
    norm_refs = [normalize_for_eval(r) for r in references]

    # Remove pairs where reference is empty after normalization
    valid = [(h, r) for h, r in zip(norm_hyps, norm_refs) if r.strip()]
    if not valid:
        return {"wer": 1.0, "cer": 1.0}
    norm_hyps, norm_refs = zip(*valid)
    norm_hyps, norm_refs = list(norm_hyps), list(norm_refs)

    # Corpus-level WER (from totals, not averaged)
    wer_output = jiwer.process_words(norm_refs, norm_hyps)
    wer = wer_output.wer

    # Corpus-level CER
    cer_output = jiwer.process_characters(norm_refs, norm_hyps)
    cer = cer_output.cer

    return {"wer": round(wer, 4), "cer": round(cer, 4)}


def compute_per_utterance_metrics(
    hypotheses: list[str],
    references: list[str],
    ids: list[str],
    metadata: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Compute per-utterance WER and CER.

    Returns a list of dicts with: id, ref, hyp, ref_norm, hyp_norm, wer, cer,
    ref_words, hyp_words, ref_chars, hyp_chars.
    """
    results = []
    for i, (hyp, ref, uid) in enumerate(zip(hypotheses, references, ids)):
        ref_norm = normalize_for_eval(ref)
        hyp_norm = normalize_for_eval(hyp)

        # Per-utterance WER
        if ref_norm.strip():
            try:
                utt_wer = jiwer.wer(ref_norm, hyp_norm)
            except Exception:
                utt_wer = 1.0
            try:
                utt_cer = jiwer.cer(ref_norm, hyp_norm)
            except Exception:
                utt_cer = 1.0
        else:
            utt_wer = 0.0 if not hyp_norm.strip() else 1.0
            utt_cer = utt_wer

        entry = {
            "id": uid,
            "ref": ref,
            "hyp": hyp,
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "wer": round(utt_wer, 4),
            "cer": round(utt_cer, 4),
            "ref_words": len(ref_norm.split()),
            "hyp_words": len(hyp_norm.split()),
            "ref_chars": len(ref_norm),
            "hyp_chars": len(hyp_norm),
        }

        # Add metadata if available
        if metadata and i < len(metadata):
            for k in ("gender", "age", "accent", "duration", "speaker",
                       "program", "n_words", "n_chars", "chars_per_sec",
                       "up_votes", "down_votes"):
                if k in metadata[i]:
                    entry[k] = metadata[i][k]

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    per_utt_results: list[dict],
    metric: str = "wer",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrap confidence interval for corpus-level WER or CER.

    Resamples utterances (with replacement) and recomputes the corpus metric
    each time. Returns mean, lower, upper bounds.
    """
    rng = np.random.RandomState(seed)
    n = len(per_utt_results)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}

    # For corpus-level WER: sum(errors) / sum(ref_words)
    if metric == "wer":
        ref_lens = np.array([r["ref_words"] for r in per_utt_results])
        # Compute errors = wer * ref_words for each utterance
        errors = np.array([r["wer"] * r["ref_words"] for r in per_utt_results])
    else:  # CER
        ref_lens = np.array([r["ref_chars"] for r in per_utt_results])
        errors = np.array([r["cer"] * r["ref_chars"] for r in per_utt_results])

    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        total_errors = errors[idx].sum()
        total_ref = ref_lens[idx].sum()
        if total_ref > 0:
            bootstrap_metrics.append(total_errors / total_ref)
        else:
            bootstrap_metrics.append(0.0)

    bootstrap_metrics = np.array(bootstrap_metrics)
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_metrics, alpha * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha) * 100)

    return {
        "mean": round(float(np.mean(bootstrap_metrics)), 4),
        "lower": round(float(lower), 4),
        "upper": round(float(upper), 4),
    }


def paired_bootstrap_test(
    results_a: list[dict],
    results_b: list[dict],
    metric: str = "wer",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap test: is system B significantly better than system A?

    Both must be evaluated on the same test set (same utterances, same order).
    Returns the proportion of bootstraps where B is better, and the mean
    difference (positive = A is worse = B is better).
    """
    assert len(results_a) == len(results_b), "Both systems must have same number of utterances"
    rng = np.random.RandomState(seed)
    n = len(results_a)

    if metric == "wer":
        errors_a = np.array([r["wer"] * r["ref_words"] for r in results_a])
        errors_b = np.array([r["wer"] * r["ref_words"] for r in results_b])
        ref_lens = np.array([r["ref_words"] for r in results_a])
    else:
        errors_a = np.array([r["cer"] * r["ref_chars"] for r in results_a])
        errors_b = np.array([r["cer"] * r["ref_chars"] for r in results_b])
        ref_lens = np.array([r["ref_chars"] for r in results_a])

    b_better_count = 0
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        total_ref = ref_lens[idx].sum()
        if total_ref == 0:
            continue
        wer_a = errors_a[idx].sum() / total_ref
        wer_b = errors_b[idx].sum() / total_ref
        diff = wer_a - wer_b  # positive means B is better
        diffs.append(diff)
        if diff > 0:
            b_better_count += 1

    diffs = np.array(diffs)
    return {
        "b_better_pct": round(b_better_count / len(diffs), 4) if diffs.size else 0.0,
        "mean_diff": round(float(diffs.mean()), 4) if diffs.size else 0.0,
        "diff_95ci_lower": round(float(np.percentile(diffs, 2.5)), 4) if diffs.size else 0.0,
        "diff_95ci_upper": round(float(np.percentile(diffs, 97.5)), 4) if diffs.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-group WER
# ---------------------------------------------------------------------------

def per_group_wer(
    per_utt_results: list[dict],
    group_key: str,
) -> dict[str, dict]:
    """
    Compute WER and CER broken down by a metadata group (e.g., gender, accent).
    """
    groups = defaultdict(list)
    for r in per_utt_results:
        val = r.get(group_key, "unknown") or "unknown"
        groups[val].append(r)

    group_metrics = {}
    for gval, items in sorted(groups.items()):
        refs = [r["ref_norm"] for r in items]
        hyps = [r["hyp_norm"] for r in items]
        valid = [(h, r) for h, r in zip(hyps, refs) if r.strip()]
        if not valid:
            continue
        hyps_v, refs_v = zip(*valid)
        wer = jiwer.wer(list(refs_v), list(hyps_v))
        cer = jiwer.cer(list(refs_v), list(hyps_v))
        group_metrics[gval] = {
            "count": len(items),
            "wer": round(wer, 4),
            "cer": round(cer, 4),
        }

    return group_metrics


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    dataset: str,
    split: str = "test",
    output_dir: Optional[str] = None,
    batch_size: int = 8,
    max_new_tokens: int = 225,
    language: str = "ar",
    seed: int = 42,
) -> dict:
    """
    Evaluate a Whisper model on a dataset split.

    Returns a dict with corpus-level metrics, confidence intervals, and
    per-group breakdowns. Saves per-utterance results to a JSONL file.
    """
    set_seed(seed)
    device = get_device()
    print(f"\nEvaluating: {model_path}")
    print(f"Dataset: {dataset}, Split: {split}, Device: {device}")

    # Load model and processor
    processor = WhisperProcessor.from_pretrained(model_path, language=language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)
    model.eval()

    # Set generation config
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_new_tokens = max_new_tokens

    # Load test data
    manifest_path = MANIFEST_DIR / f"{dataset}_{split}.jsonl"
    records = load_manifest(manifest_path)
    records = filter_manifest(records, min_dur=0.5, max_dur=30.0)
    print(f"Evaluating on {len(records)} utterances")

    # Decode
    all_hypotheses = []
    all_references = []
    all_ids = []

    start_time = time.time()
    for i in tqdm(range(0, len(records), batch_size), desc="Decoding"):
        batch_records = records[i : i + batch_size]
        batch_audio = []

        for rec in batch_records:
            audio, sr = sf.read(rec["audio_filepath"], dtype="float32")
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            batch_audio.append(audio)
            all_references.append(rec["text_norm"])
            all_ids.append(rec["id"])

        # Process audio batch
        inputs = processor.feature_extractor(
            batch_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        # Whisper expects mel input features of fixed length 3000 (5 seconds at 16kHz)
        # Pad or truncate to ensure compatibility
        input_features = inputs.input_features.to(device)
        # Whisper expects mel input features of fixed length 3000 (5 seconds at 16kHz)
        # Pad or truncate to ensure compatibility across all evaluations
        target_length = 3000
        current_length = input_features.shape[-1]
        if current_length < target_length:
            # Pad with zeros
            pad_size = target_length - current_length
            input_features = torch.nn.functional.pad(input_features, (0, pad_size))
        elif current_length > target_length:
            # Truncate
            input_features = input_features[:, :, :target_length]
        target_length = 3000
        current_length = input_features.shape[-1]
        if current_length < target_length:
            # Pad with zeros
            pad_size = target_length - current_length
            input_features = torch.nn.functional.pad(input_features, (0, pad_size))
        elif current_length > target_length:
            # Truncate
            input_features = input_features[:, :, :target_length]

        # Generate
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language=language,
                task="transcribe",
                max_new_tokens=max_new_tokens,
                num_beams=1,  # greedy decoding for all arms
            )

        # Decode tokens to text
        transcriptions = processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        all_hypotheses.extend(transcriptions)

    decode_time = time.time() - start_time
    print(f"Decoding took {decode_time:.1f}s ({decode_time/len(records)*100:.1f}s per 100 utterances)")

    # Compute metrics
    corpus_metrics = compute_wer_cer(all_hypotheses, all_references)
    print(f"Corpus WER: {corpus_metrics['wer']:.4f}")
    print(f"Corpus CER: {corpus_metrics['cer']:.4f}")

    # Per-utterance metrics
    per_utt = compute_per_utterance_metrics(
        all_hypotheses, all_references, all_ids, records
    )

    # Bootstrap CIs
    wer_ci = bootstrap_ci(per_utt, "wer", seed=seed)
    cer_ci = bootstrap_ci(per_utt, "cer", seed=seed)
    print(f"WER 95% CI: [{wer_ci['lower']:.4f}, {wer_ci['upper']:.4f}]")
    print(f"CER 95% CI: [{cer_ci['lower']:.4f}, {cer_ci['upper']:.4f}]")

    # Per-group breakdowns
    group_results = {}
    for key in ("gender", "age", "accent", "program"):
        # Only compute if the field exists in the data
        if any(key in r for r in per_utt):
            grp = per_group_wer(per_utt, key)
            if len(grp) > 1:  # Skip if all values are "unknown"
                group_results[key] = grp
                print(f"\n{key.capitalize()} breakdown:")
                for gv, gm in grp.items():
                    print(f"  {gv}: WER={gm['wer']:.4f} CER={gm['cer']:.4f} (n={gm['count']})")

    # Compile results
    results = {
        "model": model_path,
        "dataset": dataset,
        "split": split,
        "num_utterances": len(records),
        "corpus_wer": corpus_metrics["wer"],
        "corpus_cer": corpus_metrics["cer"],
        "wer_95ci": wer_ci,
        "cer_95ci": cer_ci,
        "decode_time_seconds": round(decode_time, 1),
        "per_group": group_results,
    }

    # Save results
    if output_dir is None:
        model_name = Path(model_path).name if "/" not in model_path else model_path.replace("/", "_")
        output_dir = str(RESULTS_DIR / f"{model_name}_{dataset}_{split}")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save per-utterance results
    with open(out_path / "per_utterance.jsonl", "w", encoding="utf-8") as f:
        for r in per_utt:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save summary
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {out_path}")

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper on Arabic ASR")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name/path (e.g., openai/whisper-small or runs/cv_A0_seed42/best_model)")
    parser.add_argument("--dataset", type=str, nargs="+", default=["cv"],
                        choices=["cv", "mgb2"])
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    for dataset in args.dataset:
        evaluate(
            model_path=args.model,
            dataset=dataset,
            split=args.split,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()


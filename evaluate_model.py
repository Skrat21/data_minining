#!/usr/bin/env python3
"""
evaluate_model.py - Evaluation harness for Whisper Arabic ASR (fast version).

Same outputs and metric definitions as the original harness, but much faster.
The decoding code now lives in reusable functions
(load_whisper_for_inference / transcribe_records) that score_training_data.py
also uses.

What changed and why:
  - KV-cache forced ON during generation (use_cache=True). train.py used to
    leave model.config.use_cache=False in the saved model; without the cache
    every decoding step recomputes the whole sequence, which is extremely slow.
  - fp16 inference on GPU (disable with --fp32)
  - utterances sorted by duration before batching; results are restored to the
    original manifest order afterwards
  - audio reading + mel extraction run in a thread pool that prefetches the
    next batches while the GPU decodes (threads, not processes, so it also
    works on Windows without spawn problems)
  - SDPA attention when available
  - per-utterance WER/CER computed with one jiwer call; vectorised bootstrap
    that draws the same random indices as before (same CIs for a given seed)
  - --limit runs write to a separate "_limitN" folder so a smoke test can never
    overwrite real results

Usage:
    python evaluate_model.py --model openai/whisper-small --dataset cv --split test
    python evaluate_model.py --model runs/cv_A0_seed42/best_model --dataset cv mgb2 --split test
    python evaluate_model.py --model ... --batch-size 32 --num-workers 6
    python evaluate_model.py --model ... --limit 200      # quick smoke test
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import (
    MANIFEST_DIR,
    RESULTS_DIR,
    get_device,
    set_seed,
)
from data_loader import load_manifest, filter_manifest, load_audio_16k
from normalize import normalize_for_eval

try:
    import jiwer
except ImportError:
    raise ImportError("jiwer is required: pip install jiwer")

SAMPLE_RATE = 16000
N_FRAMES = 3000  # Whisper encoder input length (30 s of mel frames)


# ---------------------------------------------------------------------------
# Core WER/CER computation
# ---------------------------------------------------------------------------

def compute_wer_cer(hypotheses: list[str], references: list[str]) -> dict:
    """
    Corpus-level WER and CER (computed from totals, not averaged per-utterance).
    Both sides are normalized identically; empty references are dropped.
    """
    norm_hyps = [normalize_for_eval(h) for h in hypotheses]
    norm_refs = [normalize_for_eval(r) for r in references]

    valid = [(h, r) for h, r in zip(norm_hyps, norm_refs) if r.strip()]
    if not valid:
        return {"wer": 1.0, "cer": 1.0}
    norm_hyps, norm_refs = zip(*valid)
    norm_hyps, norm_refs = list(norm_hyps), list(norm_refs)

    wer = jiwer.process_words(norm_refs, norm_hyps).wer
    cer = jiwer.process_characters(norm_refs, norm_hyps).cer
    return {"wer": round(wer, 4), "cer": round(cer, 4)}


def _edit_errors(out) -> int:
    return out.substitutions + out.deletions + out.insertions


def compute_per_utterance_metrics(
    hypotheses: list[str],
    references: list[str],
    ids: list[str],
    metadata: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Per-utterance WER and CER. Besides the usual fields, stores the raw error
    counts (word_errors / char_errors) so the bootstrap does not have to
    reconstruct them from rounded WER values. All original fields are kept,
    so analyze_flagged.py / compare_results.py keep working.
    """
    results = []
    for i, (hyp, ref, uid) in enumerate(zip(hypotheses, references, ids)):
        ref_norm = normalize_for_eval(ref)
        hyp_norm = normalize_for_eval(hyp)
        ref_words = len(ref_norm.split())
        ref_chars = len(ref_norm)

        if ref_norm.strip():
            try:
                w_err = _edit_errors(jiwer.process_words(ref_norm, hyp_norm))
            except Exception:
                w_err = ref_words
            try:
                c_err = _edit_errors(jiwer.process_characters(ref_norm, hyp_norm))
            except Exception:
                c_err = ref_chars
            utt_wer = w_err / max(ref_words, 1)
            utt_cer = c_err / max(ref_chars, 1)
        else:
            utt_wer = 0.0 if not hyp_norm.strip() else 1.0
            utt_cer = utt_wer
            w_err = 0 if utt_wer == 0.0 else 1
            c_err = w_err

        entry = {
            "id": uid,
            "ref": ref,
            "hyp": hyp,
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "wer": round(utt_wer, 4),
            "cer": round(utt_cer, 4),
            "word_errors": int(w_err),
            "char_errors": int(c_err),
            "ref_words": ref_words,
            "hyp_words": len(hyp_norm.split()),
            "ref_chars": ref_chars,
            "hyp_chars": len(hyp_norm),
        }

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

def _error_arrays(results: list[dict], metric: str):
    """Return (errors, ref_lens) arrays; falls back to wer*ref_words for old files."""
    if metric == "wer":
        lens = np.array([r["ref_words"] for r in results], dtype=np.float64)
        errs = np.array(
            [r.get("word_errors", r["wer"] * r["ref_words"]) for r in results],
            dtype=np.float64,
        )
    else:
        lens = np.array([r["ref_chars"] for r in results], dtype=np.float64)
        errs = np.array(
            [r.get("char_errors", r["cer"] * r["ref_chars"]) for r in results],
            dtype=np.float64,
        )
    return errs, lens


def _bootstrap_ratios(errors, ref_lens, n_bootstrap, rng, chunk=100):
    """Corpus ratios for n_bootstrap resamples, drawing indices in chunks.
    The index stream is identical to drawing one resample at a time."""
    n = len(errors)
    out = []
    done = 0
    while done < n_bootstrap:
        k = min(chunk, n_bootstrap - done)
        idx = rng.randint(0, n, size=(k, n))
        e = errors[idx].sum(axis=1)
        l = ref_lens[idx].sum(axis=1)
        out.append(np.where(l > 0, e / np.maximum(l, 1e-12), 0.0))
        done += k
    return np.concatenate(out)


def bootstrap_ci(
    per_utt_results: list[dict],
    metric: str = "wer",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for corpus-level WER or CER."""
    n = len(per_utt_results)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}

    rng = np.random.RandomState(seed)
    errors, ref_lens = _error_arrays(per_utt_results, metric)
    vals = _bootstrap_ratios(errors, ref_lens, n_bootstrap, rng)

    alpha = (1 - ci) / 2
    return {
        "mean": round(float(np.mean(vals)), 4),
        "lower": round(float(np.percentile(vals, alpha * 100)), 4),
        "upper": round(float(np.percentile(vals, (1 - alpha) * 100)), 4),
    }


def paired_bootstrap_test(
    results_a: list[dict],
    results_b: list[dict],
    metric: str = "wer",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap: is system B better than system A? (same test set/order)"""
    assert len(results_a) == len(results_b), "Both systems must have same number of utterances"
    rng = np.random.RandomState(seed)
    n = len(results_a)

    errors_a, ref_lens = _error_arrays(results_a, metric)
    errors_b, _ = _error_arrays(results_b, metric)

    diffs = []
    done = 0
    while done < n_bootstrap:
        k = min(100, n_bootstrap - done)
        idx = rng.randint(0, n, size=(k, n))
        tot = ref_lens[idx].sum(axis=1)
        ok = tot > 0
        d = (errors_a[idx].sum(axis=1) - errors_b[idx].sum(axis=1))[ok] / tot[ok]
        diffs.append(d)
        done += k
    diffs = np.concatenate(diffs) if diffs else np.array([])

    if diffs.size == 0:
        return {"b_better_pct": 0.0, "mean_diff": 0.0,
                "diff_95ci_lower": 0.0, "diff_95ci_upper": 0.0}
    return {
        "b_better_pct": round(float((diffs > 0).sum() / diffs.size), 4),
        "mean_diff": round(float(diffs.mean()), 4),
        "diff_95ci_lower": round(float(np.percentile(diffs, 2.5)), 4),
        "diff_95ci_upper": round(float(np.percentile(diffs, 97.5)), 4),
    }


# ---------------------------------------------------------------------------
# Per-group WER
# ---------------------------------------------------------------------------

def per_group_wer(per_utt_results: list[dict], group_key: str) -> dict[str, dict]:
    """WER and CER broken down by a metadata group (e.g., gender, accent)."""
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
# Fast decoding (shared with score_training_data.py)
# ---------------------------------------------------------------------------

def load_whisper_for_inference(
    model_path: str,
    language: str = "ar",
    max_new_tokens: int = 225,
    fp16: bool = True,
    device: Optional[torch.device] = None,
):
    """Load processor + model ready for fast greedy decoding. Returns (processor, model, device)."""
    device = device or get_device()
    dtype = torch.float16 if (fp16 and device.type == "cuda") else torch.float32

    processor = WhisperProcessor.from_pretrained(model_path, language=language, task="transcribe")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_path, attn_implementation="sdpa")
    except Exception:
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    # train.py saved models with use_cache=False (needed for gradient
    # checkpointing). Generation without the KV cache is extremely slow.
    model.config.use_cache = True
    model.generation_config.use_cache = True
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_new_tokens = max_new_tokens
    return processor, model, device


def _record_duration(rec: dict) -> float:
    d = rec.get("duration")
    if d is not None:
        try:
            return float(d)
        except (TypeError, ValueError):
            pass
    try:
        return float(sf.info(rec["audio_filepath"]).duration)
    except Exception:
        return 0.0


def _prepare_batch(positions: list[int], sorted_records: list[dict], feature_extractor):
    """Read audio + compute mel features for one batch (runs in a worker thread)."""
    audios = []
    for p in positions:
        path = sorted_records[p]["audio_filepath"]
        try:
            audios.append(load_audio_16k(path, SAMPLE_RATE))
        except Exception as e:
            print(f"\n[warn] failed to load {path}: {e}")
            audios.append(np.zeros(SAMPLE_RATE, dtype=np.float32))  # keeps ordering intact
    feats = feature_extractor(
        audios, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features  # padded / truncated to 3000 frames by the extractor

    cur = feats.shape[-1]  # safety net, normally a no-op
    if cur < N_FRAMES:
        feats = torch.nn.functional.pad(feats, (0, N_FRAMES - cur))
    elif cur > N_FRAMES:
        feats = feats[:, :, :N_FRAMES]
    return positions, feats


def _prefetch_batches(sorted_records, batch_size, feature_extractor, num_workers):
    """Yield (positions, features) in order, preparing upcoming batches in background threads."""
    n = len(sorted_records)
    batches = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]

    if num_workers <= 0:
        for b in batches:
            yield _prepare_batch(b, sorted_records, feature_extractor)
        return

    depth = max(2, num_workers * 2)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        pending = deque()
        it = iter(batches)

        def submit_next():
            b = next(it, None)
            if b is not None:
                pending.append(ex.submit(_prepare_batch, b, sorted_records, feature_extractor))

        for _ in range(depth):
            submit_next()
        while pending:
            fut = pending.popleft()
            result = fut.result()
            submit_next()
            yield result


def transcribe_records(
    model,
    processor,
    records: list[dict],
    batch_size: int = 16,
    max_new_tokens: int = 225,
    language: str = "ar",
    num_workers: int = 4,
    desc: str = "Decoding",
) -> tuple[list[str], float]:
    """
    Greedy-decode `records` (needs "audio_filepath"; "duration" is used for sorting).
    Returns (hypotheses in the SAME order as `records`, decode_seconds).
    """
    n = len(records)
    if n == 0:
        return [], 0.0

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Longest clips first: similar lengths per batch, and any OOM shows up immediately.
    durations = [_record_duration(r) for r in records]
    order = sorted(range(n), key=lambda i: -durations[i])
    sorted_records = [records[i] for i in order]

    hyps = [""] * n
    n_batches = (n + batch_size - 1) // batch_size
    start = time.time()

    stream = _prefetch_batches(sorted_records, batch_size, processor.feature_extractor, num_workers)
    for positions, feats in tqdm(stream, total=n_batches, desc=desc):
        feats = feats.to(device, dtype=dtype, non_blocking=True)
        with torch.inference_mode():
            predicted_ids = model.generate(
                feats,
                language=language,
                task="transcribe",
                max_new_tokens=max_new_tokens,
                num_beams=1,      # greedy decoding for all arms
                use_cache=True,   # never decode without the KV cache
            )
        texts = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        for pos, text in zip(positions, texts):
            hyps[order[pos]] = text

    return hyps, time.time() - start


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    dataset: str,
    split: str = "test",
    output_dir: Optional[str] = None,
    batch_size: int = 16,
    max_new_tokens: int = 225,
    language: str = "ar",
    seed: int = 42,
    num_workers: int = 4,
    fp16: bool = True,
    limit: Optional[int] = None,
) -> dict:
    """
    Evaluate a Whisper model on a dataset split.
    Returns corpus metrics, CIs and per-group breakdowns; saves per-utterance
    results (in the ORIGINAL manifest order) to JSONL.
    """
    set_seed(seed)

    print(f"\nEvaluating: {model_path}")
    processor, model, device = load_whisper_for_inference(
        model_path, language=language, max_new_tokens=max_new_tokens, fp16=fp16
    )
    print(f"Dataset: {dataset}, Split: {split}, Device: {device}, dtype: {next(model.parameters()).dtype}, "
          f"batch_size: {batch_size}, workers: {num_workers}")

    # ---- data ----
    manifest_path = MANIFEST_DIR / f"{dataset}_{split}.jsonl"
    records = load_manifest(manifest_path)
    records = filter_manifest(records, min_dur=0.5, max_dur=30.0)
    if limit:
        records = records[:limit]
    print(f"Evaluating on {len(records)} utterances")

    # ---- decode ----
    all_hypotheses, decode_time = transcribe_records(
        model, processor, records,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        language=language, num_workers=num_workers,
    )
    print(f"Decoding took {decode_time:.1f}s "
          f"({decode_time / max(len(records), 1) * 100:.1f}s per 100 utterances)")

    all_references = [r["text_norm"] for r in records]
    all_ids = [r["id"] for r in records]

    # ---- metrics ----
    t0 = time.time()
    corpus_metrics = compute_wer_cer(all_hypotheses, all_references)
    print(f"Corpus WER: {corpus_metrics['wer']:.4f}")
    print(f"Corpus CER: {corpus_metrics['cer']:.4f}")

    per_utt = compute_per_utterance_metrics(all_hypotheses, all_references, all_ids, records)

    # Quick health check for the "model never stops" failure mode
    long_hyp = sum(1 for r in per_utt if r["hyp_words"] > 3 * max(r["ref_words"], 1) and r["hyp_words"] > 10)
    if per_utt and long_hyp / len(per_utt) > 0.02:
        print(f"[warn] {long_hyp} hypotheses ({100 * long_hyp / len(per_utt):.1f}%) are >3x longer than the "
              f"reference - the model may not be emitting end-of-text (check how it was trained).")

    wer_ci = bootstrap_ci(per_utt, "wer", seed=seed)
    cer_ci = bootstrap_ci(per_utt, "cer", seed=seed)
    print(f"WER 95% CI: [{wer_ci['lower']:.4f}, {wer_ci['upper']:.4f}]")
    print(f"CER 95% CI: [{cer_ci['lower']:.4f}, {cer_ci['upper']:.4f}]")

    group_results = {}
    for key in ("gender", "age", "accent", "program"):
        if any(key in r for r in per_utt):
            grp = per_group_wer(per_utt, key)
            if len(grp) > 1:  # skip if all values are "unknown"
                group_results[key] = grp
                print(f"\n{key.capitalize()} breakdown:")
                for gv, gm in grp.items():
                    print(f"  {gv}: WER={gm['wer']:.4f} CER={gm['cer']:.4f} (n={gm['count']})")
    print(f"\nMetric computation took {time.time() - t0:.1f}s")

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

    # ---- save ----
    if output_dir is None:
        model_name = Path(model_path).name if "/" not in model_path else model_path.replace("/", "_")
        suffix = f"_limit{limit}" if limit else ""   # smoke tests never overwrite real results
        output_dir = str(RESULTS_DIR / f"{model_name}_{dataset}_{split}{suffix}")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "per_utterance.jsonl", "w", encoding="utf-8") as f:
        for r in per_utt:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    # ---- cleanup ----
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=225)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Prefetch threads for audio loading / mel extraction (0 = off)")
    parser.add_argument("--fp32", action="store_true",
                        help="Disable fp16 inference (bit-exact fp32 numbers, slower)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N utterances (smoke test; saved to a _limitN folder)")
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
            num_workers=args.num_workers,
            fp16=not args.fp32,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()

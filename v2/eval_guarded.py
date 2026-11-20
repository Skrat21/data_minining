#!/usr/bin/env python3
"""Compare raw and dev-configured repetition guards on one supplied manifest."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from data_loader import load_audio_16k  # noqa: E402
from evaluate_model import (  # noqa: E402
    compute_per_utterance_metrics,
    compute_wer_cer,
)
from v2.mgb2_common import (  # noqa: E402
    MAX_AUDIO_SECONDS,
    append_notes,
    gpu_preflight,
    load_inference_model,
    load_jsonl,
    write_json,
    write_jsonl,
)

# Phase-0 MGB-2 dev statistic: P99 of reference payload tokens per audio second.
DEV_P99_TOKENS_PER_SECOND = 9.4464
DEV_TOKEN_CAP_MARGIN = 16
DEV_TOKEN_CAP_MINIMUM = 32
DEV_TOKEN_CAP_MAXIMUM = 225
WHISPER_PROMPT_TOKENS = 4


def duration_token_cap(
    duration: float,
    tokens_per_second: float = DEV_P99_TOKENS_PER_SECOND,
    margin: int = DEV_TOKEN_CAP_MARGIN,
    minimum: int = DEV_TOKEN_CAP_MINIMUM,
    maximum: int = DEV_TOKEN_CAP_MAXIMUM,
) -> int:
    if (
        duration < 0
        or tokens_per_second <= 0
        or margin < 0
        or minimum <= 0
        or maximum < minimum
    ):
        raise ValueError("Duration/rate/margin must be non-negative and rate positive")
    return min(maximum, max(minimum, math.ceil(duration * tokens_per_second + margin)))


def repeated_word_loop(text: str, max_ngram: int = 12) -> tuple[bool, str]:
    if max_ngram <= 0:
        raise ValueError("max_ngram must be positive")
    words = text.split()
    candidate: tuple[int, int, int] | None = None
    for start in range(len(words)):
        for size in range(1, min(max_ngram, (len(words) - start) // 3) + 1):
            repeats = 1
            while (
                start + (repeats + 1) * size <= len(words)
                and words[start : start + size]
                == words[start + repeats * size : start + (repeats + 1) * size]
            ):
                repeats += 1
            if repeats >= 3:
                span = repeats * size
                if candidate is None or (start, -span) < (
                    candidate[0],
                    -candidate[1] * candidate[2],
                ):
                    candidate = (start, size, repeats)
    if candidate is None:
        return False, text
    start, size, _ = candidate
    return True, " ".join(words[: start + 2 * size])


def decode_records(
    model,
    processor,
    records: list[dict],
    caps: list[int],
    batch_size: int,
    no_repeat_ngram_size: int | None,
    trim_loops: bool,
    desc: str,
) -> tuple[list[dict], float]:
    if len(records) != len(caps):
        raise ValueError("Each record must have a corresponding token cap")
    if batch_size <= 0 or any(cap <= 0 for cap in caps):
        raise ValueError("Batch size and token caps must be positive")
    groups: dict[int, list[int]] = defaultdict(list)
    for index, cap in enumerate(caps):
        groups[cap].append(index)

    decoded: list[dict | None] = [None] * len(records)
    started = time.perf_counter()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    from tqdm import tqdm

    total_batches = sum(
        math.ceil(len(indexes) / batch_size) for indexes in groups.values()
    )
    bar = tqdm(total=total_batches, desc=desc)
    for cap, indexes in sorted(groups.items()):
        for offset in range(0, len(indexes), batch_size):
            batch_indexes = indexes[offset : offset + batch_size]
            batch_records = [records[i] for i in batch_indexes]
            audios = [
                load_audio_16k(r["audio_filepath"])
                for r in batch_records
            ]
            features = processor.feature_extractor(
                audios, sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device=device, dtype=dtype, non_blocking=True)
            generation_args = {
                "max_new_tokens": cap,
                "num_beams": 1,
                "use_cache": True,
            }
            if no_repeat_ngram_size is not None:
                generation_args["no_repeat_ngram_size"] = no_repeat_ngram_size
            with torch.inference_mode():
                sequences = model.generate(features, **generation_args)

            texts = processor.tokenizer.batch_decode(
                sequences, skip_special_tokens=True
            )
            for record_index, sequence, text in zip(
                batch_indexes, sequences, texts
            ):
                token_count = max(
                    0, int(sequence.shape[0]) - WHISPER_PROMPT_TOKENS
                )
                hit_cap = (
                    token_count >= cap
                    and int(sequence[-1]) != processor.tokenizer.eos_token_id
                )
                has_loop, trimmed = repeated_word_loop(text)
                decoded[record_index] = {
                    "hyp": trimmed if trim_loops else text,
                    "generation_tokens": token_count,
                    "cap": cap,
                    "hit_cap": bool(hit_cap),
                    "has_loop": has_loop,
                    "loop_trimmed": bool(trim_loops and has_loop),
                }
            bar.update(1)
    bar.close()
    if any(row is None for row in decoded):
        raise RuntimeError("Decoder did not produce output for every input record")
    return [row for row in decoded if row is not None], time.perf_counter() - started


def score_outputs(
    records: list[dict], outputs: list[dict]
) -> tuple[dict, list[dict]]:
    hypotheses = [row["hyp"] for row in outputs]
    metrics = compute_wer_cer(
        hypotheses, [record["text_norm"] for record in records]
    )
    per_utterance = compute_per_utterance_metrics(
        hypotheses,
        [record["text_norm"] for record in records],
        [record["id"] for record in records],
        records,
    )
    return metrics, per_utterance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--manifest", required=True, help="Dev or final test manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-test-eval",
        action="store_true",
        help="Required only for the final test decode after the guard is selected on dev",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-repeat-ngram-size", type=int, choices=(4, 5, 6))
    parser.add_argument("--post-hoc-loop-trim", action="store_true")
    parser.add_argument("--fixed-raw-cap", type=int, default=225)
    parser.add_argument("--tokens-per-second", type=float, default=DEV_P99_TOKENS_PER_SECOND)
    parser.add_argument("--cap-margin", type=int, default=DEV_TOKEN_CAP_MARGIN)
    parser.add_argument("--cap-minimum", type=int, default=DEV_TOKEN_CAP_MINIMUM)
    parser.add_argument("--cap-maximum", type=int, default=DEV_TOKEN_CAP_MAXIMUM)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    is_test_manifest = "test" in Path(args.manifest).stem.lower()
    if is_test_manifest and not args.allow_test_eval:
        raise ValueError("Test decoding requires --allow-test-eval after dev-only selection")
    if args.allow_test_eval and not is_test_manifest:
        raise ValueError("--allow-test-eval requires a manifest whose filename identifies test")
    if args.fixed_raw_cap <= 0 or args.fixed_raw_cap > 448:
        raise ValueError("--fixed-raw-cap must be between 1 and 448")
    if (
        args.cap_minimum <= 0
        or args.cap_maximum < args.cap_minimum
        or args.cap_maximum > 448
    ):
        raise ValueError("Dynamic cap bounds must satisfy 1 <= minimum <= maximum <= 448")
    if args.tokens_per_second <= 0 or args.cap_margin < 0:
        raise ValueError("Token rate must be positive and cap margin non-negative")
    if args.cap_maximum > MAX_AUDIO_SECONDS * args.tokens_per_second + args.cap_margin:
        print("Cap maximum exceeds the dev-derived linear estimate at 30 seconds.")

    started = time.perf_counter()
    gpu_preflight(args.output_dir)
    records = load_jsonl(args.manifest)
    if not records:
        raise ValueError(f"Manifest is empty: {args.manifest}")

    processor, model, _ = load_inference_model(
        args.model, processor_path=args.processor_path, fp16=True
    )
    raw_outputs, raw_decode_seconds = decode_records(
        model,
        processor,
        records,
        [args.fixed_raw_cap] * len(records),
        args.batch_size,
        no_repeat_ngram_size=None,
        trim_loops=False,
        desc="Raw greedy decode",
    )
    guarded_caps = [
        duration_token_cap(
            float(record["duration"]),
            args.tokens_per_second,
            args.cap_margin,
            args.cap_minimum,
            args.cap_maximum,
        )
        for record in records
    ]
    guarded_outputs, guarded_decode_seconds = decode_records(
        model,
        processor,
        records,
        guarded_caps,
        args.batch_size,
        args.no_repeat_ngram_size,
        args.post_hoc_loop_trim,
        desc="Guarded greedy decode",
    )
    raw_metrics, raw_per_utterance = score_outputs(records, raw_outputs)
    guarded_metrics, guarded_per_utterance = score_outputs(records, guarded_outputs)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_rows = [
        {
            **row,
            "generation_tokens": generation["generation_tokens"],
            "generation_cap": generation["cap"],
            "hit_cap": generation["hit_cap"],
            "has_loop": generation["has_loop"],
        }
        for row, generation in zip(raw_per_utterance, raw_outputs)
    ]
    guarded_rows = [
        {
            **row,
            "generation_tokens": generation["generation_tokens"],
            "generation_cap": generation["cap"],
            "hit_cap": generation["hit_cap"],
            "has_loop": generation["has_loop"],
            "loop_trimmed": generation["loop_trimmed"],
        }
        for row, generation in zip(guarded_per_utterance, guarded_outputs)
    ]
    write_jsonl(output / "raw_per_utterance.jsonl", raw_rows)
    write_jsonl(output / "guarded_per_utterance.jsonl", guarded_rows)
    summary = {
        "model": str(Path(args.model).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "num_clips": len(records),
        "raw": {
            **raw_metrics,
            "decode_seconds": round(raw_decode_seconds, 3),
            "outputs_hit_cap": sum(row["hit_cap"] for row in raw_outputs),
            "outputs_with_loops": sum(row["has_loop"] for row in raw_outputs),
        },
        "guarded": {
            **guarded_metrics,
            "decode_seconds": round(guarded_decode_seconds, 3),
            "outputs_hit_cap": sum(row["hit_cap"] for row in guarded_outputs),
            "outputs_with_loops": sum(row["has_loop"] for row in guarded_outputs),
            "outputs_loop_trimmed": sum(
                row["loop_trimmed"] for row in guarded_outputs
            ),
            "mean_token_cap": sum(guarded_caps) / len(guarded_caps),
            "max_token_cap": max(guarded_caps),
        },
        "guard_config": {
            "cap_formula": "min(max_cap, max(min_cap, ceil(duration_s * rate + margin)))",
            "p99_dev_payload_tokens_per_second": args.tokens_per_second,
            "margin_tokens": args.cap_margin,
            "minimum_tokens": args.cap_minimum,
            "maximum_tokens": args.cap_maximum,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "post_hoc_loop_trim": args.post_hoc_loop_trim,
            "token_rate_basis": (
                "MGB-2 dev P99 payload token/audio-second statistic; "
                "reference tokens include the Whisper task prefix"
            ),
        },
        "total_wall_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "guard_summary.json", summary)
    append_notes(
        f"### Guard evaluation: {Path(args.manifest).name}\n\n"
        f"- Model: `{Path(args.model).resolve()}`; clips: {len(records)}.\n"
        f"- Raw WER/CER: {raw_metrics['wer']}/{raw_metrics['cer']}; guarded "
        f"WER/CER: {guarded_metrics['wer']}/{guarded_metrics['cer']}.\n"
        f"- Raw and guarded decode seconds: {raw_decode_seconds:.1f} / "
        f"{guarded_decode_seconds:.1f}; total wall seconds: "
        f"{summary['total_wall_seconds']:.1f}.\n"
    )
    print(summary)


if __name__ == "__main__":
    main()

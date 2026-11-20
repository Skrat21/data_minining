#!/usr/bin/env python3
"""Select among retained checkpoints using WER on the fixed MGB-2 dev subset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_model import compute_wer_cer, transcribe_records  # noqa: E402
from v2.mgb2_common import (  # noqa: E402
    DEV_MANIFEST,
    append_notes,
    fixed_dev_records,
    gpu_preflight,
    load_inference_model,
    write_json,
)


def checkpoint_number(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Unexpected checkpoint directory name: {path.name}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dev-manifest", default=str(DEV_MANIFEST))
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=225)
    args = parser.parse_args()
    started = time.perf_counter()

    if "test" in Path(args.dev_manifest).stem.lower():
        raise ValueError("Checkpoint selection must use dev data, never test data")
    if args.dev_size <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Dev size/batch size must be positive and worker count non-negative")
    if not 1 <= args.max_new_tokens <= 448:
        raise ValueError("--max-new-tokens must be between 1 and 448")
    run_dir = Path(args.run_dir).resolve()
    gpu_preflight(run_dir)
    processor_path = run_dir / "final_model"
    if not processor_path.is_dir():
        raise FileNotFoundError(f"Final model/tokenizer directory is missing: {processor_path}")
    checkpoints = sorted(
        (p for p in run_dir.glob("checkpoint-*") if p.is_dir()),
        key=checkpoint_number,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No retained checkpoints found in {run_dir}")

    records = fixed_dev_records(args.dev_manifest, args.dev_size)
    references = [r["text_norm"] for r in records]
    scored = []
    for checkpoint in checkpoints:
        checkpoint_start = time.perf_counter()
        processor, model, _ = load_inference_model(
            checkpoint, processor_path=processor_path, fp16=True
        )
        hypotheses, decode_seconds = transcribe_records(
            model,
            processor,
            records,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            language="ar",
            num_workers=args.num_workers,
            desc=f"Dev {checkpoint.name}",
        )
        metrics = compute_wer_cer(hypotheses, references)
        row = {
            "checkpoint": str(checkpoint),
            "dev_wer": metrics["wer"],
            "dev_cer": metrics["cer"],
            "num_dev_clips": len(records),
            "decode_seconds": round(decode_seconds, 3),
            "wall_seconds": round(time.perf_counter() - checkpoint_start, 3),
        }
        scored.append(row)
        print(row)
        del model

    best = min(scored, key=lambda row: (row["dev_wer"], row["dev_cer"], row["checkpoint"]))
    result = {
        "selection_metric": "raw greedy dev WER",
        "dev_manifest": str(Path(args.dev_manifest).resolve()),
        "dev_subset_size": len(records),
        "dev_subset_seed": 123,
        "best_checkpoint": best["checkpoint"],
        "best_dev_wer": best["dev_wer"],
        "best_dev_cer": best["dev_cer"],
        "checkpoints": scored,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(run_dir / "checkpoint_selection.json", result)
    append_notes(
        f"### Dev checkpoint selection: {run_dir.name}\n\n"
        f"- Best checkpoint: `{best['checkpoint']}`; raw dev WER/CER: "
        f"{best['dev_wer']}/{best['dev_cer']} on {len(records)} clips.\n"
        f"- Checkpoints scored: {len(scored)}; wall time: "
        f"{result['wall_seconds']:.1f} s.\n"
    )
    print(result)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create the deterministic MGB-2 A0/A1/A2 training subset."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2.mgb2_common import (  # noqa: E402
    MANIFEST_DIR,
    RUNS_DIR,
    TRAIN_SUBSET_SEED,
    TRAIN_SUBSET_SIZE,
    append_notes,
    eligible_training_records,
    load_jsonl,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(MANIFEST_DIR / "mgb2_train.jsonl"),
        help="Source training manifest",
    )
    parser.add_argument(
        "--output",
        default=str(RUNS_DIR / "data" / "train_subset_seed123.jsonl"),
        help="Output manifest path",
    )
    parser.add_argument("--size", type=int, default=TRAIN_SUBSET_SIZE)
    parser.add_argument("--seed", type=int, default=TRAIN_SUBSET_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing subset: {output}")

    records = eligible_training_records(load_jsonl(args.source))
    if args.size <= 0 or args.size > len(records):
        raise ValueError(f"Requested {args.size} clips from {len(records)} eligible rows")

    import random

    chosen_indexes = set(random.Random(args.seed).sample(range(len(records)), args.size))
    subset = [r for i, r in enumerate(records) if i in chosen_indexes]
    write_jsonl(output, subset)
    duration_hours = sum(float(r["duration"]) for r in subset) / 3600
    summary = {
        "source_manifest": str(Path(args.source).resolve()),
        "output_manifest": str(output.resolve()),
        "subset_size": len(subset),
        "seed": args.seed,
        "eligible_source_count": len(records),
        "duration_hours": round(duration_hours, 4),
        "sampling": "uniform without replacement; output preserves source order",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output.with_name(output.stem + "_summary.json"), summary)
    append_notes(
        "### Fixed MGB-2 training subset\n\n"
        f"- Command settings: size={args.size}, seed={args.seed}; "
        f"eligible source clips={len(records)}.\n"
        f"- Output: `{output.resolve()}`; duration={duration_hours:.4f} h; "
        f"wall time={summary['elapsed_seconds']:.3f} s.\n"
    )
    print(summary)


if __name__ == "__main__":
    main()

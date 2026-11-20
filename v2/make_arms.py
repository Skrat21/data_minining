#!/usr/bin/env python3
"""Build matching A1 flagged-removal and A2 random-removal manifests."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2.mgb2_common import (  # noqa: E402
    RUNS_DIR,
    append_notes,
    load_jsonl,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        default=str(RUNS_DIR / "data" / "train_subset_seed123.jsonl"),
    )
    parser.add_argument(
        "--flagged-ids",
        default=str(RUNS_DIR / "scoring" / "A0" / "flagged_ids.txt"),
    )
    parser.add_argument("--output-dir", default=str(RUNS_DIR / "data"))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()

    output = Path(args.output_dir)
    a1_path = output / "train_A1_seed42.jsonl"
    a2_path = output / "train_A2_seed42.jsonl"
    if not args.overwrite and (a1_path.exists() or a2_path.exists()):
        raise FileExistsError("Refusing to overwrite an existing A1/A2 manifest")

    subset = load_jsonl(args.subset)
    flagged = {
        line.strip()
        for line in Path(args.flagged_ids).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    subset_ids = {str(r["id"]) for r in subset}
    unknown = flagged - subset_ids
    if unknown:
        raise ValueError(f"Flag list contains {len(unknown)} IDs outside the subset")
    if not flagged:
        raise ValueError("Flag list is empty")

    a1 = [r for r in subset if str(r["id"]) not in flagged]
    random_ids = set(
        random.Random(args.random_seed).sample(
            [str(r["id"]) for r in subset], len(flagged)
        )
    )
    a2 = [r for r in subset if str(r["id"]) not in random_ids]
    write_jsonl(a1_path, a1)
    write_jsonl(a2_path, a2)
    summary = {
        "subset_manifest": str(Path(args.subset).resolve()),
        "flagged_ids_file": str(Path(args.flagged_ids).resolve()),
        "a1_removed_count": len(flagged),
        "a1_remaining_count": len(a1),
        "a2_random_seed": args.random_seed,
        "a2_removed_count": len(random_ids),
        "a2_remaining_count": len(a2),
        "a2_removed_ids": sorted(random_ids),
        "a1_manifest": str(a1_path.resolve()),
        "a2_manifest": str(a2_path.resolve()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "arms_summary.json", summary)
    append_notes(
        "### A1/A2 training manifests\n\n"
        f"- A1 removes {len(flagged)} A0 worst-WER clips; A2 removes "
        f"{len(random_ids)} random clips using seed {args.random_seed}.\n"
        f"- Each remaining manifest has {len(a1)} clips; elapsed "
        f"{summary['elapsed_seconds']:.3f} s.\n"
    )
    print(
        f"A1 removed {len(flagged)}; A2 removed {len(random_ids)} "
        f"(seed {args.random_seed}); both have {len(a1)} rows."
    )


if __name__ == "__main__":
    main()

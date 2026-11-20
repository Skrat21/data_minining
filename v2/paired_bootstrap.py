#!/usr/bin/env python3
"""Paired bootstrap comparison of two saved per-utterance ASR result files."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_model import bootstrap_ci, compute_wer_cer, paired_bootstrap_test  # noqa: E402
from v2.mgb2_common import (  # noqa: E402
    RUNS_DIR,
    append_notes,
    load_jsonl,
    write_json,
)


def keyed_rows(path: str | Path) -> dict[str, dict]:
    rows = load_jsonl(path)
    keyed = {str(row["id"]): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate IDs in results file: {path}")
    return keyed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-a", required=True, help="Per-utterance JSONL")
    parser.add_argument("--system-b", required=True, help="Per-utterance JSONL")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument(
        "--output",
        default=str(RUNS_DIR / "paired_bootstrap" / "comparison.json"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    started = time.perf_counter()

    rows_a = keyed_rows(args.system_a)
    rows_b = keyed_rows(args.system_b)
    if not rows_a or rows_a.keys() != rows_b.keys():
        raise ValueError("Systems must contain the same non-empty set of utterance IDs")
    ids = sorted(rows_a)
    ordered_a = [rows_a[uid] for uid in ids]
    ordered_b = [rows_b[uid] for uid in ids]
    for left, right in zip(ordered_a, ordered_b):
        ref_a = left.get("ref_norm", left.get("ref"))
        ref_b = right.get("ref_norm", right.get("ref"))
        if ref_a != ref_b:
            raise ValueError(f"Reference mismatch for utterance {left['id']}")

    report = {
        "system_a": {
            "label": args.label_a,
            "wer_cer": compute_wer_cer(
                [row["hyp"] for row in ordered_a],
                [row["ref"] for row in ordered_a],
            ),
            "wer_95ci": bootstrap_ci(
                ordered_a, "wer", seed=args.seed
            ),
            "cer_95ci": bootstrap_ci(
                ordered_a, "cer", seed=args.seed
            ),
        },
        "system_b": {
            "label": args.label_b,
            "wer_cer": compute_wer_cer(
                [row["hyp"] for row in ordered_b],
                [row["ref"] for row in ordered_b],
            ),
            "wer_95ci": bootstrap_ci(
                ordered_b, "wer", seed=args.seed
            ),
            "cer_95ci": bootstrap_ci(
                ordered_b, "cer", seed=args.seed
            ),
        },
        "paired_bootstrap": {
            "wer": paired_bootstrap_test(
                ordered_a,
                ordered_b,
                "wer",
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed,
            ),
            "cer": paired_bootstrap_test(
                ordered_a,
                ordered_b,
                "cer",
                n_bootstrap=args.bootstrap_samples,
                seed=args.seed,
            ),
            "difference_sign": (
                f"{args.label_a} minus {args.label_b}; positive means "
                f"{args.label_b} has lower error"
            ),
        },
        "num_utterances": len(ids),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(args.output, report)
    append_notes(
        f"### Paired bootstrap: {args.label_a} vs {args.label_b}\n\n"
        f"- Utterances: {len(ids)}; replicates: {args.bootstrap_samples}; "
        f"WER mean difference (A minus B): "
        f"{report['paired_bootstrap']['wer']['mean_diff']}.\n"
    )
    print(report)


if __name__ == "__main__":
    main()

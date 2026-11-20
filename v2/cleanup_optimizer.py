#!/usr/bin/env python3
"""Delete optimizer state only after explicit approval of a run's checkpoint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2.mgb2_common import RUNS_DIR, append_notes, disk_free_gib  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--confirm-checkpoint-choice",
        action="store_true",
        help="Required: user has approved the selected checkpoint for this run",
    )
    args = parser.parse_args()
    if not args.confirm_checkpoint_choice:
        raise RuntimeError("Refusing cleanup without --confirm-checkpoint-choice")

    run_dir = Path(args.run_dir).resolve()
    runs_root = RUNS_DIR.resolve()
    try:
        run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError(f"Cleanup is restricted to {runs_root}; got {run_dir}") from exc
    if run_dir.parent != runs_root:
        raise ValueError(f"Run directory must be a direct child of {runs_root}")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    started = time.perf_counter()
    free_before = disk_free_gib(run_dir)
    removed_bytes = 0
    removed_files = []
    for checkpoint in sorted(run_dir.glob("checkpoint-*")):
        if not checkpoint.is_dir():
            continue
        optimizer_state = checkpoint / "optimizer.pt"
        if optimizer_state.is_file():
            removed_bytes += optimizer_state.stat().st_size
            optimizer_state.unlink()
            removed_files.append(str(optimizer_state))

    free_after = disk_free_gib(run_dir)
    report = (
        f"### Approved optimizer-state cleanup: {run_dir.name}\n\n"
        f"- Removed files: {len(removed_files)}; bytes removed: {removed_bytes}.\n"
        f"- Free space before/after: {free_before:.2f} / {free_after:.2f} GiB.\n"
        f"- Wall seconds: {time.perf_counter() - started:.3f}.\n"
        f"- Files: {', '.join(f'`{path}`' for path in removed_files) or 'none'}.\n"
    )
    append_notes(report)
    print(report)


if __name__ == "__main__":
    main()

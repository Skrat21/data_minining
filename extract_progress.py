#!/usr/bin/env python3
"""
extract_progress.py - Read training progress while a run is STILL going.

Safe to run at any time: it only reads the trainer_state.json files that the
Trainer writes inside runs/<run>/checkpoint-*/ (every save_steps). It never
imports torch and never touches the GPU, so it cannot slow training down.

For every run it prints the dev WER/CER at each evaluation step and writes:
  results/progress/<run>_progress.csv   (step, epoch, train loss, dev WER/CER, ...)
  results/progress/<run>_progress.png   (train loss and dev WER/CER curves)

IMPORTANT for the report: these WER/CER values are on the 500-utterance DEV
subset used for model selection during training, NOT on the test set. Report
the test-set numbers from evaluate_model.py / run_experiment.py instead.

Usage:
    python extract_progress.py                       # all runs under runs/
    python extract_progress.py --run runs/cv_A0_seed42
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def find_latest_state(run_dir: Path) -> dict | None:
    """The trainer_state.json with the highest global_step (it holds the full log history)."""
    best = None
    for p in run_dir.glob("checkpoint-*/trainer_state.json"):
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # that checkpoint is being written right now
        if best is None or state.get("global_step", 0) > best.get("global_step", 0):
            best = state
    return best


def build_rows(log_history: list[dict]) -> list[dict]:
    """Merge training-loss entries and evaluation entries by step."""
    rows: dict[int, dict] = {}
    for h in log_history:
        step = h.get("step")
        if step is None:
            continue
        row = rows.setdefault(step, {"step": step})
        row["epoch"] = h.get("epoch", row.get("epoch"))
        if "loss" in h:
            row["train_loss"] = h["loss"]
            row["learning_rate"] = h.get("learning_rate")
            row["grad_norm"] = h.get("grad_norm")
        if "eval_wer" in h:
            row["dev_wer"] = h["eval_wer"]
            row["dev_cer"] = h.get("eval_cer")
            row["dev_loss"] = h.get("eval_loss")
    return [rows[s] for s in sorted(rows)]


FIELDS = ["step", "epoch", "train_loss", "learning_rate", "grad_norm", "dev_wer", "dev_cer", "dev_loss"]


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def write_plot(rows: list[dict], title: str, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed - skipped the plot; the CSV is still written)")
        return False

    loss = [(r["step"], r["train_loss"]) for r in rows if r.get("train_loss") is not None]
    dev = [r for r in rows if r.get("dev_wer") is not None]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if loss:
        axes[0].plot(*zip(*loss))
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("step")
    axes[0].grid(alpha=0.3)

    if dev:
        axes[1].plot([r["step"] for r in dev], [r["dev_wer"] for r in dev], marker="o", label="dev WER")
        axes[1].plot([r["step"] for r in dev], [r["dev_cer"] for r in dev], marker="s", label="dev CER")
        axes[1].legend()
    axes[1].set_title("Dev-subset error rate")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def report_run(run_dir: Path, out_dir: Path) -> None:
    print(f"\n=== {run_dir.name} ===")
    state = find_latest_state(run_dir)
    if state is None:
        print("  no checkpoint with a trainer_state.json yet "
              "(the first one appears after save_steps = 500 steps)")
        return

    rows = build_rows(state.get("log_history", []))
    step, max_steps = state.get("global_step"), state.get("max_steps")
    print(f"  progress: step {step}/{max_steps}, epoch {state.get('epoch', 0):.2f}")

    losses = [r["train_loss"] for r in rows if r.get("train_loss") is not None]
    if losses:
        tail = losses[-5:]
        print(f"  train loss: first logged {losses[0]:.4f}, last {losses[-1]:.4f} "
              f"(mean of last {len(tail)}: {sum(tail) / len(tail):.4f})")

    dev = [r for r in rows if r.get("dev_wer") is not None]
    if dev:
        print("  dev subset (NOT the test set):")
        print(f"    {'step':>6} {'dev WER':>9} {'dev CER':>9}")
        for r in dev:
            print(f"    {r['step']:>6} {r['dev_wer']:>9.4f} {r['dev_cer']:>9.4f}")
        best = min(dev, key=lambda r: r["dev_wer"])
        print(f"  best dev WER so far: {best['dev_wer']:.4f} at step {best['step']}")
    else:
        print("  no dev evaluation recorded yet")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{run_dir.name}_progress.csv"
    write_csv(rows, csv_path)
    print(f"  wrote {csv_path}")
    png_path = out_dir / f"{run_dir.name}_progress.png"
    if write_plot(rows, f"{run_dir.name} (step {step})", png_path):
        print(f"  wrote {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract training progress from a running or finished run")
    parser.add_argument("--run", type=str, default=None, help="one run folder, e.g. runs/cv_A0_seed42")
    parser.add_argument("--runs-dir", type=str, default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "results" / "progress"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.run:
        run_dirs = [Path(args.run)]
    else:
        runs_dir = Path(args.runs_dir)
        run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir()) if runs_dir.exists() else []
    if not run_dirs:
        print("No run folders found.")
        return
    for d in run_dirs:
        report_run(d, out_dir)


if __name__ == "__main__":
    main()

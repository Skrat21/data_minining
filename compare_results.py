#!/usr/bin/env python3
"""
compare_results.py - Generate final comparison tables and plots (Step 12).

Reads per-utterance results from all arms and produces:
  - Summary table with WER, CER, and confidence intervals
  - Paired bootstrap tests between arms
  - Per-group comparison plots
  - LaTeX-ready table for the report

Usage:
  python compare_results.py --dataset cv
  python compare_results.py --dataset cv mgb2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR
from evaluate_model import paired_bootstrap_test


def load_arm_results(dataset: str) -> dict[str, dict]:
    """Load metrics.json from all arm result directories for a dataset."""
    arms = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(dataset):
            metrics_file = d / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    arms[d.name] = json.load(f)
    return arms


def load_per_utterance(result_dir: Path) -> list[dict]:
    """Load per-utterance results from a JSONL file."""
    path = result_dir / "per_utterance.jsonl"
    if not path.exists():
        return []
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def make_summary_table(arms: dict[str, dict], dataset: str) -> pd.DataFrame:
    """Create a summary comparison table."""
    rows = []
    a0_wer = None
    for name, m in sorted(arms.items()):
        ci = m.get("wer_95ci", {})
        cer_ci = m.get("cer_95ci", {})

        if "A0" in name:
            a0_wer = m["corpus_wer"]

        relative = ""
        if a0_wer and a0_wer > 0 and "A0" not in name and "zero_shot" not in name:
            rel = (m["corpus_wer"] - a0_wer) / a0_wer * 100
            relative = f"{rel:+.1f}%"

        rows.append({
            "Arm": name,
            "WER": f"{m['corpus_wer']:.4f}",
            "CER": f"{m['corpus_cer']:.4f}",
            "WER 95% CI": f"[{ci.get('lower', '?'):.4f}, {ci.get('upper', '?'):.4f}]" if ci else "—",
            "Rel. to A0": relative,
            "N": m.get("num_utterances", "?"),
        })

    df = pd.DataFrame(rows)
    return df


def run_all_paired_tests(dataset: str, arms: dict[str, dict]):
    """Run paired bootstrap tests between A0 and all other arms."""
    a0_dir = RESULTS_DIR / f"{dataset}_A0_seed42"
    a0_results = load_per_utterance(a0_dir)
    if not a0_results:
        print("No A0 per-utterance results found for paired tests.")
        return {}

    tests = {}
    for name in sorted(arms.keys()):
        if "A0" in name or "zero_shot" in name:
            continue
        arm_dir = RESULTS_DIR / name
        arm_results = load_per_utterance(arm_dir)
        if len(arm_results) == len(a0_results):
            test = paired_bootstrap_test(a0_results, arm_results)
            tests[name] = test
            print(f"\n{name} vs A0:")
            print(f"  {name} better: {test['b_better_pct']*100:.1f}%")
            print(f"  Mean WER diff: {test['mean_diff']:.4f}")
            print(f"  95% CI: [{test['diff_95ci_lower']:.4f}, {test['diff_95ci_upper']:.4f}]")
    return tests


def plot_comparison(arms: dict[str, dict], dataset: str, out_dir: Path):
    """Create comparison bar chart."""
    names = []
    wers = []
    cers = []
    ci_lows = []
    ci_highs = []

    for name, m in sorted(arms.items()):
        names.append(name.replace(f"{dataset}_", ""))
        wers.append(m["corpus_wer"])
        cers.append(m["corpus_cer"])
        ci = m.get("wer_95ci", {})
        ci_lows.append(m["corpus_wer"] - ci.get("lower", m["corpus_wer"]))
        ci_highs.append(ci.get("upper", m["corpus_wer"]) - m["corpus_wer"])

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
    bars_wer = ax.bar(x - width/2, wers, width, label="WER",
                       yerr=[ci_lows, ci_highs], capsize=3)
    bars_cer = ax.bar(x + width/2, cers, width, label="CER", alpha=0.7)

    ax.set_ylabel("Error Rate")
    ax.set_title(f"{dataset.upper()} - Arm Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset}_arm_comparison.png", dpi=150)
    plt.close()
    print(f"Saved comparison plot to {out_dir / f'{dataset}_arm_comparison.png'}")


def generate_latex_table(df: pd.DataFrame, dataset: str, out_dir: Path):
    """Generate a LaTeX table for the report."""
    latex = df.to_latex(index=False, escape=True)
    path = out_dir / f"{dataset}_results_table.tex"
    with open(path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {path}")


def compare(dataset: str):
    """Run full comparison for a dataset."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {dataset.upper()}")
    print(f"{'='*70}")

    out_dir = RESULTS_DIR / f"{dataset}_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = load_arm_results(dataset)
    if not arms:
        print(f"No results found for {dataset}.")
        return

    # Summary table
    df = make_summary_table(arms, dataset)
    print("\n" + df.to_string(index=False))
    df.to_csv(out_dir / f"{dataset}_summary.csv", index=False)

    # Paired bootstrap tests
    tests = run_all_paired_tests(dataset, arms)
    if tests:
        with open(out_dir / f"{dataset}_paired_tests.json", "w") as f:
            json.dump(tests, f, indent=2)

    # Plots
    plot_comparison(arms, dataset, out_dir)

    # LaTeX
    generate_latex_table(df, dataset, out_dir)

    print(f"\nAll comparison outputs saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare all experiment arms")
    parser.add_argument("--dataset", type=str, nargs="+", default=["cv"],
                        choices=["cv", "mgb2"])
    args = parser.parse_args()

    for dataset in args.dataset:
        compare(dataset)


if __name__ == "__main__":
    main()


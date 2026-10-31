#!/usr/bin/env python3
"""
analyze_flagged.py - Data mining analysis of flagged utterances (Step 13).

Analyzes the characteristics of utterances flagged by the Kedalai loop:
  - Descriptive statistics (duration, words, chars/sec, etc.)
  - Association rules over discretized features
  - Decision tree / logistic regression predicting the flag
  - Text analysis of flagged references
  - Plots and visualizations

Usage:
  python analyze_flagged.py --dataset cv
  python analyze_flagged.py --dataset mgb2
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import MANIFEST_DIR, RESULTS_DIR


def load_scores_and_manifest(dataset: str) -> pd.DataFrame:
    """Load training scores and merge with manifest metadata."""
    scores_path = RESULTS_DIR / f"{dataset}_scoring" / "train_scores.jsonl"
    manifest_path = MANIFEST_DIR / f"{dataset}_train.jsonl"

    # Load scores
    scores = []
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scores.append(json.loads(line))
    df_scores = pd.DataFrame(scores)

    # Load manifest for additional metadata
    manifest = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                manifest.append(json.loads(line))
    df_manifest = pd.DataFrame(manifest)

    # Merge
    df = df_scores.merge(
        df_manifest[["id", "duration", "gender", "age", "accent",
                      "up_votes", "down_votes", "n_words", "n_chars",
                      "chars_per_sec"]].rename(columns=lambda c: c if c not in df_scores.columns else f"m_{c}"),
        on="id",
        how="left",
    )

    # Ensure numeric columns
    for col in ["duration", "up_votes", "down_votes", "n_words", "n_chars", "chars_per_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag column
    df["flagged"] = (df["wer"] >= 1.0).astype(int)

    return df


def descriptive_analysis(df: pd.DataFrame, dataset: str, out_dir: Path):
    """Compare flagged vs unflagged utterances."""
    print("\n=== DESCRIPTIVE ANALYSIS ===")

    flagged = df[df["flagged"] == 1]
    unflagged = df[df["flagged"] == 0]

    print(f"Flagged: {len(flagged)} ({100*len(flagged)/len(df):.1f}%)")
    print(f"Unflagged: {len(unflagged)} ({100*len(unflagged)/len(df):.1f}%)")

    # Compare numeric features
    num_cols = ["duration", "ref_words", "ref_chars", "wer", "cer"]
    if "chars_per_sec" in df.columns:
        num_cols.append("chars_per_sec")

    comparison = pd.DataFrame({
        "flagged_mean": flagged[num_cols].mean(),
        "flagged_median": flagged[num_cols].median(),
        "unflagged_mean": unflagged[num_cols].mean(),
        "unflagged_median": unflagged[num_cols].median(),
    }).round(4)
    print("\nFeature comparison (flagged vs unflagged):")
    print(comparison.to_string())
    comparison.to_csv(out_dir / "feature_comparison.csv")

    # Categorical breakdowns
    for cat_col in ["gender", "age", "accent"]:
        if cat_col in df.columns and df[cat_col].notna().any():
            vals = df[cat_col].fillna("unknown").replace("", "unknown")
            vc = vals.value_counts()
            if len(vc) > 1 and not (vc.index == "unknown").all():
                print(f"\n{cat_col.capitalize()} breakdown:")
                for val in vc.index[:10]:
                    mask = vals == val
                    total = mask.sum()
                    flag_count = (df.loc[mask, "flagged"] == 1).sum()
                    flag_rate = 100 * flag_count / total if total > 0 else 0
                    share = 100 * total / len(df)
                    flag_share = 100 * flag_count / len(flagged) if len(flagged) > 0 else 0
                    print(f"  {val}: {total} ({share:.1f}% of data), "
                          f"{flag_count} flagged ({flag_rate:.1f}% flag rate, "
                          f"{flag_share:.1f}% of all flags)")

    return comparison


def plot_wer_distribution(df: pd.DataFrame, dataset: str, out_dir: Path):
    """Plot the WER distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # WER histogram
    axes[0].hist(df["wer"].clip(upper=3.0), bins=50, edgecolor="black", alpha=0.7)
    axes[0].axvline(x=1.0, color="red", linestyle="--", label="WER=1.0 (Kedalai threshold)")
    axes[0].set_xlabel("WER")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{dataset.upper()} Training Set WER Distribution")
    axes[0].legend()

    # CER histogram
    axes[1].hist(df["cer"].clip(upper=3.0), bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_xlabel("CER")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{dataset.upper()} Training Set CER Distribution")

    # WER vs duration
    sample = df.sample(min(5000, len(df)), random_state=42)
    axes[2].scatter(sample["duration"], sample["wer"].clip(upper=3.0),
                    alpha=0.3, s=5, c=sample["flagged"], cmap="coolwarm")
    axes[2].axhline(y=1.0, color="red", linestyle="--")
    axes[2].set_xlabel("Duration (s)")
    axes[2].set_ylabel("WER")
    axes[2].set_title("WER vs Duration")

    plt.tight_layout()
    plt.savefig(out_dir / f"{dataset}_wer_distribution.png", dpi=150)
    plt.close()
    print(f"Saved WER distribution plot to {out_dir / f'{dataset}_wer_distribution.png'}")


def text_analysis(df: pd.DataFrame, dataset: str, out_dir: Path):
    """Analyze text characteristics of flagged utterances."""
    print("\n=== TEXT ANALYSIS ===")
    flagged = df[df["flagged"] == 1]

    if len(flagged) == 0:
        print("No flagged utterances to analyze.")
        return

    # Short references
    short = flagged[flagged["ref_words"] <= 2]
    print(f"Flagged with <= 2 ref words: {len(short)} ({100*len(short)/len(flagged):.1f}%)")

    # Very short references (1 word)
    one_word = flagged[flagged["ref_words"] == 1]
    print(f"Flagged with 1 ref word: {len(one_word)} ({100*len(one_word)/len(flagged):.1f}%)")

    # Hypothesis/reference length ratio
    flagged_copy = flagged.copy()
    flagged_copy["hyp_ref_ratio"] = flagged_copy["hyp_words"] / flagged_copy["ref_words"].clip(lower=1)
    print(f"\nHypothesis/Reference word ratio (flagged):")
    print(f"  Mean: {flagged_copy['hyp_ref_ratio'].mean():.2f}")
    print(f"  Median: {flagged_copy['hyp_ref_ratio'].median():.2f}")
    print(f"  Hallucination-like (ratio > 3): {(flagged_copy['hyp_ref_ratio'] > 3).sum()}")
    print(f"  Deletion-like (ratio < 0.3): {(flagged_copy['hyp_ref_ratio'] < 0.3).sum()}")

    # Check for numbers in references
    import re
    has_digits = flagged["ref_norm"].str.contains(r"\d", regex=True, na=False)
    print(f"\nFlagged refs with digits: {has_digits.sum()} ({100*has_digits.mean():.1f}%)")

    # Check for very short references
    ref_len_dist = flagged["ref_words"].describe()
    print(f"\nFlagged reference word count distribution:")
    print(ref_len_dist.to_string())


def decision_tree_analysis(df: pd.DataFrame, dataset: str, out_dir: Path):
    """Train a decision tree to predict the flag from cheap features."""
    print("\n=== DECISION TREE / LOGISTIC REGRESSION ===")

    features = []
    feature_names = []

    for col in ["duration", "ref_words", "ref_chars", "chars_per_sec"]:
        if col in df.columns:
            features.append(df[col].fillna(0).values)
            feature_names.append(col)

    # Add categorical features as dummy variables
    for col in ["gender", "age", "accent"]:
        if col in df.columns:
            vals = df[col].fillna("unknown").replace("", "unknown")
            if vals.nunique() > 1 and vals.nunique() <= 20:
                dummies = pd.get_dummies(vals, prefix=col)
                for dc in dummies.columns:
                    features.append(dummies[dc].values)
                    feature_names.append(dc)

    if not features:
        print("No features available for classification.")
        return

    X = np.column_stack(features)
    y = df["flagged"].values

    # Decision tree
    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=42)
    dt_scores = cross_val_score(dt, X, y, cv=5, scoring="accuracy")
    dt.fit(X, y)
    print(f"Decision Tree accuracy (5-fold CV): {dt_scores.mean():.4f} ± {dt_scores.std():.4f}")
    print(f"Baseline (majority class): {max(y.mean(), 1-y.mean()):.4f}")

    # Feature importance
    tree_text = export_text(dt, feature_names=feature_names, max_depth=3)
    print(f"\nDecision tree (top 3 levels):\n{tree_text[:1000]}")

    importances = pd.Series(dt.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    print("\nFeature importances:")
    for name, imp in importances.head(10).items():
        print(f"  {name}: {imp:.4f}")

    # Logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_scaled, y, cv=5, scoring="accuracy")
    print(f"\nLogistic Regression accuracy (5-fold CV): {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")

    # Save feature importances
    importances.to_csv(out_dir / f"{dataset}_feature_importances.csv")


def analyze(dataset: str):
    """Run all analyses."""
    out_dir = RESULTS_DIR / f"{dataset}_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ANALYSIS: {dataset.upper()}")
    print(f"{'='*70}")

    df = load_scores_and_manifest(dataset)
    print(f"Loaded {len(df)} utterances")

    descriptive_analysis(df, dataset, out_dir)
    plot_wer_distribution(df, dataset, out_dir)
    text_analysis(df, dataset, out_dir)
    decision_tree_analysis(df, dataset, out_dir)

    print(f"\nAll analysis outputs saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze flagged utterances")
    parser.add_argument("--dataset", type=str, nargs="+", default=["cv"],
                        choices=["cv", "mgb2"])
    args = parser.parse_args()

    for dataset in args.dataset:
        analyze(dataset)


if __name__ == "__main__":
    main()


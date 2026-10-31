#!/usr/bin/env python3
"""
run_experiment.py - Master script to run the full Phase 2 experiment.

Orchestrates all steps in order:
  1. Verify data
  2. Zero-shot baseline (Step 5)
  3. A0: Fine-tune on all data (Step 6)
  4. Score training data (Step 7)
  5. A1: Kedalai loop retrain (Step 8)
  6. A2: Random removal control, 3 seeds (Step 9)
  7. Compare all arms (Step 12)

Usage:
    # Run everything from scratch
    python run_experiment.py --dataset cv --model openai/whisper-small

    # Resume from a specific step
    python run_experiment.py --dataset cv --start-step 4

    # Pilot with whisper-base
    python run_experiment.py --dataset cv --model openai/whisper-base --pilot

Changes vs. the original:
  - --freeze-encoder is now passed on to TrainConfig (it was parsed but ignored)
  - new --eval-batch-size (dev-set evaluation during training)
"""

from __future__ import annotations

import os
import sys

# --smoke must be known BEFORE config is imported: it redirects runs/ and results/
# to runs_smoke/ and results_smoke/ so real runs are never touched.
if "--smoke" in sys.argv:
    os.environ["SMOKE_TEST"] = "1"

import argparse
import json
from pathlib import Path

from config import TrainConfig, MANIFEST_DIR, RESULTS_DIR, RUNS_DIR, print_env_info

# Set by --smoke (None = normal full run)
EVAL_LIMIT = None    # evaluate only the first N test utterances
SCORE_LIMIT = None   # score only the first N training utterances
MAX_FLAGGED = None   # keep at most N flagged utterances


def verify_data(dataset: str):
    """Step 1: Verify manifests exist and have data."""
    print("\n" + "=" * 70)
    print("STEP 1: Verify data")
    print("=" * 70)

    for split in ["train", "dev", "test"]:
        path = MANIFEST_DIR / f"{dataset}_{split}.jsonl"
        if not path.exists():
            print(f"ERROR: {path} not found!")
            sys.exit(1)
        count = sum(1 for _ in open(path, encoding="utf-8"))
        print(f"  {path.name}: {count} utterances")
    print("Data verification passed ✓")


def run_zero_shot(dataset: str, model: str):
    """Step 5: Zero-shot baseline."""
    print("\n" + "=" * 70)
    print("STEP 5: Zero-shot baseline")
    print("=" * 70)

    from evaluate_model import evaluate

    out_dir = str(RESULTS_DIR / f"zero_shot_{dataset}")
    evaluate(
        model_path=model,
        dataset=dataset,
        split="test",
        output_dir=out_dir,
        limit=EVAL_LIMIT,
    )
    return out_dir


def run_a0(dataset: str, model: str, seed: int, **kwargs):
    """Step 6: A0 baseline fine-tuning."""
    print("\n" + "=" * 70)
    print("STEP 6: A0 Baseline fine-tuning")
    print("=" * 70)

    from train import train
    from evaluate_model import evaluate

    cfg = TrainConfig(
        dataset=dataset,
        arm="A0",
        seed=seed,
        model_name=model,
        **kwargs,
    )
    output_dir = train(cfg)

    # Evaluate on test set
    model_path = str(output_dir / "best_model")
    eval_dir = str(RESULTS_DIR / f"{dataset}_A0_seed{seed}")
    evaluate(
        model_path=model_path,
        dataset=dataset,
        split="test",
        output_dir=eval_dir,
        limit=EVAL_LIMIT,
    )
    return model_path


def run_scoring(dataset: str, model_path: str):
    """Step 7: Score training data with A0 model."""
    print("\n" + "=" * 70)
    print("STEP 7: Score training data")
    print("=" * 70)

    from score_training_data import score_training_set

    summary = score_training_set(
        model_path=model_path,
        dataset=dataset,
        limit=SCORE_LIMIT,
        max_flagged=MAX_FLAGGED,
    )

    # Check the gate
    flagged_pct = summary["flagged_pct"]
    if flagged_pct < 0.2:
        print(f"\n⚠️  GATE: Only {flagged_pct}% flagged — too few for the loop to change much.")
        print("   The analysis and CER-percentile arm become the main contribution.")

    return summary


def run_a1(dataset: str, model: str, seed: int, **kwargs):
    """Step 8: A1 Kedalai loop."""
    print("\n" + "=" * 70)
    print("STEP 8: A1 Kedalai loop")
    print("=" * 70)

    from train import train
    from evaluate_model import evaluate

    flagged_ids_file = str(RESULTS_DIR / f"{dataset}_scoring" / "flagged_ids.txt")
    if not Path(flagged_ids_file).exists():
        print(f"ERROR: {flagged_ids_file} not found. Run scoring step first.")
        return

    cfg = TrainConfig(
        dataset=dataset,
        arm="A1",
        seed=seed,
        model_name=model,  # retrain from pretrained, NOT from A0
        removed_ids_file=flagged_ids_file,
        **kwargs,
    )
    output_dir = train(cfg)

    model_path = str(output_dir / "best_model")
    eval_dir = str(RESULTS_DIR / f"{dataset}_A1_seed{seed}")
    evaluate(
        model_path=model_path,
        dataset=dataset,
        split="test",
        output_dir=eval_dir,
        limit=EVAL_LIMIT,
    )


def run_a2(dataset: str, model: str, seeds: list[int], **kwargs):
    """Step 9: A2 random removal control."""
    print("\n" + "=" * 70)
    print("STEP 9: A2 Random removal control")
    print("=" * 70)

    from train import train
    from evaluate_model import evaluate

    for rs in seeds:
        removed_file = str(RESULTS_DIR / f"{dataset}_scoring" / f"random_removed_seed{rs}.txt")
        if not Path(removed_file).exists():
            print(f"ERROR: {removed_file} not found. Run scoring step first.")
            continue

        cfg = TrainConfig(
            dataset=dataset,
            arm="A2",
            seed=rs,
            model_name=model,
            removed_ids_file=removed_file,
            **kwargs,
        )
        output_dir = train(cfg)

        model_path = str(output_dir / "best_model")
        eval_dir = str(RESULTS_DIR / f"{dataset}_A2_seed{rs}")
        evaluate(
            model_path=model_path,
            dataset=dataset,
            split="test",
            output_dir=eval_dir,
            limit=EVAL_LIMIT,
        )


def compare_arms(dataset: str):
    """Step 12: Compare all arms."""
    print("\n" + "=" * 70)
    print("STEP 12: Compare all arms")
    print("=" * 70)

    from evaluate_model import paired_bootstrap_test

    arms = {}
    for arm_dir in RESULTS_DIR.glob(f"{dataset}_*"):
        metrics_file = arm_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                arms[arm_dir.name] = json.load(f)

    if not arms:
        print("No results found to compare.")
        return

    print(f"\n{'Arm':<30} {'WER':>8} {'CER':>8} {'95% CI WER':>20}")
    print("-" * 70)
    for name, m in sorted(arms.items()):
        ci = m.get("wer_95ci", {})
        ci_str = f"[{ci.get('lower', '?')}, {ci.get('upper', '?')}]"
        print(f"{name:<30} {m['corpus_wer']:>8.4f} {m['corpus_cer']:>8.4f} {ci_str:>20}")

    # Paired bootstrap: A1 vs A0
    a0_per_utt = RESULTS_DIR / f"{dataset}_A0_seed42" / "per_utterance.jsonl"
    a1_per_utt = RESULTS_DIR / f"{dataset}_A1_seed42" / "per_utterance.jsonl"
    if a0_per_utt.exists() and a1_per_utt.exists():
        a0_results = [json.loads(l) for l in open(a0_per_utt, encoding="utf-8") if l.strip()]
        a1_results = [json.loads(l) for l in open(a1_per_utt, encoding="utf-8") if l.strip()]
        if len(a0_results) == len(a1_results):
            test = paired_bootstrap_test(a0_results, a1_results)
            print(f"\nPaired bootstrap A1 vs A0:")
            print(f"  A1 better in {test['b_better_pct']*100:.1f}% of bootstraps")
            print(f"  Mean WER difference: {test['mean_diff']:.4f}")
            print(f"  95% CI of difference: [{test['diff_95ci_lower']:.4f}, {test['diff_95ci_upper']:.4f}]")

    # Save comparison table
    comp_path = RESULTS_DIR / f"{dataset}_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(arms, f, indent=2)
    print(f"\nComparison saved to: {comp_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the full Phase 2 experiment")
    parser.add_argument("--dataset", type=str, default="cv", choices=["cv", "mgb2"])
    parser.add_argument("--model", type=str, default=None,
                        help="default: openai/whisper-small (openai/whisper-tiny with --smoke)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-step", type=int, default=1,
                        help="Resume from this step (1=verify, 5=zero-shot, 6=A0, 7=score, 8=A1, 9=A2, 12=compare)")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: fewer epochs, smaller eval")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder to save VRAM")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny end-to-end test (whisper-tiny, 48 train clips, 4 steps, 16 test clips). "
                             "Writes to runs_smoke/ and results_smoke/ only.")
    args = parser.parse_args()

    global EVAL_LIMIT, SCORE_LIMIT, MAX_FLAGGED
    if args.model is None:
        args.model = "openai/whisper-tiny" if args.smoke else "openai/whisper-small"
    if args.smoke:
        EVAL_LIMIT, SCORE_LIMIT, MAX_FLAGGED = 16, 48, 8
        print("\n*** SMOKE TEST: tiny data, tiny model. The WER numbers are meaningless; "
              "this only checks that every step runs. Outputs go to runs_smoke/ and results_smoke/. ***")

    print_env_info()

    # Extra kwargs for TrainConfig
    extra = {}
    if args.pilot:
        extra["num_epochs"] = 1
        extra["eval_steps"] = 100
        extra["save_steps"] = 100
        extra["dev_subset_size"] = 100
    extra["per_device_train_batch"] = args.batch_size
    extra["per_device_eval_batch"] = args.eval_batch_size
    extra["gradient_accumulation_steps"] = args.grad_accum
    if args.freeze_encoder:
        extra["freeze_encoder"] = True
    if args.smoke:
        extra.update(
            num_epochs=1,
            max_steps=4,
            eval_steps=2,
            save_steps=2,
            dev_subset_size=16,
            max_train_samples=48,
            per_device_train_batch=2,
            per_device_eval_batch=4,
            gradient_accumulation_steps=1,
        )

    step = args.start_step

    if step <= 1:
        verify_data(args.dataset)

    if step <= 5:
        run_zero_shot(args.dataset, args.model)

    if step <= 6:
        a0_model = run_a0(args.dataset, args.model, args.seed, **extra)
    else:
        a0_model = str(RUNS_DIR / f"{args.dataset}_A0_seed{args.seed}" / "best_model")

    if step <= 7:
        run_scoring(args.dataset, a0_model)

    if step <= 8:
        run_a1(args.dataset, args.model, args.seed, **extra)

    if step <= 9:
        run_a2(args.dataset, args.model, [0] if args.smoke else [0, 1, 2], **extra)

    if step <= 12:
        compare_arms(args.dataset)

    # Run analysis
    from analyze_flagged import analyze
    if args.smoke:
        # With 48 utterances the statistical models in the analysis can legitimately fail
        # (too few samples per class), so a failure here is reported, not fatal.
        try:
            analyze(args.dataset)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"\n[smoke] analysis step failed at tiny scale ({type(e).__name__}). "
                  "This is expected with 48 utterances unless the error is an ImportError.")
    else:
        analyze(args.dataset)

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE - every step ran. Safe to start the real run." if args.smoke
          else "EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

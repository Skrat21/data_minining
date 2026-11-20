#!/usr/bin/env python3
"""Train one frozen-encoder MGB-2 Whisper-small arm with loss-only validation."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import applies Hub compatibility patches before Transformers is initialized.
import hf_compat  # pyright: ignore[reportUnusedImport]
import torch  # noqa: E402
from transformers import (  # noqa: E402
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from config import set_seed  # noqa: E402
from data_loader import WhisperASRDataset, WhisperDataCollator  # noqa: E402
from v2.mgb2_common import (  # noqa: E402
    DEV_MANIFEST,
    DEV_SUBSET_SEED,
    RUNS_DIR,
    append_notes,
    fixed_dev_records,
    gpu_preflight,
    load_jsonl,
    write_json,
)


TRAIN_BATCH = 2
GRAD_ACCUM = 8
EPOCHS = 2
SAVE_EVAL_STEPS = 400
LOGGING_STEPS = 50
MODEL_NAME = "openai/whisper-small"
SEED = 42

_EVAL_STRATEGY_KW = (
    "eval_strategy"
    if "eval_strategy" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    else "evaluation_strategy"
)
_PROCESSOR_KW = (
    "processing_class"
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
    else "tokenizer"
)


class ThroughputCallback(TrainerCallback):
    def __init__(self, path: Path, presentations_per_update: int):
        self.path = path
        self.presentations_per_update = presentations_per_update
        self.previous_step = 0
        self.previous_time = 0.0
        self.clock_start = 0.0

    # Keep the TrainerCallback signature; args/kwargs are intentionally unused.
    def on_train_begin(
        self, args, state, control, **kwargs
    ):  # pyright: ignore[reportUnusedParameter]
        self.clock_start = time.perf_counter()
        self.previous_time = self.clock_start
        self.previous_step = int(state.global_step)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        print(f"Training began: {datetime.now().astimezone().isoformat(timespec='seconds')}")
        return control

    def on_step_end(
        self, args, state, control, **kwargs
    ):  # pyright: ignore[reportUnusedParameter]
        step = int(state.global_step)
        if step == 0 or (step % LOGGING_STEPS and step != state.max_steps):
            return control
        now = time.perf_counter()
        elapsed = now - self.previous_time
        step_delta = step - self.previous_step
        clips = step_delta * self.presentations_per_update
        row = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "step": step,
            "interval_seconds": round(elapsed, 3),
            "interval_clip_presentations": clips,
            "clips_per_second": round(clips / elapsed, 4) if elapsed else None,
            "seconds_per_step": round(elapsed / step_delta, 4) if step_delta else None,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[throughput] {row['timestamp']} step={step} "
            f"clips/s={row['clips_per_second']} "
            f"seconds/step={row['seconds_per_step']}"
        )
        self.previous_step = step
        self.previous_time = now
        return control


def run(args: argparse.Namespace) -> Path:
    whole_run_start = time.perf_counter()
    run_dir = Path(args.output_dir).resolve()
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 (full epochs) or a positive integer")
    if "test" in Path(args.dev_manifest).stem.lower():
        raise ValueError("Training evaluation must use dev data, never test data")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse a non-empty run directory: {run_dir}")
    if run_dir.parent.resolve() != RUNS_DIR.resolve():
        raise ValueError(f"Run output must be directly under {RUNS_DIR}")

    gpu_preflight(run_dir)
    set_seed(SEED)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_jsonl(args.train_manifest)
    dev_records = fixed_dev_records(
        args.dev_manifest, count=args.dev_subset_size, seed=DEV_SUBSET_SEED
    )
    if not train_records:
        raise ValueError(f"No training rows found in {args.train_manifest}")
    if not dev_records:
        raise ValueError(f"No dev rows found in {args.dev_manifest}")

    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    config = {
        "arm": args.arm,
        "model_name": MODEL_NAME,
        "train_manifest": str(Path(args.train_manifest).resolve()),
        "dev_manifest": str(Path(args.dev_manifest).resolve()),
        "dev_subset_size": len(dev_records),
        "dev_subset_seed": DEV_SUBSET_SEED,
        "train_clips": len(train_records),
        "learning_rate": 1e-5,
        "warmup_ratio": 0.05,
        "epochs": EPOCHS,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": TRAIN_BATCH,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "fp16": True,
        "optimizer": "adamw_torch",
        "gradient_checkpointing": True,
        "encoder_frozen": True,
        "eval_steps": SAVE_EVAL_STEPS,
        "save_steps": SAVE_EVAL_STEPS,
        "save_total_limit": 3,
        "load_best_model_at_end": False,
        "predict_with_generate": False,
        "prediction_loss_only": True,
        "seed": SEED,
        "started_at": started_at,
    }
    write_json(run_dir / "run_config.json", config)
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language="ar", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    for parameter in model.model.encoder.parameters():
        parameter.requires_grad = False
    model.generation_config.language = "ar"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.use_cache = True

    train_dataset = WhisperASRDataset(
        train_records, processor, language="ar", task="transcribe"
    )
    eval_dataset = WhisperASRDataset(
        dev_records, processor, language="ar", task="transcribe"
    )
    collator = WhisperDataCollator(processor)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        num_train_epochs=EPOCHS,
        max_steps=args.max_steps,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **{_EVAL_STRATEGY_KW: "steps"},
        eval_steps=SAVE_EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_EVAL_STEPS,
        save_total_limit=3,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=False,
        predict_with_generate=False,
        prediction_loss_only=True,
        report_to="none",
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    callback = ThroughputCallback(
        run_dir / "throughput.jsonl", TRAIN_BATCH * GRAD_ACCUM
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[callback],
        **{_PROCESSOR_KW: processor.feature_extractor},
    )

    train_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    train_start = time.perf_counter()
    result = trainer.train()
    train_seconds = time.perf_counter() - train_start
    trainer.model.config.use_cache = True
    trainer.model.generation_config.use_cache = True

    final_model = run_dir / "final_model"
    trainer.save_model(str(final_model))
    processor.save_pretrained(str(final_model))
    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    total_wall_seconds = time.perf_counter() - whole_run_start
    metrics = dict(result.metrics)
    metrics.update(
        {
            "train_wall_seconds": round(train_seconds, 3),
            "whole_run_wall_seconds": round(total_wall_seconds, 3),
            "train_started_at": train_started_at,
            "finished_at": finished_at,
            "train_clips": len(train_records),
            "train_clip_presentations_estimate": (
                int(trainer.state.global_step) * TRAIN_BATCH * GRAD_ACCUM
            ),
        }
    )
    write_json(run_dir / "train_metrics.json", metrics)
    append_notes(
        f"### {args.arm} training run\n\n"
        f"- Started: {started_at}; finished: {finished_at}.\n"
        f"- Training wall time: {train_seconds:.1f} s; total run wall time: "
        f"{total_wall_seconds:.1f} s.\n"
        f"- Clips: {len(train_records)}; epochs: {EPOCHS}; metrics: "
        f"`{run_dir / 'train_metrics.json'}`.\n"
    )
    print(json.dumps(metrics, indent=2))
    return final_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A0", "A1", "A2"), required=True)
    parser.add_argument(
        "--train-manifest",
        default=str(RUNS_DIR / "data" / "train_subset_seed123.jsonl"),
    )
    parser.add_argument("--dev-manifest", default=str(DEV_MANIFEST))
    parser.add_argument("--dev-subset-size", type=int, default=500)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="-1 trains for the configured epochs; a positive value caps optimizer steps",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

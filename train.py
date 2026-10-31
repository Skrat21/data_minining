#!/usr/bin/env python3
"""
train.py - Fine-tune Whisper on Arabic ASR data.

Supports all experimental arms:
  A0: Baseline (full training data)
  A1: Kedalai loop (remove flagged utterances)
  A2: Random removal (control)
  A3: CER percentile removal (optional)

Usage:
    python train.py --dataset cv --arm A0 --seed 42
    python train.py --config runs/cv_A1_seed42/config.yaml
    python train.py --dataset cv --arm A1 --removed-ids results/cv_flagged_ids.txt
    python train.py --dataset cv --arm A2 --removed-ids results/cv_random_removed_seed0.txt --seed 0

Changes vs. the original:
  - works together with the FIXED WhisperDataCollator (end-of-text no longer
    masked, no duplicated start token). A label sanity check runs before
    training and stops with a clear error if the labels are malformed.
  - KV cache: the saved model had config.use_cache=False (set for gradient
    checkpointing), which makes every later generate() call extremely slow.
    generation_config.use_cache stays True during training and
    config.use_cache is restored to True before saving.
  - dev-set evaluation uses its own (larger) batch size: cfg.per_device_eval_batch
  - --freeze-encoder now actually freezes the encoder (it was parsed but ignored)
  - gradient checkpointing uses use_reentrant=False (works with frozen encoder)
  - compute_metrics no longer mutates the Trainer's arrays and tolerates -100
    in the generated ids
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import time
from pathlib import Path

# Apply huggingface_hub compatibility shim BEFORE importing transformers/datasets
# to fix BucketNotFoundError import error in newer huggingface_hub versions
import hf_compat  # noqa: F401, flake8: noqa

import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Argument names that changed across transformers versions:
#   TrainingArguments: evaluation_strategy (<=4.40)  ->  eval_strategy (>=4.41)
#   Trainer:           tokenizer           (<=4.45)  ->  processing_class (>=4.46)
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

from config import TrainConfig, set_seed, print_env_info, RUNS_DIR
from data_loader import (
    get_train_dataset,
    get_eval_dataset,
    WhisperDataCollator,
)
from evaluate_model import compute_wer_cer


def compute_metrics_factory(processor: WhisperProcessor, cfg: TrainConfig):
    """Create a compute_metrics function for the Trainer."""
    pad_id = processor.tokenizer.pad_token_id

    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Replace -100 with the pad token id (copies; the Trainer's arrays stay untouched)
        pred_ids = np.where(pred_ids == -100, pad_id, pred_ids)
        label_ids = np.where(pred.label_ids == -100, pad_id, pred.label_ids)

        # Decode predictions and references
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER/CER with normalization
        return compute_wer_cer(pred_str, label_str)

    return compute_metrics


def sanity_check_labels(dataset, collator, processor, n: int = 2):
    """Fail early if the collator produces labels the model cannot learn to stop with."""
    tok = processor.tokenizer
    eot = tok.eos_token_id
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")

    batch = collator([dataset[i] for i in range(min(n, len(dataset)))])
    for row in batch["labels"]:
        ids = [t for t in row.tolist() if t != -100]
        if not ids or ids[-1] != eot:
            raise RuntimeError(
                "Label sanity check FAILED: sequences do not end with <|endoftext|>. "
                "The model would never learn to stop generating. Check data_loader.WhisperDataCollator.")
        if ids[0] == sot:
            raise RuntimeError(
                "Label sanity check FAILED: labels still start with <|startoftranscript|> "
                "(the decoder would see it twice).")
    shown = tok.decode([t for t in batch["labels"][0].tolist() if t != -100], skip_special_tokens=False)
    print(f"Label sanity check OK. Example labels: {shown!r}")


def train(cfg: TrainConfig):
    """Run one fine-tuning experiment."""
    print("=" * 70)
    print(f"Training: {cfg.run_tag}")
    print("=" * 70)
    print_env_info()

    set_seed(cfg.seed)

    # Save config
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(str(output_dir / "config.yaml"))

    # Load model and processor
    print(f"\nLoading model: {cfg.model_name}")
    processor = WhisperProcessor.from_pretrained(
        cfg.model_name,
        language=cfg.language,
        task=cfg.task,
    )
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_name)

    # Enable gradient checkpointing to save memory
    if cfg.gradient_checkpointing:
        model.config.use_cache = False  # incompatible with grad checkpointing (training only)
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:  # older transformers
            model.gradient_checkpointing_enable()

    # Set generation config (generation keeps the KV cache even while config.use_cache is False)
    model.generation_config.language = cfg.language
    model.generation_config.task = cfg.task
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_new_tokens = cfg.max_new_tokens
    model.generation_config.use_cache = True

    # Freeze the encoder to save memory (helps on 4 GB VRAM)
    if cfg.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        print("Encoder frozen to save memory")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = get_train_dataset(cfg, processor)
    eval_dataset = get_eval_dataset(cfg, processor, split="dev")

    # Data collator
    data_collator = WhisperDataCollator(processor)
    sanity_check_labels(train_dataset, data_collator, processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch,
        per_device_eval_batch_size=cfg.per_device_eval_batch,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_epochs,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.gradient_checkpointing else None,
        **{_EVAL_STRATEGY_KW: "steps"},
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=cfg.max_new_tokens,
        report_to="none",  # no wandb/tensorboard
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        dataloader_num_workers=cfg.dataloader_num_workers,  # keep at 0 on Windows
        remove_unused_columns=False,
        label_names=["labels"],
        resume_from_checkpoint=cfg.resume_from_checkpoint,
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(processor, cfg),
        **{_PROCESSOR_KW: processor.feature_extractor},
    )

    # Train
    print("\nStarting training...")
    start_time = time.time()
    result = trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    elapsed = time.time() - start_time

    # Save final metrics
    metrics = result.metrics
    metrics["train_runtime_minutes"] = elapsed / 60
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Restore the KV cache flags BEFORE saving, so evaluation / scoring of this
    # model is not slowed down by use_cache=False.
    trainer.model.config.use_cache = True
    trainer.model.generation_config.use_cache = True

    # Save model (best checkpoint was already loaded by the Trainer)
    trainer.save_model(str(output_dir / "best_model"))
    processor.save_pretrained(str(output_dir / "best_model"))

    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Final train loss: {metrics.get('train_loss', 'N/A')}")
    print(f"Output saved to: {output_dir}")

    # Cleanup to free VRAM
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for Arabic ASR")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--dataset", type=str, choices=["cv", "mgb2"], default="cv")
    parser.add_argument("--arm", type=str, default="A0",
                        choices=["A0", "A1", "A2", "A3", "A4"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="openai/whisper-small")
    parser.add_argument("--removed-ids", type=str, default=None,
                        help="File of IDs to remove from training (for A1/A2)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder to save VRAM")
    args = parser.parse_args()

    if args.config:
        cfg = TrainConfig.from_yaml(args.config)
    else:
        cfg = TrainConfig(
            dataset=args.dataset,
            arm=args.arm,
            seed=args.seed,
            model_name=args.model,
            removed_ids_file=args.removed_ids,
            num_epochs=args.epochs,
            per_device_train_batch=args.batch_size,
            per_device_eval_batch=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            eval_steps=args.eval_steps,
            save_steps=args.eval_steps,
            resume_from_checkpoint=args.resume,
            freeze_encoder=args.freeze_encoder,
        )

    train(cfg)


if __name__ == "__main__":
    main()
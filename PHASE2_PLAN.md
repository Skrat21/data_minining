# Phase 2 Plan: Fine-tuning Whisper and Testing the Kedalai-Style Loop

**Goal.** Fine-tune Whisper on Common Voice Arabic (v27.0) and an MGB-2 subset, apply the model-centric
data-selection loop from Kedalai et al. (2024), and measure whether it lowers WER compared with
(a) training on all data and (b) removing the same amount of data at random.

**Status of numbers in this plan.** Anything about run times or GPU memory is an estimate to be replaced by
measurements from the pilot in Step 3. Nothing here has been run yet.

---

## 0. Decisions and corrections to the assumptions

### 0.1 Where to run what

| Task | Where | Why |
|---|---|---|
| Data preprocessing (already done), analysis, plots, mining of removed utterances | Local laptop | CPU is enough |
| Pilot runs with `whisper-tiny` / `whisper-base` | Local RTX 3050 or Colab | Small enough for a 4-6 GB GPU (check your card's VRAM with `nvidia-smi`) |
| Main fine-tuning and scoring with `whisper-small` | **Colab (or similar)** | ~244M parameters; full fine-tuning plus optimizer states needs more memory than a 3050 laptop GPU is likely to have |

So yes, Colab is the right call for the main runs. Alternatives with free GPU quotas (check current limits
before relying on them): Kaggle notebooks. Free Colab sessions can disconnect and GPU availability varies,
so **every run must be resumable from a checkpoint saved to Drive.**

### 0.2 Model choice

| Model | Use |
|---|---|
| `openai/whisper-base` (74M) | Pilot: debug the whole pipeline end to end |
| `openai/whisper-small` (244M) | **Main model** for all arms |
| `openai/whisper-medium` (769M) | Only if time and compute allow; probably not |

The N-shot paper evaluates the Whisper family on Arabic, so using Whisper keeps us connected to it.

### 0.3 Whisper-specific facts that affect the plan

- Whisper pads every input to **30 s**, so a 3 s clip costs about as much as a 28 s clip. Compute scales
  with the **number of clips**, not hours. Common Voice (many short clips) is relatively expensive per hour.
- We train on `text_norm` (no punctuation, diacritics or hamza variants). Zero-shot Whisper outputs raw-style
  text, so **the same normalizer must be applied to every hypothesis** before computing WER.
- Whisper can hallucinate or loop, producing very long outputs (WER above 100%). Cap `max_new_tokens`
  and use identical decoding settings for all arms.
- Whisper's pretraining data is not public, so overlap with Common Voice or MGB-like data cannot be ruled out.
  State this as a limitation.

### 0.4 Storage reminders

- Free Google Drive storage is shared across the account (commonly 15 GB). Processed audio is about 8 GB,
  so Drive space is tight. Do not store full training checkpoints for every run.
- Common Voice's terms forbid re-sharing: keep Drive folders private.

---

## 1. Package the data for Colab

Many small files on Drive are slow. Pack audio into tar archives and unpack to Colab's local disk each session.

- [ ] Verify manifests exist and counts match: `cv_train/dev/test.jsonl`, `mgb2_train/dev/test.jsonl`.
- [ ] Decide training subsets (see 1.2) and write `train_subset.jsonl` files.
- [ ] Pack audio per dataset: `tar -cf cv_audio.tar -C arabic_data audio/cv` (same for MGB-2).
- [ ] Rewrite `audio_filepath` to be relative to the unpacked folder so manifests work on any machine.
- [ ] Upload tars and manifests to Drive.

```
MyDrive/arabic_asr/
  data/        cv_audio.tar  mgb2_audio.tar  manifests/*.jsonl
  runs/        <dataset>_<arm>_seed<k>/   (metrics, hypotheses, best weights only)
  logs/
```

### 1.2 Sizes to start with

| Dataset | Train | Dev (for model selection) | Test |
|---|---|---|---|
| Common Voice | official train (29,584 clips, ~34 h) | random 500-clip subset of official dev | official test (10,606 clips) |
| MGB-2 | ~50 h subset | 500-segment subset of dev | ~5 h test |

**Warning:** if the MGB-2 upload had only a train split, our dev/test were carved from train and share
programs and speakers with it. Scores on them are optimistic and not comparable with published MGB-2 numbers.
Record which case applies in the report.

**Fallback if compute is short:** cap each train set at about 15-20 h for the first full round.

---

## 2. Colab environment

- [ ] Mount Drive, copy tars to local disk, untar once per session.
- [ ] Install and **pin versions**: `transformers`, `accelerate`, `jiwer`, `soundfile`, `librosa`, `torch`.
- [ ] Log versions and GPU (`nvidia-smi`) at the top of every run.
- [ ] Set seeds (`torch`, `numpy`, `random`) from a single config value.
- [ ] Write a small `config.yaml` per run (dataset, arm, seed, lr, epochs, removed-set file).

---

## 3. Pilot (measure before committing)

- [ ] Fine-tune `whisper-base` for a few hundred steps on ~2 h of Common Voice. Confirm loss decreases and
  the pipeline works end to end (load, train, generate, normalize, score).
- [ ] On `whisper-small`, run 50 training steps and record: seconds per step, peak GPU memory,
  seconds per 100 clips of greedy inference.
- [ ] Fill in this table from measurements:

| Measure | Value |
|---|---|
| Train seconds/step (small, batch 16 effective) | |
| Steps per epoch (CV: ~1,850 at batch 16; MGB-2: compute from clip count) | |
| One full run (3 epochs) in hours | |
| Scoring the whole train set in hours | |
| Test-set decoding in minutes | |

**Gate:** if one full run takes more than your comfortable session budget, reduce the data (Step 1.2 fallback)
or switch the main model to `whisper-base` before continuing.

---

## 4. Evaluation harness (build once, use everywhere)

- [ ] `normalize_arabic` imported from the pipeline script, applied to references and hypotheses.
- [ ] Per-utterance WER and CER with `jiwer`; corpus-level WER computed from totals, not by averaging.
- [ ] Decoding settings fixed for all arms: greedy (`num_beams=1`), language `ar`, task `transcribe`,
  `max_new_tokens` capped.
- [ ] Save **per-utterance** outputs (id, ref, hyp, WER, CER, lengths) to `runs/.../test_hyp.jsonl`.
- [ ] Bootstrap confidence intervals (resample test utterances, ~1,000 times) and a **paired** bootstrap for
  differences between arms.
- [ ] Per-group WER: gender, age, accent on Common Voice; program/speaker on MGB-2 if the field exists.
- [ ] Unit-test the harness on 20 hand-checked examples.

---

## 5. Zero-shot baseline (sanity check)

- [ ] Evaluate pretrained `whisper-small` with no fine-tuning on both test sets.
- [ ] Compare against the style of numbers reported in the N-shot paper. Exact agreement is not expected
  (different normalization and splits), but huge discrepancies suggest a pipeline bug.

---

## 6. Arm A0: baseline, fine-tune on all data

For each dataset:

- [ ] Fine-tune `whisper-small` on the full training subset.
- [ ] Suggested starting settings (tune in the pilot):

| Setting | Value |
|---|---|
| Learning rate | 1e-5, linear decay, 5-10% warmup |
| Effective batch | 16 (per-device batch 8, accumulation 2; lower if memory is short) |
| Precision | fp16 |
| Memory savers | gradient checkpointing; optional 8-bit AdamW |
| Epochs | 3 |
| Dev evaluation | every ~500 steps on the 500-utterance dev subset |
| Checkpointing | save every ~500 steps to Drive; keep best by dev WER; resume on disconnect |

- [ ] Evaluate on the test set, saving per-utterance outputs.
- [ ] Keep the **best weights only** (about 1 GB) for A0; delete optimizer states.

A0 doubles as the **selection model** for the Kedalai loop (the paper fine-tunes a model, then uses it to score
its own training data).

---

## 7. Score the training data

- [ ] Run A0 on its own training set with the same decoding settings.
- [ ] Save per-utterance WER, CER, hypothesis length, and reference length.
- [ ] Plot the WER distribution; report how many utterances have WER >= 1.0, >= 0.8, >= 0.5.
- [ ] Define the Kedalai flag: **WER >= 1.0** (this includes insertions and hallucination loops, which differs
  slightly from "exactly 100%").
- [ ] Sensitivity variants to report: (a) flag only utterances with at least 3 reference words,
  (b) CER-based threshold.
- [ ] **Audit by hand:** listen to a random sample of ~50-100 flagged utterances and record how many are real
  label errors, how many are dialect/hard speech, and how many are Whisper failures. Related work found that
  only about half of algorithmically flagged samples were true errors, so this estimate matters.

**Gate:** if almost nothing is flagged (for example under 0.2% of audio), the loop will not change much.
Report the result, and move the emphasis to the CER-percentile arm and the analysis of flagged data.

---

## 8. Arm A1: Kedalai loop

- [ ] Remove flagged utterances from the training set; save the removed-ID list.
- [ ] Retrain **from the original pretrained weights** (not from A0) with identical hyperparameters.
- [ ] Keep epochs the same as A0. If more than about 2% of data was removed, also report a run with the
  same number of **steps**, so the comparison is not confounded by training length.
- [ ] Evaluate on the test set with the same harness.

## 9. Arm A2: random-removal control

- [ ] Remove the same number of **hours** (and similar clip count) at random, for **3 seeds**.
- [ ] Retrain and evaluate exactly like A0 and A1.
- [ ] This answers: is any gain due to *which* utterances were removed, or just to run-to-run noise?

## 10. Optional arms (if compute remains)

- [ ] **A3: CER percentile removal** at the same removed hours as A1.
- [ ] **A4: metadata heuristics** (duration, speaking rate, Common Voice down-votes).
- [ ] **Cross-validated scoring:** 2-fold, each half scored by a model trained on the other half.

---

## 11. Compute budget (fill in after the pilot)

Minimum set of fine-tuning runs per dataset:

| Run | Count |
|---|---|
| A0 baseline (also the selector) | 1 (2 if a second seed fits) |
| A1 Kedalai | 1 (2 with a second seed) |
| A2 random removal | 3 |
| **Total per dataset** | **5 (up to 7)** |
| **Total for two datasets** | **10 (up to 14)** |

Total time = runs x (one-run hours from Step 3) + scoring + test decoding. If this does not fit,
cut in this order: second seeds, dataset size, then model size.

---

## 12. Statistics and reporting

- [ ] Table per dataset: arm, mean WER and CER over seeds with standard deviation, bootstrap 95% CI,
  relative change vs A0.
- [ ] Report seed-to-seed variation next to the A1-vs-A0 difference. A gain smaller than the variation is
  not evidence of improvement.
- [ ] Per-group WER before/after (gender, accent, age; program for MGB-2).
- [ ] Removal rate per group compared with that group's share of the training data.
- [ ] State outcomes honestly: a flat or negative result is still a valid finding. Related work shows that removing
  even truly mislabeled utterances can hurt when it shrinks the training data.

---

## 13. Data mining analysis of removed utterances

Use the flagged set from Step 7 and the metadata already in the manifests.

- [ ] Descriptive comparison: duration, words, characters per second, hypothesis/reference length ratio,
  gender, age, accent, votes, program.
- [ ] **Association rules** over discretized features (for example "short + fast speech" -> flagged).
- [ ] **Decision tree or logistic regression** predicting the flag from cheap features; report accuracy and
  which features matter.
- [ ] **Clustering** utterance embeddings (Whisper encoder or similar) and check flag rates per cluster.
- [ ] **Anomaly detection** (for example isolation forest) on the same features, compared with the
  model-based flags.
- [ ] Text analysis of flagged references: numbers, Latin letters, rare words, very short transcripts.

---

## 14. Deliverables

- [ ] Notebook(s): data setup, fine-tuning, scoring, evaluation.
- [ ] `results/` folder: per-arm metrics, per-utterance hypotheses, removed-ID lists, plots.
- [ ] Short Phase 2 report: setup, results tables with confidence intervals, audit findings,
  mining analysis, limitations.

---

## 15. Risks to keep in mind

| Risk | Mitigation |
|---|---|
| Few utterances flagged, so no measurable effect | Report it; use CER percentile and analysis arms |
| Gains within seed noise | Multiple seeds, paired bootstrap, report variation |
| Flags are mostly hard or dialectal speech, not label errors | Manual audit in Step 7; per-group analysis |
| Whisper hallucination inflates WER | Cap generation length; identical decoding; compare with CER |
| MGB-2 dev/test carved from train | Report it as optimistic; prefer official splits if available |
| Colab disconnects | Checkpoint every ~500 steps; resumable scripts |
| Drive space | Keep best weights only for A0/A1; metrics only for random seeds |
| Possible Whisper pretraining overlap | State as a limitation |

# Latest WER and training-speed work

## What changed

The Common Voice runs spent about 74% of their wall time in repeated
generation-based evaluations (500 examples, up to 225 tokens). The MGB-2
pipeline in `train_v2.py` therefore disables generation during in-training
evaluation, evaluates dev loss every 400 optimizer steps, saves at the same
interval, and saves a final model. This targets evaluation overhead; it does
not claim to make an individual optimizer step faster.

The inference guard uses a duration-based generation cap:
`ceil(9.4464 * audio_seconds + 16)`, clamped to 32–225 tokens. The MGB-2
A0 guard settings below were compared on the fixed 500-clip DEV subset
(seed 123); no MGB-2 test data was used for guard selection.

## Common Voice test results

The full Common Voice test set contains 10,606 clips. Lower error is better.
The cap configuration was selected using DEV before test decoding.

| Model | Decode | WER (95% CI) | CER (95% CI) |
|---|---|---:|---:|
| A0 | Raw | 28.85% (28.26–29.46) | 9.94% (9.51–10.46) |
| A0 | Duration cap | 28.71% (28.17–29.29) | 9.64% (9.41–9.90) |
| A1 | Raw | 29.40% (28.68–30.15) | 10.29% (9.77–10.93) |
| A1 | Duration cap | 29.16% (28.58–29.72) | 9.80% (9.53–10.07) |

Paired raw-minus-cap changes were +0.14 percentage points (pp) WER and
+0.30 pp CER for A0; for A1 they were +0.25 pp WER and +0.49 pp CER. The
paired 95% intervals were A0 WER 0.00–0.34 pp / CER 0.00–0.68 pp, and A1
WER 0.00–0.66 pp / CER 0.13–0.95 pp. Positive values mean the cap reduced
error. These are modest gains: the WER intervals include zero at the
reported precision, and this guard did not eliminate looping outputs.

On the Common Voice DEV sweep, n-gram sizes 4 and 5 increased WER by 0.18
and 0.15 pp; size 6 and loop trimming tied raw WER. On TEST, the loop
detector counted two looped outputs per model in both decoding modes;
cap-hit counts were three for A0 and six for A1. Do not interpret the
duration cap as a general loop fix.

## Training throughput and current MGB-2 run

A prior 100-step MGB-2 A0 speed check took 415.4 seconds of training:
4.154 seconds per optimizer step, about 3.85 clip-presentations per second
(microbatch 2 × accumulation 8). The current plan uses a fixed 8,000-clip,
seed-123 subset (21.0682 audio hours), two epochs, and frozen encoder.
This is about 1,000 A0 steps and 900 steps each for A1/A2. At the measured
step time, the three training runs were initially projected at about 3.23
hours of optimization, before evaluation, saving, startup, and checkpoint
selection.

The approved A0 run has now completed all 1,000 steps. Its training wall time
was 4,300.6 seconds and total run wall time was 4,307.4 seconds (71.8
minutes), or 4.30 seconds per step and 3.72 clip-presentations per second
averaged across the run. The two 500-example loss-only DEV evaluations took
60.9 and 55.5 seconds (1.94 minutes total); the recorded losses were 0.4056
at step 400 and 0.3892 at step 800. Each scheduled interval also saved a
checkpoint. The current checkpoints are 2.045 GiB each; the final model is
0.902 GiB.

Using this full-run wall time per step, 1,000 A0 steps plus 900 steps each
for A1/A2 project to about 3.35 hours for the three training runs, including
similar scheduled-evaluation/save overhead. This is an estimate, not a
measured total; checkpoint selection and guard evaluation are additional.
The estimated training total is below the ~4-hour target, but A1/A2 throughput
has not yet been measured.

The approved DEV-only checkpoint comparison is complete (500 clips, seed
123, raw greedy WER selection):

| A0 checkpoint | DEV WER | DEV CER | Decode time |
|---|---:|---:|---:|
| Step 400 | 35.99% | 15.36% | 87.9 s |
| Step 800 | 32.15% | 12.64% | 84.8 s |

Step 800 was approved as the A0 checkpoint. After approval, only the
`optimizer.pt` files in A0's step-400 and step-800 checkpoints were removed
(2,457,781,364 bytes total); free space changed from 21.42 to 23.71 GiB.
The checkpoint weights and final model remain. This is not a test-set result.

## MGB-2 A0 DEV guard sweep

The five approved settings were run sequentially on the same 500 DEV clips.
Raw WER/CER was 32.15%/12.64% for every setting. Clips/s is 500 divided by
the corresponding raw or guarded decode time. Command wall time is the
external measured duration, including process startup.

| Setting | Guarded WER/CER | Cap hits raw/guarded | Loop flags raw/guarded | Guarded outputs changed vs raw | Decode seconds (clips/s), raw/guarded | Command wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| Duration cap only | 30.11% / 11.42% | 2 / 2 | 4 / 4 | 3 | 89.07 (5.61) / 90.72 (5.51) | 186.847 |
| Cap + no-repeat 4-gram | 30.49% / 11.34% | 2 / 0 | 4 / 0 | 80 | 92.60 (5.40) / 95.97 (5.21) | 194.390 |
| Cap + no-repeat 5-gram | 29.96% / 11.16% | 2 / 0 | 4 / 0 | 49 | 93.96 (5.32) / 95.60 (5.23) | 195.162 |
| Cap + no-repeat 6-gram | 29.57% / 11.06% | 2 / 0 | 4 / 0 | 26 | 94.80 (5.27) / 98.70 (5.07) | 199.314 |
| Cap + loop trim | 29.70% / 11.41% | 2 / 2 | 4 / 4 | 5 | 96.35 (5.19) / 94.57 (5.29) | 196.584 |

**Mechanical selection under the pre-agreed DEV rule: cap + loop trim.**
No-repeat 6-gram had the lowest WER. Loop trim was 0.13 pp higher, within
the 0.3 pp simplicity band, and was the simpler option. It also beat
cap-only WER by 0.41 pp, so it does not meet the rule for rejection. The
4-gram option is rejected because its WER is 0.38 pp higher than cap-only.
This is a DEV-only recommendation, not a test-set conclusion.

Four looped outputs were marked as trimmed. The loop-flag metric is computed
on the generated text before post-hoc trimming, so the loop-trim setting
reports four loop flags both before and after the guard. All five outputs
changed by loop trim are identified below by utterance ID and behavior rather
than reproducing dataset transcripts. The full raw and guarded rows remain in
the local JSONL outputs. One example demonstrates that the heuristic can
remove a legitimate repeated phrase, despite its lower aggregate DEV WER.

- `mgb2_validation_0001360`: a hallucinated word loop was cut down to two
  repetitions.
- `mgb2_validation_0002271`: a long repeated number phrase was cut down to
  two repetitions.
- `mgb2_validation_0002643`: the reference has a phrase repeated three times;
  the guard retained only two repetitions (a false positive).
- `mgb2_validation_0002874`: a short repeated word sequence was trimmed before
  the rest of the hypothesis.
- `mgb2_validation_0002400`: no loop was detected; the cap changed one token
  while otherwise leaving the output intact.

## Interpretation

- The measured WER reduction is from inference-time output-length control,
  not from a new trained model. Results are small and should be treated
  cautiously.
- The new trainer removes generated-text evaluation from the training loop.
  The two loss-only evaluations took under two minutes combined, compared
  with the previous 51–99 minutes per generation-based boundary. This
  demonstrates much lower evaluation overhead, though it is not a controlled
  same-settings benchmark.
- Reducing the subset from 10,000 to 8,000 clips lowers the planned work, but
  does not improve per-step throughput.
- MGB-2 A1 uses the worst 10% of the subset by A0 per-utterance WER. This is
  different from Common Voice A1, which used WER >= 1.0.
- No A1/A2 training, training-data scoring, or MGB-2 test decoding has
  started.

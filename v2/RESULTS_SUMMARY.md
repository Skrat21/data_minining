# Latest WER and training-speed work

## What changed

The Common Voice runs spent about 74% of their wall time in repeated
generation-based evaluations (500 examples, up to 225 tokens). The MGB-2
pipeline in `train_v2.py` therefore disables generation during in-training
evaluation, evaluates dev loss every 400 optimizer steps, saves at the same
interval, and saves a final model. This targets evaluation overhead; it does
not claim to make an individual optimizer step faster.

The inference guard uses a duration-based generation cap:
`ceil(9.4464 * audio_seconds + 16)`, clamped to 32–225 tokens. The cap was
chosen on DEV only. N-gram blocking and loop trimming were also evaluated;
they were not selected.

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

On the DEV sweep, n-gram sizes 4 and 5 increased WER by 0.18 and 0.15 pp;
size 6 and loop trimming tied raw WER. On TEST, the loop detector counted two
looped outputs per model in both decoding modes; cap-hit counts were three
for A0 and six for A1. Do not interpret the duration cap as a general loop
fix.

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
The checkpoint weights and final model remain. This is not a test-set result;
neither guard tuning nor MGB-2 test decoding has been run.

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
- No A1/A2 training or MGB-2 test decoding has started.

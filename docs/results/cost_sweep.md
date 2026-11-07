# Test-cost sweep results

Last updated: November 3, 2026

## Matched test results: zero-cost versus original trained model

Zero-cost evaluation job `49166548` finished on November 2 at 20:35 EST. The
user supplied `outputs/cost_sweep/cost_0/eval_49166548/metrics.json` and the
finish log. It evaluated one rollout on each of the 240 fixed test cases.

| Metric | Original trained (1.0x) | Zero-cost (0x) |
|---|---:|---:|
| Overall diagnosis accuracy | 68.75% (165/240) | 77.50% (186/240) |
| Diagnosed-only accuracy | 70.21% | 81.22% |
| Undiagnosed fraction | 2.08% (5/240) | 4.58% (11/240) |
| Action-format reward | 0.9792 | 0.9542 |
| Macro F1, all predictions | 66.27% | 74.50% |
| Hypothesis accuracy | 63.26% | 72.97% |
| Expected calibration error | 10.38% | 8.77% |
| Average tests requested | 2.73 | 2.29 |
| Average diagnostic cost | $1,137.67 | $1,627.71 |

Removing the training test-cost penalty improved overall test accuracy by
8.75 percentage points, or 21 additional correct cases, in this single run.
It also raised mean dollar cost by $490.04 (about 43%), despite reducing the
mean number of tests by about 0.44 per case. This suggests a shift toward more
expensive tests; the aggregate metrics alone do not establish which tests
changed. The undiagnosed fraction rose by six cases. The evaluation log
included repeated unknown `Complete Blood Count (CBC)` requests and invalid
action-format warnings. These are observed results from one training seed,
not evidence that the gain is stable across repeats.

## Zero-cost arm: training and validation complete

Source: training log excerpt supplied on November 2, 2026, for Slurm job
`48885778`. The run directory is
`/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK/outputs/cost_sweep/cost_0/train_48885778`.
The user verified the final `adapter_config.json` (903 bytes) and
`adapter_model.safetensors` (81 MB), both dated November 2 at 07:47, on
HiPerGator. The saved `.hydra/overrides.yaml` confirms
`reward_function.reward_functions.2.multiplier=0`, `dataset.small_sample=false`,
`training.max_steps=300`, `training.num_generations=4`, batch size 1,
128-token completions, a 2,048-token trajectory limit, checkpoint interval 25,
and vLLM GPU memory utilization 0.9. The full log and metrics file have not
been independently checked in this checkout. This is a **validation** result, not a
240-case test result. The final validation used four rollouts for each of 240
cases (960 predictions). `eval_rewards/TestCostReward` was 0.0, consistent with
the verified zero-cost override.

| Metric | Original trained validation | Zero-cost validation |
|---|---:|---:|
| Overall diagnosis accuracy | 56.46% | 56.88% |
| Diagnosed-only accuracy | 67.50% | 77.78% |
| Undiagnosed fraction | 16.35% | 26.88% |
| Action-format reward | 0.8365 | 0.73125 |
| Macro F1, all predictions | 57.59% | 60.02% |
| Hypothesis accuracy | 63.06% | 69.17% |
| Expected calibration error | 7.89% | 6.15% |

Other zero-cost validation figures: mean reward 1.3, confidence-calibration
reward 0.8366, train loss 0.0015075, training runtime 39,804.7 seconds, and
validation runtime 9,020.0 seconds. The log ended with `[SLURM] Finished: Mon
Nov 2 07:47:16 AM EST 2026`.

The overall accuracy was similar, while diagnosed-only accuracy rose and the
undiagnosed fraction also rose. These are different subsets of predictions;
diagnosed-only accuracy alone does not show an overall improvement. The log
included unknown-action and invalid-format warnings. Mean test count and mean
dollar cost were not present in the pasted validation output.

## 1.5x arm: validation failure

Training job `49166794` finished on November 2 at 22:53 EST and saved a
903-byte `adapter_config.json` and 81 MB `adapter_model.safetensors` under
`outputs/cost_sweep/cost_1p5/train_49166794/`. Its saved overrides confirm
the 1.5 multiplier and the planned 300-step, four-generation settings. The
final four-rollout validation reported `eval_all/accuracy=0`,
`eval_none_fraction=1.0`, and `eval_rewards/FormatReward=0`: none of the 960
rollouts produced a valid diagnosis. Mean test-cost reward was -0.0199 and
training loss was 0.02613. The log showed repeated invalid-format and unknown
`Complete Blood Count (CBC)` actions. A clean Slurm finish and saved adapter
show that training completed; they do not imply a useful model.

The downloaded full log at `ignore/logs/lacdm_cost_train_49166794.out` contains
300 per-step training records. `scripts/summarize_training_rewards.py` counted
steps with a positive batch-mean format or diagnosis reward:

| Training phase | Format-positive steps | Diagnosis-positive steps | Mean test-cost reward |
|---|---:|---:|---:|
| Decision Agent, steps 1–100 | 1/100 | 1/100 | -0.0553 |
| Hypothesis SFT, steps 101–200 | 0/100 | 0/100 | -0.0259 |
| Confidence calibration, steps 201–300 | 0/100 | 0/100 | -0.0213 |

The failure was present from the start of Decision Agent training, before the
HA phases. The first logged batch had zero diagnosis and format reward and a
negative cost reward. With GRPO advantages computed from the sum of these
rewards, most early learning signal could come from relative test costs rather
than valid diagnoses. This is a plausible explanation for collapse, not proof
that the 1.5 multiplier alone caused it. Initial zero-shot Llama action-format
failures and the small four-rollout groups may also contribute. The log does
not support blaming the HA phase transition.

## Next action

Do not launch 2x based on this run. First compare early DA reward traces with
the original or zero-cost training logs if available. A smaller penalty or a
cost penalty introduced after a valid-action warm-up is a more focused next
experiment. Preserve the fixed test set for a candidate selected from
validation. For the zero-cost model, inspect test choices and costs by category
to explain why fewer tests cost more.

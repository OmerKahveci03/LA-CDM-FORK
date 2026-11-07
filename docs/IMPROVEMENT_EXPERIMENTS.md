# LA-CDM improvement experiments

Last updated: November 3, 2026

## Research goal and current reference

Recreate and understand the paper's LA-CDM method, then test a change that improves the trained Llama system. The completed run used Meta-Llama-3-8B-Instruct and one 300-step cycle: 100 Decision Agent GRPO steps, 100 Hypothesis Agent SFT steps, and 100 confidence-calibration GRPO steps. It is a reproduction of the method with a different base model and shorter training, not a numerical reproduction of the paper's Qwen result.

The matched 240-case test evaluation used one rollout per case. The trained model achieved 68.75% overall diagnosis accuracy (165/240), with 2.08% undiagnosed (5/240), 2.73 tests per case, and $1,137.67 mean diagnostic cost. The untrained Llama baseline achieved 8.75% accuracy (21/240), with 87.92% undiagnosed, 1.16 tests per case, and $798.08 mean cost. These figures are recorded in `weekly_reports/week_5/WEEKLY_REPORT.tex`. The trained adapter is the reference for proposed improvements; the untrained model is not the control for comparing new training variants.

## First experiment: tune the test-cost penalty

**Hypothesis:** The current Decision Agent can retain most of its accuracy while using fewer or cheaper tests if training penalizes test cost more strongly. The default reward adds diagnosis reward (1 for correct, 0 otherwise), action-format reward (1 for valid, 0 otherwise), and per-test negative rewards. The per-test coefficients live in `configs/reward_function/defaults.yaml`. The `no_test_cost` configuration removes that reward entirely.

Run separate training arms from the same base model and data split, changing only the test-cost reward scale:

| Arm | Test-cost multiplier | Purpose |
|---|---:|---|
| Existing trained reference | 1.0 | Preserve the completed result; do not retrain unless a matched rerun is needed. |
| No-cost ablation | 0.0 | Measure the effect of removing the paper's cost term. |
| Moderate penalty | 1.5 | Test whether a small increase reduces wasteful testing. |
| Strong penalty | 2.0 | Probe the accuracy-cost tradeoff. |

These are proposed settings, not results. `TestCostReward.multiplier` multiplies the sum of per-test penalties; diagnosis and format rewards stay unchanged. The 0 arm keeps the same reward-function list but assigns zero test penalty. Do not change the dollar costs in `configs/evaluation/defaults.yaml`: those are measurement values, not training penalties. If the first stronger arm loses substantial accuracy or increases failures, review that result before spending compute on the next arm. A later, narrower sweep can explore values between the best-performing settings.

For every arm, use the same model, seed, train/validation/test splits, 300-step phase schedule, four training generations, memory settings, and one-rollout test evaluation as the reference run. Start each new arm at step 0. Use `slurm/train_cost_sweep.sbatch`, which defaults to a fresh run and stores each arm under `outputs/cost_sweep/cost_<multiplier>/train_<JOB_ID>/`. The older `slurm/train_full.sbatch` automatically resumes the latest complete `outputs/train_full_*/checkpoint-*` and is unsuitable for a fresh sweep arm. The sweep evaluation job requires an explicit adapter path.

On HiPerGator, after syncing these changes and checking that `data/` and the model environment are available, submit training arms deliberately and record each job ID. An evaluation of one arm and training of another use separate output directories and can run concurrently if the Slurm allocation permits; each job requests two L4 GPUs and 64 GB RAM. Start with the 1.5x arm before deciding whether the 2x arm is worth the compute:

```bash
sbatch --export=ALL,COST_MULTIPLIER=0 slurm/train_cost_sweep.sbatch
sbatch --export=ALL,COST_MULTIPLIER=1.5 slurm/train_cost_sweep.sbatch
sbatch --export=ALL,COST_MULTIPLIER=2 slurm/train_cost_sweep.sbatch
```

Wait for each train job's successful finish line and verify its `adapter_config.json` and `adapter_model.safetensors` (or `.bin`). Then evaluate that exact adapter; substitute the actual job ID:

```bash
sbatch --export=ALL,COST_MULTIPLIER=1.5,ADAPTER_DIR=outputs/cost_sweep/cost_1p5/train_<JOB_ID> slurm/evaluate_cost_sweep.sbatch
```

Evaluation writes `outputs/cost_sweep/cost_1p5/eval_<JOB_ID>/metrics.json`. Use `COST_MULTIPLIER=0` with `cost_0`, or `COST_MULTIPLIER=2` with `cost_2`, for the other arms. To resume an interrupted arm, pass `RESUME_FROM_CHECKPOINT=outputs/cost_sweep/cost_1p5/train_<OLD_JOB_ID>/checkpoint-<STEP>` together with the same multiplier. The job validates that the checkpoint is inside that arm and has trainer state and optimizer files. A new job writes to its own output directory.

Compare **overall diagnosis accuracy**, undiagnosed fraction, action-format reward, macro F1, mean tests, and mean dollar cost on the same 240-case test split. Also report accuracy among diagnosed cases, hypothesis accuracy, and calibration error to detect regressions. Prefer an arm that reduces cost while preserving accuracy and completion; present the full accuracy-versus-cost tradeoff rather than declaring a winner from a single metric. Use validation to choose any setting and reserve the fixed test set for the final comparison; repeated test-set selection would overfit the reported result.

## Other improvement ideas

1. **Adaptive stopping:** Train or guide the Decision Agent to diagnose when enough evidence has accumulated. First check whether hypothesis confidence is calibrated and whether high-confidence extra tests actually change outcomes. A confidence threshold is an experiment, not an assumed safe decision rule.
2. **Reward informative tests:** Reward a test when it changes the diagnosis or reduces uncertainty. This needs a carefully defined signal; otherwise the model may learn to generate superficial changes in stated confidence.
3. **Strengthen hypothesis training:** Explore additional HA supervision or a revised confidence-calibration reward if error analysis shows the HA is limiting final decisions. The trained test hypothesis accuracy (63.26%) was slightly below the untrained result (65.25%), so this is a concrete concern, but the cause is not established.
4. **Continue training:** Evaluate later checkpoints or more than one 300-step cycle to test whether training duration limits performance. Keep it separate from reward changes so the effect of each can be interpreted.

The first decision is whether the cost sweep yields a useful accuracy-cost tradeoff. Use its errors and test choices to select one subsequent method change. The existing decision-agent-only experiment can help isolate the HA contribution if that becomes the focus.

## Experiment record

For each run, record: hypothesis; code commit; config and reward multiplier; data split and seed; training and evaluation job IDs; adapter path; whether training started fresh or resumed within the same arm; validation metrics; test metrics; and representative failures. Mark planned runs as planned until logs and metrics have been checked. Results and provenance are in `docs/results/cost_sweep.md`: the zero-cost arm completed a matched test evaluation; the 1.5x arm completed training but had zero valid diagnoses in final validation. Investigate that failure before spending compute on 2x.

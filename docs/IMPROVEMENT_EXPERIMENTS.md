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

For every arm, use the same model, seed, train/validation/test splits, 300-step phase schedule, four training generations, memory settings, and one-rollout test evaluation as the reference run. Use `slurm/train_cost_sweep.sbatch` for training and `slurm/evaluate_cost_sweep.sbatch` for the test evaluation. Since November 7, 2026, each arm has one fixed directory, and no job IDs need to be recorded or passed:

```text
outputs/cost_sweep/cost_<m>/        # e.g. cost_1, cost_1p5
  jobs.log            # one line per job start, resume, finish, exit code and commit
  train_output.log    # console output of every training job, appended
  run_settings.txt    # settings fixed for this arm; a mismatched resubmit is refused
  train/              # checkpoints, final adapter, training_complete.json
    adapters/step_0300/              # adapter kept at every cycle end
    episodes/train_episodes.csv      # one row per training rollout
    episodes/val_step_0100_*.csv/.json/.jsonl   # validation episodes and summary
  test_step_0300/     # metrics.json, test_episodes.csv, test_summary.json, transcripts
```

Submit training with the arm's multiplier. After a time-out or failure, submit **the same command** again: the job resumes from the newest complete checkpoint in `train/`, removes replayed training-episode rows, and exits immediately once the arm has reached `MAX_STEPS`:

```bash
sbatch --export=ALL,COST_MULTIPLIER=1 slurm/train_cost_sweep.sbatch
```

After `train/training_complete.json` exists, evaluate the final adapter on the test split:

```bash
sbatch --export=ALL,COST_MULTIPLIER=1 slurm/evaluate_cost_sweep.sbatch
```

### Planned run: full-pass 1x baseline (decided November 7, 2026)

Train the 1x arm (the paper's unscaled test-cost penalty) for **1,920 steps**, one pass over every training patient. Loss phases switch every 100 steps; validation (all 240 cases, four sampled rollouts each, stratified by disease) runs before training and every 100 steps, plus a final pass at step 1,920. Adapters are kept at every cycle end (300, 600, …, 1,800) and at 1,920, so the step-300 adapter doubles as the matched comparison to the 0x and 1.5x arms (the learning rate is constant, so the first 300 steps do not depend on the total length). The last 120 steps are 100 DA steps and 20 HA-SFT steps of an incomplete seventh cycle; the step-1,800 adapter is the last complete cycle.

```bash
sbatch --export=ALL,COST_MULTIPLIER=1,MAX_STEPS=1920 slurm/train_cost_sweep.sbatch   # resubmit unchanged until complete
sbatch --export=ALL,COST_MULTIPLIER=1,ADAPTER_STEP=300 slurm/evaluate_cost_sweep.sbatch   # can run once adapters/step_0300 exists
sbatch --export=ALL,COST_MULTIPLIER=1 slurm/evaluate_cost_sweep.sbatch   # final (step 1,920) adapter, after training
```

Expected cost: roughly 25-35 h of training plus about 20 validation passes of about 2.5 h each, so 4-5 jobs of 24 h.

Optional variables: `MAX_STEPS` (default 300; raise it by 300 at a time to continue a finished arm in the same directory), `EVAL_ON_START=0` (skip the step-0 validation of the untrained model), `ADAPTER_STEP` (evaluate an earlier cycle-end snapshot), and `OVERWRITE=1` (replace an existing test result). Summarize any episode file with `python scripts/summarize_episodes.py <csv>`; it prints overall and per-disease accuracy, end reasons, a confusion matrix, test-ordering rates, and for training files the share of steps whose advantages came only from test cost. The completed 0x and 1.5x arms predate this layout and remain in their `train_<JOB_ID>/` and `eval_<JOB_ID>/` directories.

Compare **overall diagnosis accuracy**, undiagnosed fraction, action-format reward, macro F1, mean tests, and mean dollar cost on the same 240-case test split. Also report accuracy among diagnosed cases, hypothesis accuracy, and calibration error to detect regressions. Prefer an arm that reduces cost while preserving accuracy and completion; present the full accuracy-versus-cost tradeoff rather than declaring a winner from a single metric. Use validation to choose any setting and reserve the fixed test set for the final comparison; repeated test-set selection would overfit the reported result.

## Other improvement ideas

1. **Adaptive stopping:** Train or guide the Decision Agent to diagnose when enough evidence has accumulated. First check whether hypothesis confidence is calibrated and whether high-confidence extra tests actually change outcomes. A confidence threshold is an experiment, not an assumed safe decision rule.
2. **Reward informative tests:** Reward a test when it changes the diagnosis or reduces uncertainty. This needs a carefully defined signal; otherwise the model may learn to generate superficial changes in stated confidence.
3. **Strengthen hypothesis training:** Explore additional HA supervision or a revised confidence-calibration reward if error analysis shows the HA is limiting final decisions. The trained test hypothesis accuracy (63.26%) was slightly below the untrained result (65.25%), so this is a concrete concern, but the cause is not established.
4. **Continue training:** Evaluate later checkpoints or more than one 300-step cycle to test whether training duration limits performance. Keep it separate from reward changes so the effect of each can be interpreted.

The first decision is whether the cost sweep yields a useful accuracy-cost tradeoff. Use its errors and test choices to select one subsequent method change. The existing decision-agent-only experiment can help isolate the HA contribution if that becomes the focus.

## Experiment record

For each run, record: hypothesis; code commit; config and reward multiplier; data split and seed; training and evaluation job IDs; adapter path; whether training started fresh or resumed within the same arm; validation metrics; test metrics; and representative failures. Mark planned runs as planned until logs and metrics have been checked. Results and provenance are in `docs/results/cost_sweep.md`: the zero-cost arm completed a matched test evaluation; the 1.5x arm completed training but had zero valid diagnoses in final validation. Investigate that failure before spending compute on 2x.

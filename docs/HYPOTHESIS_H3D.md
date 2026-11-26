# Hypothesis H3d: halving the test-cost weight

**Written November 26, 2026, before training.** Fixed before any H3d result exists.

## Why

Every DA trained long enough drifted to ordering (almost) no tests: the H4c frozen arm (step 1100:
0.00 tests, 78.8% on test), the H4c joint rerun (from step 1200: ≤ 0.17 tests), and H3c (step 700: 0.33
tests). The reward explains it:
- a correct diagnosis earns +1.0, while a CT costs −0.128, an ultrasound −0.126 and an MRI −0.48
  (`configs/reward_function/defaults.yaml`);
- the separately fine-tuned HA is already about 78% right from the history alone, and tests raise it
  to about 88%, so +10 points of accuracy cost about one CT;
- under this reward, "no tests" scores at least as well. For example, frozen step 1100 (78.8%, $0)
  beats H3a (85.8%, $1,025) in reward.

A better HA made tests less valuable to a cost-sensitive DA. **Halving the cost weight** (a CT is then
worth about 6 accuracy points instead of 13) should make informative tests pay off again, trading cost
for accuracy.

"Require at least one test" was considered and rejected: a urinalysis (−0.005) or a blood count
(−0.007) is nearly free, so the DA could satisfy the rule with an uninformative test (the H3b failure mode).

## Design

- **One change from the base design:** the test-cost multiplier is **0.5** instead of 1.0
  (`reward_function.reward_functions.2.multiplier=0.5`). Everything else is identical: frozen
  separately fine-tuned HA, new DA, 700 steps, 4 rollouts, validation every 100 steps, resume fix.
- **Base design** (`DESIGN`), decided by H3c's test result before H3d starts:
  - `defer` (H3c: the HA's hypothesis is always the diagnosis) if H3c's test accuracy is at least
    H3a + defer's (88.3%);
  - otherwise `plain` (H3a: the DA makes its own diagnosis).
- **Outputs:** `outputs/h3/h3d_<design>_ha0/`.

## Checkpoint choice (validation only)

As H3c: the highest validation accuracy among steps 100–700, excluding steps with < 0.5 tests per
patient; ties go to the lower cost.

## Comparisons (240 MIMIC test patients, paired; `jobs/h3d_stats.sh`)

- **Primary:** H3d vs the same design at cost weight 1 (H3c for `defer`; H3a for `plain`), on accuracy.
  **Supported** ("a lower cost weight buys accuracy") if the 95% CI is entirely above 0. The cost
  difference is reported alongside (expected to rise).
- **Secondary:**
  - H3d vs the baseline (better if the CI is above 0);
  - H3d vs H3a + defer;
  - tests per patient, CT rate, cost.
- **Caveat:** a single run, and a different point on the accuracy–cost trade-off, not a free gain.
  Reward-based comparisons with weight-1 runs are not meaningful; compare accuracy and cost separately.

## Run

```bash
sbatch --export=ALL,DESIGN=<defer|plain>,SMOKE=1 slurm/train_h3d.sbatch     # 4 steps
sbatch --export=ALL,DESIGN=<defer|plain> slurm/train_h3d.sbatch             # 700 steps
```

**Test after choosing the step:**
```bash
# DESIGN=defer
sbatch --export=ALL,RUN_ROOT=outputs/h3/h3d_defer_ha0,ADAPTER_STEP=<step> slurm/evaluate_defer_ha.sbatch
# DESIGN=plain
sbatch --export=ALL,RUN_ROOT=outputs/h3/h3d_plain_ha0,ADAPTER_STEP=<step>,COST_MULTIPLIER=0.5 slurm/evaluate_frozen_ha.sbatch
# then
sbatch --export=ALL,SCRIPT=jobs/h3d_stats.sh,DESIGN=<design>,STEP=<step> slurm/cpu_script.sbatch
```

## Results

(to be filled in)

# Step 2: the trained Decision Agent with the frozen standalone HA

Written November 15, 2026, **before any step-2 result exists**. Sections 1–4 are fixed;
results go in section 6.

## 1. What is measured

The attempt-2 Decision Agent (DA, step 600, our reported best model) runs the normal test
evaluation on the 240 test patients. The only change: the hypotheses and confidences it
reads come from the frozen H2 HA (`outputs/ha_sft/seed_0/adapter`), not from the shared
LA-CDM adapter. No training happens.

This is evaluation only. It gives the "combined system" number, and it checks that the
frozen HA works inside the environment before H3 trains a DA against it.

## 2. How it works (`src/environment/frozen_ha_environment.py`, `src/ha/frozen.py`)

- The DA's adapter is loaded as before and drives vLLM, so its turns are unchanged.
- The frozen HA is loaded as a second LoRA adapter on the same model. At every HA turn it
  scores the four diseases on the same prompt as before.
  - The hypothesis is the most probable disease.
  - The confidence is round(10 × its probability).
- The text shown to the DA has the same format as before: "The current hypothesis is:
  pancreatitis (confidence: 7)". `tests/test_frozen_ha.py` checks this against the original
  environment.
- Every probability vector is kept for H3's reward.
- `src/evaluate.py`, `slurm/evaluate_cost_sweep.sbatch` and the original environment are
  unchanged.

## 3. Comparison and predictions

Paired on the same 240 patients against the original attempt-2 test run
(`cost_1_attempt2/test_step_0600`):

| Measure | Original (attempt 2) | Prediction for the combined system |
|---|---|---|
| Accuracy, DA's diagnosis (reported metric) | 85.4% | 87–90% |
| Accuracy, final HA hypothesis | 87.1% | 88–90% |
| Tests per case | 1.19 | ≥ 1.19 |
| Cost per case | $1,259 | about the same or higher |

**Reasons for the predictions:**
- H2 found the new HA no better than old2 on the DA's own records (+0.8 points), so a large
  accuracy gain is not expected.
- The diagnosis turn may gain more. In H1, the DA's diagnosis lost some pancreatitis cases
  that the old HA had right.
- The old HA was overconfident on short records (ECE about 18%), while the new HA's
  confidence is lower when the record is thin. A DA trained to read confidence may therefore
  order more tests, which would raise the cost.

**How the result is read** (no gate: H3 uses the frozen HA either way):
- If the 95% CI of the paired accuracy difference is above 0, the combined system is better.
- If the CI includes 0, there is no detectable change.
- If the CI is below 0, the DA depends on the old HA's way of answering, which matters for H3.

**Secondary measures:**
- cost and tests per case (paired);
- undiagnosed rate;
- pancreatitis recall;
- ECE / AUROC_correct;
- the same run with the seed-1 HA.

## 4. How to run (HiPerGator)

```bash
cd /blue/prismap-ai-core/omerkahveci/LA-CDM-FORK && git pull
PY=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
$PY -m unittest tests.test_frozen_ha tests.test_ha_sft        # runs the PEFT test too (needs no GPU)

sbatch --export=ALL,HA_SEED=0,SMOKE=1 slurm/evaluate_frozen_ha.sbatch   # 2 patients, minutes
sbatch --export=ALL,HA_SEED=0 slurm/evaluate_frozen_ha.sbatch           # primary
sbatch --export=ALL,HA_SEED=1 slurm/evaluate_frozen_ha.sbatch           # secondary

R=outputs/cost_sweep/cost_1_attempt2
for score in diagnosis final_hypothesis; do
  $PY -u scripts/test_stats.py --score $score --resamples 2000 \
    --focus-dir "combined_ha0=$R/frozen_ha_seed0_step_0600" \
    --extra "original=$R/test_step_0600" --extra "combined_ha1=$R/frozen_ha_seed1_step_0600" \
    --out-dir outputs/step2_stats/$score
done
```

The output goes to `$R/frozen_ha_seed<s>_step_0600/`, in the usual test-run format
(`test_episodes.csv`, `test_transcripts.jsonl`, `metrics.json`).

## 5. Limitations

- The DA was trained with the old HA's answers and confidences in its prompt. It now reads a
  differently calibrated HA without retraining, so a drop would show that mismatch rather
  than a weaker HA. H3a (retraining the DA against the frozen HA) addresses this.
- The DA's turns use vLLM with near-greedy sampling (temperature 0.01), as in every test
  run. The HA no longer goes through vLLM, so the HA turns are exactly deterministic.

## 6. Results (fill in)

Filled in November 15, 2026 (`outputs/step2_stats/{diagnosis,final_hypothesis}/`).

| Run | Accuracy (diagnosis) | Accuracy (final hypothesis) | Tests | Cost | Undiagnosed | ECE | AUROC_correct |
|---|---|---|---|---|---|---|---|
| Original attempt 2 | 85.4% | 87.1% | 1.19 | $1,259 | 2.1% | 14.8% | 0.750 |
| Combined, HA seed 0 | 85.4% | 89.2% | 1.20 | $1,285 | 1.7% | 5.2% | 0.936 |
| Combined, HA seed 1 | 84.2% | 90.8% | 1.16 | $1,248 | 1.7% | 3.6% | 0.924 |

| Paired difference, seed 0 minus the original | Difference [95% CI], p | Reading |
|---|---|---|
| Diagnosis accuracy | +0.0 pts [−3.8, +3.7], p = 1.00 (11 vs 11 discordant) | No detectable change |
| Final-hypothesis accuracy | +2.1 pts [−1.7, +5.8], p = 0.38 (13 vs 8) | No detectable change |
| Tests per case | +0.01 [−0.04, +0.06], p = 0.77 | No change |
| Cost | +$27 [−$30, +$91], p = 0.41 | No change |

**Predictions vs results:**

| Measure | Predicted | Result | Met? |
|---|---|---|---|
| Accuracy, DA's diagnosis | 87–90% | 85.4% | No |
| Accuracy, final hypothesis | 88–90% | 89.2% | Yes (but not significant) |
| Tests per case | ≥ 1.19 | 1.20 | Yes |
| Cost | about the same or higher | $1,285, same | Yes |

**Reading: no detectable change in accuracy or cost; calibration is much better.**
- The DA's diagnosis is unchanged (85.4%), so the DA does not depend on the old HA's way of
  answering. Swapping the HA in does no harm.
- The DA behaves the same: the same tests (1.20 vs 1.19) at the same cost.
- The HA's confidence is now meaningful. ECE fell from 14.8% to 5.2%, and AUROC_correct
  rose from 0.750 to 0.936. The 95% intervals do not overlap.
- **The DA ignores some correct HA answers.** In the same run, the HA's final hypothesis is
  right 89.2% of the time and the DA's diagnosis 85.4%: a 3.8-point gap, larger than the
  original's 1.7. The gap is mostly pancreatitis: the HA's final recall is 83.3%, the DA's
  68.5% (11 cholecystitis cases were called pancreatitis and 9 pancreatitis cases were
  called cholecystitis). This is a direct motivation for H3: a DA trained alongside this HA
  might learn to trust it. (The gap is unpaired within one run; no test was run.)
- Seed 1 agrees: diagnosis −1.2 points vs seed 0 (p = 0.65), final hypothesis +1.7 (p = 0.39).

Run log:
- Both runs completed on all 240 test patients (about 44 minutes each). `wc -l` on
  `test_episodes.csv` gives about 1,200 lines because fields hold quoted newlines; `n = 240`
  in `metrics.json`.
- ECE, Brier and AUROC are the same in the two score tables, because they use the HA's
  confidence in both.
- Paper comparison (seed 0): class-mean accuracy 84.1% and macro F1 83.0% (paper 81.3%
  both), cost $1,285 (paper $1,296); the paper's values are inside our 95% intervals.

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

| Run | Accuracy (diagnosis) | Accuracy (final hypothesis) | Tests | Cost | Undiagnosed |
|---|---|---|---|---|---|
| Original attempt 2 | 85.4% | 87.1% | 1.19 | $1,259 | |
| Combined, HA seed 0 | | | | | |
| Combined, HA seed 1 | | | | | |

| Paired difference vs the original | Difference [95% CI], p | Reading |
|---|---|---|
| Diagnosis accuracy, seed 0 | | |
| Cost, seed 0 | | |

Run log:

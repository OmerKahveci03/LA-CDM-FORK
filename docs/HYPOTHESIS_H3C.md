# Hypothesis H3c: a DA that never overrides the frozen HA

**Written November 25, 2026, before training.** Everything here is fixed before any H3c result exists.
Results go at the end.

## Why

In every system so far, the HA's final hypothesis is as accurate as or more accurate than the DA's own
diagnosis (A1, `docs/FINAL_ANALYSES.md`). Using the HA's answer after training ("defer") was ahead in all
7 systems, but those DAs were trained to make their own diagnosis. H3c trains the DA **knowing** that
the HA will diagnose, so its reward depends only on whether the **HA's** final answer is right (minus
test cost). The DA's job becomes purely choosing which tests to order and when to stop.

## Design (`src/environment/defer_ha_environment.py`, `src/train_defer_ha.py`)

- **The same as H3a in every other way:** a new DA trained from scratch; the frozen separately
  fine-tuned HA (H2 seed 0); the original reward (diagnosis, format, test cost); 4 rollouts per
  patient; validation on 240 patients every 100 steps.
- **The one change:** when the DA writes "Diagnosis: …", the recorded diagnosis (rewarded and logged)
  becomes the HA's current hypothesis. The DA's own choice is logged as `da_choice:…`.
- **Budget:** 700 steps (H3a's chosen step), one job of about 20 hours. It includes the resume fix.
- **Outputs:** `outputs/h3/h3c_ha0/train/`; test: `outputs/h3/h3c_ha0/defer_ha_seed0_step_XXXX/`.

## Checkpoint choice (validation only)

The highest validation accuracy among steps 100–700, **excluding any step whose validation DA orders
fewer than 0.5 tests per patient**. This guards against the zero-test shortcut seen in the H4c
extension: the HA's history-only accuracy is about 78%. Ties go to the lower validation cost.

## Comparisons and decision rules (240 MIMIC test patients, paired)

Computed with `scripts/defer_vs_baseline.py --h3c-dir <test folder>` (exact McNemar; paired bootstrap
95% CI):

- **Primary: H3c vs H3a + defer** (H3a's HA final hypothesis, the same HA, with deferral applied only
  after training). **Supported** ("training the DA with deferral helps") if the CI is entirely above 0.
  If it includes 0: deferring after training is enough.
- **Secondary:**
  - H3c vs baseline (better if the CI is above 0);
  - H3c vs H3a's own DA;
  - cost against both (paired permutation via `scripts/test_stats.py`);
  - tests per patient and CT rate;
  - Edward's external set, run once, if time allows.
- **Caveat:** a single run. Two identical runs differed by up to 11 points on validation (H4).

## Run

```bash
sbatch --export=ALL,SMOKE=1 slurm/train_defer_ha.sbatch          # 4 steps first
sbatch slurm/train_defer_ha.sbatch                               # 700 steps
# after choosing the step on validation:
sbatch --export=ALL,ADAPTER_STEP=<step> slurm/evaluate_defer_ha.sbatch
"$PY" scripts/defer_vs_baseline.py --h3c-dir outputs/h3/h3c_ha0/defer_ha_seed0_step_<step>   # as a CPU job
```

## Results

**Run log:** submitted on November 25, 2026: 52780633 (unit tests) → 52780634 (smoke, 4 steps) →
52780635 (700 steps) → 52780636 (spare resume after a time-out).

- **Override check (November 25, about 6 h in):** validation at step 0: 49 diagnoses, 49 overridden;
  training: 362 diagnosis-type actions, 361 overridden (the one exception is an invalid disease name or
  a turn with no HA hypothesis yet, both by design). The smoke run had 0 diagnoses, so 0 overrides
  was expected. Validation at step 100 done after about 6 h; expected to need the spare job and to
  finish around Thursday evening.

- **Validation so far** (about 8.5 h): step 0 4.9% (0.96 tests per patient), step 100 48.6% (1.67),
  step 200 58.1% (1.55). No zero-test shortcut yet. About 4 h per 100 steps, so step 700 lands
  around Thursday night, using the spare job.

- **Training finished** in one job (52780635, `train exit=0` at 2026-11-26 09:33; the spare was not needed).

**Validation curve** (DA = the HA's answer at the stop; accuracy, tests per patient, cost):

| Step | 0 | 100 | 200 | 300 | 400 | **500** | 600 | 700 |
|---|---|---|---|---|---|---|---|---|
| Accuracy | 4.9% | 48.6% | 58.1% | 56.5% | 65.5% | **70.6%** | 69.2% | 68.8% |
| Tests | 0.96 | 1.67 | 1.55 | 1.32 | 1.61 | **1.63** | 1.26 | 0.33 |
| Cost | $714 | $1,236 | $1,264 | $1,151 | $1,149 | **$941** | $484 | $86 |

**Chosen step: 500** (the highest validation accuracy; every step from 100 to 600 passes the 0.5-tests
guard). Step 700 was already drifting into the zero-test shortcut (0.33 tests, $86) and would have
been excluded by the guard anyway.

**Test (step 500; `outputs/h3/h3c_ha0/defer_ha_seed0_step_0500`, statistics `outputs/h3/h3c_ha0/stats_*`
and `outputs/final_analyses/a6_defer_vs_baseline.md`):** **83.8%** [78.6, 87.9], 1.67 tests per patient,
$1,035 per case.

| Comparison | Other | Difference [95% CI] | Wins/losses | p |
|---|---|---|---|---|
| **vs H3a + defer (primary)** | 88.3 | **−4.6 [−7.5, −1.7]** | +1/−12 | **0.003** |
| vs baseline | 85.4 | −1.7 [−7.1, +3.8] | +20/−24 | 0.65 |
| vs H3a (its own DA) | 85.8 | −2.1 [−5.8, +2.1] | +9/−14 | 0.40 |

**Verdict: not supported, and worse than the alternative.** Training the DA with deferral built in
was significantly *worse* than training it to diagnose and deferring afterwards (H3a + defer). It
ordered more tests (1.67 vs 1.27) at about the same cost. One plausible, unverified reason: when its
own diagnosis does not count, the DA's reward is a weaker, noisier signal, so it learns test selection
less well. Single run (two identical runs have differed by up to 11 points), so the size is uncertain.
Per `docs/HYPOTHESIS_H3D.md`, H3c < 88.3%, so **H3d uses DESIGN=plain**.

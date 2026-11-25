# Final analyses before the submission (A1–A4)

**Written November 25, 2026, before any of these results exist.** Each analysis has its rule fixed
here. Results go in the "Run log and results" section at the end. Background: `docs/NEXT_EXPERIMENTS.md`
items 1, 2 and 5. Deadline: the team aims to submit by Sunday, November 29.

All MIMIC results use the 240 test patients. Paired comparisons use exact McNemar for accuracy, with
Holm correction within each analysis. Nothing is tuned on test, and Edward's external set is never used
for choosing anything.

**Systems** (the validation-chosen checkpoints of H3 and H4):

| Name | Test episodes | Validation episodes (for choosing thresholds) |
|---|---|---|
| baseline | `outputs/cost_sweep/cost_1_attempt2/test_step_0600` | `…/cost_1_attempt2/train/episodes/val_step_0600_episodes.csv`, if it exists |
| step2 (baseline DA + separately fine-tuned HA) | `…/cost_1_attempt2/frozen_ha_seed0_step_0600` | none |
| h3a | `outputs/h3/h3a_ha0/frozen_ha_seed0_step_0700` | `outputs/h3/h3a_ha0/train/episodes/val_step_0700_episodes.csv` |
| h3b | `outputs/h3/h3b_ha0/frozen_ha_seed0_step_0700` | `outputs/h3/h3b_ha0/train/episodes/val_step_0700_episodes.csv` |
| h4b | `outputs/contrastive/combined/h2` | none |
| h4c_joint | `outputs/contrastive/joint/joint_h2/test_step_0600` | none (the original step-600 validation episodes were overwritten on November 24) |
| h4c_frozen | `outputs/contrastive/joint/frozen_h2/test_step_0400` | `outputs/contrastive/joint/frozen_h2/train/episodes/val_step_0400_episodes.csv` |

## A1. DA–HA disagreement and deferral rules

Code: `scripts/final_disagreement.py` (uses Edward's `edward/da_ha_disagreement.py`). Job: `jobs/final_disagreement.sh`.

- For every system: the 2×2 table (agree or disagree × HA right or wrong, plus "no DA diagnosis").
- **Primary rule, "defer"** (no parameters): the final diagnosis is the HA's final hypothesis.
  Compared with the DA's own diagnosis on the same patients (exact McNemar), with Holm across the
  7 systems. **Supported** for a system if the Holm-corrected p < 0.05 with the rule ahead.
- **Secondary, "fallback"** (no parameters): the DA's diagnosis, or the HA's hypothesis when the DA
  gave none.
- **Secondary, "threshold":** defer when the HA's final confidence (0–10) is at least t. For each
  system *with* validation episodes, t is the value in 0–10 with the highest validation accuracy
  (ties: the highest t, that is, the fewest deferrals). That t is then applied once to the same
  system's test episodes. No threshold is reported for systems without validation episodes.

## A2. Confidence bounds from an ensemble of CLM heads

Code: `src/contrastive_ha/ensemble_bounds.py`. Job: `jobs/final_ha_offline.sh`. Uses the cached
embeddings of the chosen backbone (`outputs/contrastive/embeddings/h2/`).

- **Ensemble:** K = 10 sets of heads with the stage-1 settings. Member k uses seed k and a
  **bootstrap resample of the training patients** (all of a patient's views together). Each member
  picks its own epoch on the full validation set (validation accuracy; ties go to the lower loss).
- **Per record:** the mean probability p̄_d and standard deviation s_d over members, for each disease d.
  The bounds are LCB_d = p̄_d − 2·s_d and UCB_d = p̄_d + 2·s_d. The factor 2 is fixed here, not tuned.
- **Decisive:** LCB(top) > UCB(runner-up), where top and runner-up are by p̄.
- **Questions** (on the record sets of H4a; **primary set A_da2**, the baseline DA's tests):
  1. Do the bounds flag errors? AUROC for predicting a wrong answer, using the margin
     LCB(top) − UCB(runner-up), compared with the plain confidence p̄(top) and with a single
     member's confidence. Descriptive, with a bootstrap 95% CI of the AUROC difference
     (margin minus p̄(top)).
  2. Accuracy of decisive vs non-decisive records, and the share that is decisive.
  3. **"Diagnose or test more", offline:** decisive records keep their A_da2 answer;
     non-decisive ones take the ensemble's answer with all tests (D_all) instead. Report the
     accuracy and the extra test cost, compared with "never test more" (A_da2) and "always all
     tests" (D_all). This is an upper-bound sketch (it assumes the extra tests would be ordered),
     not a deployed policy.
  4. Ensemble mean vs a single member (accuracy, ECE): does ensembling alone help?

## A3. Combined HA

Code: `scripts/combined_ha.py`. Job: `jobs/final_ha_offline.sh`.

- **Equal-weight mean of the probability vectors,** fixed here and not tuned:
  - (a) separately fine-tuned + CLM (fine-tuned backbone);
  - (b) separately fine-tuned + CLM (plain Llama);
  - (c) separately fine-tuned + baseline HA;
  - (d) all four HAs.
- On the four H4a record sets. **Primary:** A_da2 accuracy of each combination minus the separately
  fine-tuned HA (the best single HA), exact McNemar with Holm over the 4 combinations. Also ECE,
  Brier, and AUROC (confidence vs correct).
- **Expected:** (a) changes little, because its two members agree on 235/240 records. (b) and (c)
  combine HAs that differ more.

## A4. Cheap ablations (evaluation only, GPU)

- **A4a, the DA without the HA:** `src/environment/hidden_ha_environment.py`. The HA is still
  computed and logged (so `final_hypothesis` exists), but its text is never added to the DA's
  transcript. The DA's instructions are unchanged; they still mention hypotheses. Evaluated for:
  - the baseline DA (step 600) with the separately fine-tuned HA logged →
    `outputs/cost_sweep/cost_1_attempt2/hidden_ha_seed0_step_0600`;
  - the H3a DA (step 700) → `outputs/h3/h3a_ha0/hidden_ha_seed0_step_0700`.

  Compared with the same DA with the HA shown (baseline: baseline model and step2; H3a: H3a),
  paired. This measures how much these DAs **rely** on the HA. It is not a DA trained without one.
- **A4b, a weaker DA with the CLM HA:** the H4c frozen arm at **step 200** (validation 59.7%, chosen
  here as the first checkpoint where most validation episodes end in a diagnosis), evaluated with
  its (unchanged) CLM heads → `outputs/contrastive/joint/frozen_h2/test_step_0200`. Questions: how
  far does the DA fall below its HA, and does "defer" recover it (A1's script)?

## A5. Every system against the baseline model (added November 25, before running)

H3a and the H4c arms were never compared with the baseline model directly on MIMIC, only with step 2, with
H3a and with each other. Job: `jobs/final_vs_baseline_stats.sh`. Baseline minus each of step2, h3a, h3b,
h4b, h4c_joint and h4c_frozen on the 240 test patients, for the DA's diagnosis and for the HA's final
hypothesis. Exact McNemar and paired bootstrap 95% CIs for accuracy; permutation tests for cost and tests;
Holm across the 6 systems. This is descriptive: it completes the comparison table and decides no
hypothesis (H3/H4 keep their own pre-registered comparisons).

## How to run (commit, push, `git pull` first)

```bash
# 1. Unit tests (CPU)
t=$(sbatch --parsable --export=ALL,TESTS="tests.test_hidden_ha tests.test_final_analyses tests.test_frozen_ha" slurm/unit_tests.sbatch)
# 2. GPU evaluations: A4a after the tests pass; A4b needs no new code
sbatch --dependency=afterok:$t slurm/evaluate_hidden_ha.sbatch
sbatch --dependency=afterok:$t --export=ALL,RUN_ROOT=outputs/h3/h3a_ha0,ADAPTER_STEP=700 slurm/evaluate_hidden_ha.sbatch
sbatch --export=ALL,BACKBONE=h2,RUN=frozen_h2,ADAPTER_STEP=200 slurm/contrastive_evaluate.sbatch
# 3. CPU analyses (A1 now, A2 + A3 together)
sbatch --dependency=afterok:$t --export=ALL,SCRIPT=jobs/final_disagreement.sh slurm/cpu_script.sbatch
sbatch --dependency=afterok:$t --export=ALL,SCRIPT=jobs/final_ha_offline.sh slurm/cpu_script.sbatch
# 4. When the GPU evaluations are done: A4a statistics, and rerun A1 to add the A4 rows
sbatch --export=ALL,SCRIPT=jobs/final_a4_stats.sh slurm/cpu_script.sbatch
sbatch --export=ALL,SCRIPT=jobs/final_disagreement.sh slurm/cpu_script.sbatch
```
All results go to **`outputs/final_analyses/`** (`a1_disagreement.md`, `a2_ensemble_bounds.md`,
`a3_combined_ha.md`, `a4a_*/test_stats.md`).

## Run log and results

**Submitted on November 25, 2026:**

| Job | What | Waits for |
|---|---|---|
| 52771468 | unit tests (`test_hidden_ha`, `test_final_analyses`, `test_frozen_ha`) | – |
| 52771469 | A4a: baseline DA, HA hidden | tests |
| 52771470 | A4a: H3a DA, HA hidden | tests |
| 52771471 | A4b: frozen step-200 DA + CLM HA | – |
| 52771472 | A1: `jobs/final_disagreement.sh` | tests |
| 52771473 | A2 + A3: `jobs/final_ha_offline.sh` | tests |
| 52769959 | (H4c extension) frozen step-1100 test; then `jobs/contrastive_h4c_ext_stats.sh` | – |

Still to do once the GPU evaluations finish: `jobs/final_a4_stats.sh`, and rerun `jobs/final_disagreement.sh`.

(results to be filled in)

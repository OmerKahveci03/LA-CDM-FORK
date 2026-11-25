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

## A6. "H3a + defer" against the baseline (added November 25, after A1–A5, before running)

**H3a + defer:** the H3a system, whose DA chooses the tests, with the frozen separately fine-tuned HA's
final hypothesis as the diagnosis. It was chosen as the candidate "best system" **before** this
comparison, because both parts were fixed earlier: H3a was the best system on validation and cost
(H3/A5), and "defer" was A1's pre-registered rule.

- **Primary (MIMIC test, 240 patients):** H3a's HA final hypothesis vs the baseline's DA diagnosis,
  per patient. Exact McNemar, and a paired bootstrap 95% CI of the accuracy difference. Cost is H3a's
  (the tests do not change); its difference from the baseline was already tested in A5
  (−$234, Holm p = 0.003). **Better than the baseline** if the CI is entirely above 0.
- **Secondary:** the same comparison on Edward's external set (152 patients); and
  "step2 + defer" vs baseline. Step 2 had the highest "defer" accuracy in A1, so its comparison is
  **chosen after seeing test results** and is reported as such.
- Code: `scripts/defer_vs_baseline.py`. Job: `jobs/final_defer_vs_baseline.sh`. Output:
  `outputs/final_analyses/a6_defer_vs_baseline.{md,json}`.

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

- 52771468 **failed** (1 of 11 tests): `test_entry_point_swaps_the_class` imported the real
  entry point after the test file had replaced `trl` with a stub. This was a test bug; the code was
  fine (the hidden-HA environment and ensemble tests passed). The check now runs in a separate
  process. 52771469/70/72/73 never started and were cancelled; they are resubmitted after the fixed
  tests pass. 52771471 (A4b) was unaffected.
- Resubmitted: 52774123 (unit tests) → 52774124 (A4a baseline DA, HA hidden), 52774125 (A4a H3a
  DA, HA hidden), 52774126 (A1), 52774127 (A2 + A3).
- 52774124/52774125 (A4a) **failed** at start-up: the entry point reused `evaluate_frozen_ha`'s
  decorated `main`, so Hydra resolved `configs/` from the wrong module. Fixed:
  `src/evaluate_hidden_ha.py` now has its own `@hydra.main` and calls the original function. The
  failed output folders (logs only) were deleted. Rerun with a smoke gate: 52776115 (smoke,
  2 patients) → 52776116 (baseline DA) and 52776117 (H3a DA).
- 52773930: A5 (`jobs/final_vs_baseline_stats.sh`). 52780632: A6 (`jobs/final_defer_vs_baseline.sh`). 52774073: H4c-ext statistics
  (`jobs/contrastive_h4c_ext_stats.sh`, frozen step 1100 test finished).

### Results (November 25, 2026)

**A1, deferral rules (MIMIC test; `outputs/final_analyses/a1_disagreement.md`).**

| System | DA | HA final | overrode correct HA / corrected wrong HA | defer: acc (wins/losses), Holm p |
|---|---|---|---|---|
| baseline | 85.4 | 87.1 | 7 / 7 | 87.1 (+11/−7), 0.48 |
| step2 | 85.4 | 89.2 | 12 / 5 | 89.2 (+14/−5), 0.25 |
| h3a | 85.8 | 88.3 | 8 / 2 | 88.3 (+8/−2), 0.25 |
| h3b | 74.6 | 78.3 | 14 / 5 | 78.3 (+14/−5), 0.25 |
| h4b | 82.9 | 88.8 | 16 / 6 | 88.8 (+20/−6), 0.056 |
| h4c_joint | 84.2 | 87.5 | 9 / 1 | 87.5 (+9/−1), 0.11 |
| h4c_frozen | 82.1 | 88.8 | 17 / 2 | **88.8 (+18/−2), 0.003** |

- **Verdict:** "defer" is supported (Holm < 0.05) for h4c_frozen only. In every system it is
  ahead of the DA (+1.7 to +6.7 points) and never behind.
- **Threshold rule:** t was chosen on validation (baseline 6, h3a 4, h3b 5, h4c_frozen 4) and gave
  the same test result as "defer" (it defers almost always), so nothing extra.
- **Fallback rule:** no significant change anywhere.
- Secondary: ext_frozen_step1100 DA 78.8, HA 79.6 (+3/−1). The A4 rows come when A1 is rerun.

**A2, CLM head ensemble (`a2_ensemble_bounds.md`).** 10 members, each choosing its epoch at 2–11
(validation 86.0–87.2%).
- **Accuracy:** the same as one set of heads. **Calibration is better** on every record set
  (ECE: A_da2 5.4 vs 7.0; val 5.4 vs 7.9; B_none 8.8 vs 11.2).
- **Bounds:** "decisive" (LCB(top) > UCB(runner-up)) on 92.1% of A_da2. Decisive records are 93.7%
  correct, non-decisive ones 36.8% (validation: 91.9 vs 55.4).
- **But the margin flags errors no better than the mean confidence:** AUROC 0.936 vs 0.933 on A_da2,
  difference +0.003 [−0.007, +0.016]. On validation +0.006 [+0.001, +0.012], tiny.
- **"Diagnose or test more" (offline sketch, A_da2):** 7.9% of records get all tests, giving
  89.2 → 90.0% (+4/−2, p = 0.69) at $1,259 → $1,494 per case. All tests for everyone: 90.4% at $4,082.
- **Conclusion:** ensembling gives better calibration and a clean "unsure" flag, but the plain
  confidence works almost as well, and the HA's remaining errors are not fixed by more tests.

**A3, combined HAs (`a3_combined_ha.md`).** No combination beat the separately fine-tuned HA on any
record set (A_da2: all Holm p = 1.0; best d_all_four 89.6% vs 88.8%, +4/−2). Averaging it with the
baseline HA gave the best calibration on A_da2 (ECE 1.9% vs 3.5%). **Combining HAs adds no accuracy.**

**A4b, weak DA with the CLM HA** (`frozen_h2/test_step_0200`): **DA 78.3%**, undiagnosed 5.8%, 1.53
tests, $1,442, CT for 64.6% of patients. **HA final hypothesis 89.2%**, the largest DA–HA gap seen
(10.9 points). A weak DA still gathers tests from which the HA reaches about 89%. Paired statistics
come when A1 is rerun.

**A4a:** running (52776116, 52776117).

**A6, "H3a + defer" vs the baseline** (`a6_defer_vs_baseline.md`; candidate minus baseline, paired
bootstrap 95% CI, exact McNemar):

| Comparison | Candidate | Baseline | Difference [95% CI] | Wins/losses | p |
|---|---|---|---|---|---|
| **MIMIC: H3a + defer (primary)** | 88.3 | 85.4 | **+2.9 [−1.7, +7.5]** | +20/−13 | 0.30 |
| External: H3a + defer | 92.1 | 86.8 | +5.3 [0.0, +11.2] | +14/−6 | 0.12 |
| MIMIC: step2 + defer (chosen after A1) | 89.2 | 85.4 | +3.8 [−0.8, +8.3] | +20/−11 | 0.15 |

**Verdict:** not better by the rule (the CI includes 0). "H3a + defer" is at least as accurate as the
baseline (leaning positive on both data sets) and significantly cheaper (−$234, A5).

**A5, every system vs the baseline model** (MIMIC test, DA's diagnosis; `a5_vs_baseline_diagnosis/`;
system minus baseline; Holm over the 6):

| System | Accuracy [95% CI] | Cost |
|---|---|---|
| step2 | 0.0 [−3.7, +3.7] | +$27, n.s. |
| h3a | +0.4 [−3.8, +4.6] | **−$234 [−305, −170], Holm p = 0.003** |
| h3b | **−10.8 [−16.7, −5.0], Holm p = 0.004** | −$723, p < 0.001 |
| h4b | −2.5 [−6.7, +1.3] | −$24, n.s. |
| h4c_joint | −1.2 [−6.2, +4.2] | −$70, p = 0.075 |
| h4c_frozen | −3.3 [−8.3, +1.7] | +$175, p < 0.001 |

No system is more accurate than the baseline. Only H3b is less accurate. H3a's cost saving is now
confirmed against the baseline directly.

**H4c extension, frozen step 1100 test** (`outputs/contrastive/stats/h4c_ext_diagnosis/`):
- **78.8%, with 0.00 tests and $0 per case.** The DA stopped ordering tests and diagnoses from the
  history alone; the HA's history-only accuracy is about 78%.
- vs frozen 400: −3.3 [−9.6, +2.9]. vs H3a: −7.1 [−12.5, −1.7], Holm p = 0.048. vs baseline: −6.7
  [−13.3, −0.4], p = 0.052 (Holm 0.10).
- Validation accuracy (76.4%) did not reveal the collapse. Choosing on accuracy alone picked a
  degenerate zero-cost policy.

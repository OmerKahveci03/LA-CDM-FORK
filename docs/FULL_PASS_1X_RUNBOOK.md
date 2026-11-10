# Full-pass 1x baseline: runbook and record

Started: November 7, 2026. **Current position: attempt 2 running (job 50189200, started Nov 8); step-100 go/no-go
passed (42.5% val accuracy). Now Step 6 (during the run). Attempt 1 collapsed (section 0).**

## 0. Attempt 1 collapsed — what happened and what changed

Attempt 1 (training chain 50132293–50132297) never learned to diagnose:

| Val step | Undiagnosed | Accuracy | Tests/case | Turns |
|---|---:|---:|---:|---:|
| 0 (untrained) | 96.9% | 2.4% | 0.90 | 2.25 |
| 100 | 98.6% | 1.1% | 0.79 | 2.12 |
| 200 | 98.3% | 1.4% | 0.68 | 2.10 |
| 300 | 97.5% | 1.5% | 0.75 | 2.16 |
| 400 | 100.0% | 0.0% | 0.15 | 2.01 |
| 500 | 99.8% | 0.1% | 0.14 | 2.02 |

The first DA phase (steps 1–100) never escaped the untrained model's format
failures. The second DA phase (301–400) then learned to stop before ordering a test.
Hypothesis accuracy stayed at ~67%, so the HA was fine. Only the DA's actions failed.

**Cause.** In the collapsed 1.5x log, 4,012 of 4,061 format failures (99%) are on
the DA's **second** action. Either the model answers a test result as if it were the
user ("Please provide the next step."), or it asks for `Complete Blood Count (CBC)`,
is told the test is unknown (1,407 times), and replies to the error. The paper's
reward charged test cost even to invalid episodes. So in a group where all 4 rollouts
failed, the one that failed before ordering anything got the highest reward, and
GRPO turned that into a full-strength "fail fast" advantage. The successful
October 1.0x run escaped by chance in its first DA phase. This run and the 1.5x
arm did not. The November 7 logging changes did not cause this: the training
settings are identical.

**Fixes for attempt 2** (all deviations from the paper, to report as such):

| Change | Files |
|---|---|
| Test cost is not charged to invalid episodes (`charge_invalid: false`); an all-failed group ties and gets zero advantage | `src/reward_function/test_cost_reward.py`, `configs/reward_function/defaults.yaml` |
| A listed test name followed by a bracketed abbreviation (`Complete Blood Count (CBC)`) counts as that test | `src/environment/environment.py` (`_match_test_name`) |
| Collapse guard: at any validation from step 200 on, ≥95% undiagnosed stops training, writes `train/collapse_detected.json`, and queued jobs then exit | `src/train.py`, `configs/training/defaults.yaml`, `slurm/train_cost_sweep.sbatch` |
| `charge_invalid=false` is part of the arm's settings lock, so the attempt-1 directory cannot be resumed by mistake | `slurm/train_cost_sweep.sbatch` |

The CBC change also affects evaluation, so the zero-shot baseline must be run on
this code too (Step 3a).

### Step 3a — Restart (attempt 2)
```bash
squeue -u omerkahveci                                  # cancel any attempt-1 training job still listed:
scancel 50132293 50132294 50132295 50132296 50132297
git pull && git log --oneline -1 && git status
mv outputs/cost_sweep/cost_1 outputs/cost_sweep/cost_1_attempt1_collapsed
# zero-shot: if 50132298 ran before the pull, move its output and rerun on the new code
mv outputs/zero_shot outputs/zero_shot_attempt1_code    # only if it exists
$PY -m unittest tests.test_test_cost_reward_multiplier tests.test_environment_end_reasons
```
Run it with the env's Python (`PY=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python`,
see section 3). Then run Step 3 unchanged (it submits the chain and the zero-shot job).
Check validation as it comes in:
```bash
for f in outputs/cost_sweep/cost_1/train/episodes/val_step_0*_summary.json; do
  $PY -c "import json,sys; s=json.load(open(sys.argv[1])); o=s['overall']; print(s['step'], 'undiagnosed=%.3f' % o['undiagnosed_fraction'], 'acc=%.3f' % o['accuracy'], 'tests=%.2f' % o['avg_num_tests'])" "$f"
done
```
**Go/no-go at step 100 validation:** undiagnosed should be clearly below step 0's
~97%. If it isn't by step 200, the guard stops the run by itself.

This file is the single place to resume this experiment after time away. Tick the
checkboxes and fill in the record table as steps complete. Results go into
`docs/results/` and the weekly reports once they exist.

## 1. Goal and decisions

Establish the project's baseline properly, from scratch, before testing improvements.
**This run and the rerun zero-shot evaluation are the baseline; earlier runs (the
original 1.0x model, the 0x and 1.5x arms) are not used for comparison.** They were
trained on earlier code, partly across bug-fix resumes, on 300 of the 1,920 training
patients, and saved only aggregate metrics. Every later improvement is compared
against this run, on the same test cases, with per-case paired statistics.

Decisions (November 7, 2026):

| Decision | Choice | Reason |
|---|---|---|
| Cost penalty | **Multiplier 1** (paper's unscaled penalty) | The paper's method; the reference for every later change |
| Length | **1,920 steps** = one pass over every training patient | Train on the whole training split |
| Phase schedule | DA → HA-SFT → confidence calibration, **100 steps each**, repeating | Paper's schedule, unchanged |
| Validation | Before training (step 0), **every 100 steps**, and at the final step 1,920 | Learning curve over training |
| Validation protocol | All 240 validation cases × 4 sampled rollouts (temperature 0.9) | Existing validation protocol; the test set uses 1 near-deterministic rollout per case |
| Stratification | Every validation and test result broken down by the 4 diseases | Diseases differ in prevalence and difficulty |
| Resumability | One fixed directory; resubmit the same command | No job IDs to track |

Consequences to remember when reporting:

- 1,920 is not a multiple of the 300-step cycle. The run ends 120 steps into cycle 7
  (100 DA steps + 20 HA-SFT steps). The **step-1,800 adapter is the last complete
  cycle**; test it as well as the final one.
- Adapters are kept at every cycle end (300, 600, …, 1,800) and at 1,920. Choose
  which to test from the **validation curve**, not from test results.
- Statistics (95% CIs, paired McNemar / bootstrap tests, per disease) can be computed
  afterwards from the per-case episode CSVs. They cover test-set sampling only, not
  training randomness: there is one training run.
- Expected cost: ~25–35 h of training plus ~21 validation passes of ~2.5 h each,
  so about 4 days across 4–5 Slurm jobs of 24 h.

## 2. What was built (November 7, 2026)

| Change | Files |
|---|---|
| Per-episode records: end reason (`diagnosis`, `invalid_format`, `max_steps`, `max_length`), every action in order (tests, unavailable tests, misnamed tests, repeats), hypotheses and confidences per turn, cost, reward components, GRPO advantage, last model output | `src/environment/environment.py`, `src/utils/episode_log.py`, `src/trainer/cdm_grpo_trainer.py` |
| Episode CSVs for training (every step), validation (every pass) and test, plus transcripts (`.jsonl`) and summaries (`.json`) | same |
| Per-disease metrics and a confusion matrix in every validation/test `metrics.json` and log; ECE in summaries | `src/utils/episode_log.py` |
| Training-signal monitor: share of steps whose 4 rollouts tied on diagnosis and format, so advantages came only from test cost (suspected 1.5x collapse mechanism) | `src/utils/episode_log.py`, `scripts/summarize_episodes.py` |
| Validation now reports test count and USD cost (training previously never passed prices) | `src/train.py` |
| Fixed arm directory, automatic resume, settings lock, job history, step-0 and final validation, adapters kept at every cycle end | `slurm/train_cost_sweep.sbatch`, `src/train.py`, `scripts/find_latest_training_checkpoint.py` |
| Test evaluation by arm and step, never overwrites | `slurm/evaluate_cost_sweep.sbatch` |
| Zero-shot baseline and smoke tests write to fixed or timestamped folders | `slurm/zero_shot_full.sbatch`, `slurm/train_smoke.sbatch`, `slurm/adapter_smoke.sbatch` |
| Analysis scripts | `scripts/summarize_episodes.py` (tables), `scripts/validation_curve.py` (validation over steps), `scripts/plot_validation_curve.py` (PDF/PNG charts) |
| Tests (20, all passing locally) | `tests/test_episode_log.py`, `tests/test_environment_end_reasons.py`, `tests/test_find_latest_training_checkpoint.py` |

Training behaviour is unchanged from the earlier code except one relabelling: an
episode that exceeds the token limit is now recorded as `max_length` instead of
`invalid_format` (it fails the same way).

## 3. Runbook

All commands run on HiPerGator from `/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK`.
`$PY` is the la-cdm conda env's Python (the same one the Slurm scripts use); set it
once per shell:
```bash
PY=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
```

### Step 0 — Sync code ✅
```bash
git pull && git log --oneline -1 && git status   # clean tree, latest commit
```

### Step 1 — Training smoke test (~15 min) ✅ passed November 7, 2026
Run: `outputs/train_smoke/20261107-120356/`. Both steps completed; all episode,
validation (steps 0–2), adapter and completion files were written; both summaries
printed. All 4 training and 4 validation episodes ended `invalid_format` (expected
for a 2-step model; the smoke cases are appendicitis only), so cost-only signal
was 100% — see the Step 4 note.
```bash
sbatch slurm/train_smoke.sbatch
tail -3 logs/lacdm_train_smoke_<id>.err
ls outputs/train_smoke/*/episodes/
$PY scripts/summarize_episodes.py outputs/train_smoke/*/episodes/train_episodes.csv
$PY scripts/summarize_episodes.py outputs/train_smoke/*/episodes/val_step_0002_episodes.csv
```
- [x] Finished with no traceback (`[SLURM] Finished` is in the `.out` log)
- [x] `train_episodes.csv` and `val_step_0000/0001/0002_*` files exist
- [x] `training_complete.json` and `adapters/step_0002/` exist
- [x] Both summaries print tables (numbers are meaningless: 2 patients)

### Step 2 — Test-time smoke test (~15 min)
```bash
sbatch slurm/adapter_smoke.sbatch outputs/train_smoke/20261107-120356
```
- [x] `outputs/adapter_smoke/<timestamp>/` has `metrics.json`, `test_episodes.csv`, `test_summary.json`
- [x] `metrics.json` contains `eval_disease/...` and `eval_confusion/...` keys

✅ Passed November 7, 2026: `outputs/adapter_smoke/20261107-121232/` (2 cases,
both `invalid_format` after one CT, as expected for a 2-step adapter).

**If Step 1 or 2 fails, stop and debug from the `.err` file before Step 3.**

### Step 3 — Start the full run and the zero-shot baseline
```bash
jid=$(sbatch --parsable --export=ALL,COST_MULTIPLIER=1,MAX_STEPS=1920 slurm/train_cost_sweep.sbatch)
for i in 1 2 3 4; do jid=$(sbatch --parsable --dependency=afterany:$jid --export=ALL,COST_MULTIPLIER=1,MAX_STEPS=1920 slurm/train_cost_sweep.sbatch); done
sbatch slurm/zero_shot_full.sbatch
cat outputs/cost_sweep/cost_1/jobs.log          # "start from step 0 to step 1920"
```
The queued jobs each start when the previous one ends; any extra job exits
immediately once training is complete.

✅ Submitted November 7, 2026: training 50132293 (running) → 50132294 → 50132295 →
50132296 → 50132297; zero-shot 50132298. If two training jobs in a row crash (not
time out), cancel the rest (`scancel <remaining ids>`) and read `train_output.log`.
The zero-shot job waits on `QOSGrpMemLimit`: the course allocation caps total memory,
so it runs when a training job ends (~1 h, delays training by about that much).

### Step 4 — Early checks (first day)
- [ ] After ~3 h (step-0 validation + ~25 steps):
  `$PY scripts/summarize_episodes.py outputs/cost_sweep/cost_1/train/episodes/train_episodes.csv`
  **Collapse warning:** the untrained model fails format on most episodes, so a high
  "Cost-only signal" in steps 1–25 is expected (the smoke test showed 100%). The
  collapse pattern is when it **stays** high and "Undiagnosed" stays near 100%
  through steps 25–100 instead of falling (how an earlier 1.5x run collapsed).
  If so, stop and investigate before
  spending 4 days.
- [x] After step 100 (attempt 2: passed, 3.1% → 42.5%): `$PY scripts/validation_curve.py outputs/cost_sweep/cost_1/train`;
  accuracy should be clearly above step 0.
- [x] Zero-shot done (~1 h): `outputs/zero_shot/test/metrics.json` and `test_episodes.csv`.

### Step 5 — Optional: test an intermediate adapter while training continues
Not required. If the validation curve flattens early (e.g. by step 600), testing that
cycle-end adapter while training continues shows whether the rest of the run is
worth its compute. Pick the step from validation, and keep test runs to a few:
```bash
sbatch --export=ALL,COST_MULTIPLIER=1,ADAPTER_STEP=600 slurm/evaluate_cost_sweep.sbatch
```

### Step 6 — During the run (days 2–4)
- Re-run `validation_curve.py` periodically.
- Plot progress (needs matplotlib in the env, see Step 7):
  ```bash
  $PY scripts/plot_validation_curve.py --run attempt2=outputs/cost_sweep/cost_1/train       --run attempt1=outputs/cost_sweep/cost_1_attempt1_collapsed/train --out figures/validation
  $PY scripts/plot_training_curve.py outputs/cost_sweep/cost_1/train --out figures/training
  ```
- `jobs.log` should show each job resuming where the last stopped. A job that times
  out loses at most 25 steps plus any validation pass in progress.
- Two failures in a row (not time-outs): stop and read `train_output.log`.

### Step 7 — After `train/training_complete.json` exists
```bash
sbatch --export=ALL,COST_MULTIPLIER=1 slurm/evaluate_cost_sweep.sbatch                    # step 1920
sbatch --export=ALL,COST_MULTIPLIER=1,ADAPTER_STEP=1800 slurm/evaluate_cost_sweep.sbatch  # last full cycle
$PY -m pip install --no-cache-dir matplotlib   # once, into the la-cdm env (or plot locally)
$PY scripts/plot_validation_curve.py \
    --run baseline=outputs/cost_sweep/cost_1/train \
    --out figures/validation
$PY scripts/summarize_episodes.py outputs/zero_shot/test/test_episodes.csv \
    outputs/cost_sweep/cost_1/test_step_1800/test_episodes.csv \
    outputs/cost_sweep/cost_1/test_step_1920/test_episodes.csv
```
Choose which checkpoints to report from **validation**, not test results; testing many
checkpoints and reporting the best turns the test set into a validation set.

## 4. Output locations

```text
outputs/train_smoke/<timestamp>/                 Step 1
outputs/adapter_smoke/<timestamp>/               Step 2
outputs/zero_shot/test/                          zero-shot (untrained) reference
outputs/cost_sweep/cost_1/
  jobs.log, train_output.log, run_settings.txt
  train/checkpoint-*/ (newest 5), adapters/step_0300 … step_1800, step_1920/
  train/training_complete.json
  train/episodes/train_episodes.csv              1,920 steps × 4 rollouts
  train/episodes/val_step_<N>_{episodes.csv,summary.json,transcripts.jsonl}
  train/episodes/val_curve.csv                   written by validation_curve.py
  test_step_1800/, test_step_1920/ (+ any optional intermediate step)
figures/validation_{overview,per_disease}.{pdf,png}, validation_points.csv
```
Earlier runs (`outputs/cost_sweep/cost_0/`, `cost_1p5/`, `train_full_*/`, `adapter_full_*/`,
`zero_shot_full_*/`) are left untouched and are **not part of this baseline**.

## 5. Run record (fill in)

| Item | Value |
|---|---|
| Commit | Attempt 2: `d293c8c` (clean) |
| Smoke train job / result | `outputs/train_smoke/20261107-120356/` — passed (pipeline only; 0/4 diagnosed) |
| Adapter smoke job / result | `outputs/adapter_smoke/20261107-121232/` — passed (pipeline only; 0/2 diagnosed) |
| Zero-shot job / test accuracy | Rerun after the attempt-2 fixes, finished Nov 8 (`outputs/zero_shot/test/`): **8.75% (21/240)**, 87.9% undiagnosed (all `invalid_format`), 72.4% diagnosed-only, macro F1 15.5%, 1.16 tests / $798 per case, HA accuracy 65.3%. CBC returned a result in 42.9% of cases. Fixed code confirmed: metrics dated Nov 8 06:39, after the pull; only 1 of 240 transcripts has an "Unknown test" error. |
| Training jobs (from `jobs.log`) | Attempt 1 (cancelled): 50132293 → 50132297. Attempt 2: 50189200 started 2026-11-08 05:48 from step 0 (rest of the chain: fill in from `squeue`) |
| Step-0 validation accuracy | Attempt 2: 3.1%, 95.9% undiagnosed, 0.99 tests/case (attempt 1: 2.4%, 96.9%) |
| Validation curve (attempt 2) | Step 100: **42.5%** accuracy, 42.4% undiagnosed (32.5% `invalid_format`, 9.9% `max_length`), 73.8% diagnosed-only, macro F1 50.8%, 1.68 tests / $1,333, 3.4 turns. Per disease: appendicitis 52.1%, diverticulitis 47.1%, cholecystitis 41.5%, pancreatitis 24.5% (39/216 called cholecystitis). Attempt 1 at step 100: 1.1%. |
| Best validation step (chosen for testing) | |
| Step-1,800 validation / test accuracy | |
| Step-1,920 validation / test accuracy | |
| Warnings, failures, restarts | Attempt 1 (50132293–97) collapsed: 97% undiagnosed at step 0, 100% by step 400; moved to `outputs/cost_sweep/cost_1_attempt1_collapsed/`. Restarted with the section-0 fixes. |

## 6. Facts to carry into the report

- **Data.** MIMIC-IV v2.2 + MIMIC-IV-Note v2.2, primary-diagnosis cases built with the
  official MIMIC-CDM builder: appendicitis 957 (39.9%), cholecystitis 648 (27.0%),
  pancreatitis 538 (22.4%), diverticulitis 257 (10.7%); stratified 1,920/240/240 split.
  Patient history = HPI section of the **discharge summary**, summarized by Llama; labs
  = first value of each lab during the admission; imaging = abdominal radiology reports
  from the stay. A retrospective simulation of ED presentation, not a fixed time point.
- **Episode.** HA proposes a diagnosis + confidence (0–10) from history and returned
  tests; DA orders one of 12 tests or diagnoses. An episode ends at the first of: a
  valid diagnosis (right or wrong), unparseable output, ~14 DA turns, or the token
  limit. Unavailable/misnamed tests use a turn and skip the HA for that turn. The
  DA has no "decline" option, so "undiagnosed" always means a failure.
- **Reward** (end of episode): +1 correct diagnosis, +1 valid format, minus per-test
  penalties × multiplier for distinct tests that returned a result (unavailable tests
  are free). GRPO normalizes within each patient's 4 rollouts, so when all 4 tie on
  diagnosis and format, tiny cost differences become full-strength advantages and
  the multiplier's scale cancels. Early training, when most rollouts fail format,
  is therefore dominated by a "use fewer tests" signal (risk of collapse).
- **Phases.** DA GRPO; HA SFT (cross-entropy toward `Hypothesis: <true label>` on every
  HA prompt seen, confidence masked); HA confidence calibration (GRPO over 4 sampled
  confidences for one fixed hypothesis, log-score reward). One shared LoRA adapter
  (r = 8) serves both agents; KL penalty β = 0.04 to base Llama.
- **Memory-driven deviations from the paper's config** (L4 GPUs): 4 rollouts per
  patient (not 8), training microbatch 1, 128-token DA replies (test: 256),
  2,048-token transcripts (test: 4,096). The first two change GRPO's comparison
  noise; the token limits end some long training episodes early.
- **Metrics.** `all/accuracy` counts undiagnosed as wrong (headline);
  `ignore/accuracy` is diagnosed-only and must not be quoted alone;
  `FormatReward` = 1 − undiagnosed share; hypothesis accuracy and ECE are over
  every HA call, not per patient; tests/cost per case count only tests that
  returned a result and include undiagnosed cases.

## 7. Open items after this run

- Statistics script (`scripts/compare_runs.py`, not yet written): Wilson 95% CIs;
  paired McNemar tests and bootstrap CIs for accuracy differences; bootstrap CIs and
  Wilcoxon tests for cost; bootstrap CIs for macro F1 and ECE; overall and per
  disease; CI bands on validation curves (bootstrap by patient). Primary comparison:
  overall test accuracy; Holm-correct secondary tests or label them exploratory.
- Copy `train/episodes/` and `test_step_*/` off `/blue` periodically as a backup.
- Recurring unknown `Complete Blood Count (CBC)` requests: likely a missing test alias.
- Improvements to test against this baseline: `docs/IMPROVEMENT_EXPERIMENTS.md`.

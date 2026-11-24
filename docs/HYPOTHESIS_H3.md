# Hypothesis H3: train the Decision Agent against a frozen HA (H3a, H3b)

**Pre-registered November 15, 2026, before any H3 result exists.** Sections 1–6 are fixed.
H3b's reward is option 1, λ = 1. Results go in section 7.

## 1. Hypothesis

Training the DA against a fixed, well-calibrated HA gives a better DA than training it
together with the HA in one shared adapter.

- **H3a** changes only the HA: the DA is trained against the frozen H2 HA with the original
  reward.
- **H3b** also changes the reward: the DA earns credit for how much its tests move the frozen
  HA towards the correct disease.

Running both lets us tell the effect of the HA apart from the effect of the reward.

## 2. Why

- **H2:** a separately trained HA reads partial records about 15 points better than the
  shared one, and it is better calibrated.
- **Attempt 2:** the DA drifted towards ordering no tests, and the HA's confidence inflated
  over training. In the shared adapter, each role's updates can undo the other's.
- **H2 again:** with a good HA, the DA's tests beat random tests by 8 points, not 23. The DA
  has to earn its tests against a stronger "no tests" baseline.
- **Step 2** (`docs/STEP2_COMBINED.md`): with the frozen HA swapped in, the attempt-2 DA's
  diagnosis stays at 85.4%, but the HA's final hypothesis reaches 89.2%. The DA overrides
  correct HA answers, mostly on pancreatitis (68.5% vs 83.3% recall). A DA trained against
  this HA may learn to trust it.

## 3. H3a (fixed)

| Choice | Value |
|---|---|
| DA adapter | New LoRA (r = 8), from the base model, as in attempt 2 |
| HA | Frozen H2 HA, seed 0 (`outputs/ha_sft/seed_0/adapter`), at every training and validation turn |
| Loss schedule | DA phases only (lr 1e-5); no HA-SFT or calibration phases |
| Reward | Unchanged: diagnosis + format + test cost (×1), invalid episodes not charged |
| DA settings | As in attempt 2: 4 generations, batch 1, 128-token completions, 2,048-token transcripts, seed 269 |
| Length | 900 steps. That is 900 DA updates; attempt 2's best checkpoint (step 600) had 200 |
| Checkpoint choice | Best validation accuracy (DA's diagnosis), every 100 steps, never on test |

**Code:**
- New: `src/train_frozen_ha.py` and `slurm/train_frozen_ha.sbatch`.
- Reused unchanged: the frozen-HA environment from step 2, and the existing callbacks.
- Testing a chosen checkpoint: `slurm/evaluate_frozen_ha.sbatch` with
  `RUN_ROOT=outputs/h3/h3a_ha0 ADAPTER_STEP=<best>`.

**Primary comparison (paired, same 240 test patients):**
- H3a's test accuracy vs **step 2**, the attempt-2 DA with the same frozen HA.
- The HA is identical in both, so the difference measures what training the DA against the
  frozen HA did.
- **Secondary:**
  - cost and tests per case;
  - tests per case over training on validation, the drift to zero tests seen in attempt 2;
  - comparison with attempt 2's original test run (85.4% at $1,259).

**Decision rule (fixed November 15, 2026, before any H3 result).** Same form as H1 and H2.
The measure is the paired accuracy of the DA's diagnosis on the 240 test patients, with an
exact McNemar test and a paired-bootstrap 95% CI (`scripts/test_stats.py --score diagnosis`):

| 95% CI of the difference | Reading |
|---|---|
| Entirely above 0 | Supported |
| Includes 0 | No detectable gain |
| Entirely below 0 | Worse |

- **H3a:** H3a minus step 2 (seed-0 HA).
- **H3b:** the same rule twice:
  - H3b minus step 2 (the overall effect);
  - H3b minus H3a (the effect of the reward alone).
- Cost, tests per case and final-hypothesis accuracy are secondary and do not change the
  reading.

## 4. H3b's reward

**Decided November 15, 2026: option 1, λ = 1.** H3b is identical to H3a except for this
added reward term.

**Code:**
- `src/reward_function/ha_belief_reward.py` (`HABeliefReward`): weight × the frozen HA's
  last probability of the label. Invalid episodes get 0, as they get no test cost.
- `configs/reward_function/h3b.yaml`: the default reward plus this term.
- `ARM=h3b` in `slurm/train_frozen_ha.sbatch` selects it; output in `outputs/h3/h3b_ha0`.
- Tests: `tests/test_ha_belief_reward.py`, including a check that p₀ cancels within a group.
- `reward_total` in the episode CSVs includes the term. Its value per episode is
  `reward_total − reward_diagnosis − reward_format − reward_test_cost`.

All options add a term from the frozen HA's probability of the correct disease, p(y). The
HA is scored after the patient history alone (p₀) and again after each test result.

**A point that shapes the choice.** GRPO compares the 4 rollouts of the **same patient**.
Anything constant per patient, such as p₀, cancels inside the group. So the "total
information gain", p_final(y) − p₀(y), trains exactly like a reward of p_final(y).

| Option | Reward term | Pros | Cons |
|---|---|---|---|
| **1 (recommended)** | λ · p_final(y): the HA's final probability of the correct disease | Simple; no trainer change. Bounded in [0, 1], on the same scale as the diagnosis reward (1.0) and test costs (CT 0.128). Clear meaning: a test pays off if it raises p(y) by more than its cost | All credit goes to the whole episode, not to the individual test that helped |
| 2 | λ · Σ max(0, Δp_t(y)): positive per-test gains | Rewards each useful test | A test that misleads and a later test that corrects it both look helpful; not a standard quantity |
| 3 | Per-turn credit (IGPO-style): each DA turn gets its own advantage from its Δp(y) | Closest to the literature; credits the exact turn | Needs a change to the trainer's advantage and loss code, the riskiest part of this codebase; more to test |

**Recommendation: option 1 with λ = 1.**
- It is the cleanest test of the idea that a good HA's belief is a better training signal
  than right or wrong alone.
- It needs one new reward class, which reads the `ha_probs` the environment already passes,
  and no trainer changes.
- If H3b helps and per-test credit looks like the bottleneck, option 3 is the follow-up.

## 5. How to run (H3a; HiPerGator)

```bash
sbatch --export=ALL,ARM=h3a,SMOKE=1 slurm/train_frozen_ha.sbatch   # 4 steps on 2 patients
sbatch --export=ALL,ARM=h3a slurm/train_frozen_ha.sbatch           # resubmit after each 24 h time-out
sbatch --export=ALL,ARM=h3b,SMOKE=1 slurm/train_frozen_ha.sbatch   # H3b smoke
sbatch --export=ALL,ARM=h3b slurm/train_frozen_ha.sbatch           # H3b; can run alongside H3a
$PY scripts/validation_curve.py outputs/h3/h3a_ha0/train           # choose the checkpoint on validation
```

## 6. Limitations

- **One seed for the DA, and the HA is seed 0 only.** H2 showed the two HA seeds differ by 3.8
  points.
- **Different number of DA updates from attempt 2.** 900 DA-only steps are 900 DA updates, while
  attempt 2 alternated phases. Checkpoints are chosen on validation, so the comparison is
  between the best of each run.

## 7. Results (fill in)

Filled in November 17, 2026 (`outputs/h3_stats/`). All runs use the seed-0 frozen HA except
"original" (attempt 2 with its own HA). One test rollout per patient, near-greedy.

| Run | Accuracy (diagnosis) | Accuracy (final hypothesis) | Tests | Cost | Undiagnosed |
|---|---|---|---|---|---|
| Original attempt 2 | 85.4% | 87.1% | 1.19 | $1,259 | 2.1% |
| Step 2 (attempt-2 DA + frozen HA) | 85.4% | 89.2% | 1.20 | $1,285 | 1.7% |
| **H3a**, step 700 | 85.8% | 88.3% | 1.27 | $1,025 | 0.4% |
| **H3b**, step 700 | 74.6% | 78.3% | 2.10 | $536 | 0.4% |

| Comparison (paired, DA's diagnosis) | Difference [95% CI], p | Reading (section 3 rule) |
|---|---|---|
| H3a − step 2 | +0.4 [−2.5, +3.3], p = 1.00 (8 vs 7) | **No detectable gain** |
| H3b − step 2 | −10.8 [−16.7, −5.4], p < 0.001 (10 vs 36) | **Worse** |
| H3b − H3a | −11.2 [−16.2, −6.2], p < 0.001 (7 vs 34) | **Worse** |

**H3b: worse.** The HA-belief reward made the DA cheaper but less accurate.
- H3b orders more tests than step 2 (+0.90 per case) but spends $750 less per case, because
  it has almost stopped ordering CT. Share of patients with each test:

  | Test | Step 2 | H3a | H3b |
  |---|---|---|---|
  | CT | 68.3% | 66.7% | **1.7%** |
  | Ultrasound | 20.0% | 3.8% | 12.1% |
  | Radiograph | 5.0% | 2.1% | 33.3% |
  | Complete blood count | 8.3% | 30.8% | 74.2% |
  | Electrolyte panel | 0.4% | 0.4% | 35.8% |
  | Urinalysis | 1.3% | 1.7% | 27.1% |
  | Liver function panel | 15.0% | 7.9% | 18.3% |
  | Basic metabolic panel | 0.8% | 13.8% | 6.7% |
- The HA's own final hypothesis also falls, from 89.2% to 78.3% (−10.8 [−15.4, −6.7]). The
  tests H3b chooses give the HA less information, so the loss comes from the search itself,
  not from the DA ignoring the HA.
- Diverticulitis precision falls to 39.6% (step 2: 71.9%), and appendicitis recall to 85.4%
  (99.0%).
- Likely mechanism (the test rates confirm the behaviour; the cause is inferred): the
  reward makes a CT worth ordering only when it raises p(y) by more than its cost (0.128).
  After the history, the frozen HA is often already fairly sure, so a CT's measured gain is
  usually smaller than that. Cheap labs cost almost nothing and move p(y) a little. So the
  per-patient reward favours several cheap tests over one CT. The resulting errors are
  rarer per group than the savings, and the diagnosis reward does not outweigh them at 4
  rollouts per patient. The belief term rewards marginal gains in the HA's probability,
  not getting the diagnosis right.

**H3a: no detectable accuracy gain; cheaper (secondary).**
- Accuracy is unchanged: +0.4 [−2.5, +3.3].
- Cost per case is $260 lower ([−$348, −$182], p < 0.001), $1,025 vs $1,285, with slightly
  more tests (+0.07, p = 0.032). H3a keeps CT, nearly drops ultrasound (20.0% → 3.8%) and
  uses blood panels instead.
- The gap between the DA's diagnosis and the HA's final hypothesis shrinks from 3.8 to 2.5
  points (not tested).
- Diverticulitis precision falls (59.0% vs 71.9%), and pancreatitis recall rises (74.1% vs
  68.5%); neither is tested.
- Cost is secondary under the rule, so H3a's reading stays "no detectable gain". But the
  result is a same-accuracy, cheaper system: with this HA, the DA reaches the same accuracy
  for about 20% less.

**Overall.** Training the DA against the frozen HA did not raise accuracy but lowered cost
at equal accuracy. Adding the HA-belief reward traded about 11 points of accuracy for a
further $490 saving, by giving up CT. Per-test credit (option 3) would face the same
problem, since it would use the same measure of gain. Any follow-up needs a reward that
values accuracy more strongly, for example λ < 1 or a larger diagnosis reward.

Run log:
- **Nov 17, checkpoint choice (validation only, before any test run): step 700 for both.**
  Validation accuracy (sampled, 4 rollouts per case):

  | Run | Best step | Accuracy | Undiagnosed | Tests | Cost | Runner-up |
  |---|---|---|---|---|---|---|
  | H3a | 700 | 73.8% | 11.5% | 1.27 | $954 | step 400, 73.3% |
  | H3b | 700 | 70.4% | 7.3% | 1.95 | $663 | step 600, 67.8% |

- **Instability late in training (not yet explained):**
  - **H3a:** tests per case fell to 0.34 at step 800, the drift to zero tests seen in
    attempt 2. At step 900, 95.1% of validation cases were undiagnosed, matching the untrained
    model (step 0: 94.9%). No `training_complete.json` was written, so the collapse guard
    probably stopped the run.
  - **H3b:** 54.3% undiagnosed at step 800, then 19.7% at step 900.
  - **Checked against `jobs.log`: the restarts do not explain it.**
    - H3a's first job ran without a break from step 0 to 875, so the step-800 drift happened
      inside one job.
    - H3a resumed from checkpoint-875, and the collapse guard stopped it at step 900
      (`collapse_detected.json`: 95.1% ≥ 95%).
    - H3b resumed from checkpoint-700; its step-800 dip and the recovery at step 900 both
      came after that restart. A broken resume, such as one that lost the DA's weights, would
      have made step 800 look untrained (about 95% undiagnosed), not 54%, and it would not have
      recovered.
    - So the instability is real training behaviour: DA-only updates at lr 1e-5 do not stay
      stable to 900 steps.
  - Both step-700 checkpoints come before the instability. The H3a adapter is from the
    uninterrupted first job; H3b's is the checkpoint its second job resumed from.
  - **Correction (2026-11-23): the restarts *do* explain most of it.** H4's extension exposed a
    resume bug (`src/utils/peft_resume.py`): with the frozen HA loaded as a second adapter, a
    checkpoint stores it in a subfolder (`frozen_ha/`), and `transformers.Trainer` then reloads
    only that subfolder and never the DA's root adapter. A resumed job therefore continues with
    an untrained DA (optimizer and step restored). H4's training episodes show what that looks
    like: invalid formats jump to about 90% right after the resume, then the DA relearns within
    about 150 steps. So:
    - H3a's collapse at step 900 (95.1% undiagnosed, 25 steps after resuming at 875) is the bug.
    - H3b's step-800 dip (54%) and recovery at 900 is the bug plus relearning. The argument
      above ("would not have recovered") was wrong.
    - H3a's drift to 0.34 tests at step 800 happened inside its first job and is real.
    - **The H3 results are unaffected:** both chosen step-700 adapters were saved before any
      resume.
- H3b orders more tests than H3a (1.95 vs 1.27 at step 700), but they are cheaper ($663 vs
  $954). Its accuracy on diagnosed cases is lower (76.0% vs 83.3%).

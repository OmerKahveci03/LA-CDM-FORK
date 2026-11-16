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

Run log:

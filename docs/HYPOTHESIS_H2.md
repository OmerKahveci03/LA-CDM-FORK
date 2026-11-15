# Hypothesis H2: a Hypothesis Agent trained alone and then frozen is a better diagnoser

Written November 14, 2026, **before any H2 result exists**. The prediction, the primary
comparison and the decision rule below are fixed. Results go in section 9 without
changing sections 1–8.

## 1. Hypothesis

**H2.** A Hypothesis Agent (HA) with its own LoRA adapter, trained alone by supervised
fine-tuning (SFT) on full and partial patient records and then frozen, diagnoses better
than the HA inside our LA-CDM adapter, given exactly the same test results.

## 2. Why (what H1 showed)

H1 (`docs/HYPOTHESIS_H1.md`, section 6) was rejected on both adapters. The trained
Decision Agent (DA) picks useful tests: it beats random tests at the same count by 22.5
points (attempt 2) and 9.6 points (attempt 3). It also matches ordering every test at about
a quarter to a third of the cost. The limit is the HA:

- **The HA's ceiling is low.** With *every* test (arm D), the attempt-2 HA reaches 86.2%
  and the attempt-3 HA 79.2%. LDTL's separately trained diagnosis agent reaches about 94%
  with full records on the same benchmark (arXiv 2604.05116).
- **The HA is unstable.** Two near-identical training runs gave HAs 7 points apart in arm D.
  The attempt-3 HA misses most pancreatitis cases (recall 41–44%) in every arm.
- **Better search cannot help past the ceiling.** In both attempts, the DA's arm already
  matches arm D (A′ − D = +0.8 and −1.7).

Likely cause: in LA-CDM the DA and the HA share one LoRA adapter and are trained in turns
(100 DA steps, 100 HA-SFT steps, 100 HA-calibration steps). Updates for one role can undo
the other's. The HA also learns only from the records the DA happened to produce.
Multi-agent RL work (Multi², private-LoRA multi-agent RL; see `docs/LITERATURE_REVIEW.md`)
finds that per-role adapters beat a shared one.

**Order of work.** The H1 rule said to do H3 (the search reward) first. We do H2 first:
arm D shows a better search cannot lift accuracy above the HA's ceiling, and H3's reward
needs a fixed, good HA to measure information gain against. This change was made after the
H1 results and is stated in H1 section 6.

## 3. What we build (`src/ha/`, `src/train_ha_sft.py`)

| Choice | Value | Reason |
|---|---|---|
| Base model | Llama-3-8B-Instruct, bf16 (same as before) | Only the training procedure changes |
| Adapter | New LoRA, r = 8, α = 16, same target modules | Same capacity as the old shared adapter, so a gain is not just "more parameters" |
| Prompt | `prompts/hypothesis_agent.txt`, built exactly as the environment builds it | Old and new HA see identical inputs; the new HA can later be dropped into the environment |
| Target | `Hypothesis: <disease><eos>` | The confidence is read from the model's probabilities instead (section 4) |
| Training views per patient | History only, all available tests, and 2 random subsets (size uniform in 1..n, random order) | The DA obtains 1–2 results per case, so the HA must handle partial records. LDTL trained on complete records only |
| Token budget | 7,000 (tests dropped at random above it) | The same budget the H1 scripted arms used |
| Optimisation | 1 epoch, LR 1e-4, cosine, 3% warm-up, batch 1 × 8 accumulation, seed 0 | Standard LoRA SFT |
| Checkpoint choice | Lowest validation loss (validation views fixed with seed 10,000), checked every 240 steps | Chosen on validation, never on test |
| After training | Frozen | A stable diagnoser, and a fixed judge for H3 |

Only the train split is used for training and only the validation split for checkpoint
choice. The 240 test patients are untouched until section 6.

## 4. How every HA is measured (`src/score_ha.py`)

- **Fixed test sets.** For each arm, every patient's tests are copied from an H1 run, so
  every HA reads exactly the same records.

  | Arm | Tests per patient | Source |
  |---|---|---|
  | `A_da2` | The trained DA's tests, attempt 2 | `cost_1_attempt2/test_step_0600` |
  | `B_none` | None | `cost_1_attempt2/scripted_none_step_0600` |
  | `C_random2` | Random, attempt 2's count | `cost_1_attempt2/scripted_random_matched_step_0600` |
  | `D_all` | All available | `cost_1_attempt2/scripted_all_step_0600` |
  | `A_da3` | The trained DA's tests, attempt 3 | `cost_1/test_step_0600` |
  | `C_random3` | Random, attempt 3's count | `cost_1/scripted_random_matched_step_0600` |

- **Four-way scoring instead of generation.**
  - The HA's answer is whichever of the four diseases has the highest probability after
    `Hypothesis:`. That always gives a valid disease and a full probability distribution.
  - The confidence is the top probability on the 0–10 scale. The exact per-disease AUROC
    comes from the full distribution; until now we only had an approximation.
- **The same scorer for every HA.** The old attempt-2 and attempt-3 adapters (`old2`,
  `old3`) and the base model (`base`) are re-scored the same way. The comparison is
  between HAs, not between scoring methods.
- The old HAs' *generated* answers from H1 are reported alongside, for reference.

## 5. Outcomes

- **Primary:** accuracy in arm **D_all**, new HA (seed 0) minus `old2`, on the same 240
  patients.
  - Test: exact McNemar, with a 95% paired-bootstrap interval (`scripts/test_stats.py --focus-dir`).
  - `old2` is the HA of our reported best model.
- **Secondary:**
  - the same difference in `B_none`, `C_random2` and `A_da2`, where the partial-record
    training should matter most;
  - new minus `old3` in every arm;
  - exact one-vs-rest AUROC, ECE, Brier score and log loss;
  - per-disease recall, especially pancreatitis.
- **Stability:** a second run with seed 1. Arm D accuracy of seed 0 vs seed 1.
- **Earlier prediction (from H1 section 7):** arm D ≥ 90%.

## 6. Decision rule (fixed now)

| Result for new − `old2` in arm D | Verdict | What we do next |
|---|---|---|
| 95% CI lower bound > 0 (McNemar p < 0.05) | **H2 supported** | Combine the frozen HA with the DA (step 2 below), then H3 |
| 95% CI upper bound < 0 | **H2 rejected: worse** | The joint training helps the HA; find out why before H3 |
| Anything else | **No detectable gain with all tests** | Look at `B_none` and `C_random2`. If the new HA wins both (CI lower bound > 0), H2 is supported for partial records, which is where the DA works; go to step 2. If not, the ceiling is not the training procedure; try more capacity (LoRA r = 64) before H3 |

- **Stability check:** seeds 0 and 1 within 3 points in arm D counts as stable. For
  comparison, the old HAs differed by 7 points.
- **Headroom.** `old2` already scores about 86% in arm D (generated), so the most the new
  HA can gain there is about 14 points. With 240 pairs the 95% interval is roughly ±3–4
  points wide, so a real gain of 2–3 points can come out as "no detectable gain". This
  limitation is stated in advance, and it is why the partial-record arms are named in the
  third row.

## 7. How to run (HiPerGator)

```bash
cd /blue/prismap-ai-core/omerkahveci/LA-CDM-FORK && git pull
PY=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
$PY -m unittest tests.test_ha_sft tests.test_test_stats

# 1. Smoke tests (minutes): 10 training steps; scoring of 5 patients per arm with the old HA
sbatch --export=ALL,SMOKE=1 slurm/train_ha_sft.sbatch
sbatch --export=ALL,HA=old2,LIMIT=5 slurm/score_ha.sbatch

# 2. Score the existing HAs (needs no training)
for h in old2 old3 base; do sbatch --export=ALL,HA=$h slurm/score_ha.sbatch; done

# 3. Train the new HA (primary seed 0; seed 1 for stability), then score it
sbatch --export=ALL,HA_SEED=0 slurm/train_ha_sft.sbatch
sbatch --export=ALL,HA_SEED=1 slurm/train_ha_sft.sbatch
sbatch --export=ALL,HA=new0 slurm/score_ha.sbatch      # after seed 0 finishes
sbatch --export=ALL,HA=new1 slurm/score_ha.sbatch      # after seed 1 finishes
```

**Outputs:**
- Training goes to `outputs/ha_sft/seed_<s>/`:
  - `adapter/` (the frozen HA), `training_complete.json`, `log_history.json`;
  - `sft_examples_*.csv` (which tests each training row showed).
- Scores go to `outputs/ha_sft/scores/<HA>/<arm>/`: `test_episodes.csv`, `ha_probs.csv` and `summary.json`.

**Analysis:**
```bash
$PY scripts/h2_table.py                      # all HAs x all arms -> outputs/ha_sft/scores/h2_table.md
S=outputs/ha_sft/scores
for arm in D_all B_none C_random2 A_da2; do  # primary = D_all
  $PY scripts/test_stats.py --focus-dir "new0 $arm=$S/new0/$arm" \
    --extra "old2 $arm=$S/old2/$arm" --extra "old3 $arm=$S/old3/$arm" --extra "new1 $arm=$S/new1/$arm" \
    --out-dir outputs/ha_sft/h2_stats/$arm
done
```
(Leave out `--extra` for any HA not yet scored.) Ignore the "vs the paper" table in this
output: it compares against the full LA-CDM system, not an HA.

## 8. Known limitations

- **Arm A is not the combined system.** The DA chose arm A's tests while reading the *old* HA's
  hypotheses. Arm A with the new HA therefore means "the old DA's tests, read by the new HA". The
  combined system (step 2) needs the environment to call the new HA, which is built after H2.
- **Scoring vs generation.** The old HAs were used by generating text. Here they are re-scored by
  four-way comparison, and the numbers can differ slightly from H1's. That is why all HAs are re-scored.
- **The prompt has a small tokenisation difference.** The environment passes prompt text to vLLM,
  which may add a second begin-of-text token. The scorer adds none. This applies equally to every
  HA scored here.
- One primary seed; seed 1 only checks stability.

## 9. Results (fill in)

| HA | D_all | B_none | C_random2 | A_da2 | A_da3 | AUROC (D) | ECE (D) | Pancreatitis recall (D) |
|---|---|---|---|---|---|---|---|---|
| base (no adapter) | 75.4% | 61.3% | 63.3% | 82.5% | 77.1% | 0.958 | 10.8% | 68.5% |
| old2 (attempt 2, step 600) | 87.9% | 63.3% | 64.6% | 87.9% | 80.4% | 0.979 | 5.6% | 85.2% |
| old3 (attempt 3, step 600) | 78.3% | 65.8% | 66.7% | 80.4% | 75.8% | 0.976 | 5.4% | 37.0% |
| **new0** (H2, seed 0) | **90.0%** | **78.3%** | **80.8%** | **88.8%** | **85.8%** | **0.989** | **4.5%** | **83.3%** |
| new1 (H2, seed 1) | 93.8% | 76.2% | 81.7% | 91.7% | 86.7% | 0.993 | 1.2% | 88.9% |

All from four-way scoring on the same 240 test patients (`outputs/ha_sft/scores/h2_table.md`;
also has `C_random3` and ECE/Brier for every arm).

| Comparison (arm D) | Difference [95% CI], p | Verdict |
|---|---|---|
| new0 − old2 | +2.1 pts [−2.1, +6.2], p = 0.42 (15 vs 10) | No detectable gain with all tests (row 3) |

| Row 3 check: new0 − old2 | Difference [95% CI], p | Condition (CI lower bound > 0) |
|---|---|---|
| `B_none` | +15.0 pts [+8.7, +21.3], p < 0.001 (51 vs 15) | Met |
| `C_random2` | +16.2 pts [+10.0, +22.1], p < 0.001 (51 vs 12) | Met |

**Verdict (2026-11-15): H2 supported for partial records.** The new HA is not detectably better with all
tests. With no tests or with a few tests, which is how the DA works, it is about 15 points better. Next: step 2
(section 10). **Stability check: not met** (seeds 3.8 points apart in arm D, p = 0.022).

Run log:

- 2026-11-15. Training: seed 0 has 7,676 rows and seed 1 has 7,677; both have 720 validation rows. Both ran
  959 optimizer steps, and both kept the step-720 checkpoint (validation loss 0.0471 and 0.0469).
- Four-way scoring of the old HAs agrees with their generated answers in H1. In arm D, old2 gets 87.9% vs
  86.2% generated, and old3 gets 78.3% vs 79.2%.
- **Stability check: not met.** In arm D, seed 1 minus seed 0 is +3.8 points, against a pre-registered
  limit of 3 points. The old HAs differed by 9.6 points under this scoring, and by 7.0 points when generated.
  The gap is significant: new0 − new1 = −3.7 points [−6.7, −0.8], McNemar p = 0.022 (2 vs 11 discordant).
  The two seeds' validation losses are almost identical (0.0471 vs 0.0469), so validation loss does not tell
  them apart.
- Arm D primary result: new0 − old2 = +2.1 points [−2.1, +6.2], McNemar p = 0.42 (15 vs 10 discordant). The
  interval includes 0, so the decision rule's third row applies, and the verdict depends on `B_none` and `C_random2`.
- new0 − old3 in arm D = +11.7 points [+6.2, +17.1], p < 0.001.
- `B_none`: new0 − old2 = **+15.0 points [+8.7, +21.3]**, p < 0.001 (51 vs 15 discordant). This meets the first
  of the two conditions in row 3.
  - new0 − old3 = +12.5 points [+6.7, +18.3], p < 0.001.
  - new0 − new1 = +2.1 points [−2.1, +6.2], p = 0.44.
  - old2's pancreatitis recall with no tests (81.5%) comes from over-calling pancreatitis: its precision is
    50.6%, against the new HA's 80.0%.
- `C_random2`: new0 − old2 = **+16.2 points [+10.0, +22.1]**, p < 0.001 (51 vs 12 discordant). This meets the
  second condition.
  - new0 − old3 = +14.2 points [+7.9, +20.0].
  - new0 − new1 = −0.8 points [−5.0, +3.3], p = 0.85.
- **The old HA inflated H1's measure of the search.** A − C compares the DA's tests with random tests at the
  same count.
  - With old2 it is 87.9 − 64.6 = +23.3 points.
  - With new0 it is 88.8 − 80.8 = +8.0 points.
  - Most of H1's gap came from the old HA being poor at reading short records, not from the DA choosing
    better tests.
  - H1's verdict still stands, because +8 points is above its 5-point threshold. But a better search has less
    room to help than H1 suggested, which matters for H3.
  - The paired test for this new-HA comparison has not been run.
- With the new HA, random tests add little over no tests: 80.8% vs 78.3%, at $494 per case.
- `A_da2` (the DA's own tests): new0 − old2 = +0.8 points [−2.9, +4.6], p = 0.83 (12 vs 10 discordant).
  - new0 − old3 = +8.3 points [+3.3, +13.3].
  - new0 − new1 = −2.9 points [−5.8, −0.4], McNemar p = 0.065.
  - **On the records the DA actually produces, the new HA does not beat old2.** The DA's 1–2 tests (mostly
    imaging) already make those records easy to read. The new HA's gain is on records without the key test.
  - So step 2 (the old DA with the new HA) should be expected near 88–89% at the same cost, not much higher.
  - The new HA's main value is that it is:
    - robust: no old3-style blind spot;
    - calibrated: Brier 0.066 vs 0.085;
    - a fixed judge for H3's information-gain reward, which needs good readings of partial records.
- The ECE and Brier score in `test_stats.py` are computed from the 0–10 confidence of the top answer, being
  right or wrong. The table above uses the multi-class versions from `h2_table.py`, so the numbers differ.
- In every arm, the new HA has lower ECE and Brier score than every old HA. Its pancreatitis recall is
  about 83%, the same as old2's; old3 had about 37%.

## 10. What follows

- **Step 2: combine.** Run the trained DA with the frozen new HA in the environment: the DA's
  adapter makes decisions and the new HA gives the hypotheses and confidences. Report
  accuracy and cost against attempt 2's 85.4% at $1,259.
- **H3: search reward.** Retrain the DA with a per-test reward of information gain minus
  cost, where information gain is the change in the frozen HA's probability of the correct
  disease. Prediction: tests per case stay above 1 throughout training (no drift to zero
  tests), and accuracy at equal cost beats plain GRPO.

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
| base (no adapter) | | | | | | | | |
| old2 (attempt 2, step 600) | | | | | | | | |
| old3 (attempt 3, step 600) | | | | | | | | |
| **new0** (H2, seed 0) | | | | | | | | |
| new1 (H2, seed 1) | | | | | | | | |

| Comparison (arm D) | Difference [95% CI], p | Verdict |
|---|---|---|
| new0 − old2 | | |

Run log:

## 10. What follows

- **Step 2: combine.** Run the trained DA with the frozen new HA in the environment: the DA's
  adapter makes decisions and the new HA gives the hypotheses and confidences. Report
  accuracy and cost against attempt 2's 85.4% at $1,259.
- **H3: search reward.** Retrain the DA with a per-test reward of information gain minus
  cost, where information gain is the change in the frozen HA's probability of the correct
  disease. Prediction: tests per case stay above 1 throughout training (no drift to zero
  tests), and accuracy at equal cost beats plain GRPO.

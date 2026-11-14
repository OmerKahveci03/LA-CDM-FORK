# Hypothesis H1: does the trained Decision Agent's choice of tests matter?

Written November 12, 2026, **before any H1 result exists**. The prediction, the primary
comparison and the decision rule below are fixed. Results go in section 6 without
changing sections 1–5.

## 1. Hypothesis

**H1.** With our trained adapter, most of the system's accuracy comes from *how many*
tests are ordered and from the Hypothesis Agent (HA) reading them, not from *which*
tests the trained Decision Agent (DA) chooses. Ordering the same number of tests at
random gives nearly the same accuracy.

**Why we suspect it.**
- In LDTL (arXiv 2604.05116, same MIMIC-CDM benchmark), a *random* planner paired with a
  strong frozen diagnosis agent scores 84.8, above LA-CDM's 81.3.
- Our checkpoints that order no tests still reach 67–71% on test (steps 1,800 and 1,920 of attempt 2).
- HA accuracy has never improved during training (65–70% on validation, both attempts).

See `docs/LITERATURE_REVIEW.md`.

## 2. Design

The same 240 test patients; the step-600 adapter; near-greedy decoding (temperature 0.01);
256-token replies; the same HA prompt. Only the DA's turn changes
(`src/environment/scripted_environment.py`).

| Arm | Tests ordered | Prediction |
|---|---|---|
| **A** trained DA | The DA's own choices (the existing test run) | DA diagnosis; for H1, the HA's final hypothesis (**A′**) |
| **B** none | None: history only | HA's hypothesis |
| **C** random, matched | For each patient, as many tests as the trained DA obtained results for in arm A, chosen at random among that patient's available tests | HA's final hypothesis |
| **D** all | Every test with a result for that patient | HA's final hypothesis |

Arm C keeps each patient's test count the same as arm A, so A′ − C isolates *which*
tests were chosen. Tests without a result are never ordered by the scripted arms; they
return "not available" and give the HA nothing. If the HA has no valid hypothesis
when the policy diagnoses, the case counts as undiagnosed.

**Replication.** Run the design on both attempt 2's and attempt 3's step-600 adapters.
Attempt 2 is primary because it is the reported best model.

## 3. Outcomes

- **Primary:** accuracy difference **A′ − C** on the same patients, where both arms are
  scored by the HA's final hypothesis. Test: exact McNemar; 95% paired-bootstrap
  interval. Scoring both arms by the HA keeps the diagnoser the same, so only the test
  choice differs.
- **Secondary:**
  - A′ − B (what tests add at all) and D − A′ (what more tests could add);
  - the same differences scored by the DA's diagnosis (`--score diagnosis`);
  - class-mean accuracy, macro F1 and cost;
  - per-disease accuracy, especially pancreatitis vs cholecystitis.

## 4. Decision rule (fixed now)

Equivalence margin: **±5 accuracy points.**

| Result for A′ − C | Verdict | What we build next |
|---|---|---|
| 95% CI entirely inside [−5, +5] | **H1 supported**: the DA's choice adds little | Strong frozen HA first (H2), then the search reward (H3) |
| CI lower bound > 0 **and** point estimate ≥ 5 (McNemar p < 0.05) | **H1 rejected**: the DA's choices matter | Search reward (H3) first |
| Anything else | **Inconclusive** | Rerun arm C with `SCRIPTED_SEED=1` and `2` and report all three; treat as weak evidence for H1 only if every seed's CI overlaps the margin |

Arm D gives the HA's ceiling, whatever happens with A′ − C:
- **D far below LDTL's 94%:** the HA is a bottleneck, so H2 is needed regardless.
- **D within 2–3 points of A′:** more tests cannot help this HA.

With 240 paired cases, the 95% interval is roughly ±5–7 points wide, so an "inconclusive"
result is possible even if H1 is true. That limitation is stated in advance.

## 5. How to run (HiPerGator)

```bash
cd /blue/prismap-ai-core/omerkahveci/LA-CDM-FORK && git pull
PY=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
$PY -m unittest tests.test_scripted_environment tests.test_environment_end_reasons tests.test_test_stats

# Attempt 2 (primary): arm A already exists in cost_1_attempt2/test_step_0600/
for p in none random_matched all; do
  sbatch --export=ALL,COST_MULTIPLIER=1,ADAPTER_STEP=600,RUN_ROOT=outputs/cost_sweep/cost_1_attempt2,SCRIPTED_POLICY=$p slurm/evaluate_cost_sweep.sbatch
done
# Attempt 3 (replication): arm A already exists in cost_1/test_step_0600/
for p in none random_matched all; do
  sbatch --export=ALL,COST_MULTIPLIER=1,ADAPTER_STEP=600,SCRIPTED_POLICY=$p slurm/evaluate_cost_sweep.sbatch
done
```

**Outputs:** `<run root>/scripted_<policy>_step_0600/`, containing `test_episodes.csv`, `metrics.json`,
`eval_output.log` and the usual files. They are kept apart from `test_step_*`, so the normal test
figures ignore them. Check `eval_output.log` for "skipping … to keep the HA prompt under"
warnings: the `all` arm drops a test if the HA prompt would exceed 7,000 tokens.

**Analysis** (primary, then secondary):
```bash
R=outputs/cost_sweep/cost_1_attempt2
$PY scripts/test_stats.py --run-root $R --focus 600 --skip-other-steps --zero-shot none \
  --score final_hypothesis --out-dir $R/h1_step_0600 \
  --extra "B none=$R/scripted_none_step_0600" \
  --extra "C random=$R/scripted_random_matched_step_0600" \
  --extra "D all=$R/scripted_all_step_0600"
$PY scripts/test_stats.py --run-root $R --focus 600 --skip-other-steps --zero-shot none \
  --score diagnosis --out-dir $R/h1_step_0600_diagnosis \
  --extra "B none=$R/scripted_none_step_0600" \
  --extra "C random=$R/scripted_random_matched_step_0600" \
  --extra "D all=$R/scripted_all_step_0600"
```
Repeat with `R=outputs/cost_sweep/cost_1` for attempt 3. In the paired table, "step 600"
is A′ (or A), so the rows read A′ − B, A′ − C and A′ − D.

## 6. Results (fill in)

| Adapter | A′ (DA tests, HA final) | B none | C random, matched | D all | A′ − C [95% CI], p | Verdict |
|---|---|---|---|---|---|---|
| Attempt 2, step 600 | 87.1% [82.2, 90.7], 1.19 tests, $1,259 | 62.9% [56.6, 68.8], $0 | 64.6% [58.3, 70.4], 1.19 tests, $494 | 86.2% [81.3, 90.0], 9.30 tests, $4,082 | **+22.5 pts [+16.7, +28.3]**, McNemar p < 0.001 (59 vs 5 discordant) | **H1 rejected**: the DA's choices matter |
| Attempt 3, step 600 | 77.5% [71.8, 82.3], 1.50 tests, $1,115 | 67.5% [61.3, 73.1], $0 | 67.9% [61.8, 73.5], 1.50 tests, $623 | 79.2% [73.6, 83.8], 9.30 tests, $4,082 | **+9.6 pts [+4.6, +14.6]**, McNemar p < 0.001 (31 vs 8 discordant) | **H1 rejected** (replicates) |

Run log:
- 2026-11-12: arm B, attempt 2 (smoke test). 240/240 ended in a diagnosis, 0 invalid,
  0 tests. The HA over-predicts pancreatitis (85 predicted vs 54 true) and calls 22 of
  96 appendicitis cases diverticulitis.
- 2026-11-14: attempt 2 analysis (`cost_1_attempt2/h1_step_0600/test_stats.md`, scored by the
  HA's final hypothesis).
  - Other paired differences: A′ − B = +24.2 [+18.3, +30.4]; A′ − D = +0.8 [−2.5, +4.2],
    p = 0.815. The trained DA matches "all tests" with 8.1 fewer tests and $2,823 less
    per case.
  - Arm D reaches 86.2%, well below LDTL's 94%. Per the note in section 4, the HA is a
    bottleneck and H2 is needed regardless.
  - Caveat, not part of the decision rule: at the same count, the random arm spends $494
    vs the DA's $1,259. Random mostly draws cheap lab panels; the DA orders imaging.
    Follow-up check: a fixed "always order imaging" rule.
  - Side finding: the HA's final hypothesis (87.1%) beats the DA's own diagnosis (85.4%).
- 2026-11-14: secondary analysis, scored by the DA's diagnosis
  (`cost_1_attempt2/h1_step_0600_diagnosis/test_stats.md`). It agrees with the primary:
  - A − C = +20.8 [+14.2, +27.5], 65 vs 15 discordant, p < 0.001;
  - A − B = +22.5 [+15.0, +29.6];
  - A − D = −0.8 [−5.4, +4.2], p = 0.864.
  - New: the DA's diagnosis turn loses pancreatitis cases the HA had right. Recall is 64.8%
    by the DA vs 81.5% by the HA's hypothesis, while cholecystitis recall goes from 80.0% to
    84.6%. The DA also leaves 2.1% undiagnosed. Using the HA's final hypothesis as the
    answer gives 87.1% vs 85.4%, with no training (not yet tested for significance).
- 2026-11-14: attempt 3 replication (`cost_1/h1_step_0600/test_stats.md`, scored by the HA's
  final hypothesis).
  - H1 is rejected again: A′ − C = +9.6 [+4.6, +14.6]; A′ − B = +10.0 [+5.4, +15.0];
    A′ − D = −1.7 [−5.4, +2.1], p = 0.523.
  - Arm D, with all tests and no DA involved, is 7 points below attempt 2 (79.2% vs 86.2%).
    That gap matches the A′ gap between the adapters (77.5% vs 87.1%), so attempt 3's lower
    score comes from its HA.
  - Attempt 3's HA has a pancreatitis blind spot: recall is 41–44% in every arm, with or
    without tests. Its confidence tells us little: AUROC_correct is 0.52–0.59, vs 0.75 for
    attempt 2.

**Conclusion (both adapters).** The trained DA's test choice clearly beats random tests at
the same count, and roughly matches "all tests" at about ¼ to ⅓ of the cost. The HA sets the
ceiling (arm D), and that ceiling varies by 7 points between two near-identical training runs.

**Plan change, made after seeing the results and stated openly.** The decision rule in
section 4 says to build H3 (the search reward) first. We do H2 (a frozen, stronger HA) first
instead. Reason: arm D shows better search cannot lift accuracy past the HA's ceiling. H2 also
gives H3 a stable judge to compute information gain with. Section 4 itself is unchanged.

## 7. What follows

- **H2 (diagnoser):** an HA trained alone by SFT on full and random partial records, then frozen, beats the current HA in arms B, C and D. Prediction: arm D ≥ 90%.
- **H3 (search):** with that frozen HA as the judge, rewarding each test by "information gain − cost" stops the drift to no tests (tests per case stay above 1 through training) and beats plain GRPO at equal cost. Two seeds, the same test patients.

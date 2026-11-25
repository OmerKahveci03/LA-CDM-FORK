# Results story: every experiment in one line of reasoning

**The single summary of all results, in order.** Last updated **November 25, 2026**. Update it
whenever a result arrives; the details behind each bullet are in the linked documents.

Unless noted, everything is on the same **240 MIMIC test patients**.

**Statistics used throughout:**
- **Accuracy differences:** exact **McNemar** p-values and **paired bootstrap 95% CIs** (2,000
  resamples of the same patients).
- **Cost, tests and F1:** **paired permutation** tests.
- **Multiple comparisons:** **Holm** correction.
- **A single system's accuracy:** **Wilson** 95% intervals.
- Checkpoints were always chosen on **validation**, and decision rules were **written down before results**.

**Names:**
- **Baseline HA:** jointly trained in the shared adapter.
- **Separately fine-tuned HA:** H2 (called the "new HA" in Week 8).
- **CLM HA:** the contrastive HA.

---

- **Zero-shot (untrained Llama-3-8B):**
  - As the whole system (DA + HA): **8.8%** accuracy. It mostly failed the action format.
  - As an HA alone, scored by probabilities (so format can't fail), on the baseline DA's records:
    **82.5%**; with no tests 61.3%, with all tests 75.4%; ECE 10.8%.
  - **The knowledge is there; the agent behaviour isn't.** (Week 7; `docs/HYPOTHESIS_H2.md`)
- **Baseline (our LA-CDM reproduction, attempt 2, step 600):**
  - The DA and HA share one LoRA adapter and train jointly, as in the paper.
  - Test: **85.4%** [80.4, 89.3], class-mean 84.3% [79.3, 89.2], **$1,259** per case, 1.19 tests,
    HA final hypothesis 87.1%, HA ECE 14.8%, confidence AUROC 0.750.
  - **Matches the paper:** its 81.3% class-mean accuracy and $1,296 cost lie inside our 95% CIs.
  - **The reference for everything below.**
- **Attempt 3** (the baseline recipe with the HA training fix): 75.4%, $1,115, about **10 points below
  the baseline**. One run per version, so this is at least partly run-to-run noise. The HA was the
  weak spot (44% recall on pancreatitis).
- **H1, test choice** (the baseline HA kept; only the tests change; `docs/HYPOTHESIS_H1.md`):
  - With the DA's tests: **87.1%**; with random tests of the same number: 64.6%. **+22.5 points**
    [+16.7, +28.3], McNemar p < 0.001; attempt 3 replicated it (+9.6 [+4.6, +14.6]).
  - All tests: 86.2% at about 3× the cost.
  - **The DA picks good tests. The ceiling is the HA.**
- **H2, the separately fine-tuned HA** (SFT on its own, then frozen) vs the baseline HA, on identical records:
<!-- +15 points from what? +16.2 points from what? -->
  - **No tests: +15.0 points** [+8.7, +21.3]; **random tests: +16.2** [+10.0, +22.1].
  <!-- Not clear how well the H2 HA did. Is it 88.8 or 87.9? -->
  - Baseline DA's tests: 88.8% vs 87.9%; all tests: +2.1 [−2.1, +6.2], not significant.
  - ECE 4.5% vs 5.6%.
  - A pre-registered stability check failed: two seeds differ by 3.7 points on all tests (p = 0.022).
  - **A much better diagnoser from partial records.**
- **Step 2** (baseline DA + separately fine-tuned HA, no retraining; `docs/STEP2_COMBINED.md`) vs baseline:
  - DA accuracy **+0.0** [−3.8, +3.7]; HA final hypothesis +2.1 [−1.7, +5.8]; cost +$27 [−30, +91].
    None significant.
  - Calibration much better: ECE 14.8% → 5.2%, AUROC 0.750 → 0.936.
  - **A better HA alone doesn't make the system more accurate.**
- **H3a** (a new DA trained against the frozen separately fine-tuned HA; step 700 chosen;
  `docs/HYPOTHESIS_H3.md`):
  - Test **85.8%**, HA final 88.3%, 1.27 tests, **$1,025**.
  - vs step 2: accuracy +0.4 [−2.5, +3.3] (p = 1.00), **cost −$260** [−348, −182] (p < 0.001).
  - The direct MIMIC comparison with the baseline is pending (A5); on external data it was
    significantly cheaper (below).
  - **The best system so far: same accuracy, about 20% cheaper.**
- **H3b** (H3a plus a reward for raising the HA's belief; step 700):
<!-- -10.8 points from h3a I presume? -->
  - 74.6%, $536, 2.10 tests. **−10.8 points** vs step 2 [−16.7, −5.4] and −11.2 vs H3a [−16.2, −6.2],
    both p < 0.001.
  - It nearly stopped ordering CT (1.7% vs 66.7%).
  - **That reward is gamed with cheap, uninformative tests.**
- **H4a, the CLM HA alone** (frozen Llama + separately fine-tuned adapter + contrastive heads;
  `docs/HYPOTHESIS_H4.md`):
  - Baseline DA's records: **89.2%** vs the separately fine-tuned HA's 88.8%: +0.4 [−1.3, +2.1],
    **non-inferior** (margin −3). vs the baseline HA: +1.3 [−2.5, +5.0], p = 0.66.
  - ECE 6.3% vs 4.6%.
  - Accuracy mostly **inherited** from the separately fine-tuned adapter: on plain Llama the CLM HA
    reached 85.4%.
- **H4b** (baseline DA + CLM HA, no retraining) vs baseline:
  - **82.9% vs 85.4%:** −2.5 [−6.2, +1.2], p = 0.31; cost $1,235 vs $1,259, not significant.
    Also −2.5 vs step 2 (p = 0.15).
  - CLM confidence much better than the baseline HA's (AUROC 0.915 vs 0.750).
- **H4c** (a new DA trained with the CLM HA, 700 steps):
  - **Joint** (heads learn every step; step 600): **84.2%**, HA final 87.5%, $1,189.
  - **Frozen** (heads fixed; step 400): **82.1%**, $1,434.
  - Joint − frozen **+2.1** [−2.5, +6.7], p = 0.47: **no detectable gain** from joint learning. Joint
    was cheaper (−$245, p < 0.001), but the two arms were tested at different steps.
  - Joint vs H3a: −1.7 [−5.4, +2.1], +$164 (p < 0.001). Direct comparison with the baseline pending (A5).
  - Joint matches the paper's 81.3% class-mean accuracy at lower cost.
- **External validation** (Edward's 152 published case reports, not MIMIC; each system run once;
  paired against the baseline; `edward/README.md`):
  - **Baseline 86.8%.**
  - **H3a 89.5%:** +2.6 [−3.3, +7.9], not significant; **−$298** [−401, −203], **p < 0.001**. The cost
    result replicates on new data.
  - **H4b 87.5%:** −0.7, not significant.
  - **H4c joint 78.9%:** −7.9 [−15.1, −1.3], McNemar p = 0.050, Holm p = 0.15. A likely drop that isn't
    established; its HA dropped too.
  - The set is probably easier than MIMIC (radiology names the answer for 37% of patients), so the
    claim is "generalises at least as well", not "better".
- **A pattern across every system,** MIMIC and external:
  - The HA's final hypothesis is **as accurate as or more accurate than** the DA's diagnosis: 87–92% vs 79–90%.
  - The DA overrides a correct HA more often than it corrects a wrong one.
  - On external data, "always use the HA's answer" was 2.6–3.3 points better in every system. Each
    difference is small and not significant alone (for example 4 vs 0, p ≈ 0.13). Tested properly
    on MIMIC in A1.
- **The resume bug** (found during H4): restarting a two-adapter run silently reset the DA to untrained.
  - It **explains H3's late "instability"**, correcting Week 8.
  - No reported result is affected: every chosen checkpoint was saved before any restart.
- **Training longer** (exploratory; frozen arm extended to 1,900 steps; `docs/HYPOTHESIS_H4.md`):
  - Validation rose from 70.0% (best within 700 steps) to **76.4% at step 1,100**. Test pending.
  - The joint arm's accidental repeat of steps 100–700 differed from the original by up to 10.7
    points at the same step: direct evidence of **run-to-run noise**.
- **Pending, pre-registered** (`docs/FINAL_ANALYSES.md`):
  - A1: deferral rules on MIMIC.
  - A2: confidence bounds from a CLM head ensemble.
  - A3: combined HAs.
  - A4: the DA with the HA hidden, and a weak DA with the CLM HA.
  - A5: every system against the baseline.

## Where the details are

| Topic | Document |
|---|---|
| H1–H4 plans, rules, full results | `docs/HYPOTHESIS_H1.md` … `docs/HYPOTHESIS_H4.md`, `docs/STEP2_COMBINED.md` |
| Final analyses A1–A5 | `docs/FINAL_ANALYSES.md` |
| External validation | `edward/README.md` |
| Ideas not yet run | `docs/NEXT_EXPERIMENTS.md` |
| Weekly reports | `weekly_reports/week_7` … `week_9` |

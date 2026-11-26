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
- **H2, the separately fine-tuned HA** (SFT on its own, then frozen) vs the baseline HA, each alone
  on identical records (separately fine-tuned **vs** baseline HA, then the difference):
  - **History only (no tests): 78.3% vs 63.3%, +15.0 points** [+8.7, +21.3].
  - **Random tests (same number as the baseline DA ordered): 80.8% vs 64.6%, +16.2** [+10.0, +22.1].
  - The baseline DA's tests: 88.8% vs 87.9%, +0.9 (not significant).
  - All tests: 90.0% vs 87.9%, +2.1 [−2.1, +6.2], not significant.
  - ECE (all tests): 4.5% vs 5.6%.
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
  - **vs baseline directly (A5): accuracy 85.8% vs 85.4%, +0.4 [−3.8, +4.6], not significant;
    cost $1,025 vs $1,259, −$234 [−305, −170], Holm p = 0.003.** On external data it was also
    significantly cheaper (below).
  - **The best system: same accuracy, about 19% cheaper than the baseline.**
- **H3b** (H3a plus a reward for raising the HA's belief; step 700):
  - 74.6%, $536, 2.10 tests.
  - vs step 2 (85.4%): **−10.8 points** [−16.7, −5.4]; vs H3a (85.8%): −11.2 [−16.2, −6.2];
    vs baseline (85.4%, A5): −10.8 [−16.7, −5.0], Holm p = 0.004. All p < 0.001 before correction.
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
  - Joint vs H3a: −1.7 [−5.4, +2.1], +$164 (p < 0.001).
  - **vs baseline (A5):** joint 84.2% vs 85.4%, −1.2 [−6.2, +4.2]; frozen 82.1% vs 85.4%, −3.3
    [−8.3, +1.7]. Neither significant. Frozen costs $175 more (p < 0.001); joint $70 less (p = 0.075).
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
    difference is small and not significant alone (for example 4 vs 0, p ≈ 0.13).
- **A1, "defer to the HA" on MIMIC** (pre-registered; `docs/FINAL_ANALYSES.md`): using the HA's final
  hypothesis as the diagnosis was **ahead of the DA in all 7 systems** (+1.7 to +6.7 points) and never
  behind:
  - baseline 85.4 → 87.1; step 2 85.4 → **89.2**; H3a 85.8 → 88.3; H3b 74.6 → 78.3; H4b 82.9 → 88.8;
    H4c joint 84.2 → 87.5; H4c frozen 82.1 → 88.8.
  - Significant after Holm only for **H4c frozen (+18/−2 patients, Holm p = 0.003)**; H4b was
    borderline (Holm p = 0.056).
  - A validation-chosen confidence threshold deferred almost always, so it added nothing beyond "defer".
  - **The weak DA (A4b; frozen arm at step 200):** DA 78.3%, but its HA's final hypothesis 89.2%,
    a 10.9-point gap. Even a weak DA gathers tests from which the HA reaches about 89%.
  - **Takeaway:** the system's last step, the DA's final decision, loses accuracy; the HA's answer is
    the better diagnosis in every system.
- **A2, confidence bounds from 10 CLM heads** (Xuefeng's LCB/UCB idea):
  - Ensembling keeps accuracy (89.2% vs 88.8%) and **improves calibration** on every record set
    (baseline DA's records: ECE 5.4% vs 7.0%).
  - The bounds split records cleanly: "decisive" ones 93.7% correct, "non-decisive" ones 36.8%.
    But they flag errors **no better than the plain confidence** (AUROC 0.936 vs 0.933, n.s.).
  - "Get all tests when not decisive" (offline sketch): 89.2% → 90.0% (n.s.) for +$235 per case.
    The HA's remaining errors are not fixed by more tests (all tests: 90.4% at $4,082).
- **A3, combining HAs** (equal-weight averages of the HAs' probabilities): **no combination beat the
  separately fine-tuned HA** on any record set (on the baseline DA's records, best 89.6% vs 88.8%,
  n.s.). Averaging it with the baseline HA gave the best calibration (ECE 1.9% vs 3.5%).
- **A4a, the DA with the HA hidden** (the HA still computed from the DA's tests, but not shown to the DA):
  - Baseline DA: **85.4% → 59.2%** (−26.2 [−32.9, −20.0], p < 0.001), with 25% of patients undiagnosed.
    H3a's DA: **85.8% → 73.3%** (−12.5 [−17.5, −7.1], p < 0.001).
  - But the HA's answer from those same tests stayed at **89.6% / 87.5%**, unchanged.
  - **The DAs rely on the HA to diagnose, not to choose tests.** The HA's value is its diagnosis.
    (Caveat: these DAs were trained with the HA, so this measures reliance, not a DA trained without one.)
- **A6, "H3a + defer" vs the baseline** (H3a's DA chooses the tests; its frozen separately fine-tuned
  HA's final hypothesis is the diagnosis; both parts fixed before this test):
  - MIMIC: **88.3% vs 85.4%, +2.9 [−1.7, +7.5]** (+20/−13 patients), p = 0.30. External: 92.1% vs
    86.8%, +5.3 [0.0, +11.2], p = 0.12.
  - **Not significantly more accurate**, but leaning positive on both data sets, at **$1,025 vs $1,259**
    (−$234, significant). The strongest system we have: at least as accurate as the paper's method and
    about 19% cheaper.
- **A5, every system vs the baseline:** **no system is more accurate than the baseline.** Only H3b is
  less accurate (−10.8, significant). H3a is the only one significantly cheaper (−$234).
- **The resume bug** (found during H4): restarting a two-adapter run silently reset the DA to untrained.
  - It **explains H3's late "instability"**, correcting Week 8.
  - No reported result is affected: every chosen checkpoint was saved before any restart.
- **Training longer** (exploratory; frozen arm extended to 1,900 steps; `docs/HYPOTHESIS_H4.md`):
  - Validation rose from 70.0% (best within 700 steps) to **76.4% at step 1,100**.
  - **On test, step 1,100 scored 78.8% while ordering no tests at all ($0 per case).** The DA had
    learned to diagnose from the history alone, trading accuracy for zero cost: −7.1 vs H3a (Holm
    p = 0.048), −6.7 vs the baseline (p = 0.052). **Training longer did not help; it drifted to a
    degenerate zero-test policy** that choosing on validation accuracy alone did not catch.
  - The joint arm's accidental repeat of steps 100–700 differed from the original by up to 10.7
    points at the same step: direct evidence of **run-to-run noise**.
- **Pending:**
  - **H3c** (`docs/HYPOTHESIS_H3C.md`): a new DA trained against the frozen HA where the HA's hypothesis
    is always the diagnosis (the DA never overrides it). Primary comparison: vs "H3a + defer".

## Where the details are

| Topic | Document |
|---|---|
| H1–H4 plans, rules, full results | `docs/HYPOTHESIS_H1.md` … `docs/HYPOTHESIS_H4.md`, `docs/STEP2_COMBINED.md` |
| Final analyses A1–A5 | `docs/FINAL_ANALYSES.md` |
| External validation | `edward/README.md` |
| Ideas not yet run | `docs/NEXT_EXPERIMENTS.md` |
| Weekly reports | `weekly_reports/week_7` … `week_9` |

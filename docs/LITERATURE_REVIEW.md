# Literature review: improving the LA-CDM agentic loop

November 12, 2026. Covers work up to mid-2026. The summaries come from the papers' abstracts and
HTML versions. LDTL and IGPO were read in more detail; the rest come from abstracts and secondary
summaries. Numbers are the authors' own and have not been replicated unless stated.

## 1. What we need to fix (from attempts 2 and 3)

| # | Failure | Evidence |
|---|---|---|
| F1 | **The DA drifts to ordering no tests** | Tests per validation case 1.7 → ~0.1 by step 1,300 in both attempts. With no tests the model still gets 67–71% on test; at step 600, ordering tests is worth ~15 points |
| F2 | **Class bias swings between checkpoints** | Attempt 3: over-predicts diverticulitis at step 600 and pancreatitis at 900; pancreatitis recall 39% → 75% on validation. Test accuracy differs by 10 points between two runs of nearly the same code (85.4% vs 75.4%) |
| F3 | **HA confidence is uninformative** | AUROC (confidence → correct) 0.73–0.75 trained vs 0.78 zero-shot; 0.50 after step 1,500. ECE rises to 20–30 |
| F4 | **HA accuracy does not improve** | 65–70% on validation in both attempts, even after the HA-SFT fix |
| F5 | **Weak, noisy learning signal** | 4 rollouts per patient (paper: 8); one scalar reward per episode; once the 4 rollouts agree on the diagnosis, test cost is the only reward difference left |

## 2. Relevant work

### Sequential clinical diagnosis
- **LDTL** (Shen et al., arXiv 2604.05116, April 2026), on **the same MIMIC-CDM benchmark** with Llama-3-8B:
  - **Setup.** A diagnostic agent is first fine-tuned on *complete* patient information, then **frozen**. A separate planner chooses the next test category (lab, physical exam or imaging). It is trained toward a posterior that favours actions with high information gain, defined as IG = log p(y | h_{t+1}) − log p(y | h_t), where y is the true label. The episode stops when the diagnostic agent says it is confident. The planner is trained by supervised KL matching, not RL.
  - **Results.** Mean accuracy 93.4 / macro F1 91.7, against **LA-CDM 81.3 / 84.1**.
  - **Striking baselines.** A **random planner already reaches 84.8**, and the diagnostic agent fine-tuned on full information reaches 94.2.
  - **Caveats.** No seeds reported. The action space is coarse (3 categories versus our 12 tests). Test counts per case are not reported. It's unclear whether LA-CDM was rerun or its numbers were copied.
  - **Relationship to LA-CDM.** A different group (Shen and Song, UConn; Liu, Texas A&M; Min, NEC Labs America), and an arXiv preprint. It treats LA-CDM as a baseline and does not build on it.
    - **Shared with LA-CDM:** MIMIC-CDM (Hager et al., 2024), Llama-3-8B with LoRA, and the two-agent idea. It cites LA-CDM only for the standardised test-name mapping.
    - **Not described:** reuse of LA-CDM's code, test list or prices; test costs; what choosing "labs" reveals (results are just "appended to the patient context"); whether LA-CDM was rerun. Its LA-CDM row (81.3 accuracy / 84.1 macro F1) matches LA-CDM's published accuracy but not the macro F1 we recorded (81.3).
    - **Other differences:** its own 70/10/20 split, and no code link.
- **DiagGym / DiagAgent** ([arXiv 2510.24654](https://arxiv.org/abs/2510.24654)): trains a diagnosis agent with GRPO in a simulator built from MIMIC-IV. The reward combines diagnosis accuracy with **how well the ordered exams match the clinicians' real orders**. The authors' ablation found both parts were needed.
- **MAI-DxO / SDBench** ([arXiv 2506.22405](https://arxiv.org/abs/2506.22405), Microsoft): no training; prompts one model to play a panel of doctor roles. A "Dr. Hypothesis" role keeps a probability-ranked list of candidate diagnoses, other roles choose tests by expected value per cost, and a checker audits the reasoning. Reports 79.9% accuracy at $2,397 per case versus o3 alone at 78.6% / $7,850. The cases are hard NEJM teaching cases.
- **DoctorAgent-RL** ([arXiv 2505.19630](https://arxiv.org/abs/2505.19630)): multi-turn doctor–patient dialogue trained with RL against a simulated patient and an evaluator that scores several criteria.
- **Salus** (ICML 2026, [poster](https://icml.cc/virtual/2026/poster/64732)): multi-agent RL for strategic diagnostic testing. Only the abstract was seen.

### Rewarding each step of a multi-turn agent (credit assignment)
- **IGPO** ([arXiv 2510.14967](https://arxiv.org/html/2510.14967v1), ICLR 2026):
  - **Per-turn reward:** the increase in the policy's own probability of the ground-truth answer after that turn, scored with the answer forced (teacher forcing).
  - **Combined with the outcome reward:** all turn rewards in a group are z-normalised together, then summed into a cumulative advantage per turn.
  - **Results (Qwen2.5-7B, search QA):** F1 58.7 vs GRPO 51.9. Using only the information-gain reward gave 52.8; the gain comes from combining it with the outcome reward. It needs the ground-truth answer, which we have during training.
- **GiGPO** ([arXiv 2505.10978](https://arxiv.org/html/2505.10978v2), NeurIPS 2025): adds step-level advantages by grouping actions taken from the **same state** across rollouts, which it calls "anchor states". No critic and no extra rollouts. Reports +12% on ALFWorld and +9% on WebShop over GRPO.
- **A²TGPO, TIPS, turn-level reward design** ([arXiv 2505.11821](https://arxiv.org/pdf/2505.11821v2)): further variants of per-turn rewards, either normalised by turn position or scored by a teacher model.

### Choosing what to ask by expected information gain (training-free)
- **Uncertainty of Thoughts** ([NeurIPS 2024](https://arxiv.org/html/2402.03271v3)): simulates possible answers and picks the question that best reduces the remaining uncertainty. Includes a simplified medical diagnosis task.
- **BED-LLM** ([arXiv 2508.21184](https://arxiv.org/html/2508.21184v2), ICLR 2026): sequential Bayesian experimental design. It estimates expected information gain from the LLM's own probabilities over hypotheses, with a lower-variance estimator.

### Stopping too early or searching too much
- **Adaptive information control** ([arXiv 2602.01672](https://arxiv.org/html/2602.01672)) and **SAAS** ([arXiv 2605.29796](https://arxiv.org/pdf/2605.29796)): add signals that penalise stopping too early and searching too much, respectively. Survey: [arXiv 2510.16724](https://arxiv.org/pdf/2510.16724v2).

### Calibration through RL
- **RLCR** ([arXiv 2507.16806](https://arxiv.org/abs/2507.16806), ICLR 2026):
  - **Method:** the answer and its stated confidence are trained **together**, with reward = correctness + **Brier score**.
  - **Why Brier:** the authors argue a scoring rule must be both proper and *bounded*. **Log-loss is unbounded and can encourage degenerate solutions.**
  - **Results:** reported ECE 0.37 → 0.03 on HotPotQA with no loss of accuracy.
  - **Relevance:** our calibration phase uses the log-score.

### Sharing one adapter across roles
- **Multi²** ([arXiv 2606.03698](https://arxiv.org/pdf/2606.03698)): replacing role-specific adapters with one shared adapter "results in a significant degradation".
- **Small LLMs as spatial multi-agent policies** ([arXiv 2608.01425](https://arxiv.org/pdf/2608.01425)): attributes part of a training collapse to the shared setup and fixes it with a **private LoRA per agent** on a frozen shared base.
- **PettingLLMs** ([docs](https://pettingllms-docs.readthedocs.io)): describes three levels — shared model, separate LoRA per role, separate models — with separate LoRA as the middle ground.

### GRPO itself
- **DAPO** ([arXiv 2503.14476](https://arxiv.org/pdf/2503.14476v2)): **dynamic sampling** drops groups whose rewards all tie, because they give no signal. **Clip-higher** loosens the upper clipping bound to keep exploration alive.

## 3. What this suggests for us

| Idea | Source | Fixes | Cost for us |
|---|---|---|---|
| **A. Train a strong HA separately and freeze it** (SFT on full and partial information), then train only the DA | LDTL, Multi², private-LoRA MARL | F2, F3, F4 | Low–medium: SFT takes hours; needs a second adapter or a merged frozen HA |
| **B. Information-gain reward per test for the DA**: log p_HA(y \| after test) − log p_HA(y \| before test), minus cost | IGPO, LDTL | F1, F5 | Medium: one scoring pass per test. Per-episode version: a new reward function. Per-turn version: trainer change |
| **C. Step-level grouping** (all 4 rollouts start from the same history, so their first actions are directly comparable) | GiGPO | F5 | Medium: trainer change |
| **D. Brier-score calibration, trained jointly** instead of a separate log-score phase | RLCR | F3 | Low: swap the reward in the calibration phase |
| **E. Drop groups with no signal; 8 rollouts if memory allows** | DAPO, paper config | F5 | Low |
| **F. Choose tests at inference by expected information gain** | UoT, BED-LLM, MAI-DxO | F1 | Medium–high: needs simulated test results; best as a comparison |

**The idea that combines them: A + B.** Train the HA once, by SFT, then freeze it. Then train the DA with GRPO, using the frozen HA's log-probability gain on the true disease as a per-test reward, minus the test's cost.

- **Why it hangs together:** a frozen HA gives the DA a fixed, accurate judge.
  - That removes the shared-adapter interference behind F2 and F3.
  - It makes the information-gain reward stable, because the judge doesn't move during DA training.
  - Each test is then paid by what it reveals about the true disease, which counters F1. That matters most for pancreatitis vs cholecystitis, where a liver panel or ultrasound should decide it.
- **How it differs from prior work:**
  - LA-CDM trains both agents in one adapter with no per-test reward.
  - LDTL distils its planner by supervised learning over 3 coarse action categories rather than RL over 12 priced tests.
  - IGPO uses the policy's own probabilities for the reward, not a separate frozen judge.

## 4. Cheap checks before any new training (evaluation only)

LDTL's random planner beating LA-CDM raises a question about our own setup: how much does the trained DA add over the HA it is paired with? Three evaluation-only runs, about 3 h each on the test split, would answer it.

1. **Random planner + our step-600 HA.** Order k random tests, where k matches the trained DA's average, then diagnose.
2. **All tests + our HA.** The upper bound on what tests can add, and its cost.
3. **No tests + our HA** (history only). We already have a rough version from steps 1,800 and 1,920: 67–71%.

If the random planner comes close to our trained DA, the bottleneck is the HA (F4), and idea A comes first. If it is far below, the DA matters, and idea B comes first.

## 5. Lessons from LDTL for our loop

1. **Strong, separate diagnoser first.** Their random planner plus a frozen, fine-tuned diagnosis agent (84.8) is about as good as our whole trained system (84.3). Train our HA by SFT on full *and* random partial records, in its own adapter, then freeze it.
2. **A signal at every step.** Reward each test by the frozen judge's information gain on the true disease **minus its cost**, instead of one episode-level reward dominated by cost.
3. **"Try every option" is affordable.** 12 tests means 12 scoring passes per step. Build the same answer key over our 12 tests, warm-start the DA with supervised learning on it, then refine with RL.
4. **Stopping needs a calibrated signal.** Use the frozen judge's probability of its top disease, not the stated 0–10 confidence (AUROC 0.50 late in training).
5. **Cost is the gap they left.** A reward of "information gain − λ·cost" plus a sweep over λ gives an accuracy–cost curve, which revives the multiplier sweep.
6. **Report context baselines,** as they do: history only, all information, random planner, and the fine-tuned diagnosis agent with all information. Avoid their weaknesses: report cost, keep 12 individual tests, run several seeds.

**Plan:** (1) baselines, evaluation only; (2) a frozen SFT judge trained on full and partial records, with its ceiling measured; (3) supervised warm start of the DA on the information-gain-minus-cost key; (4) GRPO with the per-test information-gain-minus-cost reward plus the outcome reward; (5) compare on the same test patients with 2 seeds. Steps 1–2 are cheap and decide whether the rest is worth it.

## 6. Caveats

- LDTL's and IGPO's results come from single runs without reported variance. LDTL's comparison with LA-CDM may use different splits.
- Our own results vary by ~10 points between runs (F2), so every comparison needs 2 seeds or more.

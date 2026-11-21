# Alternative to H4 (not pursued): the HA as a *context-editing* language model

> **Not the current H4.** Written November 21, 2026, from the "Context Language Models" paper, before
> the meeting's "CLM" was identified as a **Contrastive** Language Model (Contrastive-LM/CLM).
> The current H4 is in `docs/HYPOTHESIS_H4.md`. This plan and its code (`src/clm/`,
> `slurm/clm_*.sbatch`, `docs/CLM_RUNBOOK.md`) are kept as an alternative; nothing has been run.


**DRAFT, November 21, 2026.** It replaces an earlier draft based on a misreading ("CLM" as
a contrastive model, kept in `docs/CONTRASTIVE_PRIMER.md`).

- **Source:** "Context Language Models" (`docs/literature/22968_Context_Language_Models.pdf`).
- **Background reading:** `docs/CLM_PRIMER.md`.

The decisions in section 7 come first. Sections 1–6 are then fixed **before any H4 result
exists**.

## 1. Hypothesis

An HA that **maintains its own patient-state note**, editing its context instead of only
appending to it, diagnoses better than an append-only HA. A DA trained **together** with
such an HA, as two separate adapters, beats a DA trained against a frozen append-only HA
(H3a).

| Part | Question | Training |
|---|---|---|
| **H4a** | Does context editing help the HA at all? | None: same model, editing vs append-only |
| **H4b** | Does a *trained* CLM HA beat the H2 HA (seed 0)? | HA only, on fixed records |
| **H4c** | Does learning together beat a frozen HA? | DA and CLM HA, together |

## 2. Why

- **H2 and H3:** the HA sets the ceiling. With all tests, records reach 7,000 tokens, mostly
  lab tables. A compact note of key findings and ruled-out diseases may let the HA reason
  more reliably.
- **H3 test results:** the DA's diagnosis trails the HA's last guess by 2.5–4 points.
  Co-adaptation, meaning learning together, is a candidate fix. Separate adapters avoid the
  interference seen with LA-CDM's shared adapter.
- **The paper:** context editing beats fixed harnesses on long agentic tasks, and RL with
  stepwise GRPO improves a 9B model's editing.

## 3. Stages (each gates the next)

| Stage | What | GPU cost | Gate |
|---|---|---|---|
| **0. Learn** | `docs/CLM_PRIMER.md` | none | — |
| **1. Measure context pressure** | From saved transcripts: the HA's context length per turn and its growth with tests | none (CPU) | Shows where editing could matter |
| **2. H4a, out of the box** | The untrained Llama-3-8B as HA, **with** vs **without** editing. Test results are fed one at a time from the fixed H2 record sets. Evaluation only. | a few hours | Section 5 |
| **3. H4b, train the CLM HA alone** | The DA is fixed: replay the fixed record sequences. The HA learns to edit (GRPO) and to diagnose from its own context (SFT). | ~1 day | Section 5 |
| **4. H4c, learning together** | A new DA (GRPO) and the CLM HA from stage 3, trained together | ~2 days | — |

**Independent side experiment (cheap):** the success-gated cost for the DA, below. It can run
any time.

## 4. Design

### 4.1 What the CLM HA does at each turn

1. **Context:** its context from the last turn + the new test result, appended by default.
   On the first turn, the context is the instructions + the patient history.
2. **The HA may edit** (D1). **Recommended:** it either does nothing (append-only) or emits a
   `<rewrite>…</rewrite>` block that replaces its whole context with new text (its notes).
   - The model decides **whether and when** to edit, as in the paper (Eq. 2).
   - The rewrite is capped at about 400 tokens.
3. **The hypothesis** is read from the resulting context by **four-way scoring**, as in H2:
   the probability of each disease after "Hypothesis:".
   - The confidence, the DA's transcript text and `ha_probs` for H3's machinery all stay the
     same.
   - Scoring, not generation, keeps the answer always valid.

The DA's side is unchanged. It sees "The current hypothesis is: X (confidence: c)".

### 4.2 Stage 2 (H4a): out-of-the-box comparison

- **Same model** (the untrained Llama-3-8B) **and the same records**. The only difference is
  that one arm may edit its context and the other only appends. This matches the paper's
  "CLM vs base harness" comparison.
- **Record sequences:** the "baseline DA's tests" and "all tests" sets from H2. Their results
  are fed one per turn, in the order the DA ordered them; for "all tests", in a fixed random
  order.
- **Measures:**
  - the accuracy of the final hypothesis;
  - context tokens per turn;
  - how often and how much it edits;
  - a manual review of 20 notes for lost facts.

### 4.3 Stage 3 (H4b): training the CLM HA alone

- **Weights:** a new LoRA (r = 8) on Llama-3-8B (D2), separate from any DA adapter.
- **Training signal, every step** (one patient, 4 rollouts of the HA over the same test
  sequence):
  - **Edits, by stepwise GRPO.**
    - **Outcome:** 1 if the final hypothesis is correct.
    - **Efficiency:** the paper's success-gated advantage on context tokens. Only correct
      rollouts are ranked; shorter contexts gain.
    - The advantage goes to every HA call in the rollout.
  - **Diagnosis, by SFT:** cross-entropy on "Hypothesis: <true disease>" given the HA's own
    context at every turn, as in H2. The HA learns to diagnose from its own notes.
  - **A replay batch** of mixed-disease records in each SFT update, so updates from one
    patient do not swing the HA (the attempt-3 lesson).
- **Checkpoint:** chosen on validation (final-hypothesis accuracy), never on test.

### 4.4 Stage 4 (H4c): learning together

- Each step: DA rollouts with the **current** CLM HA, then a DA GRPO update, then the HA's
  updates (section 4.3), all on that step's rollouts.
- **Two separate adapters**, optimisers and learning rates. Neither update touches the other's
  weights.
- **DA settings:** as in H3a (original reward, 4 generations, 2,048-token transcripts), 700
  steps (D4).
- **Comparison arm:** H3a already exists (a DA trained against the frozen append-only H2 HA).
  If budget allows, add a frozen-CLM-HA arm to separate "CLM" from "learning together".

### 4.5 Engineering (D3)

The HA now **generates** text (its rewrites), so it needs fast generation during training.
- Today vLLM serves the DA by **merging** the DA's LoRA into the base weights, so an HA LoRA
  cannot simply be added on top.
- **Recommended:** serve both roles from vLLM as **separate LoRA requests** on the unchanged
  base model.
  - After each update, save each adapter (small) and point vLLM at the new copy, instead of
    merging weights.
  - This is new trainer code in a new file. The current trainer stays as it is.
- **Fallback:** generate the HA's rewrites with Hugging Face `generate` on GPU 0. Simpler, but
  probably 10–20 hours slower per run.

### 4.6 Side experiment: success-gated test cost for the DA

- **Change:** the DA's test-cost term only ranks **correct** rollouts within a group,
  normalised as in the paper's Eq. 6. Wrong and invalid rollouts get no cost term, and groups
  with fewer than 2 correct rollouts get none.
- **Why:** it removes the "cheapest wrong answer wins" signal behind the drift to no tests and
  H3b's gaming.
- **Run:** one H3a-style run (frozen H2 HA, 700 steps). **Compare** with H3a, paired on test.
- **Code:** a new reward class or an advantage hook, in a new file.

## 5. Comparisons and decision rules (to be fixed with section 7)

All tests are paired on the 240 test patients, with exact McNemar and paired-bootstrap 95%
intervals.

- **H4a (gate):** editing minus append-only, same untrained model, final-hypothesis
  accuracy on the "all tests" sequences.
  - If the CI lower bound is above −3 points, continue (editing is not harmful).
  - Otherwise, stop, or revise the prompt once and rerun.
- **H4b:** the trained CLM HA minus the H2 HA (seed 0), on the same record sets.
  - If the CI lower bound is above −3 points, it is not worse: continue.
  - If the CI is entirely above 0, it is better.
- **H4c (primary):** the DA's diagnosis accuracy, joint minus H3a.
  - A CI entirely above 0 means supported; a CI that includes 0 means no detectable gain; a
    CI entirely below 0 means worse.
  - **Secondary:** cost; the gap between the DA and the HA; calibration; context tokens.
- **Side experiment:** the success-gated DA minus H3a, on accuracy and cost, with the same
  three readings.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Little context pressure in our episodes, so little room to gain | Stage 1 measures it first; stage 2 tests on the longest records |
| Llama-3-8B edits poorly out of the box (the paper's weak-model result) | Expected. Stage 2 is a "does it break things" gate; stage 3 trains the editing |
| Notes drop decisive facts (e.g. a lab value) | Manual review in stage 2; the SFT signal punishes notes that lose diagnostic information |
| The engineering of two adapters in vLLM | A smoke test first; the Hugging Face `generate` fallback |
| GPU memory: two trainable adapters on GPU 0 | Smoke test; the HA's updates use the same micro-batching as the DA |
| One seed per arm | State it; repeat the primary arm if the result is close |
| Data rules | Notes are written only by our local model; no MIMIC text goes to external APIs |

## 7a. Decisions as adopted (November 21, 2026) and implementation notes

The recommendations below were adopted (D1 a, D4 700 steps, D5 side experiment later, D6
1/B + replay), with these changes found while implementing (`src/clm/`):

- **D3 changed to Hugging Face `generate`.** vLLM serves the DA by merging the DA's LoRA into
  its weights, so an HA LoRA cannot be served next to it without rewriting the weight sync.
  The HA's replies are short, so `generate` on the training GPU is used instead. The DA's
  path is untouched.
- **D2 is decided by stage 2.**
  - Both the untrained model and the H2 HA are run as CLMs out of the box.
  - Stage 3 starts from the H2 HA (`INIT=h2`) if its edit-mode accuracy on the "all tests"
    sequences is not more than 3 points below its append-only accuracy. Otherwise it starts
    from a fresh LoRA (`INIT=base`).
  - The H2 HA already diagnoses well; a fresh LoRA would have to learn diagnosis from scratch
    with the small SFT weight of D6.
- **Stage 1 is folded into stage 2.** Every replay run reports context tokens per turn
  (`summary.json`), and the append-only run is the measurement.
- **Stage 2 arms:** A_da2, B_none, C_random2 and D_all, the same record sets as H2.
- **Commands and output folders:** `docs/CLM_RUNBOOK.md`.

## 7. Decisions needed before coding

- **D1: edit mechanism.**
  - (a) Default append, plus an optional full rewrite **(recommended)**.
  - (b) Structured edit commands (delete or replace a section), closer to the paper's shell
    edits.
  - (c) A forced rewrite every turn (MEM1-style, not a true CLM).
- **D2: the HA's starting weights.**
  - A new LoRA on base Llama **(recommended:** it keeps the ability to write notes).
  - Start from the H2 HA adapter. It was trained only to answer "Hypothesis: X", so it may
    have lost note-writing.
- **D3: generation during training.**
  - vLLM with both adapters as LoRA requests **(recommended)**.
  - Hugging Face `generate` (simpler, slower).
- **D4: the H4c length.** 700 steps (proposed).
- **D6: the strength of the HA's SFT updates in stages 3–4.**
  - **Background.** Attempt 2's HA-SFT step had a bug: only the longest row in each batch was
    scored, at about 1/B strength. That made HA updates much weaker. Attempt 3, with the fix
    (the paper's full-strength loss), scored 10 points lower. With one run each, the cause is
    not established.
  - **Why it matters here.** Each training step covers one patient, so full-strength SFT
    updates all push toward one disease.
  - **Recommended:** keep the correct loss (every row counts) and add both:
    - a mixed-disease replay batch;
    - an HA-SFT loss weight of about 1/B (B = HA records per step), so each step is roughly as
      strong as attempt 2's.

    This keeps the likely benefit of the bug without its arbitrary choice of row.
  - **Alternatives:** full strength (the paper's), or a lower HA learning rate instead of a
    loss weight.
- **D5: order.**
  - Run the cheap side experiment (success-gated DA cost) first, in parallel with stages 1–2
    **(recommended)**.
  - Or do CLM only.

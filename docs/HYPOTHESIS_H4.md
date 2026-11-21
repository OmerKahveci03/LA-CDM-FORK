# Hypothesis H4: the HA as a Contrastive Language Model, learning together with the DA

**Written November 21, 2026, before any H4 result exists.** Sections 1–6 are fixed; results
go in section 8.

- **Source:** the meeting suggestion "try a CLM model as the hypothesis agent: a mapping between
  patient state and the final output; embed the notes properly". "CLM" is the
  **Contrastive Language Model** of Contrastive-LM/CLM (Kwok, Kang, Suresh, Saad-Falcon,
  Pavone, Ré, Mirhoseini, 2026; https://github.com/Contrastive-LM/CLM; a blog post, not a
  peer-reviewed paper).
- **Background:** `docs/CONTRASTIVE_PRIMER.md`.
- **Commands and outputs:** `docs/CONTRASTIVE_RUNBOOK.md`.
- **Code:** `src/contrastive_ha/`.
- **An earlier plan** that read "CLM" as a *context-editing* model is kept as an alternative
  (`docs/HYPOTHESIS_H4_CONTEXT_EDITING.md`, `src/clm/`). It has not been run.

## 1. Hypothesis

A contrastive HA diagnoses at least as well as the H2 HA, and a DA trained while that HA keeps
learning beats one trained against a frozen copy. The contrastive HA is:
- a **frozen** Llama-3-8B-Instruct that embeds the patient state (the HA prompt);
- two small trained **projection heads** that map the state and each disease into one space;
- the most aligned disease is the hypothesis.

| Part | Question | Training |
|---|---|---|
| **H4a** | Is the contrastive HA as good a diagnoser as the H2 HA, on the same records? | Heads only, on cached embeddings |
| **H4b** | Does the baseline DA work as well with it as with the H2 HA? | None (evaluation) |
| **H4c** | Does learning together (the heads keep updating) beat frozen heads, with a new DA? | DA + heads |

## 2. Why

- **H2:** a dedicated HA is the key to accuracy on partial records. A contrastive HA is a
  cheaper, more direct "state → diagnosis" mapping, and its confidence comes straight from a
  calibrated softmax.
- **Learning together is cheap here.** Only the heads (a few million parameters) change, and
  the 8-billion-parameter backbone stays frozen. Joint training costs almost nothing beyond a
  normal DA run, and the heads cannot disturb the DA, unlike LA-CDM's shared adapter.
- **H3:** with a frozen HA, a retrained DA matched accuracy at lower cost, but the HA never
  adapted to the records the DA produces. H4c tests whether adapting helps.

## 3. Design (`src/contrastive_ha/`)

| Choice | Value |
|---|---|
| State text | The H2 HA prompt: chat template + `prompts/hypothesis_agent.txt` (history + test results) + `Hypothesis:`. These are the same tokens the H2 HA scores |
| Action text | "Final diagnosis: <disease>", one per disease |
| Pooling | The last token's final hidden state (4,096 numbers), as in CLM |
| Backbones (frozen) | **base**: Llama-3-8B-Instruct, no adapter; **h2**: the same with the frozen H2 HA adapter (seed 0) |
| Heads | One for states, one for actions: an input LayerNorm, an MLP of width 1,024 with 2 residual blocks, dropout 0.1, output 512, L2-normalised. About 7M parameters each; CLM's are 20M, trained on far more data |
| Score | exp(logit scale) × cosine; the scale is learned (initial temperature 0.07, capped at 100) |
| Loss | InfoNCE from the state side: cross-entropy of the true disease against the other three. CLM's reverse direction is left out, because with four actions it has many positives per action |
| Training data | Exactly the H2 rows: every training patient as history only, all tests and 2 random subsets (seed 0, 7,000-token budget), 7,676 rows |
| Validation | The H2 validation rows (seed 10,000, 1 random subset) |
| Optimisation | AdamW, lr 3e-4, weight decay 0.01, batch 128, 30 epochs, cosine schedule, seed 0 |
| Choice of epoch | Best validation accuracy (ties: lower validation loss); never on test |
| Confidence | round(10 × top probability), as for the H2 HA; the DA's transcript format is unchanged |

**Choice of backbone (fixed now): the higher validation accuracy of the chosen epochs** (ties:
lower validation loss).
- The h2 backbone's adapter was trained on exactly the training rows the heads are trained
  on, so its training embeddings are "seen" data.
- That can make its heads look better in training than on new data. This is why the choice is
  made on validation, and the caveat is reported.

## 4. Stages

| Stage | Jobs | GPU cost | Gate |
|---|---|---|---|
| **1a** | Embed every train, validation and test record with each backbone | ~2–4 h per backbone | — |
| **1b** | Train the heads for each backbone; choose the backbone on validation; score test (H4a) | minutes | H4a rule |
| **3** | The baseline DA (attempt 2, step 600) + the contrastive HA, test run (H4b) | ~1 h | none (informative) |
| **4** | A new DA from scratch: **joint** arm (heads updated every step) and **frozen** arm (heads fixed), 700 steps each; validation every 100 steps; checkpoint chosen on validation; test run (H4c) | ~2 days per arm | — |

**Stage-4 details:**
- **Each step:** after the DA's GRPO update, the heads take one AdamW step (lr 1e-4) on that
  step's HA embeddings, labelled with the patient's disease, plus 32 mixed-disease H2 training
  embeddings (replay).
- **Saved:** the heads with every DA snapshot and checkpoint.
- **DA settings:** as in H3a (original reward, 4 generations, 2,048-token transcripts, DA
  phases only, lr 1e-5).

## 5. Comparisons and decision rules (fixed now)

All tests are paired on the same 240 test patients, with exact McNemar and paired-bootstrap
95% intervals (`scripts/test_stats.py`).

- **H4a** (stage 1b, chosen backbone): contrastive HA minus H2 HA (`outputs/ha_sft/scores/new0`).
  - **Primary record set:** A_da2 (the baseline DA's tests, the records the system produces).
    - CI entirely above 0: better.
    - Lower bound above −3 points: not worse (non-inferior).
    - Otherwise: worse; stage 4 is then not run with this HA.
  - **Secondary record sets:** B_none, C_random2, D_all.
  - **Secondary measures:** ECE, AUROC.
- **H4b** (stage 3): the baseline DA + contrastive HA minus the baseline DA + H2 HA (step 2),
  on the DA's diagnosis. Informative only; no gate.
- **H4c** (stage 4, primary): **joint minus frozen**, the DA's diagnosis accuracy at each
  arm's validation-chosen step.
  - CI entirely above 0: supported.
  - CI includes 0: no detectable gain.
  - CI entirely below 0: worse.
  - **Secondary:** joint minus H3a (frozen H2 HA); cost; the gap between the DA and the HA's
    last hypothesis; calibration.

## 6. Risks and limitations

- **The heads may overfit.** There are about 7,700 rows for about 15M parameters. Mitigated
  by dropout, weight decay and choosing the epoch on validation.
- **The h2 backbone saw the training rows** (section 3), which is why the choice is made on
  validation.
- **Long records:** the backbone reads the full prompt (up to 7,000 tokens, within Llama's
  8,192), so nothing is cut. CLM's own default of 2,048 tokens does not apply here.
- **One seed** for the heads and for each DA. Stage-4 arms are single runs.
- **Ours vs theirs:** CLM's released heads (trained on Qwen3-8B embeddings, about 90M
  examples) are not used, because we use our own Llama. So this tests the CLM *method*, not
  their checkpoint.

## 7. How to run

See `docs/CONTRASTIVE_RUNBOOK.md`.

## 8. Results (fill in)

Run log:

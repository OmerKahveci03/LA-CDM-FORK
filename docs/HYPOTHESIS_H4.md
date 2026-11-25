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
- **Names (added 2026-11-23, after the results):** the *separately fine-tuned HA* is the H2 HA
  (fine-tuned on its own before DA training, then frozen; earlier "new HA"); the *baseline HA*
  is the one trained jointly with the DA in the shared adapter; the *CLM HA* (contrastive HA)
  is the frozen Llama + separately fine-tuned adapter with contrastive heads.
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
- 2026-11-21: smoke embedding (job 52225646) and smoke heads passed.
- 2026-11-21: full embeddings for `base` and `h2` (about 1 h and 1.5 h); full heads for both.
- 2026-11-21: H4a statistics (`jobs/contrastive_h4a_stats.sh`, B=h2) ->
  `outputs/contrastive/stats/h4a_<arm>/`.

### Stage 1b: backbone choice (validation only)

| Backbone | Chosen epoch | Val accuracy | Val loss at epoch 30 | Train acc at epoch 30 |
|---|---|---|---|---|
| base | 21 | 0.800 | 0.709 | 0.917 |
| **h2** | **2** | **0.869** | 0.981 | 0.983 |

**Chosen: h2** (higher validation accuracy). It peaks at epoch 2 and then overfits: its
embeddings already separate the diseases because the H2 adapter was fine-tuned on these
training rows. The validation patients were not in that fine-tuning.

Test accuracy of the heads alone (reference only, not used for the choice):

| Backbone | A_da2 | B_none | C_random2 | D_all |
|---|---|---|---|---|
| base | 85.4% | 70.8% | 73.3% | 85.4% |
| h2 | 89.2% | 77.1% | 80.0% | 90.8% |

### H4a (A_da2, n = 240, contrastive HA on h2)

| Run | Accuracy | ECE | Brier | AUROC (correct) |
|---|---|---|---|---|
| Contrastive HA (h2) | 89.2% [84.6, 92.5] | 6.3% [3.8, 10.0] | 0.073 | 0.883 |
| H2 HA | 88.8% [84.1, 92.2] | 4.6% [2.7, 8.1] | 0.066 | 0.906 |
| Baseline HA | 87.9% [83.2, 91.5] | 5.9% [3.6, 10.0] | 0.085 | 0.852 |

- **Contrastive minus H2 HA: +0.4 points [−1.3, +2.1]** (3 vs 2 discordant; McNemar p = 1.0).
- vs the baseline HA: +1.3 points [−2.5, +5.0] (p = 0.66).

**Verdict (rule in section 5): non-inferior.** The interval includes 0, so the contrastive HA
is not shown to be better, but its lower bound (−1.3) is above −3. Stage 4 goes ahead with
this HA. The two HAs disagree on only 5 of 240 patients, which is expected: the heads read
the same H2 adapter's representation. Calibration is slightly worse on the point estimates
(ECE 6.3% vs 4.6%) with overlapping intervals.

### H4b (test, n = 240; baseline DA step 600, no retraining; `outputs/contrastive/stats/h4b_<score>/`)

| HA with the baseline DA | DA accuracy | Tests | Cost | ECE | AUROC (correct) |
|---|---|---|---|---|---|
| Contrastive HA (h2) | 82.9% [77.6, 87.2] | 1.14 | $1,235 | 8.1% | 0.915 |
| H2 HA (step 2) | 85.4% [80.4, 89.3] | 1.20 | $1,285 | 5.2% | 0.936 |
| Baseline HA (baseline model) | 85.4% [80.4, 89.3] | 1.19 | $1,259 | 14.8% | 0.750 |

- Contrastive − step 2: −2.5 points [−5.4, +0.4] (3 vs 9 discordant; p = 0.15); cost −$51
  [−114, −1]. Contrastive − baseline model: −2.5 points [−6.2, +1.2].
- Informative only (no gate). The DA was trained with the baseline HA's messages, so a new
  HA changes what it reads; a small drop is expected and is not significant here.
- Calibration is far better than the baseline HA's (ECE 8.1% vs 14.8%, AUROC 0.915 vs 0.750),
  slightly worse than the H2 HA's.

### Stage 4: validation curves (DA diagnosis accuracy, 240 patients × 4 attempts)

Runs: joint job 52239537, frozen job 52239540 (20 h 19 min, peak RAM 7.9 GB); 700 steps each,
36% of the training patients. No extension was registered before these curves were seen.

| Step | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 |
|---|---|---|---|---|---|---|---|---|
| joint | 4.6% | 37.3% | 54.6% | 62.5% | 55.7% | 70.8% | **73.6%** | 68.5% |
| frozen | 4.6% | 39.4% | 59.7% | 52.6% | **70.0%** | 69.5% | 56.7% | 59.6% |

**Chosen steps (validation only): joint 600, frozen 400.** The heads' online loss ended at
0.05–0.06 (steps 699–700), so they did not drift.

### H4c (test, n = 240; `outputs/contrastive/stats/h4c_<score>/`)

| Run (step) | DA accuracy | Class-mean acc | Macro F1 | Tests | Cost |
|---|---|---|---|---|---|
| Joint (600) | 84.2% [79.0, 88.2] | 81.3% | 81.1% | 1.39 | $1,189 |
| Frozen (400) | 82.1% [76.7, 86.4] | 80.1% | 79.1% | 1.50 | $1,434 |
| H3a, frozen H2 HA (700) | 85.8% [80.9, 89.7] | 84.8% | 82.2% | 1.27 | $1,025 |

- **Primary, joint − frozen (DA diagnosis): +2.1 points [−2.5, +6.7]** (18 vs 13 discordant;
  McNemar p = 0.47). **Verdict: no detectable gain** (the interval includes 0).
- Cost, joint − frozen: −$245 [−336, −161], with −0.12 tests (p = 0.007). This is
  confounded: the arms are at different steps (600 vs 400), so the difference can come from
  the DA's training stage as well as from the heads.
- Joint − H3a: −1.7 points [−5.4, +2.1]; cost +$164 [+95, +235].
- HA final hypothesis, joint − frozen: −1.2 points [−5.0, +2.1]; joint − H3a −0.8 [−3.8, +2.1].
- One run per arm; the validation curves move 5–13 points between checkpoints, which is as
  large as the effects being tested.

### H4c-ext: exploratory extension to 1,900 steps (registered 2026-11-23)

**Exploratory, not pre-registered:** decided after the 700-step validation and test results
above were seen. The 700-step H4c result stays the primary one.

- Both arms (joint and frozen) resume from `checkpoint-700` to `MAX_STEPS=1900`: about all
  of the training patients once (700 steps = 36%). 1,900 is a multiple of the validation
  interval, so the last step is validated.
- Each arm's step is chosen on validation over steps 100–1900 (ties: lower cost). The
  700-step test runs are kept; new test runs go to `test_step_XXXX` as before.
- Same comparisons as H4c (joint − frozen primary; vs H3a for reference only, since H3a
  stopped at 700 steps).
- **First attempt invalid (2026-11-23).** Resuming from `checkpoint-700` silently reset the DA
  to an untrained adapter: `transformers.Trainer` reloads only the backbone adapter stored in
  the checkpoint's `contrastive_backbone/` subfolder, not the DA's root adapter. Training
  episodes went from about 15% to about 90% invalid formats immediately after step 700 in both
  arms (`figures/h4/h4_end_reasons`). Fixed by `src/utils/peft_resume.py`
  (`RestoreAdapterCallback`, logs `[resume] restored adapter 'default' ...`); the outputs after
  step 700 were moved to `train/broken_resume_after_700/` and the extension rerun from
  step 700. `save_total_limit: 5` had deleted the original `checkpoint-700`, so it was rebuilt
  from the `adapters/step_0700` snapshot (`scripts/rebuild_checkpoint.py`; `REBUILT.json` in
  the folder): same DA adapter, heads and data position, but **the AdamW moments of the DA
  and of the heads restart fresh at step 700**. The 700-step H4c results ran without a resume
  and are unaffected.
- **Second attempt (jobs 52598698/52598701, November 24).** The **frozen** arm resumed correctly from
  the rebuilt `checkpoint-700` (no jump in invalid formats) and reached 1,900 steps. Validation DA
  accuracy: 800 67.1%, 900 66.3%, 1000 70.6%, **1100 76.4%** (best), 1200 74.5%, 1300 73.9%,
  1400 73.9%, 1500 75.4%, 1600 74.9%, 1700 73.1%, 1800 70.8%, 1900 74.5%. The **joint** arm was
  submitted before the job-script fix reached the cluster and resumed from the Trainer's kept
  "best" `checkpoint-100`, so it **retrained steps 100 onward**. Its snapshots
  `adapters/step_0200`… were overwritten (step 600 at 2026-11-24 23:54), and its validation
  summaries for steps 200+ were rewritten. Consequences:
  - The pre-registered H4c results stand. The MIMIC test (`test_step_0600`, November 22) and the
    external test (`outputs/ext_v2_trained/h4c`, run at 19:14–19:45 on November 24) both used the
    **original** step-600 model, which was overwritten later. That model can no longer be
    re-tested.
  - The original joint validation curve is preserved in `figures/h4/h4_validation_points.csv`
    (2026-11-23 19:18; copied to `outputs/contrastive/joint/joint_h2/original_run_validation_points.csv`)
    and in the table above. `figures/h4_report/` (2026-11-24 11:29) was drawn before the overwrite; do
    not regenerate it.
  - **Frozen step 1100 on test** (the validation choice over 100–1900, job 52769959; statistics in
    `outputs/contrastive/stats/h4c_ext_*`): **78.8% with 0.00 tests and $0 per case.** The DA
    collapsed to diagnosing from the history alone, which validation accuracy (76.4%) did not reveal.
    vs frozen 400: −3.3 [−9.6, +2.9]; vs H3a: −7.1 [−12.5, −1.7], Holm p = 0.048; vs baseline: −6.7,
    p = 0.052. **Training longer did not help; it drifted to a degenerate zero-test policy.**
  - The current joint run is a **second joint run from step 100** (exploratory). Its steps 200–700
    repeat the original with the same settings: 58.8, 61.3, 66.4, 67.7, **67.9**, 64.3% against the
    original 54.6, 62.5, 55.7, 70.8, **73.6**, 68.5%. The two runs differ by up to 10.7 points at
    the same step, which is direct evidence of run-to-run noise.

Secondary record sets (B_none, C_random2, D_all): see
`outputs/contrastive/stats/h4a_<arm>/test_stats.md` (to be added).

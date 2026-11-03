# TRACE-CDM: evidence auditing and an adaptive Evidence Critic

TRACE-CDM (Trustworthy Reasoning through Adaptive Clinical Evidence) adds an evidence-auditing step to the trained LA-CDM Llama agent at inference time. The policy, the Hypothesis Agent, the adapter and training are unchanged. The goal is a matched comparison of the frozen agent with and without evidence feedback, on the same 240 test patients and the same test-result oracle.

Status: implemented and unit-tested on CPU with a scripted fake model. **No HiPerGator run has been made yet**, so there are no TRACE results. Nothing below is a measured improvement.

## Where it hooks in

`Environment.run` (`src/environment/environment.py`) loops over decision points. Each iteration runs the Hypothesis Agent (when enabled), then the Decision Agent, then `Environment.step` applies the action against the dataset oracle. TRACE adds one hook between the Hypothesis Agent and the Decision Agent, plus recording hooks:

| Hook | Called | Does |
|---|---|---|
| `start` | start of `run()` | registers each trajectory with its `patient_id` and label (label kept outside step state) |
| `before_decision` | after the HA, before DA generation | builds the visible state, audits it, gates the critic, and in interventional arms appends a one-line note to `trajectory.text` |
| `after_step` | after `Environment.step` | records action, input and outcome from `trajectory.action_log` |
| `finish` | end of `run()` | stores the episode: diagnosis, tests, USD cost, steps |

With `environment.trace = None` (the `baseline` arm, and always in training) none of these run. The only always-on change is `Trajectory.action_log`, which records what `step()` already decided.

The injected note is placed exactly like the existing `The current hypothesis is: ...` line, as non-trainable context before the next `Thought:`. Notes are sanitized to one line with no braces and no `Thought:`/`Action:`/`Observation:` keywords. Each finding is injected once per episode, and notes are capped by `max_flags` and `max_inject_chars`.

## Components (`src/trace_cdm/`)

- `audit.py`: the visible state (`StepState`) is built **only** from tests in `trajectory.requested_tests`; an assertion guards this. Deterministic checks:
  - `unavailable_result`: the oracle answered "not available".
  - `lab_conflict`: the same analyte has different values across panels (BMP/CMP/etc. overlap).
  - `invalid_lab_value`: empty, `nan` or `___` values.
  - `ambiguous_report`: hedging language in imaging or exam text.
  - `previous_action_error`, `repeated_request`: process errors.
  - `hypothesis_unknown`, `confidence_missing`, `hypothesis_changed`, `high_confidence_little_evidence`, `low_confidence`: Hypothesis Agent state.
  - `no_lab_yet`, `no_imaging_yet`: informational; never injected alone.

  MIMIC-CDM has no per-result timestamps, so future-evidence leakage is prevented by construction rather than by comparing times.
- `gate.py`: adaptive activation from flags that are new in the episode: critical flags, confidence/evidence disagreement, high confidence on equivocal imaging, hypothesis change, low confidence after testing, and HA failure. There is a per-episode call budget.
- `critic.py`: prompt, schema-validated JSON parsing, and a batched local Transformers backend (`local_files_only=True`, greedy, cached by prompt hash). Malformed output becomes an empty `review` result and is counted. A recommended test survives only if it is still requestable in that episode. The critic never sees the label, unrequested results or later steps.
- `controller.py`: wires the above into the hooks and writes `trace_episodes.jsonl`, `trace_critic_calls.jsonl` and `trace_summary.json` next to `metrics.json`. Exports omit history/result text unless `trace_cdm.redact_text=false`.
- `metrics.py` and `scripts/trace_compare.py`: per-arm metrics, patient-paired differences with bootstrap 95% CIs, exact McNemar tests, and verification checks.

The controller never chooses an action and never produces a test result. Abstention is not supported because the environment has no abstain action, so the critic can only suggest `proceed`, `request_test` or `review`.

## Arms (`configs/trace_cdm/`)

| Config | Arm | Agent sees | Critic | GPUs |
|---|---|---|---|---|
| `baseline` | A: native LA-CDM | nothing new | — | 2 |
| `shadow` | B, shadow | nothing new; audits are logged | — | 2 |
| `shadow_adaptive` | C, shadow | nothing new; the critic is logged | gated | 3 |
| `deterministic` | B, live | rule findings | — | 2 |
| `adaptive` | C, live | rule findings + critic | gated | 3 |
| `always` | D, live | rule findings + critic | every decision | 3 |
| `shuffled` | E, negative control | rule findings + another patient's critique | gated, same compute as C | 3 |

Select an arm with the Hydra override `trace_cdm=<config>`. Thresholds live in `baseline.yaml` under `audit` and `gate`. Tune them on the **validation** split (`dataset.test_data_file=data/val.csv`) and freeze them before the test runs.

## Running on HiPerGator

1. **Verify a critic checkpoint.** List `/data/ai/models/nlp/mistral_ai` and pick an *instruction-tuned* checkpoint in Transformers format (it has a `config.json`, tokenizer files with a chat template, and weights). Check its license. A different model family from Llama is intended; an instruction-tuned Gemma under `/data/ai/models/nlp/gemma` is the fallback. Nothing is downloaded.
2. **Smoke test** every arm on two patients:
   ```bash
   export ADAPTER_DIR=outputs/train_full_<JOB>/checkpoint-300   # the trained reference adapter
   export TRACE_CRITIC_MODEL=/data/ai/models/nlp/mistral_ai/<verified-checkpoint>
   SMALL_SAMPLE=true bash scripts/submit_trace_arms.sh
   ```
3. **Full test split**, same command without `SMALL_SAMPLE`. Outputs go to `outputs/trace_cdm/<arm>/eval_<JOB>/`, with `metrics.json`, the trace files and `run_info.yaml` (git SHA, adapter hash, critic path, GPUs).
4. **Verify and compare:**
   ```bash
   python scripts/trace_compare.py \
     --run baseline=outputs/trace_cdm/baseline/eval_<J1> \
     --run shadow=outputs/trace_cdm/shadow/eval_<J2> \
     --run shadow_adaptive=outputs/trace_cdm/shadow_adaptive/eval_<J3> \
     --run deterministic=outputs/trace_cdm/deterministic/eval_<J4> \
     --run adaptive=outputs/trace_cdm/adaptive/eval_<J5> \
     --run always=outputs/trace_cdm/always/eval_<J6> \
     --run shuffled=outputs/trace_cdm/shuffled/eval_<J7> \
     --output outputs/trace_cdm/comparison
   ```
   The script exits with status 1 if any of these checks fails:
   - the exported episodes don't reproduce that run's own `metrics.json`;
   - the `shadow` metrics differ from `baseline`;
   - `shadow` and `shadow_adaptive` took different actions.

   **Do not interpret live arms until these checks pass.** All live arms are compared with `shadow`, the instrumented reproduction of the baseline.

The critic runs on `cuda:2` by default (policy on `cuda:0`, vLLM on `cuda:1`). If a third L4 is unavailable, set `CRITIC_DEVICE=cuda:0 CRITIC_4BIT=true` to load it in 4-bit beside the policy model; this has not been tested for memory headroom.

## Report

For each arm, report:
- accuracy (all cases) and accuracy among diagnosed cases;
- undiagnosed and invalid fractions;
- per-disease recall;
- mean tests and mean USD cost;
- HA accuracy and ECE from `metrics.json`;
- critic call rate, tokens, latency, peak GPU memory, parse failures and invalid recommendations (`trace_summary.json`);
- paired Δ accuracy, Δ tests and Δ cost with 95% CIs, and the McNemar p-value.

Compare `adaptive` with `always` at the same checkpoint and prompt to measure what gating saves. Use `shuffled` to separate the effect of patient-specific critique from the effect of any extra text.

## Known limitations

- The Decision Agent was trained without audit notes. Any effect of injection, including harm from distribution shift (for example, more format errors), is a property of a frozen model reading out-of-distribution context, not of a trained TRACE agent.
- Lab parsing assumes the dataset's `Name: value, Name: value` format; values containing commas would be split.
- Hedging-term detection is lexical and will flag some routine phrasing, such as "possible". Estimate flag precision on a manually reviewed validation sample before relying on it.
- Critic self-reported `evidence_quality` and Hypothesis Agent confidence are uncalibrated.
- No controlled evidence perturbations are implemented yet. Add them only on a designated evaluation copy, with labeled perturbation types.
- Evaluation uses one near-greedy rollout (temperature 0.01) per case. vLLM numerics can vary slightly between runs, so the `shadow` check also tests run-to-run determinism.

# LA-CDM Llama Reproduction Status

Last updated: September 29, 2026

## Goal

Reproduce the LA-CDM training and evaluation pipeline on MIMIC-CDM using a
HiPerGator-hosted Meta Llama model instead of Qwen.

## Model

The project is configured to use this local Transformers-format model:

```text
/blue/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf
```

This is configured in `configs/model/defaults.yaml` and is also the default
summarization model in `scripts/prepare_data.py`.

## Completed

### 1. Extracted the candidate cohort

Script:

```text
extract_data/p01_pull_cohort.py
```

Sources:

```text
/orange/prismap-data-core/MIMIC/physionet.org/files/mimiciv/2.2
/orange/prismap-data-core/MIMIC/physionet.org/files/mimic-iv-note/2.2/note
```

Candidate output:

```text
/blue/prismap-ai-core/omerkahveci/mimic_cdm_candidates
```

The extraction found 12,441 candidate admissions and 9,336 subjects.

### 2. Built the final MIMIC-CDM cohort

The official dataset-construction repository was added as a Git submodule:

```text
extract_data/MIMIC-Clinical-Decision-Making-Dataset
```

The wrapper imports the official builder functions:

```text
extract_data/p02_build_cohort.py
```

Final cohort output:

```text
/blue/prismap-ai-core/omerkahveci/mimic_cdm_final
```

### 3. Validated the final cohort

Validation script:

```bash
python extract_data/p03_validate_cohort.py
```

Validated counts:

| Disease | Cases |
|---|---:|
| Appendicitis | 957 |
| Cholecystitis | 648 |
| Diverticulitis | 257 |
| Pancreatitis | 538 |
| Total | 2,400 |

Validation also confirmed:

- No invalid records
- No diagnosis leakage in patient histories
- No admission overlap between diseases
- All cases have histories, physical examinations, labs, and abdominal imaging
- `lab_test_mapping.csv` contains 1,104 rows

### 4. Created dataset splits

Command:

```bash
python scripts/split_dataset.py \
  --data_dir /blue/prismap-ai-core/omerkahveci/mimic_cdm_final \
  --output_dir data
```

Result:

| Split | Cases |
|---|---:|
| Train | 1,920 |
| Validation | 240 |
| Test | 240 |

The lab mapping was copied into `data/lab_test_mapping.csv`.

### 5. Improved summarization

`scripts/create_data_with_summaries.py` now:

- Uses deterministic greedy generation
- Saves a checkpoint every 25 new summaries
- Resumes rows that already contain a summary
- Passes an attention mask to generation
- Loads Llama in 4-bit NF4 mode

### 6. Completed and validated all summaries

Llama summaries are complete for all 2,400 cases:

| Split | Complete summaries |
|---|---:|
| Train | 1,920 / 1,920 |
| Validation | 240 / 240 |
| Test | 240 / 240 |

Prepared-data validation script:

```bash
python scripts/validate_prepared_data.py
```

Validation passed with no blank summaries, duplicate IDs, malformed lab fields,
malformed radiology fields, invalid labels, or cross-split overlap.

## Current Work

The two-patient zero-shot evaluation smoke test is complete:

```bash
sbatch slurm/zero_shot_smoke.sbatch
```

The first smoke attempt exposed and led to fixes for:

- Duplicate `itemid` aliases in the generated lab mapping
- Curated synonym item IDs with no standalone row in the filtered lab mapping
- RTX PRO 6000 Blackwell (`sm_120`) incompatibility with pinned PyTorch 2.6
- Implicit Accelerate defaults and incomplete Hydra tracebacks
- Missing Hydra `version_base` declarations
- Eval-only trainer unnecessarily deep-copying the full model for KL reference

The revised job requests two L4 GPUs. Transformers uses GPU 0 and vLLM uses
GPU 1.

## Environment Notes

The cohort-building Conda environment is:

```text
/home/omerkahveci/.conda/envs/mimic-cdm
```

The separate LA-CDM evaluation/training environment is stored on Blue:

```text
/blue/prismap-ai-core/omerkahveci/envs/la-cdm
```

Important compatible package versions currently used include:

```text
numpy==1.26.4
pandas==2.1.1
transformers==4.49.0
accelerate==1.5.2
bitsandbytes==0.50.2
```

The LA-CDM environment was updated to a mutually compatible vLLM stack:

```text
vllm==0.8.3
xgrammar==0.1.17
transformers==4.51.0
huggingface-hub==0.30.2
numba==0.61.0
llvmlite==0.44.0
```

`bitsandbytes >= 0.48.0` is required for the HiPerGator PyTorch CUDA 13 build.
The old submodule `requirements.txt` should not be installed as a whole because
it contains conflicting dependency pins.

Large environments and Conda package caches must be stored on Blue because the
home quota is insufficient. Use `--no-cache-dir` for large pip installs.

## Next Steps

### 1. Smoke-test evaluation (complete)

The smoke-test metrics are finite and structurally complete.

The two-L4 run successfully loaded both the Transformers and vLLM model copies
and completed CUDA graph capture, confirming that the current GPU and host-memory
requests are sufficient. The subsequent failure was a Llama tokenizer issue,
not an OOM: Llama 3 has no default padding token. The custom trainer now uses
the EOS token for padding and propagates its ID to the model configuration.
The next run passed tokenization and exposed an unconditional PEFT
`disable_adapter()` call during zero-shot evaluation. The trainer now uses that
context only when the loaded model actually has a PEFT adapter.
The following run completed inference for both smoke-test cases. It then failed
only while saving because `evaluation.metrics_output_file` was unset. Evaluation
now defaults to Hydra's run directory as `metrics.json`. Single-generation
reward normalization also uses population standard deviation to avoid NaNs.
The final smoke run completed without a traceback on September 29, 2026. Both
model copies loaded on separate L4 GPUs, CUDA graph capture completed, both
cases were evaluated, and the metrics file was written.
Inspection showed that the first successful `metrics.json` contained only the
standard Trainer loss and timing fields. The custom trainer had copied the log
dictionary before adding diagnostic metrics, so `evaluate()` could not return
them to the save step. It now updates the Trainer-owned dictionary in place so
accuracy, calibration, reward, test-count, and diagnostic-cost fields are saved.
The next smoke metrics file was complete, but both final diagnoses were
unparsed (`eval_none_fraction=1.0`, `FormatReward=0`). The action parser was
written for exact Qwen-style plain-text labels. It now accepts Llama's cosmetic
Markdown, indentation, whitespace, and trailing-period variations while still
requiring the three action fields and validating actions against the allowlists.
That tolerance change did not alter the two-case result. Targeted warnings were
therefore added for invalid format, unknown action, unknown test, and unknown
diagnosis cases. The next smoke log will include the raw invalid completion so
the remaining Llama incompatibility can be fixed from evidence.
The diagnostic run showed that Llama ended its turn before emitting the required
action: one second-step completion was empty and another contained only a
`Thought`. The 256-token limit was not exhausted. Decision-agent generation now
ignores early EOS/end-of-turn tokens while retaining the `Observation:` stop;
hypothesis and confidence-calibration generation keep normal EOS behavior.
The resulting log showed actions followed by literal Llama `<|eot_id|>` and
assistant-header tokens, confirming that forcing generation beyond EOS was
creating artificial extra turns. The parser still accepts ordinary prose
reasoning followed by valid `Action` and `Action Input` fields, since Llama does
not always label its reasoning as `Thought:`.
Further testing showed that forcing generation beyond EOS creates synthetic
chat headers and malformed actions, so that workaround was removed. Zero-shot
evaluation now uses repeatable near-greedy decoding (`temperature=0.01`) and
passes the configured seed into vLLM. Early termination and invalid actions are retained as genuine
baseline failures instead of being coerced into valid responses.
The final deterministic smoke run completed successfully. Both cases requested
CT and then ended after a `Thought` without emitting another action. These are
measured as undiagnosed zero-shot failures rather than pipeline errors. The
full 240-case job is `slurm/zero_shot_full.sbatch`; it uses two L4 GPUs and 64
GB of RAM and writes job-specific metrics under `outputs/`.

The full zero-shot baseline completed successfully as Slurm job `43908263` in
2,540.7 seconds (42.3 minutes). Key results across 240 test cases:

| Metric | Result |
|---|---:|
| Overall accuracy (counting undiagnosed as wrong) | 8.75% |
| Undiagnosed/invalid fraction | 87.92% |
| Accuracy among the 29 completed diagnoses | 72.41% |
| Macro F1 (all cases) | 15.54% |
| Macro F1 (completed diagnoses only) | 68.50% |
| Hypothesis-agent accuracy | 65.25% |
| Expected calibration error | 10.87% |
| Average tests requested | 1.16 |
| Average diagnostic cost | $798.08 |

The large gap between all-case and completed-case accuracy shows that the main
zero-shot weakness is failure to finish the required action protocol, rather
than poor diagnostic discrimination when Llama does return a diagnosis.

### 2. Full zero-shot baseline (complete)

The 240-case baseline completed successfully; results are recorded above.

### 3. Short training smoke test (training passed)

Confirm LoRA attachment, forward/backward passes, checkpoint creation, adapter
loading, and GPU memory usage before requesting a full training allocation.
The two-step job is `slurm/train_smoke.sbatch`. It uses the two-case sample,
two generations per case, two L4 GPUs, 64 GB RAM, and writes checkpoints to
`outputs/train_smoke_<JOB_ID>/`.

Job `43931304` completed both optimizer steps in 69.6 seconds. Gradients were
finite, KL became nonzero on step two, and the final LoRA adapter was saved to
`outputs/train_smoke_43931304/`. The remaining check is loading that adapter
with `slurm/adapter_smoke.sbatch` and evaluating it on two cases.
Adapter evaluation job `43933004` also completed successfully, confirming that
the saved LoRA adapter can be loaded, merged into vLLM, and used for inference.
Before the full allocation, `slurm/train_scale_smoke.sbatch` performs one step
with the production setting of eight generations to verify peak GPU memory.

### 4. Run full training and evaluation

Required experiments:

1. Full Llama LA-CDM
2. Decision-agent-only ablation
3. LA-CDM without test-cost reward
4. Evaluation of each trained adapter on the fixed test set

The Llama results reproduce the LA-CDM method and protocol, not the paper's
exact Qwen result. Compare each trained Llama run primarily against the same
Llama zero-shot baseline.

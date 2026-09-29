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

The next milestone is a two-patient zero-shot evaluation smoke test:

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

### 1. Complete the zero-shot smoke evaluation

Run `slurm/zero_shot_smoke.sbatch`, inspect both log files, and confirm that
metrics are written for two test cases.

The two-L4 run successfully loaded both the Transformers and vLLM model copies
and completed CUDA graph capture, confirming that the current GPU and host-memory
requests are sufficient. The subsequent failure was a Llama tokenizer issue,
not an OOM: Llama 3 has no default padding token. The custom trainer now uses
the EOS token for padding and propagates its ID to the model configuration.

### 2. Run the full zero-shot baseline

Record diagnostic accuracy, diagnostic cost, requested tests, interaction
length, formatting failures, and per-disease performance.

### 3. Run a short training smoke test

Confirm LoRA attachment, forward/backward passes, checkpoint creation, adapter
loading, and GPU memory usage before requesting a full training allocation.

### 4. Run full training and evaluation

Required experiments:

1. Full Llama LA-CDM
2. Decision-agent-only ablation
3. LA-CDM without test-cost reward
4. Evaluation of each trained adapter on the fixed test set

The Llama results reproduce the LA-CDM method and protocol, not the paper's
exact Qwen result. Compare each trained Llama run primarily against the same
Llama zero-shot baseline.

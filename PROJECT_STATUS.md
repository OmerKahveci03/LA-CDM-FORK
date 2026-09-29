# LA-CDM Llama Reproduction Status

Last updated: September 27, 2026

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

## Current Work

The 1,920 training histories are being summarized with Llama 3 8B. The observed
speed is approximately 5.2 seconds per case, or about 2.75 hours total.

Training summarization command:

```bash
spython --follow --gpu --cpu 2 --mem 32 --time 24 \
  scripts/create_data_with_summaries.py \
  --model /blue/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
  --input_file data/train.csv \
  --output_file data/train.csv
```

## Environment Notes

The cohort-building Conda environment is:

```text
/home/omerkahveci/.conda/envs/mimic-cdm
```

Important compatible package versions currently used include:

```text
numpy==1.26.4
pandas==2.1.1
transformers==4.49.0
accelerate==1.5.2
bitsandbytes==0.50.2
```

`bitsandbytes >= 0.48.0` is required for the HiPerGator PyTorch CUDA 13 build.
The old submodule `requirements.txt` should not be installed as a whole because
it contains conflicting dependency pins.

The custom `spython` wrapper supports `--gpu`, which requests:

```text
--gres=gpu:1
```

Use `--follow` on all future `spython` commands.

## Next Steps

### 1. Verify training summaries

```bash
python -c "import pandas as pd; d=pd.read_csv('data/train.csv'); print(d['Patient History Summary'].notna().sum(), len(d))"
```

Expected result:

```text
1920 1920
```

### 2. Summarize validation histories

```bash
spython --follow --gpu --cpu 2 --mem 32 --time 4 \
  scripts/create_data_with_summaries.py \
  --model /blue/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
  --input_file data/val.csv \
  --output_file data/val.csv
```

### 3. Summarize test histories

```bash
spython --follow --gpu --cpu 2 --mem 32 --time 4 \
  scripts/create_data_with_summaries.py \
  --model /blue/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf \
  --input_file data/test.csv \
  --output_file data/test.csv
```

### 4. Validate prepared data

Confirm that every split has a nonempty `Patient History Summary`, valid labels,
parseable lab/radiology fields, no duplicate admissions, and no cross-split
overlap. A dedicated prepared-data validation script should be added before
training.

### 5. Run a small zero-shot smoke evaluation

Use `dataset.small_sample=true` and disable Weights & Biases initially. Confirm
that Llama follows the agent response format and that vLLM, the environment,
and metrics all run successfully.

### 6. Run the full zero-shot baseline

Record diagnostic accuracy, diagnostic cost, requested tests, interaction
length, formatting failures, and per-disease performance.

### 7. Run a short training smoke test

Confirm LoRA attachment, forward/backward passes, checkpoint creation, adapter
loading, and GPU memory usage before requesting a full training allocation.

### 8. Run full training and evaluation

Required experiments:

1. Full Llama LA-CDM
2. Decision-agent-only ablation
3. LA-CDM without test-cost reward
4. Evaluation of each trained adapter on the fixed test set

The Llama results reproduce the LA-CDM method and protocol, not the paper's
exact Qwen result. Compare each trained Llama run primarily against the same
Llama zero-shot baseline.


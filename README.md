# Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning

[![](https://img.shields.io/badge/Paper-2506.13474-blue)](https://arxiv.org/abs/2506.13474)

This repository contains the official implementation of the paper

> **Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning**
>
> David Bani-Harouni, Chantal Pellegrini, Ege Özsoy, Matthias Keicher, Nassir Navab<br>
> Technical University of Munich, Munich Center for Machine Learning 
> 
> ICLR 2026

<img src="https://raw.githubusercontent.com/dharouni/LA-CDM/main/overview.png" width="750">

## Abstract

Clinical decision-making is a dynamic, interactive, and cyclic process where doctors have to repeatedly decide on which clinical action to perform and consider newly uncovered information for diagnosis and treatment. Large Language Models (LLMs) have the potential to support clinicians in this process, however, most applications of LLMs in clinical decision support suffer from one of two limitations: Either they assume the unrealistic scenario of immediate availability of all patient information and do not model the interactive and iterative investigation process, or they restrict themselves to the limited "out-of-the-box" capabilities of large pre-trained models without performing task-specific training. In contrast to this, we propose to model clinical decision-making for diagnosis with a hypothesis-driven uncertainty-aware language agent, LA-CDM, that converges towards a diagnosis via repeatedly requesting and interpreting relevant tests. Using a hybrid training paradigm combining supervised and reinforcement learning, we train LA-CDM with three objectives targeting critical aspects of clinical decision-making: accurate hypothesis generation, hypothesis uncertainty estimation, and efficient decision-making. We evaluate our methodology on MIMIC-CDM, a real-world dataset covering four abdominal diseases containing various clinical tests and show the benefit of explicitly training clinical decision-making for increasing diagnostic performance and efficiency.


## Installation

Requires Python 3.11–3.12 and a CUDA-capable GPU (tested on NVIDIA A40, 48 GB).

```bash
git clone https://github.com/dharouni/LA-CDM.git
cd LA-CDM

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync
```

## Data

LA-CDM trains on [**MIMIC-IV-Ext-CDM**](https://physionet.org/content/mimic-iv-ext-cdm/1.0/) (Hager et al.), a MIMIC-IV derived dataset with 2,400 patients across four abdominal conditions: appendicitis, cholecystitis, diverticulitis, and pancreatitis. First, follow the steps described in the official [MIMIC-CDM repository](https://github.com/paulhager/MIMIC-Clinical-Decision-Making-Dataset) to process the raw PhysioNet download.

### Data Preparation

The preparation pipeline converts the processed PhysioNet download into the CSV files expected by the training code. It has three steps:

1. **Split** the four per-condition pickle files into stratified train / val / test CSVs (80 / 10 / 10).
2. **Copy** the lab test mapping.
3. **Summarize** each patient's history of present illness into a concise summary using an LLM.

Run the full pipeline (requires a GPU for the summarization step):

```bash
python scripts/prepare_data.py \
    --data_dir /path/to/mimic-iv-ext-cdm/1.1 \
    --model Qwen/Qwen2.5-7B-Instruct
```

The resulting `data/` directory should contain:

```
data/
├── train.csv
├── val.csv
├── test.csv
└── lab_test_mapping.csv
```

The dataset includes patient history summaries, 5,959 imaging reports, and 143,191 lab results. 12 diagnostic tests are available: Physical Examination, CT, MRI, Radiograph, Ultrasound, CBC, BMP, CMP, Renal Function Panel, Liver Function Panel, Urinalysis, and Electrolyte Panel.

## Usage

All commands use [Hydra](https://hydra.cc/) for configuration. To disable Weights & Biases logging, add `wandb.mode=disabled` to any command.

Before starting any run, disable the vLLM V1 engine:

```bash
export VLLM_USE_V1=0
```

### Full LA-CDM Training

Trains both agents with the cyclic loss schedule (Decision Agent GRPO → Hypothesis Agent SFT → Confidence Calibration GRPO, 100 steps each):

```bash
python -m accelerate.commands.launch --num_processes 1 -m src.train
```

### Decision-Agent-Only Training

Trains only the Decision Agent (no hypothesis generation):

```bash
python -m accelerate.commands.launch --num_processes 1 -m src.train \
    environment=decision_agent_only experiment=decision_agent_only
```

### Without Test Costs

```bash
python -m accelerate.commands.launch --num_processes 1 -m src.train \
    reward_function=no_test_cost
```

### Evaluation

Evaluate a trained adapter:

```bash
python -m accelerate.commands.launch --num_processes 1 -m src.evaluate \
    evaluation.adapter_dir=<path-to-adapter>
```

Zero-shot evaluation (base model, no adapter):

```bash
python -m accelerate.commands.launch --num_processes 1 -m src.evaluate \
    evaluation.base_model=true
```

Metrics are saved to `evaluation/test_metrics.json` and logged to Weights & Biases.


## Citation

If you find this work useful, please cite:

```bibtex
@article{baniharouni2025language,
  title={Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning},
  author={Bani-Harouni, David and Pellegrini, Chantal and {\"O}zsoy, Ege and Keicher, Matthias and Navab, Nassir},
  journal={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```

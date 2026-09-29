#!/bin/bash
#SBATCH --job-name=lacdm-smoke
#SBATCH --output=logs/lacdm_smoke_%j.out
#SBATCH --error=logs/lacdm_smoke_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing Python executable: $PYTHON_BIN" >&2
    exit 1
fi

cd "$REPO_DIR"
export VLLM_USE_V1=0

echo "[SLURM] Started: $(date)"
echo "[SLURM] Host: $(hostname)"
echo "[SLURM] GPUs: ${CUDA_VISIBLE_DEVICES:-not-set}"

"$PYTHON_BIN" -c \
    "import torch, transformers, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('transformers', transformers.__version__, 'vllm', vllm.__version__)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    -m src.evaluate \
    evaluation.base_model=true \
    dataset.small_sample=true \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

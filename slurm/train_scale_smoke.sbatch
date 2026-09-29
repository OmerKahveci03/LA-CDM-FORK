#!/bin/bash
#SBATCH --job-name=lacdm-scale-smoke
#SBATCH --output=logs/lacdm_scale_smoke_%j.out
#SBATCH --error=logs/lacdm_scale_smoke_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
RUN_DIR="outputs/train_scale_smoke_${SLURM_JOB_ID}"

cd "$REPO_DIR"
mkdir -p logs
export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[SLURM] Started: $(date)"
echo "[SLURM] Output: $RUN_DIR"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.train \
    hydra.run.dir="$RUN_DIR" \
    dataset.small_sample=false \
    training.max_steps=1 \
    training.num_generations=8 \
    training.per_device_train_batch_size=8 \
    training.eval_strategy=no \
    training.save_steps=1 \
    training.vllm_gpu_memory_utilization=0.9 \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

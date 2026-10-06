#!/bin/bash
#SBATCH --job-name=lacdm-train-full
#SBATCH --output=logs/lacdm_train_full_%j.out
#SBATCH --error=logs/lacdm_train_full_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
RUN_DIR="outputs/train_full_${SLURM_JOB_ID}"

cd "$REPO_DIR"
mkdir -p logs
export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[SLURM] Started: $(date)"
echo "[SLURM] Host: $(hostname)"
echo "[SLURM] GPUs: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "[SLURM] Output: $RUN_DIR"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.train \
    hydra.run.dir="$RUN_DIR" \
    dataset.small_sample=false \
    training.max_steps=300 \
    training.num_generations=8 \
    training.per_device_train_batch_size=1 \
    training.save_steps=25 \
    training.vllm_gpu_memory_utilization=0.3 \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

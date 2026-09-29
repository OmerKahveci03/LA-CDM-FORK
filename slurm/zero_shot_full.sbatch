#!/bin/bash
#SBATCH --job-name=lacdm-zero-shot
#SBATCH --output=logs/lacdm_zero_shot_%j.out
#SBATCH --error=logs/lacdm_zero_shot_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
METRICS_FILE="outputs/zero_shot_full_${SLURM_JOB_ID}/metrics.json"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing Python executable: $PYTHON_BIN" >&2
    exit 1
fi

cd "$REPO_DIR"
mkdir -p logs
export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[SLURM] Started: $(date)"
echo "[SLURM] Host: $(hostname)"
echo "[SLURM] GPUs: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "[SLURM] Metrics: $METRICS_FILE"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.evaluate \
    evaluation.base_model=true \
    evaluation.metrics_output_file="$METRICS_FILE" \
    dataset.small_sample=false \
    training.vllm_gpu_memory_utilization=0.9 \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

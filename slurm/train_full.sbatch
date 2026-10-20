#!/bin/bash
#SBATCH --job-name=lacdm-train-full
#SBATCH --output=logs/lacdm_train_full_%j.out
#SBATCH --error=logs/lacdm_train_full_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
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
RESUME_ARGS=()
if [[ "${START_FRESH:-0}" == "1" ]]; then
    RESUME_FROM_CHECKPOINT=""
elif [[ -z "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    RESUME_FROM_CHECKPOINT=$("$PYTHON_BIN" scripts/find_latest_training_checkpoint.py outputs --max-steps 300)
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    if [[ ! -f "$RESUME_FROM_CHECKPOINT/trainer_state.json" || ! -f "$RESUME_FROM_CHECKPOINT/optimizer.pt" ]]; then
        echo "[SLURM] Resume checkpoint is missing trainer_state.json or optimizer.pt: $RESUME_FROM_CHECKPOINT" >&2
        exit 1
    fi
    RESUME_ARGS+=("training.resume_from_checkpoint=$RESUME_FROM_CHECKPOINT")
    echo "[SLURM] Resuming from: $RESUME_FROM_CHECKPOINT"
else
    echo "[SLURM] Starting from step 0 (no resume checkpoint selected)"
fi
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
    training.num_generations=4 \
    training.per_device_train_batch_size=1 \
    training.max_completion_length=128 \
    environment.max_length=2048 \
    training.save_steps=25 \
    training.vllm_gpu_memory_utilization=0.9 \
    "${RESUME_ARGS[@]}" \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

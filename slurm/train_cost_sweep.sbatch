#!/bin/bash
#SBATCH --job-name=lacdm-cost-train
#SBATCH --output=logs/lacdm_cost_train_%j.out
#SBATCH --error=logs/lacdm_cost_train_%j.err
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
cd "$REPO_DIR"
mkdir -p logs

# Always start a new arm from the base model. Resume only from a checkpoint
# explicitly supplied for this same arm after an interrupted job.
: "${COST_MULTIPLIER:?Set COST_MULTIPLIER to 0, 1.5, or 2}"
case "$COST_MULTIPLIER" in
    0|1.5|2) ;;
    *) echo "Unsupported COST_MULTIPLIER: $COST_MULTIPLIER" >&2; exit 2 ;;
esac
ARM="cost_${COST_MULTIPLIER//./p}"
RUN_DIR="outputs/cost_sweep/${ARM}/train_${SLURM_JOB_ID}"
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    case "$RESUME_FROM_CHECKPOINT" in
        outputs/cost_sweep/"$ARM"/train_*/checkpoint-*) ;;
        *) echo "Checkpoint must belong to $ARM: $RESUME_FROM_CHECKPOINT" >&2; exit 2 ;;
    esac
    if [[ ! -f "$RESUME_FROM_CHECKPOINT/trainer_state.json" || ! -f "$RESUME_FROM_CHECKPOINT/optimizer.pt" ]]; then
        echo "Incomplete checkpoint: $RESUME_FROM_CHECKPOINT" >&2
        exit 2
    fi
    RESUME_ARGS+=("training.resume_from_checkpoint=$RESUME_FROM_CHECKPOINT")
fi

export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[SLURM] Arm: $ARM"
echo "[SLURM] Output: $RUN_DIR"
echo "[SLURM] Resume: ${RESUME_FROM_CHECKPOINT:-none}"
echo "[SLURM] Started: $(date)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.train \
    hydra.run.dir="$RUN_DIR" \
    "experiment.name=la-cdm-${ARM}" \
    "reward_function.reward_functions.2.multiplier=$COST_MULTIPLIER" \
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

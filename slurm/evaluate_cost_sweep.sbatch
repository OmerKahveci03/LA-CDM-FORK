#!/bin/bash
#SBATCH --job-name=lacdm-cost-eval
#SBATCH --output=logs/lacdm_cost_eval_%j.out
#SBATCH --error=logs/lacdm_cost_eval_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

: "${ADAPTER_DIR:?Set ADAPTER_DIR to the completed adapter for this arm}"
: "${COST_MULTIPLIER:?Set COST_MULTIPLIER to the training arm multiplier}"
case "$COST_MULTIPLIER" in
    0|1.5|2) ;;
    *) echo "Unsupported COST_MULTIPLIER: $COST_MULTIPLIER" >&2; exit 2 ;;
esac
ARM="cost_${COST_MULTIPLIER//./p}"
case "$ADAPTER_DIR" in
    outputs/cost_sweep/"$ARM"/train_*) ;;
    *) echo "Adapter must belong to $ARM: $ADAPTER_DIR" >&2; exit 2 ;;
esac
if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ||
      ( ! -f "$ADAPTER_DIR/adapter_model.safetensors" && ! -f "$ADAPTER_DIR/adapter_model.bin" ) ]]; then
    echo "No completed adapter at $ADAPTER_DIR" >&2
    exit 2
fi
RUN_DIR="outputs/cost_sweep/${ARM}/eval_${SLURM_JOB_ID}"

export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[SLURM] Adapter: $ADAPTER_DIR"
echo "[SLURM] Metrics: $RUN_DIR/metrics.json"
echo "[SLURM] Started: $(date)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.evaluate \
    hydra.run.dir="$RUN_DIR" \
    "experiment.name=la-cdm-${ARM}" \
    "reward_function.reward_functions.2.multiplier=$COST_MULTIPLIER" \
    evaluation.base_model=false \
    evaluation.adapter_dir="$ADAPTER_DIR" \
    evaluation.metrics_output_file="$RUN_DIR/metrics.json" \
    dataset.small_sample=false \
    training.vllm_gpu_memory_utilization=0.9 \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

#!/bin/bash
#SBATCH --job-name=lacdm-adapter-smoke
#SBATCH --output=logs/lacdm_adapter_smoke_%j.out
#SBATCH --error=logs/lacdm_adapter_smoke_%j.err
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
ADAPTER_DIR=${1:?Usage: sbatch slurm/adapter_smoke.sbatch PATH_TO_ADAPTER}
METRICS_FILE="outputs/adapter_smoke_${SLURM_JOB_ID}/metrics.json"

cd "$REPO_DIR"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing Python executable: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]]; then
    echo "Missing adapter_config.json in: $ADAPTER_DIR" >&2
    exit 1
fi

mkdir -p logs
export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[SLURM] Started: $(date)"
echo "[SLURM] Adapter: $ADAPTER_DIR"
echo "[SLURM] Metrics: $METRICS_FILE"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.evaluate \
    evaluation.base_model=false \
    evaluation.adapter_dir="$ADAPTER_DIR" \
    evaluation.metrics_output_file="$METRICS_FILE" \
    dataset.small_sample=true \
    training.vllm_gpu_memory_utilization=0.9 \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

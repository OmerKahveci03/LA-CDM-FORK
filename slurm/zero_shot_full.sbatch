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

# Untrained Llama on the fixed 240-case test split, with the same evaluation
# settings as the trained arms. Results go to outputs/zero_shot/test/
# (metrics.json, test_episodes.csv, test_summary.json, transcripts).
# An existing result is never overwritten unless OVERWRITE=1.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
OUT_DIR="outputs/zero_shot/test"
METRICS_FILE="$OUT_DIR/metrics.json"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing Python executable: $PYTHON_BIN" >&2
    exit 1
fi

cd "$REPO_DIR"
mkdir -p logs
if [[ -f "$METRICS_FILE" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$METRICS_FILE already exists; set OVERWRITE=1 to replace it." >&2
    exit 2
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/eval_output.log") 2>&1
JOB_ID=${SLURM_JOB_ID:-local}
STARTED=$(date +%Y%m%d-%H%M%S)
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
echo "[SLURM] Job: $JOB_ID  Commit: $COMMIT"
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
    hydra.run.dir="$OUT_DIR" \
    "hydra.output_subdir=.hydra/${STARTED}" \
    evaluation.base_model=true \
    evaluation.metrics_output_file="$METRICS_FILE" \
    dataset.small_sample=false \
    training.vllm_gpu_memory_utilization=0.9 \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

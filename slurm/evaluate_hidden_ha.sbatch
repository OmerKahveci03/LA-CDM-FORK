#!/bin/bash
#SBATCH --job-name=lacdm-hidden-ha
#SBATCH --output=logs/lacdm_hidden_ha_%j.out
#SBATCH --error=logs/lacdm_hidden_ha_%j.out
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00

# Ablation A4a (docs/FINAL_ANALYSES.md): a trained DA on the 240-case test split with the HA computed
# and logged but hidden from the DA. Same as slurm/evaluate_frozen_ha.sbatch except for the entry point.
#
#   sbatch slurm/evaluate_hidden_ha.sbatch                                                  # baseline DA, step 600
#   sbatch --export=ALL,RUN_ROOT=outputs/h3/h3a_ha0,ADAPTER_STEP=700 slurm/evaluate_hidden_ha.sbatch  # H3a DA
#   add SMOKE=1 for 2 patients
#
# Output: $RUN_ROOT/hidden_ha_seed<s>_step_<N>/ (never overwritten unless OVERWRITE=1).

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

RUN_ROOT="${RUN_ROOT:-outputs/cost_sweep/cost_1_attempt2}"
ADAPTER_STEP="${ADAPTER_STEP:-600}"
HA_SEED="${HA_SEED:-0}"
STEP_TAG=$(printf 'step_%04d' "$ADAPTER_STEP")
ADAPTER_DIR="$RUN_ROOT/train/adapters/$STEP_TAG"
FROZEN_HA_DIR="${FROZEN_HA_DIR:-outputs/ha_sft/seed_${HA_SEED}/adapter}"
OUT_DIR="$RUN_ROOT/hidden_ha_seed${HA_SEED}_$STEP_TAG"
SMALL=false
if [[ "${SMOKE:-0}" == "1" ]]; then
    OUT_DIR="${OUT_DIR}_smoke"
    SMALL=true
fi

for d in "$ADAPTER_DIR" "$FROZEN_HA_DIR"; do
    if [[ ! -f "$d/adapter_config.json" ||
          ( ! -f "$d/adapter_model.safetensors" && ! -f "$d/adapter_model.bin" ) ]]; then
        echo "No complete adapter at $d" >&2
        exit 2
    fi
done
if [[ -f "$OUT_DIR/metrics.json" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$OUT_DIR/metrics.json already exists; set OVERWRITE=1 to replace it." >&2
    exit 2
fi
mkdir -p "$OUT_DIR"

exec > >(tee -a "$OUT_DIR/eval_output.log") 2>&1
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
export VLLM_USE_V1=0 HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[SLURM] Job: ${SLURM_JOB_ID:-local}  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] DA adapter: $ADAPTER_DIR  HA (logged, hidden): $FROZEN_HA_DIR  Output: $OUT_DIR  Started: $(date)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.evaluate_hidden_ha \
    hydra.run.dir="$OUT_DIR" \
    "hydra.output_subdir=.hydra/$(date +%Y%m%d-%H%M%S)" \
    "experiment.name=la-cdm-hidden-ha" \
    "reward_function.reward_functions.2.multiplier=1" \
    evaluation.base_model=false \
    evaluation.adapter_dir="$ADAPTER_DIR" \
    evaluation.frozen_ha_dir="$FROZEN_HA_DIR" \
    evaluation.metrics_output_file="$OUT_DIR/metrics.json" \
    dataset.small_sample=$SMALL \
    training.vllm_gpu_memory_utilization=0.9 \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

#!/bin/bash
#SBATCH --job-name=lacdm-frozen-ha
#SBATCH --output=logs/lacdm_frozen_ha_%j.out
#SBATCH --error=logs/lacdm_frozen_ha_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# Step 2 (docs/STEP2_COMBINED.md): the trained DA with the frozen standalone HA, on the
# fixed 240-case test split. The DA's adapter drives vLLM as in a normal test run; the
# frozen HA gives the hypotheses and confidences.
#
#   sbatch --export=ALL,HA_SEED=0 slurm/evaluate_frozen_ha.sbatch           # primary
#   sbatch --export=ALL,HA_SEED=0,SMOKE=1 slurm/evaluate_frozen_ha.sbatch   # 2 patients
#
# Defaults: DA = attempt 2, step 600 (RUN_ROOT, ADAPTER_STEP). Output:
# $RUN_ROOT/frozen_ha_seed<s>_step_<N>/ (or ..._smoke). slurm/evaluate_cost_sweep.sbatch
# is unchanged. An existing result is never overwritten unless OVERWRITE=1.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

RUN_ROOT="${RUN_ROOT:-outputs/cost_sweep/cost_1_attempt2}"
ADAPTER_STEP="${ADAPTER_STEP:-600}"
HA_SEED="${HA_SEED:-0}"
COST_MULTIPLIER="${COST_MULTIPLIER:-1}"
STEP_TAG=$(printf 'step_%04d' "$ADAPTER_STEP")
ADAPTER_DIR="$RUN_ROOT/train/adapters/$STEP_TAG"
FROZEN_HA_DIR="${FROZEN_HA_DIR:-outputs/ha_sft/seed_${HA_SEED}/adapter}"
OUT_DIR="$RUN_ROOT/frozen_ha_seed${HA_SEED}_$STEP_TAG"
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
JOB_ID=${SLURM_JOB_ID:-local}
STARTED=$(date +%Y%m%d-%H%M%S)
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
log_job() {
    printf '%s\tjob=%s\tcommit=%s\t%s\n' "$(date '+%F %T')" "$JOB_ID" "$COMMIT" "$1" >> "$RUN_ROOT/jobs.log"
}
EVAL_NAME="test eval of $STEP_TAG with frozen HA seed $HA_SEED${SMOKE:+ (smoke)}"
log_job "$EVAL_NAME started"
trap 'rc=$?; log_job "$EVAL_NAME exit=$rc"' EXIT

export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[SLURM] Job: $JOB_ID  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] DA adapter: $ADAPTER_DIR"
echo "[SLURM] Frozen HA: $FROZEN_HA_DIR"
echo "[SLURM] Output: $OUT_DIR"
echo "[SLURM] Started: $(date)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.evaluate_frozen_ha \
    hydra.run.dir="$OUT_DIR" \
    "hydra.output_subdir=.hydra/${STARTED}" \
    "experiment.name=la-cdm-frozen-ha-seed${HA_SEED}" \
    "reward_function.reward_functions.2.multiplier=$COST_MULTIPLIER" \
    evaluation.base_model=false \
    evaluation.adapter_dir="$ADAPTER_DIR" \
    evaluation.frozen_ha_dir="$FROZEN_HA_DIR" \
    evaluation.metrics_output_file="$OUT_DIR/metrics.json" \
    dataset.small_sample=$SMALL \
    training.vllm_gpu_memory_utilization=0.9 \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

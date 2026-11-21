#!/bin/bash
#SBATCH --job-name=lacdm-con-eval
#SBATCH --output=logs/lacdm_con_eval_%j.out
#SBATCH --error=logs/lacdm_con_eval_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# H4 test runs (240 test patients) of a DA with the contrastive HA.
#
# Stage 3, the baseline DA (attempt 2, step 600) + the stage-1 heads:
#   sbatch --export=ALL,BACKBONE=base slurm/contrastive_evaluate.sbatch
#   -> outputs/contrastive/combined/<backbone>/
# Stage 4, a snapshot of a stage-4 run (DA + the heads saved with it):
#   sbatch --export=ALL,BACKBONE=base,RUN=joint_base,ADAPTER_STEP=700 slurm/contrastive_evaluate.sbatch
#   -> outputs/contrastive/joint/<RUN>/test_step_XXXX/
# add SMOKE=1 for the small sample (output ..._smoke). Never overwrites unless OVERWRITE=1.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

BACKBONE="${BACKBONE:?Set BACKBONE=base or BACKBONE=h2}"
if [[ -n "${RUN:-}" ]]; then
    : "${ADAPTER_STEP:?Set ADAPTER_STEP (chosen on validation)}"
    STEP_TAG=$(printf 'step_%04d' "$ADAPTER_STEP")
    RUN_ROOT="outputs/contrastive/joint/$RUN"
    DA_DIR="$RUN_ROOT/train/adapters/$STEP_TAG"
    HEADS="$DA_DIR/contrastive_heads.pt"
    OUT_DIR="$RUN_ROOT/test_$STEP_TAG"
else
    RUN_ROOT="outputs/contrastive/combined"
    DA_DIR=outputs/cost_sweep/cost_1_attempt2/train/adapters/step_0600
    HEADS="outputs/contrastive/heads/$BACKBONE/heads.pt"
    OUT_DIR="$RUN_ROOT/$BACKBONE"
fi
SMALL=false
if [[ "${SMOKE:-0}" == "1" ]]; then
    OUT_DIR="${OUT_DIR}_smoke"
    SMALL=true
fi
[[ -f "$DA_DIR/adapter_config.json" ]] || { echo "No DA adapter at $DA_DIR" >&2; exit 2; }
[[ -f "$HEADS" ]] || { echo "No heads at $HEADS" >&2; exit 2; }
if [[ -f "$OUT_DIR/metrics.json" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$OUT_DIR/metrics.json already exists; set OVERWRITE=1 to replace it." >&2
    exit 2
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/eval_output.log") 2>&1
STARTED=$(date +%Y%m%d-%H%M%S)
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"

export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[SLURM] Job: ${SLURM_JOB_ID:-local}  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] DA: $DA_DIR  Backbone: $BACKBONE  Heads: $HEADS  Output: $OUT_DIR  Started: $(date)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.contrastive_ha.evaluate \
    hydra.run.dir="$OUT_DIR" \
    "hydra.output_subdir=.hydra/${STARTED}" \
    "experiment.name=la-cdm-contrastive-eval" \
    +contrastive=defaults \
    "contrastive.backbone=$BACKBONE" \
    "contrastive.heads_path=$HEADS" \
    "reward_function.reward_functions.2.multiplier=1" \
    evaluation.base_model=false \
    evaluation.adapter_dir="$DA_DIR" \
    evaluation.metrics_output_file="$OUT_DIR/metrics.json" \
    dataset.small_sample=$SMALL \
    training.vllm_gpu_memory_utilization=0.9 \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

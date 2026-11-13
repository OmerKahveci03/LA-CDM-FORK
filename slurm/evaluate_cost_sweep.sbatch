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

# Evaluate one arm's adapter on the fixed 240-case test split.
#
#   sbatch --export=ALL,COST_MULTIPLIER=1 slurm/evaluate_cost_sweep.sbatch
#
# Uses the adapter saved at the arm's final training step. Set ADAPTER_STEP to
# evaluate an earlier cycle-end snapshot (for example 300 after extending an
# arm to 600 steps). Results go to outputs/cost_sweep/cost_<m>/test_step_<N>/.
# An existing result is never overwritten unless OVERWRITE=1.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

: "${COST_MULTIPLIER:?Set COST_MULTIPLIER, for example 1}"
if [[ ! "$COST_MULTIPLIER" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "COST_MULTIPLIER must be a non-negative number: $COST_MULTIPLIER" >&2
    exit 2
fi
ARM="cost_${COST_MULTIPLIER//./p}"
# RUN_ROOT overrides the arm folder, e.g. a moved earlier attempt:
#   --export=ALL,COST_MULTIPLIER=1,ADAPTER_STEP=300,RUN_ROOT=outputs/cost_sweep/cost_1_attempt2
RUN_ROOT="${RUN_ROOT:-outputs/cost_sweep/${ARM}}"
TRAIN_DIR="$RUN_ROOT/train"

if [[ -z "${ADAPTER_STEP:-}" ]]; then
    if [[ ! -f "$TRAIN_DIR/training_complete.json" ]]; then
        echo "Training for $ARM has not finished ($TRAIN_DIR/training_complete.json is missing)." >&2
        exit 2
    fi
    ADAPTER_STEP=$("$PYTHON_BIN" -c "import json,sys; print(json.load(open(sys.argv[1]))['global_step'])" "$TRAIN_DIR/training_complete.json")
fi
STEP_TAG=$(printf 'step_%04d' "$ADAPTER_STEP")
ADAPTER_DIR="$TRAIN_DIR/adapters/$STEP_TAG"
OUT_DIR="$RUN_ROOT/test_$STEP_TAG"

# Hypothesis H1 (docs/HYPOTHESIS_H1.md): SCRIPTED_POLICY=none|all|random_matched
# replaces the Decision Agent with a fixed test policy; the adapter's Hypothesis
# Agent diagnoses. Output: $RUN_ROOT/scripted_<policy>_step_<N>/ (kept apart from
# test_step_* so the normal test figures ignore it). random_matched copies each
# patient's test count from SCRIPTED_REFERENCE (default: this step's normal test run).
SCRIPTED_ARGS=()
if [[ -n "${SCRIPTED_POLICY:-}" ]]; then
    if [[ ! "$SCRIPTED_POLICY" =~ ^(none|all|random_matched)$ ]]; then
        echo "SCRIPTED_POLICY must be none, all or random_matched: $SCRIPTED_POLICY" >&2
        exit 2
    fi
    OUT_DIR="$RUN_ROOT/scripted_${SCRIPTED_POLICY}_$STEP_TAG"
    SCRIPTED_ARGS+=("evaluation.scripted_policy=$SCRIPTED_POLICY" "evaluation.scripted_seed=${SCRIPTED_SEED:-0}")
    if [[ "$SCRIPTED_POLICY" == "random_matched" ]]; then
        SCRIPTED_REFERENCE="${SCRIPTED_REFERENCE:-$RUN_ROOT/test_$STEP_TAG/test_episodes.csv}"
        if [[ ! -f "$SCRIPTED_REFERENCE" ]]; then
            echo "random_matched needs the reference episodes: $SCRIPTED_REFERENCE is missing" >&2
            exit 2
        fi
        SCRIPTED_ARGS+=("evaluation.scripted_reference_episodes=$SCRIPTED_REFERENCE")
    fi
fi

if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ||
      ( ! -f "$ADAPTER_DIR/adapter_model.safetensors" && ! -f "$ADAPTER_DIR/adapter_model.bin" ) ]]; then
    echo "No complete adapter at $ADAPTER_DIR" >&2
    exit 2
fi
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
EVAL_NAME="test eval of $STEP_TAG${SCRIPTED_POLICY:+ (scripted $SCRIPTED_POLICY)}"
log_job "$EVAL_NAME started"
trap 'rc=$?; log_job "$EVAL_NAME exit=$rc"' EXIT

export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[SLURM] Job: $JOB_ID  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] Adapter: $ADAPTER_DIR"
echo "[SLURM] Output: $OUT_DIR"
echo "[SLURM] Started: $(date)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.evaluate \
    hydra.run.dir="$OUT_DIR" \
    "hydra.output_subdir=.hydra/${STARTED}" \
    "experiment.name=la-cdm-${ARM}" \
    "reward_function.reward_functions.2.multiplier=$COST_MULTIPLIER" \
    evaluation.base_model=false \
    evaluation.adapter_dir="$ADAPTER_DIR" \
    evaluation.metrics_output_file="$OUT_DIR/metrics.json" \
    dataset.small_sample=false \
    training.vllm_gpu_memory_utilization=0.9 \
    "${SCRIPTED_ARGS[@]}" \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

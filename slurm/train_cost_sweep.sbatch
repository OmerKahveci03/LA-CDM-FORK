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

# Train one test-cost arm in a fixed directory, resuming automatically.
#
#   sbatch --export=ALL,COST_MULTIPLIER=1 slurm/train_cost_sweep.sbatch
#
# Submit the same command again after a time-out or failure: the job resumes
# from the newest complete checkpoint in outputs/cost_sweep/cost_<m>/train/.
# Once training has reached MAX_STEPS, resubmitting exits without work.
# Validation runs on all 240 validation cases every 100 steps (fixed), plus
# once before training on a fresh start.
# Optional: MAX_STEPS (default 300; raise it by 300 at a time to continue a
# finished arm), EVAL_ON_START=0 (skip the step-0 validation of the untrained
# model), START_FRESH=1 (refuses if checkpoints exist).
# Validation over time: python scripts/validation_curve.py outputs/cost_sweep/cost_<m>/train
# and scripts/plot_validation_curve.py for charts.

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
MAX_STEPS=${MAX_STEPS:-300}
EVAL_STEPS=100  # fixed, so every arm is validated at the same steps
ARM="cost_${COST_MULTIPLIER//./p}"
RUN_ROOT="outputs/cost_sweep/${ARM}"
TRAIN_DIR="$RUN_ROOT/train"
mkdir -p "$TRAIN_DIR"

# Everything this job prints is also appended to one log for the arm.
exec > >(tee -a "$RUN_ROOT/train_output.log") 2>&1
JOB_ID=${SLURM_JOB_ID:-local}
STARTED=$(date +%Y%m%d-%H%M%S)
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
log_job() {
    printf '%s\tjob=%s\tcommit=%s\t%s\n' "$(date '+%F %T')" "$JOB_ID" "$COMMIT" "$1" >> "$RUN_ROOT/jobs.log"
}
trap 'rc=$?; log_job "train exit=$rc"' EXIT

# Settings that change what the model learns and so must not change between
# jobs of the same arm. MAX_STEPS is excluded so a finished arm can be extended.
SETTINGS="cost_multiplier=$COST_MULTIPLIER num_generations=4 batch_size=1 max_completion_length=128 max_length=2048"
if [[ -f "$RUN_ROOT/run_settings.txt" ]]; then
    if [[ "$(cat "$RUN_ROOT/run_settings.txt")" != "$SETTINGS" ]]; then
        echo "[SLURM] Settings differ from this arm's first job:" >&2
        echo "  saved:   $(cat "$RUN_ROOT/run_settings.txt")" >&2
        echo "  current: $SETTINGS" >&2
        exit 2
    fi
else
    echo "$SETTINGS" > "$RUN_ROOT/run_settings.txt"
fi

if [[ -f "$TRAIN_DIR/training_complete.json" ]]; then
    DONE_STEP=$("$PYTHON_BIN" -c "import json,sys; print(json.load(open(sys.argv[1]))['global_step'])" "$TRAIN_DIR/training_complete.json")
    if (( DONE_STEP >= MAX_STEPS )); then
        echo "[SLURM] $ARM already completed $DONE_STEP steps (MAX_STEPS=$MAX_STEPS); nothing to do."
        log_job "already complete at step $DONE_STEP"
        trap - EXIT
        exit 0
    fi
fi

CHECKPOINT=$("$PYTHON_BIN" scripts/find_latest_training_checkpoint.py "$TRAIN_DIR" \
    --max-steps "$MAX_STEPS" --pattern 'checkpoint-*')
RESUME_ARGS=()
if [[ "${START_FRESH:-0}" == "1" ]]; then
    if [[ -n "$CHECKPOINT" ]]; then
        echo "[SLURM] START_FRESH=1 but $TRAIN_DIR has checkpoints. Move that directory first." >&2
        exit 2
    fi
fi
if [[ -n "$CHECKPOINT" ]]; then
    RESUME_ARGS+=("training.resume_from_checkpoint=$CHECKPOINT")
    log_job "resume from $(basename "$CHECKPOINT") to step $MAX_STEPS"
else
    log_job "start from step 0 to step $MAX_STEPS"
    # Validate the untrained model once, so the validation curve has a step-0
    # point measured exactly like the later ones (~2.5 h; EVAL_ON_START=0 skips).
    if [[ "${EVAL_ON_START:-1}" == "1" && ! -f "$TRAIN_DIR/episodes/val_step_0000_summary.json" ]]; then
        RESUME_ARGS+=("training.eval_on_start=true")
    fi
fi

export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[SLURM] Job: $JOB_ID  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] Arm: $ARM  Output: $TRAIN_DIR"
echo "[SLURM] Resume: ${CHECKPOINT:-none (step 0)}  Target: $MAX_STEPS steps"
echo "[SLURM] Started: $(date)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.train \
    hydra.run.dir="$TRAIN_DIR" \
    "hydra.output_subdir=.hydra/${STARTED}" \
    "experiment.name=la-cdm-${ARM}" \
    "reward_function.reward_functions.2.multiplier=$COST_MULTIPLIER" \
    dataset.small_sample=false \
    "training.max_steps=$MAX_STEPS" \
    "training.eval_steps=$EVAL_STEPS" \
    training.num_generations=4 \
    training.per_device_train_batch_size=1 \
    training.max_completion_length=128 \
    environment.max_length=2048 \
    training.save_steps=25 \
    training.vllm_gpu_memory_utilization=0.9 \
    "${RESUME_ARGS[@]}" \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

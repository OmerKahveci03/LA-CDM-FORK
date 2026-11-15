#!/bin/bash
#SBATCH --job-name=lacdm-h3-train
#SBATCH --output=logs/lacdm_h3_train_%j.out
#SBATCH --error=logs/lacdm_h3_train_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# H3 (docs/HYPOTHESIS_H3.md): train only the Decision Agent, against the frozen H2 HA.
#
#   sbatch --export=ALL,ARM=h3a slurm/train_frozen_ha.sbatch            # frozen HA, original reward
#   sbatch --export=ALL,ARM=h3a,SMOKE=1 slurm/train_frozen_ha.sbatch    # 4 steps, 2 val patients
#
# ARM=h3b (information-gain reward) is not implemented yet and is refused.
# Resubmit the same command after a time-out: it resumes from the newest checkpoint in
# outputs/h3/<arm>_ha<seed>/train/. Same DA settings as attempt 2 (4 generations,
# batch 1, 128-token completions, 2048-token transcripts, invalid episodes not charged),
# but every phase is a DA phase. Validation on all 240 validation cases every 100 steps.
# Optional: HA_SEED (default 0), MAX_STEPS (default 900), EVAL_ON_START=0.
# slurm/train_cost_sweep.sbatch and src/train.py are unchanged.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

ARM="${ARM:?Set ARM=h3a}"
case "$ARM" in
    h3a) ;;
    h3b) echo "ARM=h3b (information-gain reward) is not implemented yet" >&2; exit 2 ;;
    *) echo "ARM must be h3a: $ARM" >&2; exit 2 ;;
esac
HA_SEED="${HA_SEED:-0}"
FROZEN_HA_DIR="outputs/ha_sft/seed_${HA_SEED}/adapter"
MAX_STEPS=${MAX_STEPS:-900}
EVAL_STEPS=100
RUN_ROOT="outputs/h3/${ARM}_ha${HA_SEED}"
SMOKE_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    RUN_ROOT="outputs/h3/smoke_${ARM}_ha${HA_SEED}"
    MAX_STEPS=4
    EVAL_STEPS=2
    SMOKE_ARGS+=("dataset.small_sample=true")
else
    SMOKE_ARGS+=("dataset.small_sample=false")
fi
TRAIN_DIR="$RUN_ROOT/train"
if [[ ! -f "$FROZEN_HA_DIR/adapter_config.json" ]]; then
    echo "No frozen HA at $FROZEN_HA_DIR" >&2
    exit 2
fi
mkdir -p "$TRAIN_DIR"

exec > >(tee -a "$RUN_ROOT/train_output.log") 2>&1
JOB_ID=${SLURM_JOB_ID:-local}
STARTED=$(date +%Y%m%d-%H%M%S)
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
log_job() {
    printf '%s\tjob=%s\tcommit=%s\t%s\n' "$(date '+%F %T')" "$JOB_ID" "$COMMIT" "$1" >> "$RUN_ROOT/jobs.log"
}
trap 'rc=$?; log_job "train exit=$rc"' EXIT

# Settings that must not change between jobs of the same arm (MAX_STEPS may grow).
SETTINGS="arm=$ARM frozen_ha=$FROZEN_HA_DIR schedule=da_only cost_multiplier=1 num_generations=4 batch_size=1 max_completion_length=128 max_length=2048 charge_invalid=false"
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

if [[ -f "$TRAIN_DIR/collapse_detected.json" ]]; then
    echo "[SLURM] $ARM was stopped by the collapse guard; not resuming:" >&2
    cat "$TRAIN_DIR/collapse_detected.json" >&2
    trap - EXIT
    exit 0
fi
if [[ -f "$TRAIN_DIR/training_complete.json" ]]; then
    DONE_STEP=$("$PYTHON_BIN" -c "import json,sys; print(json.load(open(sys.argv[1]))['global_step'])" "$TRAIN_DIR/training_complete.json")
    if (( DONE_STEP >= MAX_STEPS )); then
        echo "[SLURM] $ARM already completed $DONE_STEP steps (MAX_STEPS=$MAX_STEPS); nothing to do."
        trap - EXIT
        exit 0
    fi
fi

CHECKPOINT=$("$PYTHON_BIN" scripts/find_latest_training_checkpoint.py "$TRAIN_DIR" \
    --max-steps "$MAX_STEPS" --pattern 'checkpoint-*')
RESUME_ARGS=()
if [[ -n "$CHECKPOINT" ]]; then
    RESUME_ARGS+=("training.resume_from_checkpoint=$CHECKPOINT")
    log_job "resume from $(basename "$CHECKPOINT") to step $MAX_STEPS"
else
    log_job "start from step 0 to step $MAX_STEPS"
    if [[ "${EVAL_ON_START:-1}" == "1" && ! -f "$TRAIN_DIR/episodes/val_step_0000_summary.json" ]]; then
        RESUME_ARGS+=("training.eval_on_start=true")
    fi
fi

export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[SLURM] Job: $JOB_ID  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] Arm: $ARM  Frozen HA: $FROZEN_HA_DIR  Output: $TRAIN_DIR"
echo "[SLURM] Resume: ${CHECKPOINT:-none (step 0)}  Target: $MAX_STEPS steps"
echo "[SLURM] Started: $(date)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.train_frozen_ha \
    hydra.run.dir="$TRAIN_DIR" \
    "hydra.output_subdir=.hydra/${STARTED}" \
    "experiment.name=la-cdm-${ARM}-ha${HA_SEED}" \
    evaluation.frozen_ha_dir="$FROZEN_HA_DIR" \
    "reward_function.reward_functions.2.multiplier=1" \
    "training.loss_schedule.phases=[{name:da,steps:100,learning_rate:1e-5}]" \
    "${SMOKE_ARGS[@]}" \
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

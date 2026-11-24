#!/bin/bash
#SBATCH --job-name=lacdm-con-joint
#SBATCH --output=logs/lacdm_con_joint_%j.out
#SBATCH --error=logs/lacdm_con_joint_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=24:00:00

# H4 stage 4 (docs/HYPOTHESIS_H4.md): a new DA trained with the contrastive HA.
#
#   sbatch --export=ALL,ARM=joint,BACKBONE=base  slurm/contrastive_train_joint.sbatch  # heads keep learning
#   sbatch --export=ALL,ARM=frozen,BACKBONE=base slurm/contrastive_train_joint.sbatch  # heads frozen
#   add SMOKE=1 for 4 steps on the small sample (outputs/contrastive/joint/smoke_<arm>_<backbone>)
#
# The heads start from stage 1 (outputs/contrastive/heads/<backbone>/heads.pt); the backbone stays
# frozen. Same DA settings as H3a (original reward, 4 generations, 2048-token transcripts, da phases
# only, validation every 100 steps). Output: outputs/contrastive/joint/<arm>_<backbone>/train/; the
# heads of each snapshot: train/adapters/step_XXXX/contrastive_heads.pt. Resubmit after a time-out.
# Optional: MAX_STEPS (default 700), EVAL_ON_START=0.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

ARM="${ARM:?Set ARM=joint or ARM=frozen}"
case "$ARM" in
    joint) UPDATE=true ;;
    frozen) UPDATE=false ;;
    *) echo "ARM must be joint or frozen: $ARM" >&2; exit 2 ;;
esac
BACKBONE="${BACKBONE:?Set BACKBONE=base or BACKBONE=h2 (the one chosen in stage 1)}"
HEADS="outputs/contrastive/heads/$BACKBONE/heads.pt"
REPLAY="outputs/contrastive/embeddings/$BACKBONE/train.pt"
MAX_STEPS=${MAX_STEPS:-700}
EVAL_STEPS=100
RUN_ROOT="outputs/contrastive/joint/${ARM}_${BACKBONE}"
SMOKE_ARGS=("dataset.small_sample=false")
if [[ "${SMOKE:-0}" == "1" ]]; then
    RUN_ROOT="outputs/contrastive/joint/smoke_${ARM}_${BACKBONE}"
    MAX_STEPS=4
    EVAL_STEPS=2
    SMOKE_ARGS=("dataset.small_sample=true")
fi
TRAIN_DIR="$RUN_ROOT/train"
for f in "$HEADS" "$REPLAY"; do
    [[ -f "$f" ]] || { echo "Missing $f (run stage 1 first)" >&2; exit 2; }
done
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
SETTINGS="arm=$ARM backbone=$BACKBONE heads=$HEADS update=$UPDATE schedule=da_only cost_multiplier=1 num_generations=4 batch_size=1 max_completion_length=128 max_length=2048 charge_invalid=false"
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
# A rebuilt checkpoint (scripts/rebuild_checkpoint.py) has no optimizer.pt, so the search above
# skips it (and may instead find an older, complete one, e.g. the Trainer's kept "best"
# checkpoint-100). RESUME_CHECKPOINT is used whenever it is further along than what was found,
# so later jobs switch to the run's own newer checkpoints automatically.
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    [[ -f "$RESUME_CHECKPOINT/trainer_state.json" ]] || { echo "No trainer_state.json in $RESUME_CHECKPOINT" >&2; exit 2; }
    RESUME_STEP=${RESUME_CHECKPOINT##*-}
    FOUND_STEP=${CHECKPOINT##*-}
    if [[ -z "$CHECKPOINT" ]] || (( ${FOUND_STEP:-0} < RESUME_STEP )); then
        CHECKPOINT="$RESUME_CHECKPOINT"
    fi
fi
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
echo "[SLURM] Arm: $ARM  Backbone: $BACKBONE  Heads: $HEADS (update=$UPDATE)  Output: $TRAIN_DIR"
echo "[SLURM] Resume: ${CHECKPOINT:-none (step 0)}  Target: $MAX_STEPS steps"
echo "[SLURM] Started: $(date)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.contrastive_ha.train_joint \
    hydra.run.dir="$TRAIN_DIR" \
    "hydra.output_subdir=.hydra/${STARTED}" \
    "experiment.name=la-cdm-contrastive-${ARM}" \
    +contrastive=defaults \
    "contrastive.backbone=$BACKBONE" \
    "contrastive.heads_path=$HEADS" \
    "contrastive.replay_path=$REPLAY" \
    "contrastive.update=$UPDATE" \
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

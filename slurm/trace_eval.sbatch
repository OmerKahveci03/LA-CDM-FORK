#!/bin/bash
#SBATCH --job-name=trace-cdm-eval
#SBATCH --output=logs/trace_cdm_eval_%j.out
#SBATCH --error=logs/trace_cdm_eval_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Evaluate one TRACE-CDM arm with the trained LA-CDM adapter on the test split.
#   ARM=baseline|shadow|deterministic        -> 2 GPUs (policy + vLLM)
#   ARM=shadow_adaptive|adaptive|always|shuffled -> 3 GPUs (+ critic on cuda:2):
#       sbatch --gres=gpu:l4:3 --export=ALL,ARM=adaptive,ADAPTER_DIR=...,TRACE_CRITIC_MODEL=... slurm/trace_eval.sbatch
# scripts/submit_trace_arms.sh submits every arm with the right GPU count.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python

cd "$REPO_DIR"
mkdir -p logs

: "${ARM:?Set ARM to one of baseline, shadow, shadow_adaptive, deterministic, adaptive, always, shuffled}"
case "$ARM" in
    baseline|shadow|deterministic) NEEDS_CRITIC=0 ;;
    shadow_adaptive|adaptive|always|shuffled) NEEDS_CRITIC=1 ;;
    *) echo "Unknown ARM: $ARM" >&2; exit 2 ;;
esac

ADAPTER_DIR=${ADAPTER_DIR:-$("$PYTHON_BIN" scripts/find_latest_training_adapter.py outputs)}
if [[ -z "$ADAPTER_DIR" || ! -f "$ADAPTER_DIR/adapter_config.json" ||
      ( ! -f "$ADAPTER_DIR/adapter_model.safetensors" && ! -f "$ADAPTER_DIR/adapter_model.bin" ) ]]; then
    echo "No completed adapter at '$ADAPTER_DIR'" >&2
    exit 2
fi

CRITIC_ARGS=()
if [[ "$NEEDS_CRITIC" == 1 ]]; then
    : "${TRACE_CRITIC_MODEL:?Set TRACE_CRITIC_MODEL to a verified local instruction checkpoint directory}"
    if [[ ! -f "$TRACE_CRITIC_MODEL/config.json" ]]; then
        echo "TRACE_CRITIC_MODEL has no config.json: $TRACE_CRITIC_MODEL" >&2
        exit 2
    fi
    GPUS=$(nvidia-smi -L | wc -l)
    CRITIC_DEVICE=${CRITIC_DEVICE:-cuda:2}
    if [[ "$CRITIC_DEVICE" == cuda:2 && "$GPUS" -lt 3 ]]; then
        echo "ARM=$ARM needs a third GPU for the critic (found $GPUS). Submit with --gres=gpu:l4:3," >&2
        echo "or set CRITIC_DEVICE=cuda:0 CRITIC_4BIT=true to share the policy GPU." >&2
        exit 2
    fi
    CRITIC_ARGS=("trace_cdm.critic.device=$CRITIC_DEVICE" "trace_cdm.critic.load_in_4bit=${CRITIC_4BIT:-false}")
fi

# Never download models or datasets from compute nodes.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="outputs/trace_cdm/${ARM}/eval_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
{
    echo "arm: $ARM"
    echo "git_commit: $(git rev-parse HEAD)"
    echo "git_dirty: $([[ -n $(git status --porcelain --untracked-files=no) ]] && echo true || echo false)"
    echo "adapter_dir: $ADAPTER_DIR"
    echo "adapter_sha256: $(sha256sum "$ADAPTER_DIR"/adapter_model.* | head -1 | cut -d' ' -f1)"
    echo "critic_model: ${TRACE_CRITIC_MODEL:-none}"
    echo "slurm_job_id: $SLURM_JOB_ID"
    echo "gpus: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';')"
    echo "started: $(date -Iseconds)"
} > "$RUN_DIR/run_info.yaml"

echo "[SLURM] Arm: $ARM  Adapter: $ADAPTER_DIR  Output: $RUN_DIR"
echo "[SLURM] Started: $(date)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.evaluate \
    hydra.run.dir="$RUN_DIR" \
    "experiment.name=la-cdm-trace-${ARM}" \
    trace_cdm="$ARM" \
    "${CRITIC_ARGS[@]}" \
    evaluation.base_model=false \
    evaluation.adapter_dir="$ADAPTER_DIR" \
    evaluation.metrics_output_file="$RUN_DIR/metrics.json" \
    dataset.small_sample="${SMALL_SAMPLE:-false}" \
    training.vllm_gpu_memory_utilization=0.9 \
    wandb.mode=disabled

echo "finished: $(date -Iseconds)" >> "$RUN_DIR/run_info.yaml"
echo "[SLURM] Finished: $(date)"

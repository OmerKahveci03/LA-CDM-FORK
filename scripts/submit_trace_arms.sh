#!/usr/bin/env bash
# Submit TRACE-CDM evaluation arms with the GPU count each one needs.
#
#   ADAPTER_DIR=outputs/train_full_<JOB>/checkpoint-300 \
#   TRACE_CRITIC_MODEL=/data/ai/models/nlp/mistral_ai/<verified-instruct-checkpoint> \
#   bash scripts/submit_trace_arms.sh [arm ...]
#
# With no arguments: baseline shadow deterministic shadow_adaptive adaptive always shuffled.
# Set SMALL_SAMPLE=true for a 2-patient smoke run of every arm.
set -euo pipefail

: "${ADAPTER_DIR:?Set ADAPTER_DIR to the trained adapter (the same one for every arm)}"
ARMS=("$@")
if [[ ${#ARMS[@]} -eq 0 ]]; then
    ARMS=(baseline shadow deterministic shadow_adaptive adaptive always shuffled)
fi

for ARM in "${ARMS[@]}"; do
    case "$ARM" in
        baseline|shadow|deterministic) GRES=gpu:l4:2 ;;
        shadow_adaptive|adaptive|always|shuffled)
            : "${TRACE_CRITIC_MODEL:?Set TRACE_CRITIC_MODEL for critic arms}"
            GRES=gpu:l4:3 ;;
        *) echo "Unknown arm: $ARM" >&2; exit 2 ;;
    esac
    sbatch --gres="$GRES" --job-name="trace-${ARM}" \
        --export=ALL,ARM="$ARM",ADAPTER_DIR="$ADAPTER_DIR",SMALL_SAMPLE="${SMALL_SAMPLE:-false}" \
        slurm/trace_eval.sbatch
done

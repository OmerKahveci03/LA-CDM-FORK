#!/bin/bash
#SBATCH --job-name=lacdm-clm-eval
#SBATCH --output=logs/lacdm_clm_eval_%j.out
#SBATCH --error=logs/lacdm_clm_eval_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# H4 stage 4: test evaluation (240 test patients) of a stage-4 checkpoint: the DA adapter and
# the CLM HA adapter saved at the same step.
#
#   sbatch --export=ALL,RUN=joint_h2init_seed0,ADAPTER_STEP=700 slurm/clm_evaluate_joint.sbatch
#   add SMOKE=1 for the small sample (output ..._smoke)
#
# Reads outputs/clm/joint/<RUN>/train/adapters/step_XXXX (DA) and .../step_XXXX/clm_ha (HA).
# Output: outputs/clm/joint/<RUN>/test_step_XXXX/ (test_episodes.csv, metrics.json,
# clm_notes.jsonl, ...). Never overwrites a finished result unless OVERWRITE=1.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

RUN="${RUN:?Set RUN to a run under outputs/clm/joint/}"
ADAPTER_STEP="${ADAPTER_STEP:?Set ADAPTER_STEP (chosen on validation)}"
RUN_ROOT="outputs/clm/joint/$RUN"
STEP_TAG=$(printf 'step_%04d' "$ADAPTER_STEP")
DA_DIR="$RUN_ROOT/train/adapters/$STEP_TAG"
HA_DIR="$DA_DIR/clm_ha"
OUT_DIR="$RUN_ROOT/test_$STEP_TAG"
SMALL=false
if [[ "${SMOKE:-0}" == "1" ]]; then
    OUT_DIR="${OUT_DIR}_smoke"
    SMALL=true
fi
for d in "$DA_DIR" "$HA_DIR"; do
    if [[ ! -f "$d/adapter_config.json" ]]; then
        echo "No adapter at $d" >&2
        exit 2
    fi
done
if [[ -f "$OUT_DIR/metrics.json" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$OUT_DIR/metrics.json already exists; set OVERWRITE=1 to replace it." >&2
    exit 2
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/eval_output.log") 2>&1
STARTED=$(date +%Y%m%d-%H%M%S)
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
printf '%s\tjob=%s\tcommit=%s\ttest eval of %s\n' "$(date '+%F %T')" "${SLURM_JOB_ID:-local}" "$COMMIT" "$STEP_TAG" >> "$RUN_ROOT/jobs.log"

export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[SLURM] Job: ${SLURM_JOB_ID:-local}  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] DA: $DA_DIR  CLM HA: $HA_DIR  Output: $OUT_DIR  Started: $(date)"

"$PYTHON_BIN" -m accelerate.commands.launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    -m src.clm.evaluate_joint \
    hydra.run.dir="$OUT_DIR" \
    "hydra.output_subdir=.hydra/${STARTED}" \
    "experiment.name=la-cdm-clm-eval" \
    +clm=defaults \
    "clm.ha_dir=$HA_DIR" \
    "reward_function.reward_functions.2.multiplier=1" \
    evaluation.base_model=false \
    evaluation.adapter_dir="$DA_DIR" \
    evaluation.metrics_output_file="$OUT_DIR/metrics.json" \
    dataset.small_sample=$SMALL \
    training.vllm_gpu_memory_utilization=0.9 \
    wandb.mode=disabled

echo "[SLURM] Finished: $(date)"

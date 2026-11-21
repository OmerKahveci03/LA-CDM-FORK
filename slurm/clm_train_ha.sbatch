#!/bin/bash
#SBATCH --job-name=lacdm-clm-ha
#SBATCH --output=logs/lacdm_clm_ha_%j.out
#SBATCH --error=logs/lacdm_clm_ha_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# H4 stage 3 (H4b, docs/HYPOTHESIS_H4.md): train the CLM HA alone on replayed test sequences.
#
#   sbatch --export=ALL,INIT=h2   slurm/clm_train_ha.sbatch          # start from the H2 HA (seed 0)
#   sbatch --export=ALL,INIT=base slurm/clm_train_ha.sbatch          # start from a fresh LoRA
#   add SMOKE=1 for 4 steps on 8 patients (outputs/clm/ha_train/smoke_<init>)
#   optional: HA_SEED (default 0), STEPS (default 600)
#
# Output: outputs/clm/ha_train/<init>init_seed<s>/ (adapter/, val_curve.csv, updates.jsonl, ...).
# Resubmit the same command after a time-out: it resumes from <run>/checkpoint/.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

INIT="${INIT:?Set INIT=h2 or INIT=base}"
HA_SEED="${HA_SEED:-0}"
case "$INIT" in
    h2) INIT_ADAPTER="outputs/ha_sft/seed_0/adapter" ;;
    base) INIT_ADAPTER=none ;;
    *) echo "INIT must be h2 or base: $INIT" >&2; exit 2 ;;
esac
if [[ "$INIT_ADAPTER" != none && ! -f "$INIT_ADAPTER/adapter_config.json" ]]; then
    echo "No adapter at $INIT_ADAPTER" >&2
    exit 2
fi
EXTRA=(--steps "${STEPS:-600}")
OUT_DIR="outputs/clm/ha_train/${INIT}init_seed${HA_SEED}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    OUT_DIR="outputs/clm/ha_train/smoke_${INIT}"
    EXTRA=(--smoke)
fi
if [[ -f "$OUT_DIR/training_complete.json" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$OUT_DIR/training_complete.json exists; nothing to do (OVERWRITE=1 to start again)." >&2
    exit 0
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/train_output.log") 2>&1

COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
printf '%s\tjob=%s\tcommit=%s\tstart (init=%s)\n' "$(date '+%F %T')" "${SLURM_JOB_ID:-local}" "$COMMIT" "$INIT" >> "$OUT_DIR/jobs.log"
echo "[SLURM] Job: ${SLURM_JOB_ID:-local}  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] Init: $INIT ($INIT_ADAPTER)  Seed: $HA_SEED  Output: $OUT_DIR  Started: $(date)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_BIN" -m src.clm.train_ha --output-dir "$OUT_DIR" --init "$INIT_ADAPTER" --seed "$HA_SEED" "${EXTRA[@]}"

echo "[SLURM] Finished: $(date)"

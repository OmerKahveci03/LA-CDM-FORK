#!/bin/bash
#SBATCH --job-name=lacdm-ha-sft
#SBATCH --output=logs/lacdm_ha_sft_%j.out
#SBATCH --error=logs/lacdm_ha_sft_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Hypothesis H2 (docs/HYPOTHESIS_H2.md): train the standalone Hypothesis Agent by SFT.
#
#   sbatch --export=ALL,HA_SEED=0 slurm/train_ha_sft.sbatch          # primary run
#   sbatch --export=ALL,HA_SEED=0,SMOKE=1 slurm/train_ha_sft.sbatch  # 10-step smoke test
#
# Output: outputs/ha_sft/seed_<s>/ (smoke: outputs/ha_sft/smoke/). The model is split over
# both L4s. Resubmitting the same command resumes from the newest checkpoint.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

HA_SEED="${HA_SEED:-0}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    OUT_DIR=outputs/ha_sft/smoke
    EXTRA=(--smoke)
else
    OUT_DIR="outputs/ha_sft/seed_${HA_SEED}"
    EXTRA=()
fi
if [[ -f "$OUT_DIR/training_complete.json" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$OUT_DIR/training_complete.json exists; set OVERWRITE=1 to train again." >&2
    exit 2
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/train_output.log") 2>&1

COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
echo "[SLURM] Job: ${SLURM_JOB_ID:-local}  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] Output: $OUT_DIR  Seed: $HA_SEED  Started: $(date)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_BIN" -m src.train_ha_sft --output-dir "$OUT_DIR" --seed "$HA_SEED" "${EXTRA[@]}"

echo "[SLURM] Finished: $(date)"

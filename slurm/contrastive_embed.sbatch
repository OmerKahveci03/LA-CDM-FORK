#!/bin/bash
#SBATCH --job-name=lacdm-con-embed
#SBATCH --output=logs/lacdm_con_embed_%j.out
#SBATCH --error=logs/lacdm_con_embed_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=12:00:00

# H4 stage 1a (docs/HYPOTHESIS_H4.md, docs/CONTRASTIVE_RUNBOOK.md): embed every H2 training and
# validation record and the fixed test records with the frozen backbone.
#
#   sbatch --export=ALL,BACKBONE=base slurm/contrastive_embed.sbatch   # Llama-3-8B-Instruct, no adapter
#   sbatch --export=ALL,BACKBONE=h2   slurm/contrastive_embed.sbatch   # + the frozen H2 HA adapter
#   add SMOKE=1 for 8 patients per split/arm (outputs/contrastive/embeddings_smoke/<backbone>)
#
# Output: outputs/contrastive/embeddings/<backbone>/{train.pt,val.pt,actions.pt,test/<arm>.pt}.
# Finished files are skipped, so resubmitting continues an interrupted job.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

BACKBONE="${BACKBONE:?Set BACKBONE=base or BACKBONE=h2}"
case "$BACKBONE" in base|h2) ;; *) echo "BACKBONE must be base or h2" >&2; exit 2 ;; esac
OUT_DIR="outputs/contrastive/embeddings/$BACKBONE"
LIMIT_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    OUT_DIR="outputs/contrastive/embeddings_smoke/$BACKBONE"
    LIMIT_ARGS=(--limit 8)
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/embed_output.log") 2>&1

A2=outputs/cost_sweep/cost_1_attempt2
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
echo "[SLURM] Job: ${SLURM_JOB_ID:-local}  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] Backbone: $BACKBONE  Output: $OUT_DIR  Started: $(date)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_BIN" -m src.contrastive_ha.embed --backbone "$BACKBONE" --out-dir "$OUT_DIR" "${LIMIT_ARGS[@]}" \
    --arm "A_da2=$A2/test_step_0600/test_episodes.csv" \
    --arm "B_none=$A2/scripted_none_step_0600/test_episodes.csv" \
    --arm "C_random2=$A2/scripted_random_matched_step_0600/test_episodes.csv" \
    --arm "D_all=$A2/scripted_all_step_0600/test_episodes.csv"

echo "[SLURM] Finished: $(date)"

#!/bin/bash
#SBATCH --job-name=lacdm-con-heads
#SBATCH --output=logs/lacdm_con_heads_%j.out
#SBATCH --error=logs/lacdm_con_heads_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# H4 stage 1b: train the contrastive heads on the cached embeddings, choose the epoch on
# validation, then score the fixed test records.
#
#   sbatch --export=ALL,BACKBONE=base slurm/contrastive_heads.sbatch
#   sbatch --export=ALL,BACKBONE=h2   slurm/contrastive_heads.sbatch
#   add SMOKE=1 to use the smoke embeddings (outputs/contrastive/*_smoke/)
#
# Output: outputs/contrastive/heads/<backbone>/{heads.pt,curve.csv,summary.json} and
# outputs/contrastive/scores/<backbone>/<arm>/{test_episodes.csv,ha_probs.csv,summary.json}.
# Never overwrites finished heads unless OVERWRITE=1.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

BACKBONE="${BACKBONE:?Set BACKBONE=base or BACKBONE=h2}"
SUFFIX=""
[[ "${SMOKE:-0}" == "1" ]] && SUFFIX="_smoke"
EMB_DIR="outputs/contrastive/embeddings$SUFFIX/$BACKBONE"
OUT_DIR="outputs/contrastive/heads$SUFFIX/$BACKBONE"
SCORES_DIR="outputs/contrastive/scores$SUFFIX/$BACKBONE"
for f in train.pt val.pt actions.pt; do
    [[ -f "$EMB_DIR/$f" ]] || { echo "Missing $EMB_DIR/$f (run slurm/contrastive_embed.sbatch first)" >&2; exit 2; }
done
if [[ -f "$OUT_DIR/heads.pt" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$OUT_DIR/heads.pt exists; set OVERWRITE=1 to train again." >&2
    exit 2
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/heads_output.log") 2>&1
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
echo "[SLURM] Job: ${SLURM_JOB_ID:-local}  Commit: $COMMIT  Backbone: $BACKBONE  Started: $(date)"

"$PYTHON_BIN" -m src.contrastive_ha.train_heads --emb-dir "$EMB_DIR" --out-dir "$OUT_DIR" --scores-dir "$SCORES_DIR"

echo "[SLURM] Finished: $(date)"

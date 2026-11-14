#!/bin/bash
#SBATCH --job-name=lacdm-score-ha
#SBATCH --output=logs/lacdm_score_ha_%j.out
#SBATCH --error=logs/lacdm_score_ha_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# Hypothesis H2 (docs/HYPOTHESIS_H2.md): score one Hypothesis Agent on the fixed H1 test sets.
#
#   sbatch --export=ALL,HA=new0 slurm/score_ha.sbatch    # outputs/ha_sft/seed_0/adapter
#   sbatch --export=ALL,HA=old2 slurm/score_ha.sbatch    # attempt 2 step 600 (shared adapter)
#   sbatch --export=ALL,HA=old3 slurm/score_ha.sbatch    # attempt 3 step 600
#   sbatch --export=ALL,HA=base slurm/score_ha.sbatch    # base model, no adapter
#   add LIMIT=5 for a smoke test (5 patients per arm, written to scores_smoke/)
#
# Output: outputs/ha_sft/scores/<HA>/<arm>/{test_episodes.csv,ha_probs.csv,summary.json}

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

A2=outputs/cost_sweep/cost_1_attempt2
A3=outputs/cost_sweep/cost_1
: "${HA:?Set HA to new<seed> (e.g. new0), old2, old3 or base}"
case "$HA" in
    new*) ADAPTER="outputs/ha_sft/seed_${HA#new}/adapter" ;;
    old2) ADAPTER="$A2/train/adapters/step_0600" ;;
    old3) ADAPTER="$A3/train/adapters/step_0600" ;;
    base) ADAPTER=none ;;
    *) echo "Unknown HA=$HA" >&2; exit 2 ;;
esac
if [[ "$ADAPTER" != none && ! -f "$ADAPTER/adapter_config.json" ]]; then
    echo "No adapter at $ADAPTER" >&2
    exit 2
fi

LIMIT_ARGS=()
SCORES=outputs/ha_sft/scores
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")
    SCORES=outputs/ha_sft/scores_smoke
fi
OUT_DIR="$SCORES/$HA"
if [[ -f "$OUT_DIR/overview.json" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$OUT_DIR/overview.json exists; set OVERWRITE=1 to replace it." >&2
    exit 2
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/score_output.log") 2>&1

COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
echo "[SLURM] Job: ${SLURM_JOB_ID:-local}  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] HA: $HA  Adapter: $ADAPTER  Output: $OUT_DIR  Started: $(date)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Arms: the test lists of the H1 runs. A = the trained DA's tests (attempt 2 / 3),
# B = none, C = random with the DA's count (attempt 2 / 3), D = all available.
"$PYTHON_BIN" -m src.score_ha --adapter "$ADAPTER" --out-dir "$OUT_DIR" "${LIMIT_ARGS[@]}" \
    --arm "A_da2=$A2/test_step_0600/test_episodes.csv" \
    --arm "B_none=$A2/scripted_none_step_0600/test_episodes.csv" \
    --arm "C_random2=$A2/scripted_random_matched_step_0600/test_episodes.csv" \
    --arm "D_all=$A2/scripted_all_step_0600/test_episodes.csv" \
    --arm "A_da3=$A3/test_step_0600/test_episodes.csv" \
    --arm "C_random3=$A3/scripted_random_matched_step_0600/test_episodes.csv"

echo "[SLURM] Finished: $(date)"

#!/bin/bash
#SBATCH --job-name=lacdm-clm-replay
#SBATCH --output=logs/lacdm_clm_replay_%j.out
#SBATCH --error=logs/lacdm_clm_replay_%j.err
#SBATCH --account=cai6734
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# H4 (docs/HYPOTHESIS_H4.md, docs/CLM_RUNBOOK.md): the CLM HA on the fixed H1/H2 test
# sequences, one test result per turn. Stage 2 (untrained) and test scoring of stage 3.
#
#   sbatch --export=ALL,MODEL=base,MODE=edit   slurm/clm_replay.sbatch   # stage 2: untrained, may rewrite
#   sbatch --export=ALL,MODEL=base,MODE=append slurm/clm_replay.sbatch   # stage 2: untrained, append-only
#   sbatch --export=ALL,MODEL=h2,MODE=edit     slurm/clm_replay.sbatch   # stage 2: H2 HA (seed 0), may rewrite
#   sbatch --export=ALL,MODEL=h2,MODE=append   slurm/clm_replay.sbatch   # = the H2 scores (sanity check)
#   sbatch --export=ALL,MODEL=stage3,HA_RUN=h2init_seed0,MODE=edit slurm/clm_replay.sbatch  # trained CLM HA
#   add SMOKE=1 for 5 patients per arm (written to outputs/clm/replay_smoke/)
#
# Output: outputs/clm/replay/<name>/<arm>/{test_episodes.csv,ha_probs.csv,notes.jsonl,summary.json}
# with <name> = base_edit, base_append, h2_edit, h2_append or stage3-<HA_RUN>_<MODE>.
# Never overwrites a finished result unless OVERWRITE=1.

set -euo pipefail

REPO_DIR=/blue/prismap-ai-core/omerkahveci/LA-CDM-FORK
PYTHON_BIN=/blue/prismap-ai-core/omerkahveci/envs/la-cdm/bin/python
cd "$REPO_DIR"
mkdir -p logs

A2=outputs/cost_sweep/cost_1_attempt2
: "${MODEL:?Set MODEL=base, h2 or stage3 (with HA_RUN)}"
: "${MODE:?Set MODE=edit or append}"
case "$MODEL" in
    base) ADAPTER=none; NAME="base_${MODE}" ;;
    h2) ADAPTER="outputs/ha_sft/seed_${HA_SEED:-0}/adapter"; NAME="h2_${MODE}" ;;
    stage3) : "${HA_RUN:?Set HA_RUN to a run under outputs/clm/ha_train/}"
            ADAPTER="outputs/clm/ha_train/${HA_RUN}/adapter"; NAME="stage3-${HA_RUN}_${MODE}" ;;
    *) echo "Unknown MODEL=$MODEL" >&2; exit 2 ;;
esac
if [[ "$ADAPTER" != none && ! -f "$ADAPTER/adapter_config.json" ]]; then
    echo "No adapter at $ADAPTER" >&2
    exit 2
fi

ROOT=outputs/clm/replay
LIMIT_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
    ROOT=outputs/clm/replay_smoke
    LIMIT_ARGS=(--limit 5)
fi
OUT_DIR="$ROOT/$NAME"
if [[ -f "$OUT_DIR/overview.json" && "${OVERWRITE:-0}" != "1" ]]; then
    echo "$OUT_DIR/overview.json exists; set OVERWRITE=1 to replace it." >&2
    exit 2
fi
mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/replay_output.log") 2>&1

COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs prompts || echo '-dirty')"
echo "[SLURM] Job: ${SLURM_JOB_ID:-local}  Commit: $COMMIT  Host: $(hostname)"
echo "[SLURM] Model: $MODEL ($ADAPTER)  Mode: $MODE  Output: $OUT_DIR  Started: $(date)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Arms (as in H2): A = the baseline DA's tests, B = none, C = random with the DA's count, D = all.
"$PYTHON_BIN" -m src.clm.replay --model "$ADAPTER" --mode "$MODE" --out-dir "$OUT_DIR" "${LIMIT_ARGS[@]}" \
    --arm "A_da2=$A2/test_step_0600/test_episodes.csv" \
    --arm "B_none=$A2/scripted_none_step_0600/test_episodes.csv" \
    --arm "C_random2=$A2/scripted_random_matched_step_0600/test_episodes.csv" \
    --arm "D_all=$A2/scripted_all_step_0600/test_episodes.csv"

echo "[SLURM] Finished: $(date)"

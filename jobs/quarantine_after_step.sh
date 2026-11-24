#!/bin/bash
# Move a training run's outputs after STEP into a side folder (nothing is deleted), so the run
# can resume from checkpoint-STEP again. Only moves files; safe to run on a login node.
#   bash jobs/quarantine_after_step.sh outputs/contrastive/joint/joint_h2/train 700 broken_resume
# Moves checkpoint-N, adapters/step_N and episodes/val_step_N_* with N > STEP into
# <train dir>/<label>_after_<STEP>/. train_episodes.csv is trimmed by the next job itself.
set -euo pipefail
TRAIN_DIR="${1:?train dir}"; STEP="${2:?step}"; LABEL="${3:-quarantine}"
DEST="$TRAIN_DIR/${LABEL}_after_${STEP}"
[[ -d "$TRAIN_DIR/checkpoint-$STEP" ]] || { echo "No $TRAIN_DIR/checkpoint-$STEP; refusing" >&2; exit 2; }
mkdir -p "$DEST/adapters" "$DEST/episodes"
moved=0
for c in "$TRAIN_DIR"/checkpoint-*; do
  n=${c##*-}; [[ "$n" =~ ^[0-9]+$ ]] && (( n > STEP )) && { mv "$c" "$DEST/"; moved=$((moved+1)); }
done
for a in "$TRAIN_DIR"/adapters/step_*; do
  [[ -e "$a" ]] || continue
  n=$((10#${a##*_})); (( n > STEP )) && { mv "$a" "$DEST/adapters/"; moved=$((moved+1)); }
done
for f in "$TRAIN_DIR"/episodes/val_step_*; do
  [[ -e "$f" ]] || continue
  n=$(basename "$f" | cut -d_ -f3); n=$((10#$n)); (( n > STEP )) && { mv "$f" "$DEST/episodes/"; moved=$((moved+1)); }
done
log="$TRAIN_DIR/contrastive_heads_updates.jsonl"
if [[ -f "$log" ]]; then  # keep the update log consistent with the checkpoint we resume from
  cp "$log" "$DEST/"
  grep -E "\"step\": ([0-9]+)," "$log" | awk -v s="$STEP" -F'"step": ' '{split($2,a,","); if (a[1]+0 <= s) print}' > "$log.tmp"
  mv "$log.tmp" "$log"
fi
echo "Moved $moved items after step $STEP to $DEST"
ls "$TRAIN_DIR" | grep checkpoint- | sort -t- -k2 -n | tail -3

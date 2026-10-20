"""Print the most advanced complete full-training checkpoint, if one exists."""

import argparse
import json
from pathlib import Path


def find_latest_checkpoint(outputs_dir: Path, max_steps: int) -> Path | None:
    candidates = []
    for checkpoint in outputs_dir.glob("train_full_*/checkpoint-*"):
        if not checkpoint.is_dir() or not checkpoint.name.removeprefix("checkpoint-").isdigit():
            continue
        state_path = checkpoint / "trainer_state.json"
        optimizer_path = checkpoint / "optimizer.pt"
        if not state_path.is_file() or not optimizer_path.is_file():
            continue
        try:
            step = int(checkpoint.name.removeprefix("checkpoint-"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if state.get("global_step") != step or step > max_steps:
            continue
        candidates.append((step, state_path.stat().st_mtime, checkpoint))
    return max(candidates, default=(None, None, None))[2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_dir", type=Path)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args()
    checkpoint = find_latest_checkpoint(args.outputs_dir, args.max_steps)
    if checkpoint is not None:
        print(checkpoint.resolve())

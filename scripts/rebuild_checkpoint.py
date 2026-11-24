"""Rebuild a resumable ``checkpoint-STEP`` from the adapter snapshot ``adapters/step_STEP``.

Needed when the Trainer's ``save_total_limit`` has deleted the checkpoint (2026-11-23, H4
extension). The snapshot holds the DA adapter (and, for H4, the heads); ``trainer_state.json``
is copied from the oldest later checkpoint with ``global_step`` set back to STEP and the log
history cut at STEP, so the Trainer skips the right number of patients. There is no
``optimizer.pt``: the resumed job starts the DA's (and the heads') AdamW moments fresh.

    python scripts/rebuild_checkpoint.py outputs/contrastive/joint/joint_h2/train 700

Standard library only. Writes ``checkpoint-STEP/REBUILT.json`` describing what was done.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

COPY = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "contrastive_heads.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("train_dir", type=Path)
    parser.add_argument("step", type=int)
    args = parser.parse_args()
    train, step = args.train_dir, args.step
    snapshot = train / "adapters" / f"step_{step:04d}"
    target = train / f"checkpoint-{step}"
    if target.exists():
        raise SystemExit(f"{target} already exists; not touching it")
    if not (snapshot / "adapter_config.json").is_file():
        raise SystemExit(f"No adapter snapshot at {snapshot}")
    later = sorted((p for p in train.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()
                    and int(p.name.split("-")[-1]) > step and (p / "trainer_state.json").is_file()),
                   key=lambda p: int(p.name.split("-")[-1]))
    if not later:
        raise SystemExit(f"No later checkpoint with trainer_state.json in {train} to copy the state from")
    source = later[0]
    state = json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
    per_step = state["epoch"] / state["global_step"] if state.get("global_step") else 0.0
    state["global_step"] = step
    state["epoch"] = step * per_step
    state["log_history"] = [e for e in state.get("log_history", []) if e.get("step", 0) <= step]

    target.mkdir()
    copied = []
    for name in COPY:
        if (snapshot / name).is_file():
            shutil.copy2(snapshot / name, target / name)
            copied.append(name)
    (target / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (target / "REBUILT.json").write_text(json.dumps({
        "from_snapshot": str(snapshot), "state_from": str(source / "trainer_state.json"), "files": copied,
        "missing": ["optimizer.pt", "scheduler.pt", "rng_state.pth", "contrastive_heads_optimizer.pt"],
        "note": "save_total_limit deleted the original; optimizers restart fresh at this step",
        "built_at": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    print(f"Rebuilt {target} from {snapshot} (state from {source.name}): {', '.join(copied)}")


if __name__ == "__main__":
    main()

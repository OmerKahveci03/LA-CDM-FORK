"""Print the newest completed full-training adapter directory, if present."""

import argparse
from pathlib import Path


def find_latest_adapter(outputs_dir: Path) -> Path | None:
    candidates = []
    for run_dir in outputs_dir.glob("train_full_*"):
        config = run_dir / "adapter_config.json"
        weights = next(
            (path for path in (run_dir / "adapter_model.safetensors", run_dir / "adapter_model.bin") if path.is_file()),
            None,
        )
        if config.is_file() and weights is not None:
            candidates.append((weights.stat().st_mtime, run_dir))
    return max(candidates, default=(None, None))[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_dir", type=Path)
    args = parser.parse_args()
    adapter = find_latest_adapter(args.outputs_dir)
    if adapter is not None:
        print(adapter.resolve())

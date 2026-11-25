"""Evaluate a DA with the frozen HA's hypothesis as the diagnosis (H3c, docs/HYPOTHESIS_H3C.md).

``src/evaluate_frozen_ha.py`` (unchanged) with ``DeferHAEnvironment``. Same arguments:
``evaluation.adapter_dir`` (the DA) and ``evaluation.frozen_ha_dir`` (the frozen HA).
"""

import hydra
from omegaconf import DictConfig

import src.evaluate_frozen_ha as frozen
from src.environment.defer_ha_environment import DeferHAEnvironment


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    frozen.FrozenHAEnvironment = DeferHAEnvironment
    print("[H3c] evaluation: the frozen HA's hypothesis is the diagnosis", flush=True)
    return frozen.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()

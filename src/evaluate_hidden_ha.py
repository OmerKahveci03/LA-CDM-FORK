"""Evaluate a trained DA with the HA computed but hidden from it (ablation A4a, docs/FINAL_ANALYSES.md).

Identical to ``src/evaluate_frozen_ha.py`` (unchanged) except for the environment class: the frozen
HA still scores every turn and is logged, but its hypothesis is never added to the DA's transcript
(``src/environment/hidden_ha_environment.py``). Same arguments: ``evaluation.adapter_dir`` (the DA)
and ``evaluation.frozen_ha_dir`` (the HA that is logged).
"""

import src.evaluate_frozen_ha as frozen
from src.environment.hidden_ha_environment import HiddenHAEnvironment

# evaluate_frozen_ha.main looks the class up in its own module at call time.
frozen.FrozenHAEnvironment = HiddenHAEnvironment
main = frozen.main

if __name__ == "__main__":
    print("[hidden HA] the HA is computed and logged but not shown to the DA", flush=True)
    main()

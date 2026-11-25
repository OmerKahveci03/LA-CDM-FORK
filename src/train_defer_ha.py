"""Train a DA whose diagnosis is always the frozen HA's (H3c, docs/HYPOTHESIS_H3C.md).

Runs the unchanged training function of ``src/train_frozen_ha.py`` (H3) with two swaps:
- the environment is ``DeferHAEnvironment`` (the HA's current hypothesis is the diagnosis that counts);
- the trainer also gets ``RestoreAdapterCallback``, so a resumed job restores the DA's adapter
  (the multi-adapter resume bug, src/utils/peft_resume.py; the H3 script itself lacks the fix).
Same arguments as src/train_frozen_ha.py.
"""

import hydra
from omegaconf import DictConfig

import src.train_frozen_ha as h3
from src.environment.defer_ha_environment import DeferHAEnvironment
from src.trainer.cdm_grpo_trainer import CDMGRPOTrainer
from src.utils.peft_resume import RestoreAdapterCallback


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    resume = cfg.training.resume_from_checkpoint

    class Trainer(CDMGRPOTrainer):
        def __init__(self, *args, callbacks=None, **kwargs):
            super().__init__(*args, callbacks=[RestoreAdapterCallback(resume), *(callbacks or [])], **kwargs)

    # train_frozen_ha's function looks both names up in its own module at call time.
    h3.FrozenHAEnvironment = DeferHAEnvironment
    h3.CDMGRPOTrainer = Trainer
    print("[H3c] the DA chooses the tests; the frozen HA's hypothesis is the diagnosis", flush=True)
    return h3.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()

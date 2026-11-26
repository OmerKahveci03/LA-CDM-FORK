"""H3 training (``src/train_frozen_ha.py``, unchanged) plus the multi-adapter resume fix.

Used by H3d with DESIGN=plain (docs/HYPOTHESIS_H3D.md). The H3 script loads the frozen HA as a second
adapter, so a resumed job would otherwise continue with an untrained DA (src/utils/peft_resume.py).
"""

import hydra
from omegaconf import DictConfig

import src.train_frozen_ha as h3
from src.trainer.cdm_grpo_trainer import CDMGRPOTrainer
from src.utils.peft_resume import RestoreAdapterCallback


@hydra.main(version_base=None, config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    resume = cfg.training.resume_from_checkpoint

    class Trainer(CDMGRPOTrainer):
        def __init__(self, *args, callbacks=None, **kwargs):
            super().__init__(*args, callbacks=[RestoreAdapterCallback(resume), *(callbacks or [])], **kwargs)

    h3.CDMGRPOTrainer = Trainer
    return h3.main.__wrapped__(cfg)


if __name__ == "__main__":
    main()

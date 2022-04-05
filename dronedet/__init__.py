import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from dronedet.utils import import_object, import_objects, patch_empty_params, patch_relative_paths


def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.training.seed)
    torch.backends.cudnn.enabled = cfg.general.get("use_cudnn", True)

    callbacks = import_objects(cfg.get("callback", {}))
    loggers = import_objects(cfg.get("logger", {}))

    trainer = pl.Trainer(logger=loggers, callbacks=callbacks, **cfg.trainer)

    dm = import_object(cfg.datamodule.data_module_name)(cfg=cfg, trainer=trainer)
    module = import_object(cfg.training.lightning_module_name)(cfg=cfg, datamodule=dm)
    trainer.fit(module, dm)


# config_path is relative path to config folder (from script), config_name is name of main config in that folder
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = patch_empty_params(cfg)
    cfg = patch_relative_paths(cfg, hydra.utils.get_original_cwd())
    train(cfg=cfg)


if __name__ == "__main__":
    main()

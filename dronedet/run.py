import hydra
from omegaconf import DictConfig, OmegaConf

from dronedet.utils import patch_empty_params, patch_relative_paths


def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


# config_path is relative path to config folder (from script), config_name is name of main config in that folder
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = patch_empty_params(cfg)
    cfg = patch_relative_paths(cfg, hydra.utils.get_original_cwd())
    run(cfg=cfg)


if __name__ == "__main__":
    main()

from omegaconf import DictConfig

from dronedet.run import run


def test_run(cfg: DictConfig) -> None:
    run(cfg)

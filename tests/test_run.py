import tempfile
from pathlib import Path
from typing import Generator

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from dronedet.run import run
from dronedet.utils import patch_empty_params


@pytest.fixture(scope="session")
def custom_tmp_path() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as dirname:
        yield Path(dirname)


@pytest.fixture(scope="session")
def logs_folder(custom_tmp_path: Path) -> Path:
    return custom_tmp_path / "logs"


@pytest.fixture(scope="session")
def cfg(logs_folder: Path) -> DictConfig:
    initialize(config_path="../dronedet/conf", job_name="tests")
    cfg = compose(
        config_name="config",
        overrides=[f"hydra.run.dir={logs_folder}"],
    )
    cfg = patch_empty_params(cfg)
    return cfg


def test_run(cfg: DictConfig) -> None:
    run(cfg)

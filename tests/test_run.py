import os
import tempfile
from pathlib import Path
from typing import Generator, List

import pytest


@pytest.fixture(scope="session")
def custom_tmp_path() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as dirname:
        yield Path(dirname)


@pytest.fixture(scope="session")
def logs_folder(custom_tmp_path: Path) -> Path:
    return custom_tmp_path / "logs"


@pytest.fixture(scope="session")
def overrides(logs_folder: Path) -> List[str]:
    overrides = [f"hydra.run.dir={logs_folder}"]
    return overrides


def test_run(overrides: List[str]) -> None:
    command = "python -m dronedet.run " + " ".join(overrides)
    assert os.system(command) == 0

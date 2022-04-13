import os
import tempfile
from pathlib import Path
from typing import Generator, List

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def custom_tmp_path() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as dirname:
        yield Path(dirname)


@pytest.fixture(scope="session")
def logs_folder(custom_tmp_path: Path) -> Path:
    return custom_tmp_path / "logs"


@pytest.fixture(scope="session")
def video(custom_tmp_path: Path) -> Path:
    video_path = custom_tmp_path / "video.mkv"
    width, height = 1280, 720
    video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"XVID"), 30, (1280, 720), isColor=True)
    number_of_frames = 150
    for index in range(number_of_frames):
        sample = np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8)
        video_writer.write(sample)
    return video_path


@pytest.fixture(scope="session")
def overrides(logs_folder: Path, video: Path) -> List[str]:
    overrides = [f"hydra.run.dir={logs_folder}"]
    overrides.append(f"cameras.video.path={video}")
    return overrides


def test_run(overrides: List[str]) -> None:
    command = "python -m dronedet.run " + " ".join(overrides)
    assert os.system(command) == 0

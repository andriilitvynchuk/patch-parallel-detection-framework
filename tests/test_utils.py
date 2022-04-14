import pytest
import torch

from dronedet.utils import crop_n_parts


@pytest.fixture(scope="session")
def batch() -> torch.Tensor:
    return torch.randn(2, 3, 8, 16)


@pytest.mark.parametrize("n_crops", [1, 4])
def test_crop_n_parts(batch: torch.Tensor, n_crops: int) -> None:
    result = crop_n_parts(batch, n_crops=n_crops)
    b, c, h, w = batch.shape
    assert result.shape == (b, n_crops, c, h // (int(n_crops**0.5)), w // (int(n_crops**0.5)))

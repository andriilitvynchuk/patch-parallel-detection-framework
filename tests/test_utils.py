import pytest
import torch

from dronedet.utils import crop_n_parts


@pytest.fixture(scope="session")
def batch() -> torch.Tensor:
    return torch.randn(2, 3, 8, 16)


@pytest.mark.parametrize("n_crops", [1, 4])
def test_crop_n_parts(batch: torch.Tensor, n_crops: int) -> None:
    result, meta = crop_n_parts(batch, n_crops=n_crops)
    b, c, h, w = batch.shape
    assert result.shape == (b, n_crops, c, h // (int(n_crops**0.5)), w // (int(n_crops**0.5)))

    reconstruct = torch.zeros_like(batch)
    for index, crop_meta in enumerate(meta):
        reconstruct[
            ..., crop_meta.height_start : crop_meta.height_end, crop_meta.width_start : crop_meta.width_end
        ] = result[:, index]
    assert (batch == reconstruct).all()


@pytest.mark.parametrize("n_crops", [2, 9])
def test_crop_n_parts_erros(batch: torch.Tensor, n_crops: int) -> None:
    with pytest.raises(ValueError):
        crop_n_parts(batch, n_crops=n_crops)

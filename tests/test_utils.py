import pytest
import torch

from dronedet.utils import calculate_overlap_value, crop_n_parts


@pytest.fixture(scope="session")
def batch() -> torch.Tensor:
    return torch.randn(2, 3, 8, 16)


@pytest.mark.parametrize("n_crops,overlap_percent", [(1, 0.0), (4, 0.2)])
def test_crop_n_parts(batch: torch.Tensor, n_crops: int, overlap_percent: float) -> None:
    result, meta = crop_n_parts(batch, n_crops=n_crops, overlap_percent=overlap_percent)
    b, c, h, w = batch.shape
    new_height = h // (int(n_crops**0.5)) + calculate_overlap_value(h, overlap_percent)
    new_width = w // (int(n_crops**0.5)) + calculate_overlap_value(w, overlap_percent)
    assert result.shape == (b, n_crops, c, new_height, new_width)

    reconstruct = torch.zeros_like(batch)
    for index, crop_meta in enumerate(meta):
        reconstruct[
            ..., crop_meta.height_start : crop_meta.height_end, crop_meta.width_start : crop_meta.width_end
        ] = result[:, index]
    assert (batch == reconstruct).all()


@pytest.mark.parametrize("n_crops,overlap_percent", [(2, 0.2), (9, 0.2), (4, 0.6)])
def test_crop_n_parts_erros(batch: torch.Tensor, n_crops: int, overlap_percent: float) -> None:
    with pytest.raises(ValueError):
        crop_n_parts(batch, n_crops=n_crops, overlap_percent=overlap_percent)

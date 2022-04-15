from dataclasses import dataclass
from itertools import product
from typing import List, Tuple

import torch


@dataclass
class CropMeta:
    height_start: int
    width_start: int
    height_end: int
    width_end: int


def modify_with_overlap(begin: int, end: int, current_index: int, total_indexes: int, overlap: int) -> Tuple[int, int]:
    if current_index == 0:
        end += overlap
    elif 0 < current_index < total_indexes - 1:
        begin = max(begin - overlap // 2, 0)
        end += overlap // 2
    else:
        begin = max(begin - overlap, 0)
    return begin, end


def calculate_overlap_value(side: int, overlap_percent: float) -> int:
    return (int(side * overlap_percent) // 2) * 2  # must be even


def crop_n_parts(
    tensor: torch.Tensor, n_crops: int = 4, overlap_percent: float = 0.1
) -> Tuple[torch.Tensor, List[CropMeta]]:
    """
    Split image into N crops.
    Input:
        tensor: tensor with shape [B, ..., H, W]
        n_crops: int which must be square of int (1, 4, 9, ...)
        overlap_percent: float (percent of image)
    Output:
        tensor with shape [B, n_crops, ..., C, H // n_crops ** 0.5, W_new // n_crops ** 0.5]
        meta: List[CropMeta]], can be used for original image reconstruction (example in test)

    """
    height, width = tensor.size(-2), tensor.size(-1)
    n_splits_by_side = n_crops**0.5
    if not n_splits_by_side.is_integer():
        raise ValueError("Currently, n_crops should be square of a whole number")

    n_splits_by_side = int(n_splits_by_side)
    if not (height / n_splits_by_side).is_integer() or not (width / n_splits_by_side).is_integer():
        raise ValueError(
            f"Cannot divide image into {n_crops} parts, height or width is not divisible by {n_splits_by_side}"
        )
    if not (0 <= overlap_percent <= 1 / n_splits_by_side):
        raise ValueError(f"Overlap value should be in [0, {( 1 / n_splits_by_side):.2f} range]")
    overlap_height = calculate_overlap_value(height, overlap_percent)
    overlap_width = calculate_overlap_value(width, overlap_percent)
    height_linspace = torch.linspace(0, height, steps=n_splits_by_side + 1, dtype=torch.int)
    width_linspace = torch.linspace(0, width, steps=n_splits_by_side + 1, dtype=torch.int)
    results, meta = [], []
    for height_index, width_index in product(range(n_splits_by_side), range(n_splits_by_side)):
        height_value = int(height_linspace[height_index])
        end_height_value = int(height_linspace[height_index + 1])
        width_value = int(width_linspace[width_index])
        end_width_value = int(width_linspace[width_index + 1])
        # modfify respect to overlap
        height_value, end_height_value = modify_with_overlap(
            height_value, end_height_value, height_index, n_splits_by_side, overlap_height
        )
        width_value, end_width_value = modify_with_overlap(
            width_value, end_width_value, width_index, n_splits_by_side, overlap_width
        )
        results.append(tensor[..., height_value:end_height_value, width_value:end_width_value])  # type: ignore
        meta.append(CropMeta(height_value, width_value, end_height_value, end_width_value))
    return torch.stack(results, dim=1), meta

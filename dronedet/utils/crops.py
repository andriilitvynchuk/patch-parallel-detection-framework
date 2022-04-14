import torch


def crop_n_parts(tensor: torch.Tensor, n_crops: int = 4) -> torch.Tensor:
    """
    Split image into N crops.
    Input:
        tensor: tensor with shape [B, C, H, W]
        n_crops: int which must be square of int (1, 4, 9, ...)
    Output:
        tensor with shape [B, n_crops, C, H_new, W_new]

    TODO: add overlappings on edges
    """
    _, _, height, width = tensor.shape
    n_splits_by_side = n_crops**0.5
    if not n_splits_by_side.is_integer():
        raise ValueError("Currently, n_crops should be square of a whole number")

    n_splits_by_side = int(n_splits_by_side)
    if not (height / n_splits_by_side).is_integer() or not (width / n_splits_by_side).is_integer():
        raise ValueError(
            f"Cannot divide image into {n_crops} parts, height or width is not divisible by {n_splits_by_side}"
        )

    height_linspace = torch.linspace(0, height, steps=n_splits_by_side + 1, dtype=torch.int)
    width_linspace = torch.linspace(0, width, steps=n_splits_by_side + 1, dtype=torch.int)
    results = []
    for height_index in range(len(height_linspace[:-1])):
        for width_index in range(len((width_linspace[:-1]))):
            height_value = height_linspace[height_index]
            end_height_value = height_linspace[height_index + 1]
            width_value = width_linspace[width_index]
            end_width_value = width_linspace[width_index + 1]
            results.append(tensor[..., height_value:end_height_value, width_value:end_width_value])  # type: ignore
    return torch.stack(results, dim=1)

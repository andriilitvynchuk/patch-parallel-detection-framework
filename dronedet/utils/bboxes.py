from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch


def get_index(element: Any, element_list: List[Any]) -> Optional[Any]:
    try:
        index_element = element_list.index(element)
        return index_element
    except ValueError:
        return None


def draw_bbox(
    image: np.ndarray,
    left_top: Tuple[int, int],
    right_bottom: Tuple[int, int],
    text: str,
    bbox_color: Union[str, Tuple[int, int, int]],
    text_color: Union[str, Tuple[int, int, int]],
    thickness: int = 4,
    font_thickness: int = 2,
    font_scale: float = 2,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> np.ndarray:
    cv2.rectangle(image, left_top, right_bottom, bbox_color, thickness=thickness)
    cv2.putText(
        image,
        text,
        (left_top[0], left_top[1] - 2),
        font,
        font_scale,
        text_color,
        font_thickness,
    )
    return image


def scale_bboxes_torch(
    bboxes: torch.Tensor, input_size: Tuple[int, int], output_size: Tuple[int, int]
) -> torch.Tensor:
    h_scale = output_size[0] / input_size[0]
    w_scale = output_size[1] / input_size[1]
    scale_tensor = torch.tensor([w_scale, h_scale, w_scale, h_scale]).view(1, -1)
    bboxes[:, :4] *= scale_tensor.to(bboxes.device).to(bboxes.dtype)
    return bboxes


def merge_bboxes_torch(
    bboxes: torch.Tensor, input_size: Tuple[int, int], output_size: Tuple[int, int]
) -> torch.Tensor:
    pass

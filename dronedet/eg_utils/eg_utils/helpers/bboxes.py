from typing import Tuple, Union

import cv2
import numpy as np


def nms(
    bboxes: np.ndarray, iou_threshold: float, sigma: float = 0.3, method: str = "nms"
) -> np.ndarray:
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = bboxes[:, 5] == cls
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate(
                [cls_bboxes[:max_ind], cls_bboxes[max_ind + 1 :]]
            )
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ["nms", "soft-nms"]

            if method == "nms":
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == "soft-nms":
                weight = np.exp(-(1.0 * iou**2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.0
            cls_bboxes = cls_bboxes[score_mask]

    return np.array(best_bboxes)


def bboxes_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def draw_bbox(
    image: np.ndarray,
    left_top: Tuple[int, int],
    right_bottom: Tuple[int, int],
    text: str,
    bbox_color: Union[str, Tuple[int, int, int]],
    text_color: Union[str, Tuple[int, int, int]],
    thickness: int = 4,
    font_thickness: int = 2,
    font_scale: int = 2,
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

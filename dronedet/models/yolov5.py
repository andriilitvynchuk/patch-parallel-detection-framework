from typing import Any, Dict, List

import numpy as np
import torch

from dronedet.base.simple_deep_model import SimpleDeepModel
from dronedet.utils import non_max_suppression


# from .yolov5.models.common import DetectMultiBackend


# from .yolov5.utils.augmentations import letterbox
# from .yolov5.utils.general import scale_coords


class Yolov5Detector(SimpleDeepModel):
    def __init__(self, config: Dict):
        self._load_cfg(config)
        self._load_model()

    def _load_cfg(self, config: Dict[str, Any]) -> None:
        super()._load_cfg(config)
        self._nms_conf_thres = config["nms_conf_thres"]  # confidence threshold
        self._iou_thres = config["iou_thres"]  # NMS IOU threshold
        self._iou_thres_post = config["iou_thres_post"]
        self._max_det = config["max_det"]  # maximum detections per image
        self._classes = config["classes"]  # filter by class: --class 0, or --class 0 2 3
        self._agnostic_nms = config["agnostic_nms"]  # class-agnostic NMS
        self._classify = config["classify"]  # False
        self._stride = config["stride"]

    def _load_model(self) -> None:
        self._model = torch.jit.load(self._model_path)
        self._warmup()

    def _warmup(self) -> None:
        if self._input_size is not None:
            print("Warming up ... ")
            check_array = torch.randn(1, 3, *self._input_size[:2])
            self.forward_batch(check_array)
            print("Done")

    def _gpu_preprocess(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore
        batch = images.to(self._precision)
        if self._need_resize:
            batch = torch.nn.functional.interpolate(batch, size=self._input_size, mode="nearest")
        if self._need_norm:
            batch = (batch.to(self._norm_device) - self._norm_mean) / self._norm_std
        return batch.to(self._device)

    def _preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        gpu_preprocessed_images = self._gpu_preprocess(images=batch)
        return gpu_preprocessed_images

    # def _warmup(self) -> None:
    #     if self._input_size is not None:
    #         self._model.warmup(imgsz=(1, 3, self._input_size[0], self._input_size[1]))
    #     else:
    #     x    self._model.warmup()

    # def _preprocess(self, img0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Args:
    #         img: 3-dimentional image in BGR format
    #     """

    #     # Padded resize
    #     # img = letterbox(img0, self._input_size, stride=self._stride, auto=self._auto_letterbox)[0]
    #     img =

    #     # Convert
    #     img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    #     img = np.ascontiguousarray(img)

    #     im, im0s = img, img0

    #     im = torch.from_numpy(im).to(self._device)
    #     im = im.half() if self._half else im.float()  # uint8 to fp16/32
    #     im /= 255  # 0 - 255 to 0.0 - 1.0
    #     if len(im.shape) == 3:
    #         im = im[None]  # expand for batch dim

    #     return im, im0s

    # @staticmethod
    # def _postprocess_detections(pred: Sequence, im: np.ndarray, im0s: np.ndarray) -> List[List]:
    #     detections = list()
    #     for i, det in enumerate(pred):
    #         result = list()
    #         if len(det):
    #             # Rescale boxes from img_size to im0 size
    #             det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()

    #             for *xyxy, conf, cls in reversed(det.cpu().numpy()):
    #                 x1, y1, x2, y2 = xyxy
    #                 result.append([x1, y1, x2, y2, conf, cls])

    #         detections.append(result)
    #     return detections

    def forward_batch(self, batch: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            preprocessed_input = self._preprocess_batch(batch)
            print(preprocessed_input.size())
            (results,) = self._model(preprocessed_input)
            print(results.size())

    def forward_image(self, image: torch.Tensor) -> List[np.ndarray]:
        """
        returns: [[x, y, w, h, conf, cls], ...]
        """
        # preprocess
        im, im0s = self._preprocess(img)

        # Inference
        pred = self._model(im, augment=self._augment, visualize=False)

        # NMS
        pred = non_max_suppression(
            pred, self._nms_conf_thres, self._iou_thres, self._classes, self._agnostic_nms, max_det=self._max_det
        )

        # Process predictions
        pred = self._postprocess_detections(pred, im, im0s)[0]

        return pred

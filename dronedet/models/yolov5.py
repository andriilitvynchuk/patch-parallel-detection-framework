from typing import Any, Dict, List

import numpy as np
import torch

from dronedet.base.simple_deep_model import SimpleDeepModel
from dronedet.utils import non_max_suppression


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

    def _load_model(self) -> None:
        self._model = torch.jit.load(self._model_path, map_location=self._device)
        self._model = self._model.eval().to(self._precision)
        self._warmup()

    def _warmup(self) -> None:
        if self._input_size is not None:
            print("Warming up ... ")
            check_array = torch.randn(1, 3, *self._input_size[:2])
            self.forward_batch(check_array)
            print("Done")

    def _gpu_preprocess(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore
        if self._need_resize:
            images = torch.nn.functional.interpolate(
                images.to(torch.float32), size=self._input_size, mode="bilinear", align_corners=True
            )
        batch = images.to(self._precision)
        if self._need_norm:
            batch = (batch.to(self._norm_device) - self._norm_mean) / self._norm_std
        return batch.to(self._device)

    def _preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        gpu_preprocessed_images = self._gpu_preprocess(images=batch)
        return gpu_preprocessed_images

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

    def forward_batch(self, batch: torch.Tensor) -> List[np.ndarray]:  # type: ignore
        with torch.no_grad():
            preprocessed_input = self._preprocess_batch(batch)
            (results,) = self._model(preprocessed_input)
            results = non_max_suppression(results, self._nms_conf_thres, self._iou_thres, max_det=self._max_det)
        return [result.cpu().numpy() for result in results]

    def forward_image(self, image: torch.Tensor) -> np.ndarray:  # type: ignore
        output = self.forward_batch(image.unsqueeze(0))
        return output[0]

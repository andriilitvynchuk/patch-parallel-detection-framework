from typing import Any, Dict, List, Tuple

import torch

from dronedet.base.simple_deep_model import SimpleDeepModel
from dronedet.utils import non_max_suppression, scale_bboxes_torch


class Yolov5Detector(SimpleDeepModel):
    def __init__(self, config: Dict):
        self._load_cfg(config)
        self._load_model()

    def _load_cfg(self, config: Dict[str, Any]) -> None:
        super()._load_cfg(config)
        self._detection_threshold = config["detection_threshold"]  # confidence threshold
        self._iou_threshold = config["iou_threshold"]  # NMS IOU threshold
        self._max_det = config["max_det"]  # maximum detections per image

    def _load_model(self) -> None:
        self._model = torch.jit.load(self._model_path, map_location=self._device)
        self._model = self._model.eval().to(self._precision)
        self._warmup()

    def _warmup(self) -> None:
        if self._input_size is not None:
            print("Warming up ... ")
            check_array = torch.randn(1, 3, *self._input_size)
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

    def _postprocess_image_bboxes(self, image_bboxes: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        postprocessed = scale_bboxes_torch(
            image_bboxes, input_size=self._input_size, output_size=output_size  # type: ignore
        )
        return postprocessed

    def forward_batch(self, batch: torch.Tensor) -> List[torch.Tensor]:  # type: ignore
        with torch.no_grad():
            preprocessed_input = self._preprocess_batch(batch)
            (results,) = self._model(preprocessed_input)
            results = non_max_suppression(
                results, self._detection_threshold, self._iou_threshold, max_det=self._max_det
            )
            results = [self._postprocess_image_bboxes(result, (batch.size(-2), batch.size(-1))) for result in results]
        return results

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        output = self.forward_batch(image.unsqueeze(0))
        return output[0]

from abc import abstractmethod
from typing import Any, Dict, Sequence

import cv2
import numpy as np
import torch


class SimpleDeepModel:
    """Simple class for running Deep Learning models.
    For base use it needs config's dict structure like this:
        model_path: /path/to/model
        precision: FP16
        device: cuda:0
        preprocessing:
            colorspace: RGB
            resize:
                height: 736
                width: 1280
            normalize:
                device: cuda:0
                (optional) mean: [0.485, 0.456, 0.406]
                (optional) std: [0.229, 0.224, 0.225]
    """

    def __init__(self, config: Dict[str, Any]):
        self._load_cfg(config)
        self._load_model()
        self._warmup()

    def _load_cfg(self, config: Dict[str, Any]) -> None:
        self._model_path = config["model_path"]
        self._device = torch.device(config["device"])
        self._precision = self._string_to_torch_dtype(config["precision"])
        self._load_preprocessing(config["preprocessing"])

    def _string_to_torch_dtype(self, string: str) -> torch.dtype:
        if string.lower() == "fp32" or self._device == "cpu":
            return torch.float32
        elif string.lower() == "fp16":
            return torch.float16
        else:
            raise ValueError("Unsupportable precision")

    def _load_preprocessing(self, config: Dict[str, Any]) -> None:
        """Loads preprocessing.
        If resize is not needed, leave resize as None.
        If normalize is not needed, leave normalize as None.
        In case normalize is not None and mean or std wasn't declared, ImageNet mean and std will be used.
        """
        self._colorspace = config["colorspace"]

        self._need_resize = config["resize"] is not None
        self._input_size = (config["resize"]["height"], config["resize"]["width"]) if self._need_resize else None

        self._need_norm = config["normalize"] is not None
        self._norm_mean, self._norm_std, self._norm_device = None, None, None
        if self._need_norm:
            self._norm_device = torch.device(config["normalize"]["device"])
            self._norm_mean = torch.tensor(config["normalize"].get("mean", [0.485, 0.456, 0.406]))
            self._norm_mean = 255 * self._norm_mean.to(self._norm_device).to(self._precision).view(1, 3, 1, 1)
            self._norm_std = torch.tensor(config["normalize"].get("std", [0.229, 0.224, 0.225]))
            self._norm_std = 255 * self._norm_std.to(self._norm_device).to(self._precision).view(1, 3, 1, 1)

    @abstractmethod
    def _load_model(self) -> None:
        """Loading of your model from model_path."""
        raise NotImplementedError("Implement loading of model")

    def _warmup(self) -> None:
        if self._input_size is not None:
            print("Warming up ... ")
            check_array = np.random.randint(0, 255, size=(1, 3, *self._input_size[:2])).astype(np.uint8)
            self.forward_batch(check_array)  # type: ignore
            print("Done")

    def _cpu_image_preprocess(self, image: np.ndarray) -> np.ndarray:
        if self._need_resize and (image.shape != self._input_size):
            image = cv2.resize(image, dsize=(self._input_size[1], self._input_size[0]))  # type: ignore
        if self._colorspace == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def _gpu_preprocess(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        batch = torch.stack([torch.tensor(image, dtype=self._precision).permute(2, 0, 1) for image in images])
        if self._need_norm:
            batch = (batch.to(self._norm_device) - self._norm_mean) / self._norm_std
        return batch.to(self._device)

    def _preprocess_batch(self, batch: Sequence[np.ndarray]) -> torch.Tensor:
        """Preprocess batch of images
        Args:
            batch (Sequence[np.ndarray]): batch of images in format of (B, H, W, C) np.ndarray
            or any iterable sequence of (H, W, C) np.ndarrays
        Returns:
            torch.Tensor: preprocessed batch of images with size (B, C, H, W)
        """
        cpu_preprocessed_images = [self._cpu_image_preprocess(image) for image in batch]
        gpu_preprocessed_images = self._gpu_preprocess(images=cpu_preprocessed_images)
        return gpu_preprocessed_images

    @abstractmethod
    def forward_batch(self, batch: Sequence[np.ndarray], *args: Any, **kwargs: Any) -> np.ndarray:
        """Full forward of images.
        Args:
            batch (Sequence[np.ndarray]): batch of images in format of (B, H, W, C) np.ndarray
            or any iterable sequence of (H, W, C) np.ndarrays
        """
        raise NotImplementedError("Implement forward of batch")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward_batch(*args, **kwargs)

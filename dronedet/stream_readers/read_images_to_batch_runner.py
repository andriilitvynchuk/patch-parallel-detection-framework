from typing import Any, Dict

import numpy as np
import torch

import shared_numpy as snp
from dronedet.base import SimpleRunner
from dronedet.utils import calculate_overlap_value, crop_n_parts, import_object


class ReadImagesToBatchRunner(SimpleRunner):
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        super().__init__()
        self._load_cfg(config)
        self._load_global_cfg(global_config)

    def _load_cfg(self, config: Dict[str, Any]) -> None:
        self._stream_reader_class = import_object(config["class"])
        self._n_buffers = config["n_buffers"]
        self._n_crops = config["n_crops"]
        self._overlap_p = config["overlap_percent"]
        self._verbose = config.get("verbose", True)

    def _load_global_cfg(self, config: Dict[str, Any]) -> None:
        self._cameras = list(config["cameras"].values())  # cameras is list of dicts (e.g. video: {})

    def _init_sources(self) -> None:
        self._sources = [self._stream_reader_class(camera_params) for camera_params in self._cameras]

    def _init_buffers(self) -> None:
        camera_height, camera_width = self._cameras[0]["height"], self._cameras[0]["width"]
        height = camera_height // int(self._n_crops**0.5) + calculate_overlap_value(camera_height, self._overlap_p)
        width = camera_width // int(self._n_crops**0.5) + calculate_overlap_value(camera_width, self._overlap_p)
        future_buffer_size = (len(self._cameras), self._n_crops, 3, height, width)
        self._image_buffers = [
            torch.empty(size=future_buffer_size, device=self._cameras[0]["device"], dtype=torch.uint8).share_memory_()
            for _ in range(self._n_buffers)
        ]

    def _init_run(self) -> None:
        self._init_sources()
        self._init_buffers()

    def _process(self, share_data: Dict[str, Any]) -> Dict[str, Any]:
        # get images from streams in GPU
        read_list = [source.read() for source in self._sources]
        # add copy of images on CPU in shared memory
        share_data["images_cpu"] = [snp.from_array(image.cpu().numpy()) for (image, _) in read_list]

        share_data["images_gpu"] = self._image_buffers[len(self._timers["main_work_time"]) % self._n_buffers]
        share_data["crop_meta"] = []
        # memory is already allocated, just copy
        for index, (image, _) in enumerate(read_list):
            cropped_image, meta = crop_n_parts(
                image.unsqueeze(0), n_crops=self._n_crops, overlap_percent=self._overlap_p
            )
            share_data["images_gpu"][index] = cropped_image[0]  # cropped image has shape [1, N_crops, 3, H, W]
            share_data["crop_meta"].append(meta)

        share_data["meta"] = [camera_meta for (_, camera_meta) in read_list]
        # if all cameras crashed - close pipeline
        self.is_running = any([camera_meta["success"] for camera_meta in share_data["meta"]])
        if not self.is_running:
            print("All cameras are not available")
        share_data["bboxes"] = [np.array([100, 100, 1280, 1280, 0.8, 0]).reshape(1, -1)] * len(self._cameras)
        return share_data

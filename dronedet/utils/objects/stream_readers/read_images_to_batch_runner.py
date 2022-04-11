# from os import read
from typing import Any, Dict

import shared_numpy as snp
import torch

from dronedet.eg_utils.eg_utils.helpers.config import import_object  # type: ignore
from dronedet.utils.objects.base.simple_runner import SimpleRunner  # type: ignore


class ReadImagesToBatchRunner(SimpleRunner):
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        super().__init__()
        self._vms = None
        self._load_cfg(config)
        self._load_global_cfg(global_config)

    def _load_cfg(self, config: Dict[str, Any]) -> None:
        self._stream_reader_class = import_object(config["class"])
        self._n_buffers = config["n_buffers"]
        self._verbose = config.get("verbose", True)

    def _load_global_cfg(self, config: Dict[str, Any]) -> None:
        self._cameras = config["cameras"]

    def _init_sources(self) -> None:
        self._sources = [self._stream_reader_class(camera_params) for camera_params in self._cameras]

    def _init_buffers(self) -> None:
        future_buffer_size = (len(self._cameras), 3, self._cameras[0]["height"], self._cameras[0]["width"])
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
        batch_tensor = self._image_buffers[len(self._timers["main_work_time"]) % self._n_buffers]
        # memory is already allocated, just copy
        for index, (image, _) in enumerate(read_list):
            batch_tensor[index] = image
        share_data["images_gpu"] = batch_tensor
        # add copy of images on CPU in shared memory
        share_data["images_cpu"] = [snp.from_array(image_tensor.cpu().numpy()) for image_tensor in batch_tensor]

        meta = [camera_meta for (_, camera_meta) in read_list]
        # if all cameras crashed - close pipeline
        self.is_running = any([camera_meta["success"] for camera_meta in meta])
        if not self.is_running:
            print("All cameras are not available")
        share_data["meta"] = meta
        return share_data

from typing import Any, Dict

import torch

from dronedet.base import SimpleRunner
from dronedet.utils import create_shared_array, crop_n_parts, import_object


class ReadImagesToBatchRunner(SimpleRunner):
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        super().__init__()
        self._load_cfg(config)
        self._load_global_cfg(global_config)

    def _load_cfg(self, config: Dict[str, Any]) -> None:
        self._stream_reader_class = import_object(config["class"])
        self._n_buffers = config["n_buffers"]
        self._n_crops = config["n_crops"]
        self._verbose = config.get("verbose", True)

    def _load_global_cfg(self, config: Dict[str, Any]) -> None:
        self._cameras = list(config["cameras"].values())  # cameras is list of dicts (e.g. video: {})

    def _init_sources(self) -> None:
        self._sources = [self._stream_reader_class(camera_params) for camera_params in self._cameras]

    def _init_buffers(self) -> None:
        height = self._cameras[0]["height"] // int(self._n_crops**0.5)
        width = self._cameras[0]["width"] // int(self._n_crops**0.5)
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
        share_data["images_cpu"] = [create_shared_array(image.cpu().numpy()) for (image, _) in read_list]

        # TODO: check if we need buffers at all
        batch_tensor = self._image_buffers[len(self._timers["main_work_time"]) % self._n_buffers]
        # memory is already allocated, just copy
        for index, (image, _) in enumerate(read_list):
            batch_tensor[index] = crop_n_parts(image.unsqueeze(0))[0]
        share_data["images_gpu"] = batch_tensor

        share_data["meta"] = [camera_meta for (_, camera_meta) in read_list]
        # if all cameras crashed - close pipeline
        self.is_running = any([camera_meta["success"] for camera_meta in share_data["meta"]])
        if not self.is_running:
            print("All cameras are not available")
        return share_data

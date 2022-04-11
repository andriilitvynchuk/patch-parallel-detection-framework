import queue
import threading
import time
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
import torch


def patch_video_path(video_path: str) -> Union[str, int]:
    if len(video_path) == 1:
        try:
            return int(video_path)
        except ValueError:
            raise ValueError("Strange video path")
    else:
        return video_path


class MultithreadingOpencvStreamCapture:
    def __init__(self, config: Dict[str, Any]):
        self._load_cfg(config)
        self._init_source()

        self._last_connection_time: float = time.time()

        self._queue: queue.Queue = queue.Queue(1)
        self._read_thread = threading.Thread(target=self._reader, daemon=True)
        self._read_thread.start()

    def _load_cfg(self, config: Dict[str, Any]) -> None:
        self._name = config["name"]
        self._url = config["url"]
        self._colorspace = config["colorspace"].upper()
        self._channel_order = config["channel_order"].upper()
        self._device = torch.device(config["device"])
        self._height = config.get("height")
        self._width = config.get("width")
        self._framerate = config.get("framerate")
        self._rotate = config.get("rotate")
        self._reconnect_time: int = config.get("reconnect_time", 300)

    def _init_source(self) -> None:
        self._source = cv2.VideoCapture(patch_video_path(self._url))

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        if self._height is not None and self._width is not None and image.shape[:2] != (self._height, self._width):
            image = cv2.resize(image, dsize=(self._width, self._height))
        if self._colorspace == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image)
        if self._channel_order == "CHW":
            image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor.to(self._device)

    def _reader(self) -> None:
        while True:
            meta_information: Dict[str, Any] = dict()
            ret, image = self._source.read()
            if ret:
                image = self._process_image(image)
                meta_information.update(dict(time=time.time(), success=True))
                self._last_connection_time = meta_information["time"]
            else:
                image = torch.zeros((3, self._height, self._width), torch.uint8, self._device)  # type: ignore
                if self._channel_order == "HWC":
                    image = image.permute(1, 2, 0)
                meta_information.update(dict(time=time.time(), success=False))

                if meta_information["time"] - self._last_connection_time > self._reconnect_time:
                    self._source.release()
                    self._init_source()

            if self._framerate != "every_frame":
                if not self._queue.empty():
                    self._queue.get_nowait()  # discard previous (unprocessed) frame
                self._queue.put_nowait((image, meta_information))
            else:
                self._queue.put((image, meta_information))

    def read(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self._queue.get()

import threading
import time
from queue import Queue
from typing import Any, Dict, Tuple

import torch
from kornia.geometry import Rotate
from tensor_stream import FourCC, FrameRate, Planes, ResizeType, TensorStreamConverter  # type: ignore


class MultithreadingTensorStream:
    def __init__(self, config: Dict[str, Any]):
        self._load_cfg(config)
        self._init_source()
        self._last_connection_time: float = time.time()

        self._queue: Queue = Queue(1)
        self._read_thread = threading.Thread(target=self._reader, daemon=True)
        self._read_thread.start()

    def _load_cfg(self, config: Dict[str, Any]) -> None:
        self._name = config["name"]
        self._url = config["url"]
        self._height = config["height"]
        self._width = config["width"]
        self._framerate = FrameRate.BLOCKING if config["framerate"].lower() == "every_frame" else FrameRate.FAST
        self._colorspace = FourCC.RGB24 if config["colorspace"].upper() == "RGB" else FourCC.BGR24
        self._channel_order = Planes.PLANAR if config["channel_order"].upper() == "CHW" else Planes.MERGED
        self._device = config["device"]
        self._rotate = (
            Rotate(angle=torch.tensor(-config["rotate"]).to(self._device))
            if config.get("rotate") is not None
            else None
        )
        self._timeout = config.get("timeout", -1)
        self._reconnect_time: int = config.get("reconnect_time", 300)

    def _init_source(self) -> None:
        self._source: TensorStreamConverter = TensorStreamConverter(
            stream_url=self._url, cuda_device=self._device, framerate_mode=self._framerate, timeout=self._timeout
        )
        self._source.initialize()
        self._source.start()

    def _reader(self) -> None:
        while True:
            meta_information: Dict[str, Any] = dict()
            try:
                tensor = self._source.read(
                    height=self._height,
                    width=self._width,
                    planes_pos=self._channel_order,
                    pixel_format=self._colorspace,
                    resize_type=ResizeType.BILINEAR,
                )  # get last available frame from stream
                if self._rotate is not None:
                    tensor = self._rotate(torch.stack([tensor.to(torch.float)]))[0].to(torch.uint8)
                meta_information.update(dict(time=time.time(), success=True))
                self._last_connection_time = meta_information["time"]
            except RuntimeError:
                # black frame in case of camera fail
                tensor = torch.zeros(size=(3, self._height, self._width), dtype=torch.uint8)
                if self._channel_order == Planes.MERGED:
                    tensor = tensor.permute(1, 2, 0)
                meta_information.update(dict(time=time.time(), success=False))

                if meta_information["time"] - self._last_connection_time > self._reconnect_time:
                    self._source.stop()
                    self._init_source()

            if self._framerate != FrameRate.BLOCKING:
                if not self._queue.empty():
                    self._queue.get_nowait()  # discard previous (unprocessed) frame
                self._queue.put_nowait((tensor, meta_information))
            else:
                self._queue.put((tensor, meta_information))

    def read(self) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self._queue.get()

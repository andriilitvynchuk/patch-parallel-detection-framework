import datetime
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np

from dronedet.objects.base import SimpleRunnerManager  # type: ignore
from dronedet.objects.shared_numpy_utils import unlink_dict  # type: ignore


class VisualizationRunnerManager(SimpleRunnerManager):
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any], name: Optional[str] = None):
        super().__init__(name)
        self._load_cfg(config)
        self._load_global_cfg(global_config)

        self._writers: Dict[str, Any] = dict()

    def _load_cfg(self, config: Dict[str, Any]) -> None:
        self._resize = config.get("resize")
        self._save = config.get("save")
        self._verbose = config.get("verbose", True)

    def _load_global_cfg(self, config: Dict[str, Any]) -> None:
        self._cameras = config["cameras"]

    def _get_number_of_mini_runners(self) -> int:
        return len(self._cameras)

    def _write_image(
        self,
        image: np.ndarray,
        camera_index: int,
        meta_information: Dict[str, Any],
    ) -> None:
        camera_name = self._cameras[camera_index]["name"]
        file_name = datetime.datetime.fromtimestamp(meta_information["time"]).strftime("%Y-%m-%d_%H:%M:%S.%f.jpg")
        cv2.imwrite(
            os.path.join(self._writers[camera_name]["events_upper_not_safe_folder"], file_name),
            image,
        )
        if self._writers[self._cameras[camera_index]["name"]].get("video") is not None:
            self._writers[self._cameras[camera_index]["name"]]["video"].write(image)

    def _init_run(self, camera_index: int) -> None:
        self._init_drawing_one_camera(camera_params=self._cameras[camera_index])

    def _process(self, share_data: Dict[str, Any], camera_index: int) -> None:
        image = share_data["images_cpu"]
        meta_information = share_data["meta"]
        split_screen = self._split_screen.get(self._cameras[camera_index]["name"])
        original_image = image.transpose(1, 2, 0)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        debug_image = original_image.copy()
        if self._resize is not None:
            debug_image = cv2.resize(debug_image, dsize=(self._resize[1], self._resize[0]))

        debug_image = self._visualize_split_screen(debug_image, split_screen)

        self._write_image(
            image=debug_image,
            camera_index=camera_index,
            meta_information=meta_information,
        )
        # it is final Runner, delete shared memory
        unlink_dict(share_data)

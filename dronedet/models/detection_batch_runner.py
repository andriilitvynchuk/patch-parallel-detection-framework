from typing import Any, Dict, List, Optional

import numpy as np

import shared_numpy as snp
from dronedet.base import SimpleRunner
from dronedet.utils import get_index, import_object


class DetectionBatchRunner(SimpleRunner):
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any], name: Optional[str] = None):
        super().__init__(name)
        self._load_cfg(config)
        self._load_global_cfg(global_config)

        self._last_time_empty = {index: 0 for index in range(len(self._cameras))}

    def _load_cfg(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._model_class = import_object(config["class"])
        self._lazy_mode_time = config.get("lazy_mode_time", 0)
        self._verbose = config.get("verbose", True)

    def _load_global_cfg(self, config: Dict[str, Any]) -> None:
        self._cameras = list(config["cameras"].values())  # cameras is list of dicts (e.g. video: {})

    def _init_run(self) -> None:
        self._model = self._model_class(self._config["class_params"])

    def _process(self, share_data: Dict[str, Any]) -> Dict[str, Any]:
        batch_tensor = share_data["images_gpu"]
        meta = share_data["meta"]

        leave_images_for_model = [
            index
            for index in range(batch_tensor.size(0))
            if meta[index]["success"] and meta[index]["time"] - self._last_time_empty[index] > self._lazy_mode_time
        ]
        subbatch_tensor = batch_tensor[leave_images_for_model]
        forwarded_bboxes = self._model(subbatch_tensor) if subbatch_tensor.size(0) > 0 else []

        bboxes: List[np.ndarray] = []
        for index in range(batch_tensor.size(0)):
            index_in_forwarded_bboxes = get_index(element=index, element_list=leave_images_for_model)
            if index_in_forwarded_bboxes is not None:
                image_bboxes = forwarded_bboxes[index_in_forwarded_bboxes]
                if len(image_bboxes) == 0:
                    self._last_time_empty[index] = meta[index]["time"]
            else:
                image_bboxes = np.empty((0, 6))
            bboxes.append(image_bboxes)

        # add bboxes to shared memory to avoid extra pickles-unpickles.
        share_data["bboxes"] = [
            snp.from_array(image_bboxes) if len(image_bboxes) > 0 else image_bboxes for image_bboxes in bboxes
        ]
        return share_data

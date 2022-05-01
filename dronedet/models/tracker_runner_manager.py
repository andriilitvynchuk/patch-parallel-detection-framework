from typing import Any, Dict, Optional

import shared_numpy as snp
from dronedet.base import SimpleRunnerManager  # type: ignore
from dronedet.utils import import_object


class TrackerRunnerManager(SimpleRunnerManager):
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any], name: Optional[str] = None):
        super().__init__(name)
        self._load_cfg(config)
        self._load_global_cfg(global_config)

    def _load_cfg(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._tracker_class = import_object(config["class"])
        self._verbose = config.get("verbose", True)

    def _load_global_cfg(self, config: Dict[str, Any]) -> None:
        self._cameras = list(config["cameras"].values())  # cameras is list of dicts (e.g. video: {})

    def _get_number_of_mini_runners(self) -> int:
        return len(self._cameras)

    def _init_run(self, camera_index: int) -> None:
        self._tracker = self._tracker_class(self._config["params"])

    def _process(self, share_data: Dict[str, Any], camera_index: int) -> Dict[str, Any]:
        bboxes = share_data["bboxes"]

        # get tracks
        tracker_results, according_index = self._tracker.update(bboxes)
        bboxes_with_tracks = self._tracker.match_tracks_with_boxes(tracker_results, according_index, bboxes)
        # smooth labels
        bboxes_with_tracks[:, -1] = self._tracker.update_labels(
            tracks=bboxes_with_tracks[:, 4], labels=bboxes_with_tracks[:, -1]
        )
        # delete old bboxes from shared memory because if tracker exists
        # we will not send old bboxes to anyone
        if len(bboxes) > 0 and type(bboxes) is snp.shared_numpy.SharedNDArray:
            bboxes.unlink()
        # move new bboxes to shared memory
        bboxes_with_tracks = snp.from_array(bboxes_with_tracks) if len(bboxes_with_tracks) > 0 else bboxes_with_tracks
        share_data["bboxes_with_tracks"] = bboxes_with_tracks
        return share_data

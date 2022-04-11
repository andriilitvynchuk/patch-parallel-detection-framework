import multiprocessing as mp
import time
from abc import abstractclassmethod
from functools import partial
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from .simple_runner import SimpleRunner


# The main idea of RunnerManager is to work with cameras in parallel.
# (e.g Runner is mostly used for BATCH work - models inference, getting images from camera, but
# after models inference we need to do tracking, events checking and visualization and it all can be
# optimized to run in parallel per each camera.
# This manager runs several mini-runners inside of it (number pf mini-runners == number of cameras)
# It implements senders and receivers for connecting with Runner or another RunnerManager
# The methods are the same as in SimpleRunner, the main difference is that some methods have extra "camera_index" param


class SimpleRunnerManager(SimpleRunner):
    @abstractclassmethod
    def _get_number_of_mini_runners(self) -> int:
        raise NotImplementedError

    def start(self) -> None:
        self._proc_list: List[mp.Process] = [
            mp.Process(target=partial(self._run, camera_index=index), daemon=True)
            for index in range(self._get_number_of_mini_runners())
        ]
        for process in self._proc_list:
            process.start()
        self._is_running.value = True

    def join(self) -> None:
        for process in self._proc_list:
            process.join()

    def _create_connector(self, parent_class: "SimpleRunner") -> List[mp.Queue]:
        return [mp.Queue(1) for _ in range(self._get_number_of_mini_runners())]

    def _unbatch(self, data: Sequence[Any], index: int) -> Any:
        unbatched_data = data[index]
        if type(unbatched_data) is torch.Tensor and unbatched_data.device.type != "cpu":
            unbatched_data = unbatched_data.cpu()
        return unbatched_data

    def send_data_to_connector(self, share_data: Dict[str, Any], connector: List[mp.Queue], **kwargs: Any) -> None:
        camera_index = kwargs.get("camera_index")
        if camera_index is not None:
            camera_connector = connector[camera_index]
            camera_connector.put(share_data)
        else:
            unbatch_keys = kwargs.get("unbatch_keys")
            for index, camera_connector in enumerate(connector):
                camera_share_data = dict()
                for key, value in share_data.items():
                    if unbatch_keys is not None and key in unbatch_keys:
                        camera_share_data[key] = self._unbatch(data=value, index=index)
                    else:
                        camera_share_data[key] = value
                camera_connector.put(camera_share_data)

    def _receive_from_parents(self, camera_index: int) -> Dict[str, Any]:
        share_data = dict()
        for parent_info in self._parents.values():
            share_data.update(parent_info["connector"][camera_index].get())
        return share_data

    @abstractclassmethod
    def _init_run(self, camera_index: int) -> None:
        raise NotImplementedError

    @abstractclassmethod
    def _process(self, share_data: Dict[str, Any], camera_index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def _run(self, camera_index: int) -> None:
        self._init_run(camera_index=camera_index)

        timers: Dict[str, List[float]] = {"total_runner_time": [], "main_work_time": []}
        while self._is_running.value:
            total_begin = time.time()
            share_data = self._receive_from_parents(camera_index=camera_index)

            work_begin = time.time()
            result_share_data = self._process(share_data=share_data, camera_index=camera_index)
            timers["main_work_time"].append(time.time() - work_begin)

            self._send_to_children(share_data=result_share_data, camera_index=camera_index)
            timers["total_runner_time"].append(time.time() - total_begin)

            # print timers
            if len(timers["total_runner_time"]) % 100 == 0:
                if self._verbose:
                    total_mean = np.mean(timers["total_runner_time"])
                    work_mean = np.mean(timers["main_work_time"])
                    print(f"{self._name} #{camera_index} time: {total_mean:.5f} (Work time: {work_mean:.5f})")
                timers = {"total_runner_time": [], "main_work_time": []}
        print(f"Closing {self._name}")

import multiprocessing as mp
import time
from abc import abstractclassmethod
from ctypes import c_bool
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


# The SimpleRunner separates process(es) that receive something,
# process it and send result to the next Runners.
# It has parents from which it RECEIVES information (_receive_from_parents).
# Also it has children which it SENDS data that it has (_send_to_children).
# Each Runner implements the interface which defines
# how it wants other runners to connect with them (_create_connector)
# and how to send information to the other Runners (send_data_to_connector) to make it independent from each other.
# In most cases you will only need to add params loading from config and define
# _init_run and _process functions. If you want to create your own type of Runner you can
# look at implementation of RunnerManager as example of extending functionality.
# The examples of realization of Runner can be found in objects/models or objects/stream_readers


class SimpleRunner:
    def __init__(self, name: Optional[str] = None) -> None:
        """
        self._name is used as key in children and parents dicts; it can be changed after initiation
        self._receiving_params are used to save some parameters which are used in _receive_from_parents
        self._is_running - shared value that shows Runner if there is need to process information
        """
        self._name = name if name is not None else self.__class__.__name__

        self._children: Dict[str, Dict[str, Any]] = dict()
        self._parents: Dict[str, Dict[str, Any]] = dict()
        self._receiving_params: Dict[str, Any] = dict()

        self._is_running = mp.Value(c_bool, False)

    def start(self) -> None:
        """
        Start a process(es)
        """
        self._proc: mp.Process = mp.Process(target=self._run, daemon=True)
        self._proc.start()
        self._is_running.value = True

    def join(self) -> None:
        self._proc.join()

    def _get_number_of_mini_runners(self) -> int:
        """
        Number of processes that are runned inside of Runner (is needed for extensions)
        """
        return 1

    # create input queue(s) for connecting with this Runner
    def _create_connector(self, parent_class: "SimpleRunner") -> Union[mp.Queue, List[mp.Queue]]:
        """
        Create connector for interacting with parent (connector is designed to be Queue).
        If our parent is SimpleRunner create only one Queue. If our parent is RunnerManager -
        create the exact number of Queues as number of processes it has, so every mini-runner
        will have it's own Queue (can be changed and for example all mini-runners will send data in
        one queue, but it's very specific case)
        """
        if parent_class._get_number_of_mini_runners() == 1:
            return mp.Queue(1)
        else:
            return [mp.Queue(1) for _ in range(parent_class._get_number_of_mini_runners())]

    @staticmethod
    def send_data_to_connector(
        share_data: Dict[str, Any], connector: Union[mp.Queue, List[mp.Queue]], **kwargs: Any
    ) -> None:
        """
        Previous runner(parent) used to send data to child (method belongs to child).
        """
        camera_index = kwargs.get("camera_index")
        if camera_index is not None and type(connector) is list:
            connector[camera_index].put(share_data)  # type: ignore
        else:
            connector.put(share_data)  # type: ignore

    def add_child(
        self,
        child_class: "SimpleRunner",
        dict_keys: Sequence[str],
        share_connector_with: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        We connect this Runner with his child.
        Parameters:
            child_class - child class to which we will send data
            dict_keys - keys of "share_data" dictionary that we want to send him
            share_connector_with - if we want to share connector with other parent
                (e.g. 2 runners send data to same connector)
            kwargs - other realization specific parameters. This possibility is left for easier extension
        """
        connector = child_class.add_parent(parent_class=self, share_connector_with=share_connector_with)
        self._children[child_class.name] = dict(
            send_data_method=child_class.send_data_to_connector,
            connector=connector,
            dict_keys=dict_keys,
            pipeline_sending_kwargs=kwargs,
        )

    def add_parent(
        self, parent_class: "SimpleRunner", share_connector_with: Optional[str] = None
    ) -> Union[mp.Queue, List[mp.Queue]]:
        """
        This method is called by parent (Look at add_child). We create connector, save info to _parents
        and return connector so parent will also have possibility to save info to _children
        """
        if share_connector_with is not None:
            connector = self._parents[share_connector_with]["connector"]
        else:
            connector = self._create_connector(parent_class)
        self._parents[parent_class.name] = dict(connector=connector)
        return connector

    def add_receiving_params(self, params: Dict[str, Any]) -> None:
        """
        Update receiving params
        """
        self._receiving_params.update(params)

    def _receive_from_parents(self) -> Dict[str, Any]:
        """
        Receive all dicts from parents and combine it
        to "share_data" dict (which is given in _process method further)
        """
        share_data = dict()
        if self._parents:
            all_dicts_from_parents: List[Dict[str, Any]] = []
            for parent_info in self._parents.values():
                if type(parent_info["connector"]) is list:
                    for camera_connector in parent_info["connector"]:
                        all_dicts_from_parents.append(camera_connector.get())
                else:
                    all_dicts_from_parents.append(parent_info["connector"].get())

            # first batch back in list all keys that are in "batch_keys"
            batch_keys = self._receiving_params.get("batch_keys", set())
            for batch_key in batch_keys:
                share_data[batch_key] = [
                    dict_from_parent[batch_key]
                    for dict_from_parent in all_dicts_from_parents
                    if batch_key in dict_from_parent.keys()
                ]

            # add to share data all unique keys
            for dict_from_parent in all_dicts_from_parents:
                for key, value in dict_from_parent.items():
                    if key not in batch_keys:
                        share_data[key] = value

        return share_data

    def _send_to_children(self, share_data: Dict[str, Any], **runner_sending_kwargs: Any) -> None:
        """
        Send data to all children, using their send_data_to_connector method.
        """
        for child_info in self._children.values():
            share_data_subset = {key: share_data.get(key) for key in child_info["dict_keys"]}
            child_info["send_data_method"](
                share_data=share_data_subset,
                connector=child_info["connector"],
                **child_info["pipeline_sending_kwargs"],
                **runner_sending_kwargs,
            )

    @abstractclassmethod
    def _init_run(self) -> None:
        """
        This method is runned when process is starting. Declare here heavy to pickle things
        (e.g TensorRT and PyTorch models)
        """
        raise NotImplementedError

    @abstractclassmethod
    def _process(self, share_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main work. Share_data contains all data from parents. Get data that you need,
        process it, put result back to share_data and return it.
        """
        raise NotImplementedError

    def _run(self) -> None:
        """
        This method is main while loop with timers of different parts.
        Generally you don't need to change it.
        """
        self._init_run()

        self._timers: Dict[str, List[float]] = {"total_runner_time": [], "main_work_time": []}
        while self._is_running.value:
            total_begin = time.time()
            share_data = self._receive_from_parents()

            work_begin = time.time()
            result_share_data = self._process(share_data)
            self._timers["main_work_time"].append(time.time() - work_begin)

            self._send_to_children(share_data=result_share_data)
            self._timers["total_runner_time"].append(time.time() - total_begin)

            # print timers
            if len(self._timers["total_runner_time"]) % 100 == 0:
                total_mean = np.mean(self._timers["total_runner_time"])
                work_mean = np.mean(self._timers["main_work_time"])
                print(f"{self._name} time: {total_mean:.5f} (Work time: {work_mean:.5f})")
                self._timers = {"total_runner_time": [], "main_work_time": []}
        print(f"Closing {self._name}")

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def children(self) -> Any:
        return self._children

    @property
    def parents(self) -> Any:
        return self._parents

    @property
    def is_running(self) -> bool:
        return self._is_running.value

    @is_running.setter
    def is_running(self, value: bool) -> None:
        self._is_running.value = value

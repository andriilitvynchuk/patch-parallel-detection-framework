from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Tuple, Union

import numpy as np


class SharedNDArray:
    def __init__(self, array: np.ndarray, shm: SharedMemory):
        self.array = array
        self.shm = shm

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()


def create_shared_array(array: np.ndarray) -> Tuple[np.ndarray, SharedMemory]:
    # shm = SharedMemory(create=True, size=array.nbytes)
    # shm_array = SharedNDArray(array.shape, dtype=array.dtype, buffer=shm.buf)
    # shm_array[:] = array[:]
    # shm_array.set_shm(shm)
    shm = SharedMemory(create=True, size=array.nbytes)
    shm_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shm_array[:] = array[:]
    return shm_array, shm


def share_dict(dictionary: Dict[Any, Union[Any, np.ndarray]]) -> Dict[Any, Union[Any, SharedNDArray]]:
    for key, value in dictionary.items():
        if type(value) is np.ndarray and len(value) > 0:
            dictionary[key] = create_shared_array(value)
        if type(value) is dict:
            dictionary[key] = share_dict(value)
    return dictionary


def close_dict(dictionary: Dict[Any, Union[Any, SharedNDArray]]) -> None:
    for key, value in dictionary.items():
        if type(value) is SharedNDArray:
            dictionary[key].close()
        if type(value) is dict:
            close_dict(value)


def unlink_dict(dictionary: Dict[Any, Union[Any, SharedNDArray]]) -> None:
    for key, value in dictionary.items():
        if type(value) is SharedNDArray:
            dictionary[key].unlink()
        if type(value) is dict:
            unlink_dict(value)

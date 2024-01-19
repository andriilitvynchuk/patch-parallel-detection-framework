from typing import Any, Dict, Union

from shared_numpy.shared_numpy import SharedNDArray  # type: ignore


def unlink_dict(dictionary: Dict[Any, Union[Any, SharedNDArray]]) -> None:
    for key, value in dictionary.items():
        if type(value) is SharedNDArray:
            dictionary[key].unlink()
        if type(value) is dict:
            unlink_dict(value)

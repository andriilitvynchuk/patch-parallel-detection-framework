from typing import Any, Dict, Union

import numpy as np
import shared_numpy as snp
from shared_numpy.shared_numpy import SharedNDArray


def share_dict(dictionary: Dict[Any, Union[Any, np.ndarray]]) -> Dict[Any, Union[Any, SharedNDArray]]:
    for key, value in dictionary.items():
        if type(value) is np.ndarray and len(value) > 0:
            dictionary[key] = snp.from_array(value)
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

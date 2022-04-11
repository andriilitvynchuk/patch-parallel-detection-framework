from typing import Any, Dict

import numpy as np


def rle_encode(mask: np.ndarray) -> np.ndarray:
    pixels = mask.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.array(np.where(pixels[1:] != pixels[:-1])[0] + 2, dtype=np.int32)
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def rle_decode(encoded_mask: np.ndarray, shape: int) -> np.ndarray:
    mask = np.zeros(shape=shape + 2, dtype=np.uint8)
    if encoded_mask.size != 0:
        start = encoded_mask[0]
        for i, start in enumerate(encoded_mask[::2]):
            length = encoded_mask[2 * i + 1]
            mask[start : start + length] = 1
    return mask[1:-1]


def masks_in_dict_rle_encode(dictionary: Dict[Any, Any]) -> Dict[Any, Any]:
    for key, value in dictionary.items():
        # find masks
        if type(value) is np.ndarray and value.dtype == np.uint8:
            # encode in format {0: rle_encoding, 1: rle_encoding, ..., N-1: rle_encoding, shape: mask_shape}
            if len(value.shape) == 3:
                dictionary[key] = dict()
                dictionary[key]["rle"] = {
                    index: rle_encode(mask).tolist() for index, mask in enumerate(value)
                }
            elif len(value.shape) == 2:
                dictionary[key] = dict()
                dictionary[key]["rle"] = {0: rle_encode(value).tolist()}
            else:
                raise ValueError("Not support such masks or images")
            dictionary[key]["shape"] = value.shape[-2:]
        if type(value) == dict:
            dictionary[key] = masks_in_dict_rle_encode(value)
    return dictionary

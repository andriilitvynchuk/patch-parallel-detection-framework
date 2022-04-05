import os

from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig


def patch_empty_params(cfg: DictConfig) -> DictConfig:
    """
    Find all key-value pair where value is None and input empty dict instead of None
    """
    for key, value in cfg.items():
        cfg[key] = dict() if "params" in key and value is None else value  # type: ignore
        value_type = type(value)
        if value_type is DictConfig:
            cfg[key] = patch_empty_params(value)
        elif value_type is ListConfig:
            cfg[key] = [patch_empty_params(element) if type(element) is DictConfig else element for element in value]
    return cfg


def patch_relative_paths(cfg: DictConfig, original_cwd: str) -> DictConfig:
    """
    Final all keys which has "path" word inside of it and extend it to absolute path
    """
    for key, value in cfg.items():
        if "path" in key and key != "dirpath" and value is not None:  # type: ignore
            if type(value) == ListConfig:
                cfg[key] = [os.path.join(original_cwd, item) if not os.path.isabs(item) else item for item in value]
            elif not os.path.isabs(value):
                cfg[key] = os.path.join(original_cwd, value)
        value_type = type(value)
        if (value_type is DictConfig or value_type is dict) and len(value) > 0:
            cfg[key] = patch_relative_paths(value, original_cwd)
    return cfg

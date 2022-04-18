import importlib
import os
from typing import Any

from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig


def patch_empty_params(cfg: DictConfig) -> DictConfig:
    """
    Find all key-value pair where value is None and input empty dict instead of None
    """
    for key, value in cfg.items():
        value_type = type(value)
        if value_type is DictConfig:
            cfg[key] = patch_empty_params(value)
        elif value_type is ListConfig:
            cfg[key] = [patch_empty_params(element) if type(element) is DictConfig else element for element in value]
        else:
            cfg[key] = dict() if "params" in key and value is None else value  # type: ignore
    return cfg


def patch_single_path(key: Any, value: Any, original_cwd: str) -> str:
    if "path" in key and key != "dirpath" and value is not None and not os.path.isabs(value):  # type: ignore
        value = os.path.join(original_cwd, value)
    return value


def patch_relative_paths(cfg: DictConfig, original_cwd: str) -> DictConfig:
    """
    Change all relative paths -> absolute paths to make it work with hydra.
    key must contain "path" string
    """
    for key, value in cfg.items():
        value_type = type(value)
        if value_type is DictConfig:
            cfg[key] = patch_relative_paths(value, original_cwd)
        elif value_type is ListConfig:
            cfg[key] = [
                patch_relative_paths(element, original_cwd)
                if type(element) is DictConfig
                else patch_single_path(key, element, original_cwd)
                for element in value
            ]
        else:
            cfg[key] = patch_single_path(key, value, original_cwd)
    return cfg


def import_object(name: str) -> Any:
    """
    This function can import any Python object from path. Example:

    timer_class = import_object("leiadlutils.general.Timer")
    timer = timer_class()

    or

    timer = import_object("leiadlutils.general.Timer")()

    """
    components = name.split(".")
    mod = importlib.import_module(".".join(components[:-1]))
    return getattr(mod, components[-1])

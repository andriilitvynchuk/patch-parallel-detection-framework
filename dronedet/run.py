from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

from dronedet.base import SimplePipeline
from dronedet.stream_readers import ReadImagesToBatchRunner
from dronedet.utils import patch_empty_params, patch_relative_paths
from dronedet.visualization import VisualizationRunnerManager


class DroneDetPipeline(SimplePipeline):
    def __init__(self, config: Dict[str, Any]):
        self.read_images_to_batch_runner = ReadImagesToBatchRunner(config=config["read_images"], global_config=config)
        self.visualization_runner_manager = VisualizationRunnerManager(
            config=config["visualization"], global_config=config
        )

    def connect_runners(self) -> None:
        self.read_images_to_batch_runner.add_child(
            self.visualization_runner_manager,
            dict_keys=["images_cpu", "bboxes", "meta"],
            unbatch_keys=["images_cpu", "bboxes", "meta"],
        )

    def start(self) -> None:
        self._recursive_start(self.read_images_to_batch_runner)

    def join(self) -> None:
        self._recursive_join(self.read_images_to_batch_runner)


def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    pipeline = DroneDetPipeline(config=cfg)  # type: ignore
    pipeline.connect_runners()
    pipeline.start()
    pipeline.join()


# config_path is relative path to config folder (from script), config_name is name of main config in that folder
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = patch_empty_params(cfg)
    cfg = patch_relative_paths(cfg, hydra.utils.get_original_cwd())
    run(cfg=cfg)


if __name__ == "__main__":
    main()

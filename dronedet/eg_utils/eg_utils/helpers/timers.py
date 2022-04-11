import time
from collections import defaultdict
from pprint import pprint
from typing import Dict, List, Optional

import numpy as np
import torch


class Timer:
    def __init__(self) -> None:
        self._zero_timers()

    def _zero_timers(self) -> None:
        self._timers: Dict[str, List[float]] = defaultdict(list)

    def start_countdown(self) -> None:
        self._t_global = time.time()
        self._t_local = time.time()

    def update_local_timer(self, key: str, divide: float = 1) -> None:
        torch.cuda.synchronize()
        self._timers[key].append((time.time() - self._t_local) / divide)
        self._t_local = time.time()

    def update_global_timer(self, key: str, divide: float = 1) -> None:
        self._timers[key].append((time.time() - self._t_global) / divide)

    def get_mean_results(self, last: Optional[int] = None) -> Dict[str, float]:
        last_t = (
            last
            if last is not None
            else max(map(lambda x: len(self._timers[x]), self._timers))
        )
        mean_results = {
            key: np.mean(value[-last_t:]) for key, value in self._timers.items()
        }
        self._zero_timers()
        return mean_results

    def print_mean_results(self, last: Optional[int] = None) -> None:
        mean_results = self.get_mean_results()
        sorted_results = sorted(
            [(key, value) for key, value in mean_results.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        pprint(sorted_results)

    @property
    def timers(self) -> Dict[str, List[float]]:
        return self._timers

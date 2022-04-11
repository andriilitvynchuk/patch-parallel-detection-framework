import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional


class Buffer:
    def __init__(self, buffer: int = 1):
        """
        Params:
            buffer: int - discrete number of elements for making choice
        """
        self._buffer = buffer
        self._history: deque = deque(maxlen=buffer)

    def _choose_value(self) -> Any:
        return Counter(self._history).most_common(1)[0][0]

    def update(
        self, value: Any, not_full_return_value: Optional[Any] = None
    ) -> Optional[Any]:
        self._history.append(value)
        if len(self._history) == self._buffer:
            return self._choose_value()
        else:
            return not_full_return_value


class TimeBuffer:
    def __init__(self, buffer: float = 1):
        """
        Params:
            buffer: float - time in seconds to store values for making choice
        """
        self._buffer = buffer
        self._history: List[Dict[str, Any]] = []
        self._first_time: Optional[float] = None

    def _choose_value(self) -> Any:
        return Counter([info_dict["value"] for info_dict in self._history]).most_common(
            1
        )[0][0]

    def update(
        self, value: Any, not_full_return_value: Optional[Any] = None
    ) -> Optional[Any]:
        self._history.append(dict(value=value, time=time.time()))
        if self._first_time is None:
            self._first_time = self._history[-1]["time"]
        if self._history[-1]["time"] - self._first_time >= self._buffer:
            # filter old values
            self._history = [
                info_dict
                for info_dict in self._history
                if self._history[-1]["time"] - info_dict["time"] < self._buffer
            ]
            return self._choose_value()
        else:
            return not_full_return_value

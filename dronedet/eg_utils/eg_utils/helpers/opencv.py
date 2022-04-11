import os
import threading
from queue import Empty, Queue
from typing import Any

import cv2
import numpy as np

from .utils import listdir


def rotate_image(
    image: np.ndarray, angle: float, warp_flags: int = cv2.INTER_LINEAR
) -> np.ndarray:
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=warp_flags)
    return result


def leave_biggest_instance(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    _, label_instances = cv2.connectedComponents(mask, connectivity=connectivity)
    labels, counts = np.unique(label_instances, return_counts=True)
    # except zero
    if len(labels) > 1:
        max_label = labels[1 + np.argmax(counts[1:])]
        max_instance = (label_instances == max_label).astype(np.uint8)
    else:
        max_instance = mask
    return max_instance


def delete_smallest_instances(
    mask: np.ndarray, threshold: float, connectivity: int = 8
) -> np.ndarray:
    _, label_instances = cv2.connectedComponents(mask, connectivity=connectivity)
    labels, counts = np.unique(label_instances, return_counts=True)
    if len(labels) > 1:
        mask_square = mask.shape[0] * mask.shape[1]
        delete_labels = [
            label
            for label, count in zip(labels[1:], counts[1:])
            if count / mask_square < threshold
        ]
        for delete_label in delete_labels:
            label_instances[label_instances == delete_label] = 0
        output_mask = (label_instances > 0).astype(np.uint8)
    else:
        output_mask = mask
    return output_mask


class StreamCapture:
    def __init__(self, name: str):
        self.cap = cv2.VideoCapture(name)
        self.queue: Queue = Queue(1)
        read_thread = threading.Thread(target=self._reader, daemon=True)
        read_thread.start()

    def _reader(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not self.queue.empty():
                try:
                    self.queue.get_nowait()  # discard previous (unprocessed) frame
                except Empty:
                    pass
            self.queue.put_nowait((ret, frame))
            if not ret:
                break

    def read(self) -> np.ndarray:
        return self.queue.get()

    def get(self, item: Any) -> Any:
        return self.cap.get(item)


class VideoCapture:
    def __init__(self, name: str):
        self.cap = cv2.VideoCapture(name)
        self.queue: Queue = Queue(1)
        read_thread = threading.Thread(target=self._reader, daemon=True)
        read_thread.start()

    def _reader(self) -> None:
        while True:
            ret, frame = self.cap.read()
            self.queue.put((ret, frame))
            if not ret:
                break

    def read(self) -> np.ndarray:
        return self.queue.get()

    def get(self, item: Any) -> Any:
        return self.cap.get(item)


class ImageFolderReader:
    def __init__(self, path: str):
        self.path = path
        self.images_list = listdir(path)
        self.index = 0
        self.queue: Queue = Queue(1)
        read_thread = threading.Thread(target=self._reader, daemon=True)
        read_thread.start()

    def _reader(self) -> None:
        while self.index < len(self.images_list):
            frame = cv2.imread(os.path.join(self.path, self.images_list[self.index]))
            self.queue.put((True, frame))
            self.index += 1
        self.queue.put((False, None))

    def get(self) -> np.ndarray:
        return self.queue.get()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameSourceError(RuntimeError):
    pass


class FrameSource:
    """Abstract frame source wrapper for image/video/camera inputs."""

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self) -> None:
        return


@dataclass
class SourceSpec:
    mode: str
    image_path: str
    video_path: str
    camera_index: int
    loop_video: bool


class ImageSource(FrameSource):
    def __init__(self, image_path: str):
        path = Path(image_path)
        frame = cv2.imread(str(path))
        if frame is None:
            raise FrameSourceError(f"Failed to open image: {path}")
        self._frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        # Return a copy so downstream code can draw overlays safely.
        return True, self._frame.copy()


class VideoSource(FrameSource):
    def __init__(self, video_path: str, loop: bool = True):
        self._path = str(Path(video_path))
        self._loop = bool(loop)
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise FrameSourceError(f"Failed to open video: {self._path}")

    def _rewind(self) -> bool:
        self._cap.release()
        self._cap = cv2.VideoCapture(self._path)
        return self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self._cap.read()
        if ok:
            return True, frame
        if not self._loop:
            return False, None
        if not self._rewind():
            return False, None
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()


class CameraSource(FrameSource):
    def __init__(self, camera_index: int):
        self._camera_index = int(camera_index)
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise FrameSourceError(f"Failed to open camera index {self._camera_index}")
        # Keep only the latest frame to reduce stutter.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()


def create_source(spec: SourceSpec) -> FrameSource:
    mode = spec.mode
    if mode == "image":
        return ImageSource(spec.image_path)
    if mode == "video":
        return VideoSource(spec.video_path, loop=spec.loop_video)
    if mode == "camera":
        return CameraSource(spec.camera_index)
    raise FrameSourceError(f"Unsupported source mode: {mode}")

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2


class VideoSource:
    def __init__(self, source_type: str, source_name: str, source_value: str | int):
        self.source_type = source_type
        self.source_name = source_name
        self.source_value = str(source_value)
        self.capture: Optional[cv2.VideoCapture] = None

    def open(self) -> "VideoSource":
        if self.source_type == "usb_camera":
            self.capture = cv2.VideoCapture(int(self.source_value))
        elif self.source_type == "video_file":
            path = Path(self.source_value)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {path}")
            self.capture = cv2.VideoCapture(str(path))
        elif self.source_type == "rtsp":
            self.capture = cv2.VideoCapture(self.source_value)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")

        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError(f"Unable to open source: {self.source_name}")
        return self

    def read(self) -> Tuple[bool, Optional[object]]:
        if self.capture is None:
            return False, None
        return self.capture.read()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    @property
    def is_open(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

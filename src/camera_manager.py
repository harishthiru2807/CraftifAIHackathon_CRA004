from __future__ import annotations

from typing import Optional, Tuple

from src.video_source import VideoSource


class CameraManager:
    def __init__(self, source_type: str = "usb_camera", source_name: str = "Camera 0", source_value: str | int = 0):
        self.source_type = source_type
        self.source_name = source_name
        self.source_value = source_value
        self.video_source: Optional[VideoSource] = None
        self.status = "idle"
        self.paused = False

    def configure(self, source_type: str, source_name: str, source_value: str | int) -> None:
        self.source_type = source_type
        self.source_name = source_name
        self.source_value = source_value
        if self.video_source is not None:
            self.video_source.close()
            self.video_source = None
        self.status = "configured"

    def start(self) -> None:
        self.video_source = VideoSource(self.source_type, self.source_name, self.source_value)
        self.video_source.open()
        self.status = "running"
        self.paused = False

    def stop(self) -> None:
        if self.video_source is not None:
            self.video_source.close()
        self.video_source = None
        self.status = "stopped"
        self.paused = False

    def pause(self) -> None:
        self.paused = True
        self.status = "paused"

    def resume(self) -> None:
        self.paused = False
        self.status = "running"

    def reconnect(self) -> None:
        if self.video_source is not None:
            self.video_source.close()
        self.start()

    def read_frame(self) -> Tuple[bool, Optional[object]]:
        if self.paused or self.video_source is None:
            return False, None
        return self.video_source.read()

    @property
    def active_source(self) -> str:
        return self.source_name

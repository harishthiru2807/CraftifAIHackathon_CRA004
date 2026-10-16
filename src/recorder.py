from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoRecorder:
    def __init__(self, output_dir: str | Path = "recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer: Optional[cv2.VideoWriter] = None
        self.output_path: Optional[Path] = None
        self.started_at: Optional[datetime] = None
        self.is_recording = False

    def start(self, frame_shape: Tuple[int, int, int], fps: float = 20.0, file_prefix: str = "recording") -> Optional[Path]:
        if self.writer is not None:
            self.stop()
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("Recording dimensions and FPS must be positive.")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.output_path = self.output_dir / f"{file_prefix}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        self.started_at = datetime.now(timezone.utc)
        self.is_recording = self.writer.isOpened()
        if not self.is_recording:
            self.writer = None
            self.output_path = None
            self.started_at = None
        return self.output_path

    def write(self, frame: np.ndarray) -> bool:
        if self.writer is None or not self.is_recording:
            return False
        if frame.shape[1] <= 0 or frame.shape[0] <= 0:
            raise ValueError("Cannot record an empty video frame.")
        self.writer.write(frame)
        return True

    def stop(self) -> Optional[Path]:
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.is_recording = False
        result = self.output_path
        self.output_path = None
        self.started_at = None
        return result

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

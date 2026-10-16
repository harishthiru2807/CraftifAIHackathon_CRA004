from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class SnapshotManager:
    def __init__(self, output_dir: str | Path = "snapshots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, frame: np.ndarray, label: str = "snapshot") -> Optional[Path]:
        if frame is None:
            return None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"{label}_{timestamp}.jpg"
        success = cv2.imwrite(str(path), frame)
        if not success:
            return None
        return path

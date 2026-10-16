from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from src.alert_manager import AlertManager
from src.detector import Detector
from src.event_manager import EventManager
from src.metrics import MetricsTracker
from src.recorder import VideoRecorder
from src.snapshot_manager import SnapshotManager


class VisionPipeline:
    def __init__(
        self,
        project_root: str | Path,
        confidence_threshold: float = 0.25,
        alert_cooldown_seconds: float = 10.0,
        auto_snapshot_on_event: bool = True,
        source_name: str = "Camera 0",
        model_name: str = "yolov8n.pt",
        device: str = "cpu",
    ):
        self.project_root = Path(project_root)
        self.confidence_threshold = confidence_threshold
        self.source_name = source_name
        self.detector = Detector(model_name=model_name, confidence_threshold=confidence_threshold, device=device)
        self.alert_manager = AlertManager(cooldown_seconds=alert_cooldown_seconds)
        self.event_manager = EventManager(self.project_root)
        self.metrics_tracker = MetricsTracker()
        self.snapshot_manager = SnapshotManager(self.project_root / "snapshots")
        self.recorder = VideoRecorder(self.project_root / "recordings")
        self.auto_snapshot_on_event = auto_snapshot_on_event
        self.frame_count = 0

    def process_frame(
        self,
        frame: np.ndarray,
        source_name: str | None = None,
        source_type: str = "usb_camera",
    ) -> Tuple[np.ndarray, List[dict], List[dict]]:
        if frame is None:
            return None, [], []
        source_name = source_name or self.source_name
        self.detector.confidence_threshold = self.confidence_threshold
        inference_started = time.perf_counter()
        detections = [
            detection
            for detection in self.detector.detect(frame)
            if float(detection.get("confidence", 0.0)) >= self.confidence_threshold
        ]
        self.inference_ms = (time.perf_counter() - inference_started) * 1000
        annotated = self.detector.annotate_frame(frame, detections)
        cv2.putText(
            annotated,
            datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        events: List[dict] = []
        alerts: List[dict] = []

        for detection in detections:
            class_name = detection.get("class_name", "object")
            confidence = float(detection.get("confidence", 0.0))
            bbox = detection.get("bounding_box", {})
            should_trigger = self.alert_manager.should_trigger(class_name)
            if not should_trigger:
                continue
            alert_record = self.alert_manager.add_alert(class_name, confidence, source_name)
            alerts.append(alert_record)
            snapshot = self.snapshot_manager.save_snapshot(annotated, label=class_name) if self.auto_snapshot_on_event else None
            event = self.event_manager.create_event(
                source_type=source_type,
                source_name=source_name,
                class_name=class_name,
                confidence=confidence,
                bounding_box=bbox,
                snapshot_path=str(snapshot.relative_to(self.project_root)) if snapshot else "",
                alert_triggered=True,
            )
            self.event_manager.record_event(event)
            events.append(event)

        self.metrics_tracker.update(detections, source_name)
        self.frame_count += 1
        if self.recorder.is_recording:
            self.recorder.write(annotated)
        return annotated, detections, events

    def start_recording(self, frame_shape: Tuple[int, ...], fps: float = 20.0) -> str | None:
        output = self.recorder.start(frame_shape, fps=fps)
        if output is None:
            return None
        return str(output)

    def stop_recording(self) -> str | None:
        output = self.recorder.stop()
        if output is None:
            return None
        return str(output)

    def stats(self) -> Dict[str, Any]:
        return self.metrics_tracker.snapshot()

from __future__ import annotations

import numpy as np

import src.pipeline as pipeline_module
from src.pipeline import VisionPipeline


class StubDetector:
    def __init__(self, **kwargs):
        self.model_name = kwargs["model_name"]
        self.confidence_threshold = kwargs["confidence_threshold"]
        self.error_message = ""

    def detect(self, frame):
        return [
            {
                "class_name": "person",
                "confidence": 0.91,
                "bounding_box": {"x1": 1, "y1": 2, "x2": 20, "y2": 40},
            }
        ]

    def annotate_frame(self, frame, detections):
        return frame.copy()


def test_pipeline_records_one_cooldown_event_with_actual_source_type(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "Detector", StubDetector)
    pipeline = VisionPipeline(
        project_root=tmp_path,
        auto_snapshot_on_event=False,
        alert_cooldown_seconds=30,
    )
    frame = np.zeros((32, 48, 3), dtype=np.uint8)

    _, _, first_events = pipeline.process_frame(frame, "sample.mp4", source_type="video_file")
    _, _, duplicate_events = pipeline.process_frame(frame, "sample.mp4", source_type="video_file")

    assert len(first_events) == 1
    assert duplicate_events == []
    persisted = pipeline.event_manager.list_events()
    assert len(persisted) == 1
    assert persisted[0]["source_type"] == "video_file"
    assert persisted[0]["source_name"] == "sample.mp4"

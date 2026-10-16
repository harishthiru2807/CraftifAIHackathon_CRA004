from __future__ import annotations

from src.metadata_exporter import export_csv, export_json, export_jsonl, validate_event_schema


def test_metadata_exports(tmp_path):
    events = [
        {
            "event_id": "evt-1",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "source_type": "usb_camera",
            "source_name": "Camera 0",
            "class_name": "person",
            "confidence": 0.88,
            "bounding_box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
            "snapshot_path": "snapshots/sample.jpg",
            "alert_triggered": True,
        }
    ]
    json_path = export_json(tmp_path / "events.json", events)
    csv_path = export_csv(tmp_path / "events.csv", events)
    jsonl_path = export_jsonl(tmp_path / "events.jsonl", events)

    assert validate_event_schema(events[0])
    assert json_path.exists()
    assert csv_path.exists()
    assert jsonl_path.exists()

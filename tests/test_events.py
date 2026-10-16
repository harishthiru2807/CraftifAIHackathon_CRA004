from __future__ import annotations

from src.event_manager import EventManager


def test_event_recording(tmp_path):
    manager = EventManager(tmp_path)
    event = manager.create_event(
        source_type="usb_camera",
        source_name="Camera 0",
        class_name="person",
        confidence=0.92,
        bounding_box={"x1": 100, "y1": 80, "x2": 300, "y2": 450},
        snapshot_path="snapshots/test.jpg",
        alert_triggered=True,
    )
    manager.record_event(event)
    events = manager.list_events()
    assert len(events) == 1
    assert events[0]["class_name"] == "person"
    assert events[0]["alert_triggered"] is True
    assert events[0]["source_name"] == "Camera 0"

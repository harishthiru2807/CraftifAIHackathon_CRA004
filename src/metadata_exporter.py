from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List


REQUIRED_FIELDS = {
    "event_id",
    "timestamp",
    "source_type",
    "source_name",
    "class_name",
    "confidence",
    "bounding_box",
    "snapshot_path",
    "alert_triggered",
}


def validate_event_schema(event: dict) -> bool:
    if not isinstance(event, dict):
        return False
    if not REQUIRED_FIELDS.issubset(event.keys()):
        return False
    bbox = event.get("bounding_box", {})
    if not isinstance(bbox, dict):
        return False
    required_bbox = {"x1", "y1", "x2", "y2"}
    if not required_bbox.issubset(bbox.keys()):
        return False
    return True


def export_jsonl(path: str | Path, events: Iterable[dict]) -> Path:
    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    records = list(events)
    with export_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if validate_event_schema(record):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return export_path


def export_json(path: str | Path, events: Iterable[dict]) -> Path:
    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    data = list(events)
    with export_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return export_path


def export_csv(path: str | Path, events: Iterable[dict]) -> Path:
    export_path = Path(path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    records = list(events)
    fieldnames = [
        "event_id",
        "timestamp",
        "source_type",
        "source_name",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "snapshot_path",
        "alert_triggered",
    ]
    with export_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            bbox = record.get("bounding_box", {})
            row = {
                "event_id": record.get("event_id", ""),
                "timestamp": record.get("timestamp", ""),
                "source_type": record.get("source_type", ""),
                "source_name": record.get("source_name", ""),
                "class_name": record.get("class_name", ""),
                "confidence": record.get("confidence", 0.0),
                "x1": bbox.get("x1", 0),
                "y1": bbox.get("y1", 0),
                "x2": bbox.get("x2", 0),
                "y2": bbox.get("y2", 0),
                "snapshot_path": record.get("snapshot_path", ""),
                "alert_triggered": record.get("alert_triggered", False),
            }
            writer.writerow(row)
    return export_path

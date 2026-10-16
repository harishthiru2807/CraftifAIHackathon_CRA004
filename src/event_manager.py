from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from src.metadata_exporter import export_csv, export_json, export_jsonl, validate_event_schema


class EventManager:
    def __init__(self, storage_dir: str | Path = "."):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.event_file = self.storage_dir / "events.jsonl"
        self.event_file.touch(exist_ok=True)
        self._event_cache: List[Dict[str, Any]] = []

    def create_event(
        self,
        source_type: str,
        source_name: str,
        class_name: str,
        confidence: float,
        bounding_box: Dict[str, float],
        snapshot_path: str | None = None,
        alert_triggered: bool = False,
    ) -> Dict[str, Any]:
        event = {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_type": source_type,
            "source_name": source_name,
            "class_name": class_name,
            "confidence": float(confidence),
            "bounding_box": {
                "x1": float(bounding_box.get("x1", 0.0)),
                "y1": float(bounding_box.get("y1", 0.0)),
                "x2": float(bounding_box.get("x2", 0.0)),
                "y2": float(bounding_box.get("y2", 0.0)),
            },
            "snapshot_path": snapshot_path or "",
            "alert_triggered": bool(alert_triggered),
        }
        self._event_cache.append(event)
        return event

    def record_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if not validate_event_schema(event):
            raise ValueError("Event schema validation failed before persistence.")
        with self.event_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._event_cache.append(event)
        return event

    def list_events(self) -> List[Dict[str, Any]]:
        if self.event_file.exists() and self.event_file.stat().st_size > 0:
            events: List[Dict[str, Any]] = []
            with self.event_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if validate_event_schema(record):
                        events.append(record)
            return events
        return list(self._event_cache)

    def export_json(self, target_path: str | Path) -> Path:
        events = self.list_events()
        return export_json(target_path, events)

    def export_jsonl(self, target_path: str | Path) -> Path:
        events = self.list_events()
        return export_jsonl(target_path, events)

    def export_csv(self, target_path: str | Path) -> Path:
        events = self.list_events()
        return export_csv(target_path, events)

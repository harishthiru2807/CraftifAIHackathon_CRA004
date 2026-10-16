from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List


class MetricsTracker:
    def __init__(self):
        self.total_detections = 0
        self.detections_by_class: Counter[str] = Counter()
        self.confidence_values: List[float] = []
        self.recent_events: List[Dict[str, Any]] = []
        self.session_started = datetime.now(timezone.utc)

    def update(self, detections: List[dict], source_name: str = "") -> None:
        for detection in detections:
            class_name = detection.get("class_name", "unknown")
            confidence = float(detection.get("confidence", 0.0))
            self.total_detections += 1
            self.detections_by_class[class_name] += 1
            self.confidence_values.append(confidence)
            self.recent_events.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_name": source_name,
                    "class_name": class_name,
                    "confidence": confidence,
                }
            )
            if len(self.recent_events) > 50:
                self.recent_events = self.recent_events[-50:]

    def snapshot(self) -> Dict[str, Any]:
        if not self.confidence_values:
            avg_confidence = 0.0
            max_confidence = 0.0
        else:
            avg_confidence = sum(self.confidence_values) / len(self.confidence_values)
            max_confidence = max(self.confidence_values)
        return {
            "total_detections": self.total_detections,
            "counts_by_class": dict(self.detections_by_class),
            "average_confidence": round(avg_confidence, 4),
            "max_confidence": round(max_confidence, 4),
            "recent_events": self.recent_events[-10:],
            "session_duration_seconds": round((datetime.now(timezone.utc) - self.session_started).total_seconds(), 2),
        }

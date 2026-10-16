from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List


class AlertManager:
    def __init__(self, cooldown_seconds: float = 10.0, max_history: int = 50):
        self.cooldown_seconds = cooldown_seconds
        self.max_history = max_history
        self.alert_history: List[Dict[str, Any]] = []
        self._last_triggered: Dict[str, datetime] = {}

    def should_trigger(self, class_name: str, now: datetime | None = None) -> bool:
        if now is None:
            now = datetime.now(timezone.utc)
        last_seen = self._last_triggered.get(class_name)
        if last_seen is not None:
            elapsed = (now - last_seen).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False
        self._last_triggered[class_name] = now
        return True

    def add_alert(self, class_name: str, confidence: float, source_name: str, timestamp: str | None = None) -> Dict[str, Any]:
        payload = {
            "class_name": class_name,
            "confidence": float(confidence),
            "source_name": source_name,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "message": f"{class_name} detected with confidence {confidence:.2f} on {source_name}.",
        }
        self.alert_history.append(payload)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        return payload

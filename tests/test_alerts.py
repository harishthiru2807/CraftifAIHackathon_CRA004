from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.alert_manager import AlertManager


def test_alert_cooldown():
    alert_manager = AlertManager(cooldown_seconds=10.0)
    base = datetime.now(timezone.utc)

    assert alert_manager.should_trigger("person", base) is True
    assert alert_manager.should_trigger("person", base + timedelta(seconds=3)) is False
    assert alert_manager.should_trigger("vehicle", base + timedelta(seconds=6)) is True

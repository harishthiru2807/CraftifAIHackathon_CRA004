from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4


VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]


def persist_uploaded_video(uploaded_file, output_dir: str | Path) -> Path:
    """Persist a video upload under a unique, sanitized local filename."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix.lstrip(".") not in VIDEO_TYPES:
        raise ValueError(f"Unsupported video format: {suffix or 'no extension'}")

    original_name = uploaded_file.name.replace("\\", "/").split("/")[-1]
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", original_name).strip("._")
    if not safe_name:
        safe_name = f"video{suffix}"

    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{uuid4().hex}_{safe_name}"
    with destination.open("xb") as output:
        output.write(uploaded_file.getvalue())
    return destination

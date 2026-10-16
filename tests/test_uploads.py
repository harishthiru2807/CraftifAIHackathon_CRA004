from __future__ import annotations

import pytest

from src.uploads import persist_uploaded_video


class FakeUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_uploaded_video_is_saved_with_safe_unique_name(tmp_path):
    upload = FakeUpload(r"..\folder\sample clip.mp4", b"sample-video")

    saved_path = persist_uploaded_video(upload, tmp_path)

    assert saved_path.parent == tmp_path
    assert saved_path.name.endswith("_sample_clip.mp4")
    assert saved_path.read_bytes() == b"sample-video"


def test_uploaded_video_rejects_unsupported_extension(tmp_path):
    upload = FakeUpload("sample.txt", b"not a video")

    with pytest.raises(ValueError, match="Unsupported video format"):
        persist_uploaded_video(upload, tmp_path)

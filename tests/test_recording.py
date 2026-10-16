from __future__ import annotations

import numpy as np

from src.recorder import VideoRecorder


def test_recording_creates_file(tmp_path):
    recorder = VideoRecorder(tmp_path)
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    output_path = recorder.start((60, 80, 3), fps=10.0)
    assert output_path is not None
    assert recorder.write(frame) is True
    saved = recorder.stop()
    assert saved is not None
    assert saved.exists()
    capture = __import__("cv2").VideoCapture(str(saved))
    success, recorded_frame = capture.read()
    capture.release()
    assert success
    assert recorded_frame.shape[:2] == frame.shape[:2]

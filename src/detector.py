from __future__ import annotations

from typing import List

import numpy as np
import cv2


class Detector:
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.25, device: str = "cpu"):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.error_message = ""
        self._initialization_attempted = False

    def _initialize(self) -> None:
        if self._initialization_attempted:
            return
        self._initialization_attempted = True
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_name)
        except Exception as exc:  # pragma: no cover - environment-dependent
            self.model = None
            self.error_message = str(exc)

    def detect(self, frame: np.ndarray | None) -> List[dict]:
        if frame is None:
            return []
        self._initialize()
        if self.model is None:
            return []

        try:
            results = self.model(frame, conf=self.confidence_threshold, device=self.device, verbose=False)
        except Exception as exc:  # pragma: no cover - runtime-model errors
            self.error_message = str(exc)
            return []

        detections: List[dict] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", {})
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = names.get(class_id, f"class_{class_id}")
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": confidence,
                        "bounding_box": {
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                        },
                    }
                )
        return detections

    def annotate_frame(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        annotated = frame.copy()
        for detection in detections:
            bbox = detection.get("bounding_box", {})
            x1, y1, x2, y2 = [int(float(bbox.get(key, 0))) for key in ("x1", "y1", "x2", "y2")]
            label = f"{detection.get('class_name', 'object')} {detection.get('confidence', 0.0):.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.camera_manager import CameraManager
from src.pipeline import VisionPipeline

PROJECT_ROOT = Path(__file__).resolve().parent


def ensure_pipeline() -> VisionPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = VisionPipeline(
            project_root=PROJECT_ROOT,
            confidence_threshold=0.25,
            alert_cooldown_seconds=10.0,
            auto_snapshot_on_event=True,
            model_name="yolov8n.pt",
            device="cpu",
        )
    return st.session_state.pipeline


def ensure_camera_manager() -> CameraManager:
    if "camera_manager" not in st.session_state:
        st.session_state.camera_manager = CameraManager(source_type="usb_camera", source_name="Camera 0", source_value=0)
    return st.session_state.camera_manager


st.set_page_config(page_title="EnviroGuard Vision AI", layout="wide")
st.title("EnviroGuard Vision AI")
st.caption("Real-time edge vision monitoring dashboard")

pipeline = ensure_pipeline()
camera_manager = ensure_camera_manager()

with st.sidebar:
    st.header("Input settings")
    source_type = st.selectbox("Input source", ["usb_camera", "video_file", "rtsp"], index=0)
    camera_index = st.number_input("Camera index", min_value=0, max_value=20, value=0, step=1)
    video_path = st.text_input("Video file path", value="")
    rtsp_url = st.text_input("RTSP URL", value="")
    confidence_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
    alert_cooldown = st.slider("Alert cooldown (seconds)", 1.0, 30.0, 10.0, 1.0)
    enable_snapshot = st.checkbox("Auto snapshot on qualifying event", value=True)

    if st.button("Start"):
        chosen_source = source_type
        chosen_value = camera_index
        if chosen_source == "video_file":
            chosen_value = video_path or ""
        elif chosen_source == "rtsp":
            chosen_value = rtsp_url or ""
        camera_manager.configure(chosen_source, f"Camera {camera_index}" if chosen_source == "usb_camera" else chosen_value, chosen_value)
        try:
            camera_manager.start()
            st.session_state.running = True
            pipeline.confidence_threshold = confidence_threshold
            pipeline.alert_manager.cooldown_seconds = alert_cooldown
            pipeline.auto_snapshot_on_event = enable_snapshot
            pipeline.source_name = camera_manager.active_source
            st.success("Camera stream started.")
        except Exception as exc:  # pragma: no cover - UI feedback path
            st.error(f"Unable to start stream: {exc}")
            st.session_state.running = False

    if st.button("Stop"):
        camera_manager.stop()
        st.session_state.running = False

    if st.button("Pause"):
        camera_manager.pause()

    if st.button("Resume"):
        camera_manager.resume()

    if st.button("Reconnect"):
        try:
            camera_manager.reconnect()
            st.session_state.running = True
        except Exception as exc:  # pragma: no cover - UI feedback path
            st.error(f"Reconnect failed: {exc}")

    if st.button("Take snapshot"):
        if "last_frame" in st.session_state:
            snapshot = pipeline.snapshot_manager.save_snapshot(st.session_state.last_frame, label="manual_snapshot")
            if snapshot:
                st.success(f"Snapshot saved: {snapshot}")
            else:
                st.warning("Snapshot could not be saved.")
        else:
            st.warning("No frame available to snapshot.")

    if st.button("Start recording"):
        try:
            if "last_frame" in st.session_state and st.session_state.last_frame is not None:
                output = pipeline.start_recording(fps=20.0)
                st.session_state.recording = True
                st.success(f"Recording started: {output}")
            else:
                st.warning("A frame is required before recording begins.")
        except Exception as exc:  # pragma: no cover - UI feedback path
            st.error(f"Recording failed: {exc}")

    if st.button("Stop recording"):
        saved_path = pipeline.stop_recording()
        st.session_state.recording = False
        if saved_path:
            st.success(f"Recording ended: {saved_path}")

    if st.button("Export JSON"):
        target = PROJECT_ROOT / "exports" / "events.json"
        exported = pipeline.event_manager.export_json(target)
        st.success(f"JSON export saved to {exported}")

    if st.button("Export CSV"):
        target = PROJECT_ROOT / "exports" / "events.csv"
        exported = pipeline.event_manager.export_csv(target)
        st.success(f"CSV export saved to {exported}")

left_col, right_col = st.columns([2, 1])

with left_col:
    live_slot = st.empty()
    status_slot = st.empty()
    if st.session_state.get("running"):
        try:
            while st.session_state.get("running"):
                ok, frame = camera_manager.read_frame()
                if not ok or frame is None:
                    status_slot.warning("Stream unavailable or paused.")
                    break
                annotated, detections, events = pipeline.process_frame(frame, source_name=camera_manager.active_source)
                if annotated is None:
                    continue
                display_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                live_slot.image(display_frame, channels="RGB", use_container_width=True)
                st.session_state.last_frame = annotated
                status_slot.write(
                    f"Status: {camera_manager.status} | Detections: {len(detections)} | Source: {camera_manager.active_source}"
                )
                if events:
                    st.session_state.last_alerts = events[-5:]
                time.sleep(0.03)
        except Exception as exc:  # pragma: no cover - UI path
            status_slot.error(f"Processing error: {exc}")
            st.session_state.running = False
    else:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Stream idle", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        live_slot.image(placeholder, channels="RGB", use_container_width=True)
        status_slot.write("Status: idle")

with right_col:
    st.subheader("Detection analytics")
    metrics = pipeline.stats()
    st.metric("Total detections", metrics.get("total_detections", 0))
    st.metric("Average confidence", round(metrics.get("average_confidence", 0.0), 4))
    st.write("Counts by class")
    st.json(metrics.get("counts_by_class", {}))
    st.write("Recent events")
    st.json(metrics.get("recent_events", [])[-10:])

    st.write("Alert history")
    st.json(pipeline.alert_manager.alert_history[-10:])

    st.write("Saved recordings")
    recordings = sorted(PROJECT_ROOT.joinpath("recordings").glob("*.mp4"))
    st.json([str(path.relative_to(PROJECT_ROOT)) for path in recordings[-5:]])

    st.write("Saved snapshots")
    snapshots = sorted(PROJECT_ROOT.joinpath("snapshots").glob("*.jpg"))
    st.json([str(path.relative_to(PROJECT_ROOT)) for path in snapshots[-5:]])

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.camera_manager import CameraManager
from src.detector import Detector
from src.pipeline import VisionPipeline
from src.uploads import VIDEO_TYPES, persist_uploaded_video

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"


def ensure_runtime() -> tuple[VisionPipeline, CameraManager]:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = VisionPipeline(
            project_root=PROJECT_ROOT,
            confidence_threshold=0.25,
            alert_cooldown_seconds=10.0,
            auto_snapshot_on_event=True,
            model_name="yolov8n.pt",
            device="cpu",
        )
    if "camera_manager" not in st.session_state:
        st.session_state.camera_manager = CameraManager()
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("source_type", "usb_camera")
    st.session_state.setdefault("source_name", "Camera 0")
    st.session_state.setdefault("latest_detections", [])
    st.session_state.setdefault("latest_frame", None)
    st.session_state.setdefault("inference_ms", 0.0)
    st.session_state.setdefault("measured_fps", 0.0)
    st.session_state.setdefault("last_frame_at", None)
    return st.session_state.pipeline, st.session_state.camera_manager


def save_uploaded_video(uploaded_file) -> Path:
    return persist_uploaded_video(uploaded_file, UPLOAD_DIR)


def start_selected_source(source_type: str, camera_index: int, uploaded_video, rtsp_url: str) -> None:
    pipeline, camera = ensure_runtime()
    try:
        if source_type == "usb_camera":
            value = int(camera_index)
            source_name = f"Camera {value}"
        elif source_type == "video_file":
            if uploaded_video is None:
                st.error("Upload a video file before starting video analysis.")
                return
            value = str(save_uploaded_video(uploaded_video))
            source_name = uploaded_video.name
        else:
            value = rtsp_url.strip()
            if not value:
                st.error("Enter an RTSP URL before starting the stream.")
                return
            source_name = value

        camera.configure(source_type, source_name, value)
        camera.start()
        pipeline.confidence_threshold = st.session_state.confidence_threshold
        if pipeline.detector.model_name != st.session_state.model_choice:
            pipeline.detector = Detector(
                model_name=st.session_state.model_choice,
                confidence_threshold=st.session_state.confidence_threshold,
                device="cpu",
            )
        pipeline.alert_manager.cooldown_seconds = st.session_state.alert_cooldown
        pipeline.auto_snapshot_on_event = st.session_state.auto_snapshot_on_event
        pipeline.source_name = source_name
        st.session_state.source_type = source_type
        st.session_state.source_name = source_name
        st.session_state.running = True
        st.session_state.latest_detections = []
        st.session_state.latest_frame = None
        st.session_state.last_frame_at = None
        st.session_state.status_message = f"Connected to {source_name}"
    except (OSError, RuntimeError, ValueError) as exc:
        camera.stop()
        st.session_state.running = False
        st.session_state.status_message = f"Could not start input: {exc}"


st.set_page_config(page_title="EnviroGuard Vision AI", page_icon="🌿", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
      [data-testid="stMetric"] {background: #10251f; border: 1px solid #214638; padding: 14px 18px; border-radius: 12px;}
      div[data-testid="stStatusWidget"] {visibility: hidden;}
      .eg-caption {color: #9cb5ab; margin-top: -0.7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

pipeline, camera_manager = ensure_runtime()

st.title("🌿 EnviroGuard Vision AI")
st.markdown('<p class="eg-caption">Live object detection, event alerts, and saved video evidence — on your computer.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Monitor setup")
    input_choice = st.radio(
        "Video source",
        ["USB camera", "Upload a video", "RTSP stream"],
        key="input_choice",
    )
    source_type = {
        "USB camera": "usb_camera",
        "Upload a video": "video_file",
        "RTSP stream": "rtsp",
    }[input_choice]

    camera_index = 0
    uploaded_video = None
    rtsp_url = ""
    if source_type == "usb_camera":
        camera_index = st.number_input("Camera device index", min_value=0, max_value=20, value=0, step=1)
        st.caption("Use 0 for the default webcam. Camera access depends on OS permissions.")
    elif source_type == "video_file":
        uploaded_video = st.file_uploader(
            "Choose a sample video",
            type=VIDEO_TYPES,
            help="Upload an MP4, AVI, MOV, MKV, or WebM clip from your computer.",
        )
        if uploaded_video is not None:
            st.success(f"Ready: {uploaded_video.name} ({uploaded_video.size / (1024 * 1024):.1f} MB)")
    else:
        rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://user:password@host:554/stream")

    st.divider()
    st.subheader("Detection settings")
    st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05, key="confidence_threshold")
    st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt"], key="model_choice")
    st.slider("Alert cooldown (seconds)", 1.0, 60.0, 10.0, 1.0, key="alert_cooldown")
    st.checkbox("Save snapshot for each new alert", value=True, key="auto_snapshot_on_event")
    st.caption("Inference runs on CPU. PipeGen, TensorRT, and DeepStream are not configured in this prototype.")

    if st.button("▶ Start monitoring", type="primary", use_container_width=True):
        start_selected_source(source_type, camera_index, uploaded_video, rtsp_url)
    col_pause, col_stop = st.columns(2)
    with col_pause:
        if st.button("⏸ Pause", use_container_width=True, disabled=not st.session_state.running):
            camera_manager.pause()
            st.session_state.status_message = "Monitoring paused"
    with col_stop:
        if st.button("■ Stop", use_container_width=True, disabled=not st.session_state.running):
            camera_manager.stop()
            st.session_state.running = False
            st.session_state.status_message = "Monitoring stopped"
    if st.button("↻ Reconnect", use_container_width=True, disabled=not st.session_state.running):
        try:
            camera_manager.reconnect()
            st.session_state.status_message = f"Reconnected to {camera_manager.active_source}"
        except (OSError, RuntimeError, ValueError) as exc:
            st.session_state.running = False
            st.session_state.status_message = f"Reconnect failed: {exc}"

    if camera_manager.status == "paused" and st.button("▶ Resume", use_container_width=True):
        camera_manager.resume()
        st.session_state.status_message = "Monitoring resumed"


@st.fragment(run_every=0.15)
def render_live_monitor() -> None:
    pipeline, camera = ensure_runtime()
    status = st.session_state.get("status_message", "Choose an input source and start monitoring.")
    if st.session_state.running and camera.status == "paused":
        status = "Monitoring paused"
    st.caption(f"**Source:** {st.session_state.source_name}　·　**Status:** {status}")

    if st.session_state.running and camera.status != "paused":
        ok, frame = camera.read_frame()
        if not ok or frame is None:
            camera.stop()
            st.session_state.running = False
            message = "Video finished." if st.session_state.source_type == "video_file" else "Input disconnected or no frame received."
            st.session_state.status_message = message
            st.warning(message)
            st.rerun(scope="app")
        else:
            started = time.perf_counter()
            annotated, detections, events = pipeline.process_frame(
                frame,
                source_name=camera.active_source,
                source_type=st.session_state.source_type,
            )
            st.session_state.inference_ms = (time.perf_counter() - started) * 1000
            now = time.perf_counter()
            previous = st.session_state.last_frame_at
            st.session_state.measured_fps = 1 / (now - previous) if previous else 0.0
            st.session_state.last_frame_at = now
            st.session_state.latest_frame = annotated
            st.session_state.latest_detections = detections
            if events:
                for event in events:
                    st.toast(
                        f"{event['class_name']} detected ({event['confidence']:.0%})",
                        icon="🔔",
                    )

    last_frame = st.session_state.latest_frame
    if last_frame is None:
        preview = np.zeros((450, 800, 3), dtype=np.uint8)
        cv2.putText(preview, "Start a camera or upload a video to begin", (80, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 220, 210), 2)
        st.image(preview, channels="BGR", use_container_width=True)
    else:
        st.image(last_frame, channels="BGR", use_container_width=True)

    metrics = pipeline.stats()
    values = st.columns(4)
    values[0].metric("Processing FPS", f"{st.session_state.measured_fps:.1f}")
    values[1].metric("Frame latency", f"{st.session_state.inference_ms:.0f} ms")
    values[2].metric("Objects in frame", len(st.session_state.latest_detections))
    values[3].metric("Session detections", metrics["total_detections"])
    if pipeline.detector.error_message:
        st.warning(f"Detector status: {pipeline.detector.error_message}")


def stop_recording_message() -> str:
    result = pipeline.stop_recording()
    st.session_state.recording = False
    if result:
        return f"Recording saved: {result}"
    return "No recording was active."


monitor_col, evidence_col = st.columns([1.8, 1])
with monitor_col:
    st.subheader("Live monitoring")
    render_live_monitor()
with evidence_col:
    st.subheader("Capture & recording")
    frame = st.session_state.latest_frame
    if st.button("📸 Save snapshot", use_container_width=True, disabled=frame is None):
        path = pipeline.snapshot_manager.save_snapshot(frame, label="manual_snapshot")
        if path:
            st.success(f"Snapshot saved: {path.relative_to(PROJECT_ROOT)}")
        else:
            st.error("Snapshot could not be written.")

    if not st.session_state.get("recording", False):
        if st.button("⏺ Start recording", use_container_width=True, disabled=frame is None):
            result = pipeline.start_recording(frame.shape, fps=20.0)
            if result:
                st.session_state.recording = True
                st.success(f"Recording overlay video to {Path(result).name}")
            else:
                st.error("Video writer failed to initialize. Check codec support and output permissions.")
    elif st.button("■ Stop recording", use_container_width=True):
        st.info(stop_recording_message())

    st.caption("Recordings contain the processed frames with detection overlays.")
    st.divider()
    st.subheader("Exports")
    events = pipeline.event_manager.list_events()
    json_bytes = ("\n".join(json.dumps(event, ensure_ascii=False) for event in events)).encode("utf-8")
    csv_buffer = io.StringIO()
    csv_fields = [
        "event_id", "timestamp", "source_type", "source_name", "class_name",
        "confidence", "x1", "y1", "x2", "y2", "snapshot_path", "alert_triggered",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=csv_fields)
    writer.writeheader()
    for event in events:
        row = {key: event.get(key, "") for key in writer.fieldnames}
        row.update(event.get("bounding_box", {}))
        writer.writerow(row)
    download_col1, download_col2 = st.columns(2)
    download_col1.download_button("Download JSONL", json_bytes, "enviroguard-events.jsonl", "application/x-ndjson", use_container_width=True)
    download_col2.download_button("Download CSV", csv_buffer.getvalue(), "enviroguard-events.csv", "text/csv", use_container_width=True)
    st.caption(f"{len(events)} persisted event records")

st.divider()
analytics_col, alerts_col, saved_col = st.columns([1, 1, 1.2])
with analytics_col:
    st.subheader("Detection analytics")
    metrics = pipeline.stats()
    class_counts = metrics.get("counts_by_class", {})
    if class_counts:
        st.bar_chart(class_counts)
    else:
        st.caption("Class totals appear after detections are processed.")
    st.caption(f"Average detection confidence: {metrics.get('average_confidence', 0):.0%}")
with alerts_col:
    st.subheader("Recent alerts")
    alerts = pipeline.alert_manager.alert_history[-8:]
    if alerts:
        for alert in reversed(alerts):
            st.write(f"🔔 **{alert['class_name']}** · {alert['confidence']:.0%} · {alert['timestamp']}")
    else:
        st.caption("No alerts yet.")
with saved_col:
    st.subheader("Saved evidence")
    recordings = sorted((PROJECT_ROOT / "recordings").glob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    snapshots = sorted((PROJECT_ROOT / "snapshots").glob("*.jpg"), key=lambda path: path.stat().st_mtime, reverse=True)
    st.write(f"**Recordings:** {len(recordings)}")
    if recordings:
        latest_recording = recordings[0]
        st.video(str(latest_recording))
        with latest_recording.open("rb") as file_handle:
            st.download_button("Download latest recording", file_handle, latest_recording.name, "video/mp4")
    st.write(f"**Snapshots:** {len(snapshots)}")
    for snapshot in snapshots[:4]:
        st.image(str(snapshot), caption=snapshot.name, width=150)

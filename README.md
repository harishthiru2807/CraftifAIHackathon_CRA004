# EnviroGuard Vision AI

A real-time computer vision prototype for local edge monitoring, built for ordinary Windows laptops with a USB webcam or built-in camera. This project demonstrates how a lightweight AI pipeline can detect common objects, trigger alerts, save evidence, and present a polished monitoring dashboard without requiring custom hardware.

## Project summary

EnviroGuard Vision AI is designed for the CraftifAI PipeGen hackathon track as a software-only prototype. It captures live video input, runs YOLO-based object detection, overlays annotations, records video, stores evidence snapshots, and logs structured metadata for later analysis.

The system is intentionally built to be practical on a normal PC: CPU inference is supported as the default, GPU acceleration is optional, and the project is structured so official NVIDIA or PipeGen workflows can be added later without pretending they are already working.

## Why this matters

Many operational environments need immediate awareness of activity from a video stream: a workshop floor, a warehouse entrance, a campus corridor, or a temporary monitoring site. In these scenarios, a simple, reliable monitoring dashboard can provide value even before a full industrial deployment exists.

This prototype focuses on the core functions that are useful and measurable in a real environment:

- live visual monitoring
- object detection with confidence thresholds
- alerting with cooldown rules to reduce duplicates
- timestamped event logging
- saved recordings and evidence snapshots
- exportable metadata for reporting and follow-up

## What the application does

The app allows users to:

- choose a USB camera, a local video file, or an RTSP stream
- start, pause, resume, stop, and reconnect the feed
- run object detection on incoming frames in real time
- view FPS, inference latency, source information, and detection counts
- save detection events with bounding boxes and confidence scores
- trigger alert notifications based on configurable rules
- record processed video and save snapshot evidence
- export event metadata in JSON and CSV formats

## Technology stack

- Python 3.10+
- OpenCV for webcam access, frame processing, and recording
- Ultralytics YOLO with a small pretrained model for baseline detection
- Streamlit for the dashboard interface
- NumPy for frame handling and image processing
- JSON / CSV export for events and analytics
- Pytest for automated validation

The project does not claim that PipeGen, TensorRT, or DeepStream is fully configured unless it has been installed and validated in the environment. A dedicated integration scaffold is included in the repository, but it remains clearly separated from the working runtime.

## System architecture

The project is organized into a small modular pipeline:

- `app.py` manages the Streamlit dashboard and user controls
- `src/camera_manager.py` handles source selection and stream lifecycle
- `src/video_source.py` opens and reads USB, file, or RTSP inputs
- `src/detector.py` runs YOLO inference and annotates frames
- `src/alert_manager.py` filters repeated alerts with cooldown logic
- `src/event_manager.py` stores event metadata and exports
- `src/recorder.py` writes processed video files to `recordings/`
- `src/snapshot_manager.py` stores evidence images in `snapshots/`
- `src/metrics.py` tracks detections and analytics summaries
- `src/metadata_exporter.py` validates and exports JSON / CSV data
- `src/pipeline.py` ties the runtime components together

## Input sources supported

This prototype supports three input categories:

1. USB webcam / built-in camera
2. Local video file
3. RTSP URL

The current implementation handles failures gracefully by surfacing messages for unavailable cameras, invalid files, bad RTSP URLs, and missing model issues without crashing the whole app.

## Real-time detection workflow

The live processing loop works as follows:

1. Open the selected input source
2. Read frames from the camera or file
3. Pass each frame through the YOLO detector
4. Filter detections by confidence threshold
5. Draw bounding boxes and labels on the image
6. Evaluate alert rules and cooldown timers
7. Save a snapshot when configured for a qualifying event
8. Log a JSON event with timestamp, source, class, confidence, and bounding box
9. Update the monitoring dashboard metrics
10. Optionally record the processed video stream

## Alerting model

Alerts are generated based on configurable rules:

- a selected object class is detected
- confidence exceeds the configured threshold
- the same class is not repeatedly firing inside the cooldown window

This prevents noisy duplicate alerts while still generating useful evidence for real detections.

## Evidence and exports

Every cooldown-qualified alert creates a structured event. The JSON record includes fields such as:

```json
{
  "event_id": "unique-event-id",
  "timestamp": "2026-10-16T12:00:00+00:00",
  "source_type": "usb_camera",
  "source_name": "Camera 0",
  "class_name": "person",
  "confidence": 0.92,
  "bounding_box": {
    "x1": 100,
    "y1": 80,
    "x2": 300,
    "y2": 450
  },
  "snapshot_path": "snapshots/example.jpg",
  "alert_triggered": true
}
```

Event data is stored as real JSONL records and can be exported to:

- JSON
- JSONL
- CSV

The project prevents accidental data loss by creating directories if needed and validating event structure before persistence.

## Dashboard features

The dashboard includes the following sections:

### Live monitoring

- live camera stream with bounding boxes
- start / stop / pause / resume / reconnect controls
- source selection and active source display
- FPS, latency, and detection count indicators
- current status panel

### Detection analytics

- total detections
- class breakdowns
- recent event activity
- confidence summaries
- session statistics

### Recordings and evidence

- saved video files in `recordings/`
- saved JPEG evidence in `snapshots/`
- timestamps and metadata association
- export buttons for JSON and CSV history

## Project directory structure

```text
EnviroGuard-Vision-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   └── settings.yaml
├── src/
│   ├── camera_manager.py
│   ├── video_source.py
│   ├── detector.py
│   ├── pipeline.py
│   ├── recorder.py
│   ├── snapshot_manager.py
│   ├── event_manager.py
│   ├── metadata_exporter.py
│   ├── alert_manager.py
│   └── metrics.py
├── tests/
│   ├── test_events.py
│   ├── test_metadata.py
│   ├── test_alerts.py
│   └── test_recording.py
├── integration/
│   ├── README.md
│   └── pipegen_integration.py
├── models/
│   └── README.md
├── recordings/
│   └── .gitkeep
├── snapshots/
│   └── .gitkeep
├── exports/
│   └── .gitkeep
└── README.md
```

## Setup and running

### 1. Clone or download the project

```bash
git clone <repository-url>
cd EnviroGuard-Vision-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

## Deploying on Streamlit Community Cloud

1. Confirm the GitHub repository is public and the latest commit is on `main`.
2. In Streamlit Community Cloud, create an app for `harishthiru2807/Hack-with-Hyderabad-Harish`.
3. Select branch `main` and main file path `app.py` (it is at the repository root).
4. Deploy and check the app's build/runtime logs if startup fails.

The dashboard loads without initializing the YOLO model. Ultralytics and its weights are loaded only when monitoring starts, which lets the page and sample-video upload render before model initialization. The first inference can take longer while model weights are fetched. The cloud host processes uploaded video; it cannot directly access a webcam attached to your personal computer. Use **Upload a video** for cloud demos. Local webcam monitoring is for running the app on the computer connected to the camera.

### 5. Use the app

In the sidebar, select **Upload a video** and choose a sample clip (`.mp4`, `.avi`, `.mov`, `.mkv`, or `.webm`). Then select **Start monitoring** to process it. Uploaded files are copied into the local `uploads/` folder using a unique, sanitized filename; video files are ignored by Git.

You can also select:

- **USB camera** and its device index, or
- **RTSP stream** and a reachable stream URL

Then configure:

- confidence threshold
- YOLO model
- alert cooldown
- whether to auto-save snapshots for events

Use **Pause**, **Resume**, **Stop**, and **Reconnect** for a live source. A local sample video stops automatically when it reaches the end. The dashboard displays the current annotated frame, per-frame latency, processing FPS, detection counts, recent alerts, saved evidence, and event downloads.

## Testing

The repository includes a small automated test suite to validate the core logic:

```bash
pip install -r requirements-dev.txt
pytest tests -q
```

Validated behaviors include:

- event generation and persistence
- metadata export integrity
- alert cooldown logic
- recording file creation

## PipeGen and GPU integration note

This repository includes a dedicated integration area for official CraftifAI PipeGen workflows and future NVIDIA execution paths. It does not claim that PipeGen, TensorRT, or DeepStream is already operating in this environment.

Instead, the code is organized so those integrations can be added and tested properly when the official tooling is installed and configured. This keeps the prototype honest and avoids overstating what was actually executed.

## Design constraints and honest limitations

- CPU inference is the default and works on standard Windows hardware
- GPU acceleration is optional and not required for the prototype to operate
- detection is limited by the pretrained model and the video feed quality
- RTSP streams require valid network access and a reachable endpoint
- recording is only as reliable as the local disk and codec support on the machine
- the project is a software prototype and not a production-grade industrial safety system

## Future improvements

Possible next steps for a real-world version include:

- multi-camera tiled display with source health checks
- persistent user configuration and alert rules
- dashboard search, filtering, and historical trend charts
- model selection for different object categories
- better packaging for deployment in a local lab or edge environment
- official PipeGen and NVIDIA integration when toolchains are available

## Conclusion

EnviroGuard Vision AI is a working, modular, and honest software prototype for live vision monitoring. It demonstrates real-time object detection, alert logic, evidence capture, and reporting in a way that is easy to run on a normal Windows laptop while remaining extensible for future optional GPU and PipeGen integrations.

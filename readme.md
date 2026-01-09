# Valmet Hackathon — Field Device Detector

Hackathon prototype for the "3D Industrial Object Detection" challenge.

We detect industrial field devices in a 2D video stream and (as a next step) map them into a 3D digital twin of the plant. The 3D scene is represented as a Gaussian splat point cloud (`.ply`).

## What's included

- **2D detection (baseline):** TensorFlow.js model for a small set of known device classes (meant to be swapped or upgraded later).
- **Frame extraction:** Python utilities to sample sharp frames from video for inspection + annotation.
- **3D viewer:** `viser` viewer to inspect the Gaussian splat `.ply`.
- **Web demo:** Plays the video and overlays bounding boxes using a precomputed detections JSON.

## Workflow

1. Extract frames from a source video.
2. Run inference on frames → write detections to JSON.
3. Open the demo page → load video + JSON → see overlays.

## Setup

### Prerequisites

- Python 3.8+
- Node.js + npm

### Install

```bash
pip install -r requirements.txt
npm install
```

## Run

### 1) Sample frames

```bash
python scripts/frame_sampler.py \
  --input-video assets/videos/Pilot_plant.mp4 \
  --frame-interval 12 \
  --frames-dir outputs/frames/highres \
  --manifest outputs/frame_manifest.json
```

### 2) Run detection

Uses per-class thresholds from `config/thresholds.json`.

```bash
python scripts/tfjs_inference.py \
  --manifest outputs/frame_manifest.json \
  --frames-dir outputs/frames/highres \
  --output outputs/detections/pilot_plant.json \
  --thresholds config/thresholds.json
```

### 3) View the overlay

```bash
python3 -m http.server 8080
```

Open http://localhost:8080/web/

The demo loads:
- **Detections:** `outputs/detections/pilot_plant.json`
- **Video:** `assets/videos/Pilot_plant.mp4` (or whatever you configured)

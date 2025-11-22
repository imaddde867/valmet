# Valmet Hackathon: Field Device Detector

This repository contains a solution for the "3D Industrial Object Detection" challenge. The goal is to create a high-impact demo that detects industrial field devices in 2D video streams and localizes them within a 3D digital twin of the plant, represented by a Gaussian splatting (`.ply`) file.

The project aims to deliver a polished demo and pitch, showcasing a clear business case for an "Ask Your Plant" assistant and inventory tool as part of Valmet's Industrial Metaverse.

## Core Functionality

*   **2D Object Detection**: A baseline TensorFlow.js model detects known device classes. The system is designed to be extended with more advanced open-vocabulary models.
*   **Frame Processing**: Python scripts are used to sample high-quality frames from source videos for analysis and annotation.
*   **3D Visualization**: A `viser`-based viewer allows for inspection of the Gaussian splat point cloud data.
*   **Web-Based Demo**: A simple web interface overlays the 2D bounding box detections on a video player, demonstrating the model's output in real-time.

## Project Workflow

The end-to-end workflow is as follows:

1.  **Setup**: Install Python and Node.js dependencies.
2.  **Frame Sampling**: A source video (e.g., `Pilot_plant.mp4`) is processed to extract individual frames at a specified interval.
3.  **Inference**: The object detection model is run on the extracted frames to generate a JSON file containing all detections (class, confidence, bounding box).
4.  **Visualization**: The results can be viewed in the web demo, which loads the video and the corresponding detection data.

## Setup

### Prerequisites

*   Python 3.8+
*   Node.js and npm

### Installation

1.  **Python Dependencies**: Install the required Python packages, including `torch`, `viser`, and `gsplat`.

    ```bash
    pip install -r requirements.txt
    ```

2.  **Node.js Dependencies**: Install the TensorFlow.js package for the inference script.

    ```bash
    npm install
    ```

## Usage

### 1. Frame Sampling

Extract frames from a source video. The output is a directory of images and a `manifest.json` file.

```bash
python scripts/frame_sampler.py \
  --input-video assets/videos/Pilot_plant.mp4 \
  --frame-interval 12 \
  --frames-dir outputs/frames/highres \
  --manifest outputs/frame_manifest.json
```

### 2. Object Detection

Run the TF.js baseline model on the sampled frames. This script uses per-class thresholds defined in `config/thresholds.json`.

```bash
python scripts/tfjs_inference.py \
  --manifest outputs/frame_manifest.json \
  --frames-dir outputs/frames/highres \
  --output outputs/detections/pilot_plant.json \
  --thresholds config/thresholds.json
```

### 3. Visualization

Launch a local web server to view the results.

```bash
python3 -m http.server 8080
```

Navigate to **http://localhost:8080/web/** in your browser. The interface will load the video and overlay the detections from `outputs/detections/pilot_plant.json`.

## Next Steps

To improve model accuracy, the next phase involves manual annotation of the high-quality frames (`outputs/frames/highres`) using a tool like LabelMe, Roboflow, or CVAT. This annotated dataset will be used to fine-tune the detector and deliver a more robust and "elite" solution for the challenge.
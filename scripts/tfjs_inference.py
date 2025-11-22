#!/usr/bin/env python3
"""
Wrapper script to run TFJS inference on sampled frames via tfjs-node.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_model = repo_root / (
        "web/models/tfjs_baseline/tensorflow_automl_model/"
        "model-197536060022980608_tf-js_2023-05-04T05_50_49.038047Z_model.json"
    )
    default_manifest = repo_root / "outputs/frame_manifest.json"
    default_frames = repo_root / "outputs/frames/model"
    default_output = repo_root / "outputs/detections/pilot_plant.json"
    default_labels = repo_root / (
        "web/models/tfjs_baseline/tensorflow_automl_model/"
        "model-197536060022980608_tf-js_2023-05-04T05_50_49.038047Z_dict.txt"
    )

    parser = argparse.ArgumentParser(
        description="Run TFJS baseline detector over sampled frames."
    )
    parser.add_argument("--model", type=Path, default=default_model, help="Path to TFJS model.json.")
    parser.add_argument("--manifest", type=Path, default=default_manifest, help="Frame manifest JSON.")
    parser.add_argument("--frames-dir", type=Path, default=default_frames, help="Directory with sampled frames.")
    parser.add_argument("--output", type=Path, default=default_output, help="Output detections JSON path.")
    parser.add_argument("--labels", type=Path, default=default_labels, help="Label dictionary text file.")
    parser.add_argument("--video-name", default="Pilot_plant.mp4", help="Video name recorded in output schema.")
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional JSON string or path to JSON file with per-class thresholds.",
    )
    parser.add_argument(
        "--max-dets",
        type=int,
        default=200,
        help="Maximum detections to retain per frame after NMS (default: 200).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS (default: 0.5).",
    )
    parser.add_argument(
        "--node-script",
        type=Path,
        default=Path(__file__).with_name("tfjs_inference_node.js"),
        help="Internal use: path to the Node.js inference script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_script = args.node_script
    if not node_script.exists():
        raise FileNotFoundError(f"Missing Node.js inference helper at {node_script}")

    cmd = [
        "node",
        str(node_script),
        "--model",
        str(args.model),
        "--manifest",
        str(args.manifest),
        "--frames-dir",
        str(args.frames_dir),
        "--output",
        str(args.output),
        "--labels",
        str(args.labels),
        "--video-name",
        args.video_name,
    ]
    if args.thresholds:
        cmd.extend(["--thresholds", str(args.thresholds)])
    cmd.extend(["--max-dets", str(args.max_dets)])
    cmd.extend(["--iou-threshold", str(args.iou_threshold)])

    print("Running:", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Inference pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)

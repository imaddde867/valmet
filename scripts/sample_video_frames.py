#!/usr/bin/env python3
"""Sample frames from a video, resize them, and emit a manifest for inference."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample frames from a video for downstream TFJS inference."
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=Path("assets/videos/Pilot_plant.mp4"),
        help="Path to the source video (default: assets/videos/Pilot_plant.mp4).",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(224, 224),
        help="Resize width and height expected by the TFJS model (default: 224 224).",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="Sample one frame every N frames (default: 30).",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("outputs/frames"),
        help="Directory where sampled frames will be stored.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/frame_manifest.json"),
        help="Manifest JSON output path.",
    )
    return parser.parse_args()


def ensure_dirs(frames_dir: Path, manifest_path: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)


def sample_frames(
    video_path: Path,
    resize: Tuple[int, int],
    frame_interval: int,
    frames_dir: Path,
) -> List[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    sampled: List[dict] = []
    frame_idx = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_interval <= 1 or frame_idx % frame_interval == 0:
            resized = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            filename = f"frame_{frame_idx:06d}.png"
            frame_path = frames_dir / filename
            cv2.imwrite(str(frame_path), resized)

            timestamp = frame_idx / fps if fps > 0 else None
            sampled.append(
                {
                    "frame_index": frame_idx,
                    "timestamp_sec": round(timestamp, 3) if timestamp is not None else None,
                    "filename": filename,
                }
            )
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Processed {frame_idx} frames, saved {saved_count} samples.")
    return sampled


def write_manifest(entries: List[dict], manifest_path: Path) -> None:
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")


def main() -> None:
    args = parse_args()
    ensure_dirs(args.frames_dir, args.manifest)
    entries = sample_frames(
        video_path=args.input_video,
        resize=tuple(args.resize),
        frame_interval=args.frame_interval,
        frames_dir=args.frames_dir,
    )
    write_manifest(entries, args.manifest)


if __name__ == "__main__":
    main()

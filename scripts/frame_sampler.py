#!/usr/bin/env python3
"""Sample frames from a video with configurable quality profiles."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

INTERPOLATION_MAP = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def compute_megapixels(width: int, height: int) -> float:
    return round((width * height) / 1_000_000.0, 3)


def resize_for_model(
    frame, target_size: Tuple[int, int], keep_aspect: bool, interpolation: int
) -> Tuple:
    original_height, original_width = frame.shape[:2]
    target_width, target_height = target_size
    if keep_aspect:
        scale = min(target_width / original_width, target_height / original_height)
        new_width = max(1, int(round(original_width * scale)))
        new_height = max(1, int(round(original_height * scale)))
    else:
        new_width, new_height = target_width, target_height
    resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    return resized, new_width, new_height, scale_x, scale_y


def prepare_annotation_frame(
    frame,
    interpolation: int,
    max_dimension: Optional[int],
    target_megapixels: Optional[float],
):
    original_height, original_width = frame.shape[:2]
    scale = 1.0
    if max_dimension:
        max_dim_scale = max_dimension / max(original_width, original_height)
        scale = min(scale, max_dim_scale)
    if target_megapixels:
        target_pixels = target_megapixels * 1_000_000.0
        original_pixels = original_width * original_height
        if original_pixels > target_pixels:
            megapixel_scale = math.sqrt(target_pixels / original_pixels)
            scale = min(scale, megapixel_scale)
    scale = min(scale, 1.0)
    if scale < 1.0:
        new_width = max(1, int(round(original_width * scale)))
        new_height = max(1, int(round(original_height * scale)))
        resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
    else:
        resized = frame
        new_width, new_height = original_width, original_height
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    return resized, new_width, new_height, scale_x, scale_y


def save_image(
    path: Path,
    image,
    image_format: str,
    jpeg_quality: int,
    png_compression: int,
) -> None:
    params: List[int] = []
    if image_format == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, clamp(jpeg_quality, 1, 100)]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, clamp(png_compression, 0, 9)]
    success = cv2.imwrite(str(path), image, params)
    if not success:
        raise RuntimeError(f"Failed to write frame to {path}")


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
        type=positive_int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(224, 224),
        help="Resize width and height expected by the TFJS model (default: 224 224).",
    )
    parser.add_argument(
        "--keep-aspect",
        action="store_true",
        help="Preserve the original aspect ratio when resizing model frames.",
    )
    parser.add_argument(
        "--interpolation",
        choices=sorted(INTERPOLATION_MAP.keys()),
        default="area",
        help="Interpolation filter used during resizing (default: area).",
    )
    parser.add_argument(
        "--frame-interval",
        type=positive_int,
        default=30,
        help="Sample one frame every N frames (default: 30).",
    )
    parser.add_argument(
        "--frames-format",
        choices=["png", "jpg"],
        default="png",
        help="File format for model/inference frames (default: png).",
    )
    parser.add_argument(
        "--frames-jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality (0-100) for model frames when --frames-format=jpg.",
    )
    parser.add_argument(
        "--frames-png-compression",
        type=int,
        default=3,
        help="PNG compression level (0-9) for model frames when --frames-format=png.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("outputs/frames/model"),
        help="Directory where sampled frames will be stored.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/frame_manifest.json"),
        help="Manifest JSON output path.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Optional directory to store high-quality annotation frames.",
    )
    parser.add_argument(
        "--annotation-max-dim",
        type=positive_int,
        default=None,
        help="Keep the longest side of annotation frames under this size (pixels).",
    )
    parser.add_argument(
        "--annotation-target-megapixels",
        type=positive_float,
        default=None,
        help="Approximate megapixel budget for annotation frames (preserves aspect).",
    )
    parser.add_argument(
        "--annotation-format",
        choices=["png", "jpg"],
        default="jpg",
        help="File format for annotation frames (default: jpg).",
    )
    parser.add_argument(
        "--annotation-jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality (0-100) for annotation frames when using jpg.",
    )
    parser.add_argument(
        "--annotation-png-compression",
        type=int,
        default=2,
        help="PNG compression level (0-9) for annotation frames when using png.",
    )
    args = parser.parse_args()
    if args.annotation_dir is None and (
        args.annotation_max_dim or args.annotation_target_megapixels
    ):
        parser.error(
            "--annotation-dir must be provided when annotation-specific options are set."
        )
    return args


def ensure_dirs(
    frames_dir: Path, manifest_path: Path, annotation_dir: Optional[Path]
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if annotation_dir:
        annotation_dir.mkdir(parents=True, exist_ok=True)


def sample_frames(
    video_path: Path,
    resize: Tuple[int, int],
    frame_interval: int,
    frames_dir: Path,
    interpolation: int,
    keep_aspect: bool,
    frame_encoding: Dict[str, object],
    annotation_options: Dict[str, object],
) -> List[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video at {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    sampled: List[dict] = []
    frame_idx = 0
    saved_count = 0
    annotation_saved = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_interval <= 1 or frame_idx % frame_interval == 0:
            (
                model_frame,
                model_width,
                model_height,
                model_scale_x,
                model_scale_y,
            ) = resize_for_model(frame, resize, keep_aspect, interpolation)
            model_filename = f"frame_{frame_idx:06d}.{frame_encoding['format']}"
            frame_path = frames_dir / model_filename
            save_image(
                frame_path,
                model_frame,
                frame_encoding["format"],
                frame_encoding["jpeg_quality"],
                frame_encoding["png_compression"],
            )

            annotation_variant = None
            annotation_dir = annotation_options.get("dir")
            if annotation_dir:
                (
                    annotation_frame,
                    annotation_width,
                    annotation_height,
                    annotation_scale_x,
                    annotation_scale_y,
                ) = prepare_annotation_frame(
                    frame,
                    interpolation,
                    annotation_options.get("max_dim"),
                    annotation_options.get("target_megapixels"),
                )
                annotation_filename = (
                    f"frame_{frame_idx:06d}_hq.{annotation_options['format']}"
                )
                annotation_path = annotation_dir / annotation_filename
                save_image(
                    annotation_path,
                    annotation_frame,
                    annotation_options["format"],
                    annotation_options["jpeg_quality"],
                    annotation_options["png_compression"],
                )
                annotation_saved += 1
                annotation_variant = {
                    "filename": annotation_filename,
                    "frames_dir": str(annotation_dir),
                    "width": annotation_width,
                    "height": annotation_height,
                    "scale": {"x": annotation_scale_x, "y": annotation_scale_y},
                    "megapixels": compute_megapixels(annotation_width, annotation_height),
                    "max_dimension": annotation_options.get("max_dim"),
                    "target_megapixels": annotation_options.get("target_megapixels"),
                }

            timestamp = frame_idx / fps if fps > 0 else None
            original_height, original_width = frame.shape[:2]
            entry: Dict[str, object] = {
                "frame_index": frame_idx,
                "timestamp_sec": round(timestamp, 3) if timestamp is not None else None,
                "filename": model_filename,
                "original_size": {"width": original_width, "height": original_height},
                "variants": {
                    "model": {
                        "filename": model_filename,
                        "frames_dir": str(frames_dir),
                        "width": model_width,
                        "height": model_height,
                        "scale": {"x": model_scale_x, "y": model_scale_y},
                        "megapixels": compute_megapixels(model_width, model_height),
                        "keep_aspect": keep_aspect,
                    }
                },
            }
            if annotation_variant:
                entry["variants"]["annotation"] = annotation_variant
                entry["annotation_filename"] = annotation_variant["filename"]
            sampled.append(entry)
            saved_count += 1

        frame_idx += 1

    cap.release()
    if annotation_options["dir"]:
        print(
            f"Processed {frame_idx} frames, saved {saved_count} model samples and "
            f"{annotation_saved} annotation frames."
        )
    else:
        print(f"Processed {frame_idx} frames, saved {saved_count} samples.")
    return sampled


def write_manifest(entries: List[dict], manifest_path: Path) -> None:
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")


def main() -> None:
    args = parse_args()
    interpolation_flag = INTERPOLATION_MAP[args.interpolation]
    frame_encoding = {
        "format": args.frames_format,
        "jpeg_quality": clamp(args.frames_jpeg_quality, 1, 100),
        "png_compression": clamp(args.frames_png_compression, 0, 9),
    }
    annotation_options: Dict[str, object] = {
        "dir": args.annotation_dir,
        "max_dim": args.annotation_max_dim,
        "target_megapixels": args.annotation_target_megapixels,
        "format": args.annotation_format,
        "jpeg_quality": clamp(args.annotation_jpeg_quality, 1, 100),
        "png_compression": clamp(args.annotation_png_compression, 0, 9),
    }
    ensure_dirs(args.frames_dir, args.manifest, args.annotation_dir)
    entries = sample_frames(
        video_path=args.input_video,
        resize=tuple(args.resize),
        frame_interval=args.frame_interval,
        frames_dir=args.frames_dir,
        interpolation=interpolation_flag,
        keep_aspect=args.keep_aspect,
        frame_encoding=frame_encoding,
        annotation_options=annotation_options,
    )
    write_manifest(entries, args.manifest)


if __name__ == "__main__":
    main()

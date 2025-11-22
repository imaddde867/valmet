#!/usr/bin/env python3
"""Convert LabelMe bounding-box annotations into the repo detection schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to annotations/{name}.json"
    )
    parser.add_argument(
        "--labelme-dir",
        type=Path,
        required=True,
        help="Directory containing LabelMe JSON files (one per frame).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Frame manifest JSON produced by sample_video_frames.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output annotations JSON path (e.g., annotations/pilot_eval.json)",
    )
    parser.add_argument(
        "--class-map",
        type=str,
        default=None,
        help="Optional JSON dict mapping LabelMe labels to detector labels",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict[str, Dict[str, object]]:
    data = json.loads(path.read_text())
    return {entry["filename"]: entry for entry in data}


def load_class_map(path_str: str | None) -> Dict[str, str]:
    if not path_str:
        return {}
    path = Path(path_str)
    return json.loads(path.read_text())


def bbox_from_points(points: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return ymin, xmin, ymax, xmax  # match TFJS detector order


def to_relative_bbox(bbox, width: float, height: float):
    ymin, xmin, ymax, xmax = bbox
    return (ymin / height, xmin / width, ymax / height, xmax / width)


def convert(labelme_dir: Path, manifest: Path, output: Path, class_map: Dict[str, str]):
    manifest_map = load_manifest(manifest)
    results = []
    for label_path in sorted(labelme_dir.glob("*.json")):
        payload = json.loads(label_path.read_text())
        filename = payload.get("imagePath") or label_path.with_suffix(".png").name
        frame_info = manifest_map.get(filename)
        if not frame_info:
            raise RuntimeError(f"No manifest entry for {filename}; did you label sampled frames?")
        dets = []
        width = payload.get("imageWidth")
        height = payload.get("imageHeight")
        if not width or not height:
            raise RuntimeError(f"Missing imageWidth/Height in {label_path}")
        for shape in payload.get("shapes", []):
            label = shape.get("label")
            if not label:
                continue
            mapped_label = class_map.get(label, label)
            bbox = to_relative_bbox(bbox_from_points(shape["points"]), width, height)
            dets.append({"cls": mapped_label, "bbox": list(bbox)})
        results.append({
            "i": frame_info["frame_index"],
            "t": frame_info.get("timestamp_sec"),
            "dets": dets,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"video": str(manifest), "frames": results}
    output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote annotations to {output}")


def main():
    args = parse_args()
    convert(args.labelme_dir, args.manifest, args.output, load_class_map(args.class_map))


if __name__ == "__main__":
    main()

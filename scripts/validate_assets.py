#!/usr/bin/env python3
"""
Scan the assets directory, collect metadata for videos and 3D files,
validate references against docs/PLAN.md, and emit assets_manifest.json.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
PLAN_PATH = REPO_ROOT / "docs" / "PLAN.md"
MANIFEST_PATH = REPO_ROOT / "assets_manifest.json"


@dataclass
class VideoMetadata:
    path: str
    width: Optional[int]
    height: Optional[int]
    frame_count: Optional[int]
    fps: Optional[float]
    duration_seconds: Optional[float]


@dataclass
class PointCloudMetadata:
    path: str
    vertex_count: Optional[int]


@dataclass
class MeshMetadata:
    path: str
    geometry_count: int
    total_vertices: int
    total_faces: int


def _read_plan_text() -> str:
    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"PLAN file missing at {PLAN_PATH}")
    return PLAN_PATH.read_text(encoding="utf-8")


def _in_plan(asset_name: str, plan_text: str) -> bool:
    return asset_name in plan_text


def _video_stats(path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return VideoMetadata(path=str(path.relative_to(REPO_ROOT)),
                             width=None, height=None,
                             frame_count=None, fps=None,
                             duration_seconds=None)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps > 0 else None
    return VideoMetadata(
        path=str(path.relative_to(REPO_ROOT)),
        width=width,
        height=height,
        frame_count=frame_count,
        fps=round(fps, 3) if fps else None,
        duration_seconds=round(duration, 3) if duration else None,
    )


def _ply_vertex_count(path: Path) -> Optional[int]:
    try:
        with path.open("rb") as f:
            for raw_line in f:
                line = raw_line.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex"):
                    parts = line.split()
                    if len(parts) == 3 and parts[2].isdigit():
                        return int(parts[2])
                    try:
                        return int(parts[-1])
                    except ValueError:
                        continue
                if line == "end_header":
                    break
    except Exception:
        return None
    return None


def _mesh_stats(path: Path) -> MeshMetadata:
    total_vertices = 0
    total_faces = 0
    geometry_count = 0
    try:
        loaded = trimesh.load(path, force="scene")
        if hasattr(loaded, "geometry"):
            geometries = loaded.geometry.values()
            geometry_count = len(loaded.geometry)
        else:
            geometries = [loaded]
            geometry_count = 1
        for geom in geometries:
            verts = getattr(geom, "vertices", [])
            faces = getattr(geom, "faces", [])
            total_vertices += len(verts)
            total_faces += len(faces)
    except Exception:
        geometry_count = 0
        total_vertices = 0
        total_faces = 0
    return MeshMetadata(
        path=str(path.relative_to(REPO_ROOT)),
        geometry_count=geometry_count,
        total_vertices=total_vertices,
        total_faces=total_faces,
    )


def build_manifest() -> Tuple[Dict[str, List[dict]], List[str]]:
    manifest: Dict[str, List[dict]] = {"videos": [], "pointclouds": [], "meshes": []}
    missing_in_plan: List[str] = []
    plan_text = _read_plan_text()

    video_paths = sorted((ASSETS_DIR / "videos").glob("*.mp4"))
    ply_paths = sorted((ASSETS_DIR / "pointclouds").glob("*.ply"))
    glb_paths = sorted((ASSETS_DIR / "meshes").glob("*.glb"))

    for path in video_paths:
        meta = _video_stats(path)
        manifest["videos"].append(asdict(meta))
        if not _in_plan(path.name, plan_text):
            missing_in_plan.append(meta.path)

    for path in ply_paths:
        vertex_count = _ply_vertex_count(path)
        manifest["pointclouds"].append(
            {"path": str(path.relative_to(REPO_ROOT)), "vertex_count": vertex_count}
        )
        if not _in_plan(path.name, plan_text):
            missing_in_plan.append(str(path.relative_to(REPO_ROOT)))

    for path in glb_paths:
        meta = _mesh_stats(path)
        manifest["meshes"].append(asdict(meta))
        if not _in_plan(path.name, plan_text):
            missing_in_plan.append(meta.path)

    return manifest, missing_in_plan


def main() -> None:
    if not ASSETS_DIR.exists():
        raise FileNotFoundError(f"Assets directory not found at {ASSETS_DIR}")

    manifest, missing = build_manifest()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if missing:
        print("⚠️ Assets not referenced in docs/PLAN.md:")
        for item in missing:
            print(f"  - {item}")
    else:
        print("✅ All assets referenced in docs/PLAN.md.")

    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Asset validation failed: {exc}", file=sys.stderr)
        sys.exit(1)

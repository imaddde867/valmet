#!/usr/bin/env python3
"""Compute precision/recall and mAP for detector outputs against ground truth."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    frame: int
    cls: str
    bbox: BBox
    score: float


@dataclass
class GroundTruth:
    frame: int
    cls: str
    bbox: BBox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detections JSON against ground-truth annotations."
    )
    parser.add_argument("--detections", type=Path, required=True, help="Detections JSON path.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Ground truth JSON path.")
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("config/thresholds.json"),
        help="Optional per-class threshold JSON (default: config/thresholds.json).",
    )
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default: 0.5).")
    parser.add_argument(
        "--suggest-thresholds",
        action="store_true",
        help="Sweep thresholds per class to maximize F1 (prints suggestions).",
    )
    parser.add_argument("--thr-min", type=float, default=0.1, help="Grid search min threshold (default 0.1).")
    parser.add_argument("--thr-max", type=float, default=0.9, help="Grid search max threshold (default 0.9).")
    parser.add_argument("--thr-step", type=float, default=0.05, help="Grid search step (default 0.05).")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file at {path}")
    return json.loads(path.read_text())


def load_detections(path: Path) -> List[Detection]:
    payload = load_json(path)
    frames = payload.get("frames", [])
    detections: List[Detection] = []
    for frame in frames:
        frame_idx = frame.get("i", frame.get("frame_index"))
        if frame_idx is None:
            continue
        for det in frame.get("dets", frame.get("detections", [])):
            bbox = det.get("bbox") or det.get("box")
            if bbox is None or len(bbox) < 4:
                continue
            score = det.get("conf", det.get("score"))
            if score is None:
                continue
            detections.append(
                Detection(frame=int(frame_idx), cls=str(det.get("cls") or det.get("class")), bbox=tuple(map(float, bbox)), score=float(score))
            )
    return detections


def load_ground_truth(path: Path) -> List[GroundTruth]:
    payload = load_json(path)
    entries: List[GroundTruth] = []
    for frame in payload.get("frames", []):
        frame_idx = frame.get("i", frame.get("frame_index"))
        if frame_idx is None:
            continue
        for det in frame.get("dets", frame.get("detections", [])):
            bbox = det.get("bbox") or det.get("box")
            if bbox is None or len(bbox) < 4:
                continue
            entries.append(
                GroundTruth(frame=int(frame_idx), cls=str(det.get("cls") or det.get("class")), bbox=tuple(map(float, bbox)))
            )
    return entries


def load_thresholds(path: Path) -> Dict[str, float]:
    if path and path.exists():
        return json.loads(path.read_text())
    return {"_default": 0.15}


def compute_iou(box_a: BBox, box_b: BBox) -> float:
    aymin, axmin, aymax, axmax = box_a
    bymin, bxmin, bymax, bxmax = box_b
    inter_ymin = max(aymin, bymin)
    inter_xmin = max(axmin, bxmin)
    inter_ymax = min(aymax, bymax)
    inter_xmax = min(axmax, bxmax)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_area = inter_h * inter_w
    area_a = max(0.0, aymax - aymin) * max(0.0, axmax - axmin)
    area_b = max(0.0, bymax - bymin) * max(0.0, bxmax - bxmin)
    union = area_a + area_b - inter_area + 1e-9
    return inter_area / union


def filter_detections(detections: Iterable[Detection], thresholds: Dict[str, float]) -> List[Detection]:
    default_thr = thresholds.get("_default", 0.0)
    return [det for det in detections if det.score >= thresholds.get(det.cls, default_thr)]


def evaluate_class(detections: List[Detection], ground_truth: List[GroundTruth], iou_thr: float) -> Dict[str, float]:
    if not ground_truth:
        return {"tp": 0, "fp": len(detections), "fn": 0, "precision": 0.0, "recall": 0.0, "ap": None, "f1": 0.0}

    gt_map: Dict[int, List[Dict[str, object]]] = {}
    for gt in ground_truth:
        gt_map.setdefault(gt.frame, []).append({"bbox": gt.bbox, "matched": False})

    detections_sorted = sorted(detections, key=lambda d: d.score, reverse=True)
    tp_flags: List[int] = []
    fp_flags: List[int] = []

    for det in detections_sorted:
        matches = gt_map.get(det.frame, [])
        best_iou = 0.0
        best_idx = -1
        for idx, gt_entry in enumerate(matches):
            if gt_entry["matched"]:
                continue
            iou = compute_iou(det.bbox, gt_entry["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_thr and best_idx >= 0:
            matches[best_idx]["matched"] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    tp_cum: List[int] = []
    fp_cum: List[int] = []
    running_tp = 0
    running_fp = 0
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        running_tp += tp_flag
        running_fp += fp_flag
        tp_cum.append(running_tp)
        fp_cum.append(running_fp)

    total_gt = len(ground_truth)
    precisions: List[float] = []
    recalls: List[float] = []
    for tp_val, fp_val in zip(tp_cum, fp_cum):
        precisions.append(tp_val / (tp_val + fp_val) if (tp_val + fp_val) else 0.0)
        recalls.append(tp_val / total_gt)

    ap = voc_ap(recalls, precisions)
    precision = precisions[-1] if precisions else 0.0
    recall = recalls[-1] if recalls else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fn = total_gt - (tp_cum[-1] if tp_cum else 0)
    return {"tp": tp_cum[-1] if tp_cum else 0, "fp": fp_cum[-1] if fp_cum else len(detections_sorted), "fn": fn, "precision": precision, "recall": recall, "ap": ap, "f1": f1}


def voc_ap(recalls: List[float], precisions: List[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = 0.0
    for i in range(len(mrec) - 1):
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


def evaluate_dataset(
    detections: List[Detection],
    ground_truth: List[GroundTruth],
    thresholds: Dict[str, float],
    iou_thr: float,
) -> Dict[str, object]:
    filtered = filter_detections(detections, thresholds)
    classes = sorted({det.cls for det in filtered} | {gt.cls for gt in ground_truth})
    per_class: Dict[str, Dict[str, float]] = {}
    ap_values: List[float] = []
    tp_total = fp_total = fn_total = 0
    for cls in classes:
        cls_dets = [det for det in filtered if det.cls == cls]
        cls_gts = [gt for gt in ground_truth if gt.cls == cls]
        metrics = evaluate_class(cls_dets, cls_gts, iou_thr)
        per_class[cls] = metrics
        if metrics["ap"] is not None:
            ap_values.append(metrics["ap"])
        tp_total += metrics["tp"]
        fp_total += metrics["fp"]
        fn_total += metrics["fn"]
    overall_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    overall_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    macro_map = sum(ap_values) / len(ap_values) if ap_values else 0.0
    return {"per_class": per_class, "precision": overall_precision, "recall": overall_recall, "mAP": macro_map}


def suggest_thresholds(
    detections: List[Detection],
    ground_truth: List[GroundTruth],
    classes: List[str],
    iou_thr: float,
    thr_min: float,
    thr_max: float,
    thr_step: float,
) -> Dict[str, Dict[str, float]]:
    suggestions: Dict[str, Dict[str, float]] = {}
    thr_values: List[float] = []
    value = thr_min
    while value <= thr_max + 1e-9:
        thr_values.append(round(value, 4))
        value += thr_step
    for cls in classes:
        cls_dets = [det for det in detections if det.cls == cls]
        cls_gts = [gt for gt in ground_truth if gt.cls == cls]
        best: Optional[Dict[str, float]] = None
        for thr in thr_values:
            filtered = [det for det in cls_dets if det.score >= thr]
            metrics = evaluate_class(filtered, cls_gts, iou_thr)
            candidate = {"threshold": thr, **metrics}
            if best is None or candidate["f1"] > best["f1"]:
                best = candidate
        if best is None:
            best = {"threshold": thr_min, "precision": 0.0, "recall": 0.0, "ap": 0.0, "f1": 0.0}
        suggestions[cls] = best
    return suggestions


def print_report(result: Dict[str, object]) -> None:
    print("\nPer-class metrics (IoU >= threshold):")
    print(f"{'Class':<12}{'P':>8}{'R':>8}{'AP':>8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}")
    for cls, metrics in result["per_class"].items():
        ap = metrics["ap"] if metrics["ap"] is not None else 0.0
        print(
            f"{cls:<12}{metrics['precision']*100:8.2f}{metrics['recall']*100:8.2f}{ap*100:8.2f}{metrics['f1']*100:8.2f}{metrics['tp']:6}{metrics['fp']:6}{metrics['fn']:6}"
        )
    print(
        f"\nOverall Precision: {result['precision']*100:.2f}% | Overall Recall: {result['recall']*100:.2f}% | mAP: {result['mAP']*100:.2f}%"
    )


def main() -> None:
    args = parse_args()
    detections = load_detections(args.detections)
    ground_truth = load_ground_truth(args.ground_truth)
    thresholds = load_thresholds(args.thresholds)
    result = evaluate_dataset(detections, ground_truth, thresholds, args.iou)
    print_report(result)

    if args.suggest_thresholds:
        classes = sorted({det.cls for det in detections} | {gt.cls for gt in ground_truth})
        suggestions = suggest_thresholds(
            detections,
            ground_truth,
            classes,
            args.iou,
            args.thr_min,
            args.thr_max,
            args.thr_step,
        )
        print("\nSuggested thresholds (max F1 search):")
        for cls, info in suggestions.items():
            print(
                f"  {cls}: {info['threshold']:.2f} | P={info['precision']*100:.1f}% R={info['recall']*100:.1f}% F1={info['f1']*100:.1f}%"
            )
        print(
            "\nUpdate config/thresholds.json with the values above once you have enough samples."
        )


if __name__ == "__main__":
    main()

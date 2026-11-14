"""False-positive, false-negative, and threshold analysis for YOLO detectors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from config import DATA_DIR, INFERENCE, RESULTS_DIR
from dataset import load_label_file
from evaluate import extract_metrics
from model import class_names, load_model
from train import resolve_data

DEFAULT_CONF = float(INFERENCE.get("conf", 0.25))
DEFAULT_IOU = float(INFERENCE.get("iou", 0.45))
DEFAULT_IMGSZ = int(INFERENCE.get("imgsz", 640))

CONF_SWEEP_VALUES = [round(0.05 * index, 2) for index in range(1, 20)]
IOU_SWEEP_VALUES = [round(0.1 * index, 2) for index in range(3, 10)]
SWEEP_METRICS = ("precision", "recall", "mAP50", "mAP50-95")

SIZE_BUCKETS = (
    (0.0, 0.01, "tiny"),
    (0.01, 0.05, "small"),
    (0.05, 0.2, "medium"),
    (0.2, float("inf"), "large"),
)

TOP_ITEMS = 20


def _class_label(names: list[str], class_id: int) -> str:
    """Return a class name, falling back to the id when names are missing."""
    return names[class_id] if class_id < len(names) else str(class_id)


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return the pairwise IoU matrix for two sets of xyxy boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    sizes = np.clip(bottom_right - top_left, 0.0, None)
    intersection = sizes[..., 0] * sizes[..., 1]

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0.0, None)
    area_a *= np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0.0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0.0, None)
    area_b *= np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0.0, None)

    union = area_a[:, None] + area_b[None, :] - intersection
    return np.where(union > 0.0, intersection / union, 0.0).astype(np.float32)


def load_ground_truth(
    label_path: str | Path,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized YOLO labels as pixel xyxy boxes and class ids."""
    labels = load_label_file(Path(label_path))
    if len(labels) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=int)

    height, width = image_shape
    scale = np.array([width, height, width, height], dtype=np.float32)
    boxes = labels[:, 1:5] * scale
    xyxy = np.column_stack(
        (
            boxes[:, 0] - boxes[:, 2] / 2.0,
            boxes[:, 1] - boxes[:, 3] / 2.0,
            boxes[:, 0] + boxes[:, 2] / 2.0,
            boxes[:, 1] + boxes[:, 3] / 2.0,
        )
    )
    return xyxy.astype(np.float32), labels[:, 0].astype(int)


def match_detections(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_classes: np.ndarray,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> dict[str, Any]:
    """Greedily match predictions to ground truth by descending confidence."""
    ious = box_iou(gt_boxes, pred_boxes)
    matched_gt: set[int] = set()
    true_positives: list[tuple[int, int]] = []
    false_positives: list[int] = []

    for pred_index in np.argsort(-pred_scores):
        if len(gt_boxes) == 0:
            false_positives.append(int(pred_index))
            continue

        candidate = ious[:, pred_index].copy()
        if class_aware:
            same_class = gt_classes == pred_classes[pred_index]
            candidate = np.where(same_class, candidate, 0.0)
        if matched_gt:
            candidate[list(matched_gt)] = -1.0

        gt_index = int(np.argmax(candidate))
        if candidate[gt_index] >= iou_threshold:
            matched_gt.add(gt_index)
            true_positives.append((gt_index, int(pred_index)))
        else:
            false_positives.append(int(pred_index))

    false_negatives = [
        index for index in range(len(gt_boxes)) if index not in matched_gt
    ]

    return {
        "ious": ious,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def size_bucket(area: float) -> str:
    """Return the size label for a normalized box area."""
    for lower, upper, label in SIZE_BUCKETS:
        if lower <= area < upper:
            return label
    return SIZE_BUCKETS[-1][2]


def _normalized_area(box: np.ndarray, image_shape: tuple[int, int]) -> float:
    """Return the box area normalized by the image area."""
    height, width = image_shape
    if height <= 0 or width <= 0:
        return 0.0
    area = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    return float(area / (width * height))


def collect_error_analysis(
    weights: str | Path = "models/yolov8n_voc_best.pt",
    data_dir: str | Path = DATA_DIR,
    split: str = "val",
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    device: str | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run predictions on a split and rank false positives and negatives."""
    data_dir = Path(data_dir)
    images_dir = data_dir / "images" / split
    labels_dir = data_dir / "labels" / split
    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    model = load_model(weights)
    names = class_names(model)
    predict_args = {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device is not None:
        predict_args["device"] = device

    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    buckets = {
        label: {"total": 0, "matched": 0}
        for _, _, label in SIZE_BUCKETS
    }
    total_gt = 0
    total_predictions = 0
    total_true_positives = 0

    results = model.predict(
        source=[str(path) for path in image_paths],
        stream=True,
        **predict_args,
    )
    for result in results:
        image_path = Path(result.path)
        image_shape = tuple(int(value) for value in result.orig_shape)
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_boxes, gt_classes = load_ground_truth(label_path, image_shape)

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)
            pred_classes = np.zeros((0,), dtype=int)
        else:
            pred_boxes = boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_scores = boxes.conf.cpu().numpy().astype(np.float32)
            pred_classes = boxes.cls.cpu().numpy().astype(int)

        match = match_detections(
            gt_boxes,
            gt_classes,
            pred_boxes,
            pred_scores,
            pred_classes,
            iou_threshold=iou_threshold,
        )
        ious = match["ious"]

        total_gt += len(gt_boxes)
        total_predictions += len(pred_boxes)
        total_true_positives += len(match["true_positives"])

        for gt_index, _ in match["true_positives"]:
            area = _normalized_area(gt_boxes[gt_index], image_shape)
            bucket = size_bucket(area)
            buckets[bucket]["total"] += 1
            buckets[bucket]["matched"] += 1

        for gt_index in match["false_negatives"]:
            area = _normalized_area(gt_boxes[gt_index], image_shape)
            buckets[size_bucket(area)]["total"] += 1

            best_confidence = 0.0
            best_iou = 0.0
            best_class = None
            if len(pred_scores) > 0:
                pred_index = int(np.argmax(ious[gt_index]))
                best_iou = float(ious[gt_index, pred_index])
                best_confidence = float(pred_scores[pred_index])
                best_class = _class_label(names, int(pred_classes[pred_index]))

            class_id = int(gt_classes[gt_index])
            false_negatives.append(
                {
                    "image": image_path.name,
                    "class_id": class_id,
                    "class_name": _class_label(names, class_id),
                    "confidence": best_confidence,
                    "best_iou": best_iou,
                    "best_class": best_class,
                    "area": area,
                }
            )

        for pred_index in match["false_positives"]:
            class_id = int(pred_classes[pred_index])
            false_positives.append(
                {
                    "image": image_path.name,
                    "class_id": class_id,
                    "class_name": _class_label(names, class_id),
                    "confidence": float(pred_scores[pred_index]),
                    "area": _normalized_area(pred_boxes[pred_index], image_shape),
                }
            )

        if len(gt_boxes) > 0 and len(pred_scores) > 0:
            for gt_index, gt_class_id in enumerate(gt_classes):
                pred_index = int(np.argmax(ious[gt_index]))
                if ious[gt_index, pred_index] < iou_threshold:
                    continue
                actual = _class_label(names, int(gt_class_id))
                predicted = _class_label(names, int(pred_classes[pred_index]))
                confusion[(actual, predicted)] += 1

    size_summary = {}
    for label, counts in buckets.items():
        total = counts["total"]
        matched = counts["matched"]
        size_summary[label] = {
            "total": total,
            "matched": matched,
            "recall": matched / total if total else 0.0,
        }

    return {
        "weights": str(weights),
        "split": split,
        "images": len(image_paths),
        "ground_truth": total_gt,
        "predictions": total_predictions,
        "true_positives": total_true_positives,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "precision": (
            total_true_positives / total_predictions
            if total_predictions
            else 0.0
        ),
        "recall": total_true_positives / total_gt if total_gt else 0.0,
        "false_positives": sorted(
            false_positives,
            key=lambda item: item["confidence"],
            reverse=True,
        ),
        "false_negatives": sorted(
            false_negatives,
            key=lambda item: (item["confidence"], -item["area"]),
            reverse=True,
        ),
        "class_confusion": [
            {"actual": actual, "predicted": predicted, "count": count}
            for (actual, predicted), count in confusion.most_common()
        ],
        "size_buckets": size_summary,
    }


def sweep_thresholds(
    weights: str | Path = "models/yolov8n_voc_best.pt",
    data: str | Path | None = None,
    parameter: str = "conf",
    values: list[float] | None = None,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = 16,
    device: str | None = None,
) -> list[dict[str, float]]:
    """Run validation across confidence or NMS IoU values."""
    if parameter not in {"conf", "iou"}:
        raise ValueError(f"Unsupported sweep parameter: {parameter}")

    data_path = resolve_data(data)
    model = load_model(weights)
    sweep_values = values or (
        CONF_SWEEP_VALUES if parameter == "conf" else IOU_SWEEP_VALUES
    )

    rows = []
    for value in sweep_values:
        kwargs = {
            "data": str(data_path),
            "imgsz": imgsz,
            "batch": batch,
            "plots": False,
            "verbose": False,
            "project": str(RESULTS_DIR / "runs"),
            "name": f"error_sweep_{parameter}_{value:g}",
            "exist_ok": True,
        }
        kwargs[parameter] = float(value)
        if device is not None:
            kwargs["device"] = device

        metrics = model.val(**kwargs)
        row = {"value": float(value)}
        row.update(extract_metrics(metrics))
        rows.append(row)

    return rows


def best_threshold(rows: list[dict[str, float]]) -> dict[str, float]:
    """Return the sweep row with the highest mAP50-95."""
    if not rows:
        raise ValueError("No sweep rows available.")
    return max(
        rows,
        key=lambda row: (row.get("mAP50-95", 0.0), row.get("mAP50", 0.0)),
    )


def plot_sweep(
    rows: list[dict[str, float]],
    parameter: str,
    out_path: str | Path | None = None,
) -> Path:
    """Plot validation metrics against a confidence or IoU sweep."""
    if not rows:
        raise ValueError("No sweep rows to plot.")

    out_path = (
        Path(out_path)
        if out_path is not None
        else RESULTS_DIR / f"{parameter}_sweep.png"
    )
    values = [row["value"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric in SWEEP_METRICS:
        if metric not in rows[0]:
            continue
        ax.plot(
            values,
            [row[metric] for row in rows],
            marker="o",
            label=metric,
        )
    ax.set_xlabel(parameter)
    ax.set_ylabel("Score")
    ax.set_title(f"{parameter} threshold sweep")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_analysis(
    analysis: dict[str, Any],
    out_path: str | Path = RESULTS_DIR / "error_analysis.json",
) -> Path:
    """Write the error-analysis payload as indented JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(analysis, indent=2) + "\n", encoding="utf-8"
    )
    return out_path


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _score(value: float) -> str:
    return f"{value:.4f}"


def write_report(
    analysis: dict[str, Any],
    conf_sweep: list[dict[str, float]] | None = None,
    iou_sweep: list[dict[str, float]] | None = None,
    out_path: str | Path = RESULTS_DIR / "error_analysis.md",
) -> Path:
    """Write a markdown report of false positives, negatives, and sweeps."""
    conf_sweep = conf_sweep or []
    iou_sweep = iou_sweep or []

    lines = [
        "# Error analysis",
        "",
        f"- Weights: `{analysis['weights']}`",
        f"- Split: `{analysis['split']}`",
        f"- Images: {analysis['images']}",
        "",
        "## Summary",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Ground-truth boxes", analysis["ground_truth"]],
                ["Predictions", analysis["predictions"]],
                ["True positives", analysis["true_positives"]],
                ["False positives", analysis["false_positive_count"]],
                ["False negatives", analysis["false_negative_count"]],
                ["Precision", _score(analysis["precision"])],
                ["Recall", _score(analysis["recall"])],
            ],
        ),
        "",
        "## Top false positives by confidence",
        "",
    ]

    if analysis["false_positives"]:
        lines.append(
            _markdown_table(
                ["Image", "Class", "Confidence", "Area"],
                [
                    [
                        item["image"],
                        item["class_name"],
                        _score(item["confidence"]),
                        _score(item["area"]),
                    ]
                    for item in analysis["false_positives"][:TOP_ITEMS]
                ],
            )
        )
    else:
        lines.append("None.")

    lines.extend(["", "## Top false negatives by confidence", ""])
    if analysis["false_negatives"]:
        lines.append(
            _markdown_table(
                [
                    "Image",
                    "Class",
                    "Confidence",
                    "Best IoU",
                    "Best prediction",
                    "Area",
                ],
                [
                    [
                        item["image"],
                        item["class_name"],
                        _score(item["confidence"]),
                        _score(item["best_iou"]),
                        item["best_class"] or "none",
                        _score(item["area"]),
                    ]
                    for item in analysis["false_negatives"][:TOP_ITEMS]
                ],
            )
        )
    else:
        lines.append("None.")

    lines.extend(["", "## Class confusion", ""])
    if analysis["class_confusion"]:
        lines.append(
            _markdown_table(
                ["Actual", "Predicted", "Count"],
                [
                    [item["actual"], item["predicted"], item["count"]]
                    for item in analysis["class_confusion"][:TOP_ITEMS]
                ],
            )
        )
    else:
        lines.append("None.")

    lines.extend(["", "## Recall by object size", ""])
    lines.append(
        _markdown_table(
            ["Bucket", "Ground truth", "Matched", "Recall"],
            [
                [
                    label,
                    values["total"],
                    values["matched"],
                    _score(values["recall"]),
                ]
                for label, values in analysis["size_buckets"].items()
            ],
        )
    )

    for title, rows, parameter in (
        ("Confidence sweep", conf_sweep, "conf"),
        ("IoU sweep", iou_sweep, "iou"),
    ):
        lines.extend(["", f"## {title}", ""])
        if not rows:
            lines.append("Not run.")
            continue

        best = best_threshold(rows)
        lines.append(f"Best {parameter}: {best['value']:g}")
        lines.append("")
        lines.append(
            _markdown_table(
                [parameter, *SWEEP_METRICS],
                [
                    [
                        f"{row['value']:g}",
                        *[
                            _score(row[metric]) if metric in row else "—"
                            for metric in SWEEP_METRICS
                        ],
                    ]
                    for row in rows
                ],
            )
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze YOLO false positives, negatives, and thresholds."
    )
    parser.add_argument(
        "--weights",
        default="models/yolov8n_voc_best.pt",
        help="Variant key (n/s/m), checkpoint name, or path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Dataset root containing images/ and labels/.",
    )
    parser.add_argument("--data", type=Path, default=None, help="Dataset YAML.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="'cpu' or a GPU index.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run confidence and IoU validation sweeps.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=RESULTS_DIR / "error_analysis.md",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "error_analysis.json",
    )
    args = parser.parse_args()

    analysis = collect_error_analysis(
        weights=args.weights,
        data_dir=args.data_dir,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        iou_threshold=args.iou_threshold,
    )

    conf_sweep: list[dict[str, float]] = []
    iou_sweep: list[dict[str, float]] = []
    if args.sweep:
        conf_sweep = sweep_thresholds(
            weights=args.weights,
            data=args.data,
            parameter="conf",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        iou_sweep = sweep_thresholds(
            weights=args.weights,
            data=args.data,
            parameter="iou",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        plot_sweep(conf_sweep, "conf", RESULTS_DIR / "conf_sweep.png")
        plot_sweep(iou_sweep, "iou", RESULTS_DIR / "iou_sweep.png")

    json_path = save_analysis(analysis, args.out_json)
    report_path = write_report(
        analysis,
        conf_sweep=conf_sweep,
        iou_sweep=iou_sweep,
        out_path=args.out_md,
    )

    print("Images:", analysis["images"])
    print("False positives:", analysis["false_positive_count"])
    print("False negatives:", analysis["false_negative_count"])
    print("JSON:", json_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()

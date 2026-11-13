"""Validation metrics for trained YOLO detectors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from config import RESULTS_DIR, ROOT_DIR
from model import load_model
from train import resolve_data

METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}

ARTIFACT_FILES = {
    "pr_curve": "BoxPR_curve.png",
    "f1_curve": "BoxF1_curve.png",
    "precision_curve": "BoxP_curve.png",
    "recall_curve": "BoxR_curve.png",
    "confusion_matrix": "confusion_matrix.png",
    "confusion_matrix_normalized": "confusion_matrix_normalized.png",
    "results_csv": "results.csv",
}


def extract_metrics(metrics: Any) -> dict[str, float]:
    """Map Ultralytics result keys to stable, short metric names."""
    results = getattr(metrics, "results_dict", {})
    return {
        name: float(results[key])
        for name, key in METRIC_KEYS.items()
        if key in results
    }


def per_class_ap(metrics: Any) -> dict[str, float]:
    """Return per-class mAP@0.5:0.95 from a validation result."""
    names = getattr(metrics, "names", [])
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]

    maps = getattr(getattr(metrics, "box", None), "maps", None)
    if maps is None or len(names) != len(maps):
        return {}

    return {str(name): float(ap) for name, ap in zip(names, maps)}


def _display_path(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def export_artifacts(
    run_dir: str | Path,
    out_dir: str | Path = RESULTS_DIR,
) -> dict[str, str]:
    """Copy PR curves, confusion matrices, and results.csv into results/."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = {}
    for key, filename in ARTIFACT_FILES.items():
        source = run_dir / filename
        if not source.exists():
            continue
        destination = out_dir / filename
        shutil.copy2(source, destination)
        exported[key] = _display_path(destination)

    return exported


def save_metrics(payload: dict[str, Any], out_path: Path) -> Path:
    """Write a metrics payload as indented JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return out_path


def evaluate_model(
    weights: str | Path = "models/yolov8n_voc_best.pt",
    data: str | Path | None = None,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    project: str | Path | None = None,
    name: str = "yolov8n_voc_val",
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run validation and write results/metrics.json."""
    data_path = resolve_data(data)
    project = Path(project) if project is not None else RESULTS_DIR / "runs"

    model = load_model(weights)
    kwargs = {
        "data": str(data_path),
        "imgsz": imgsz,
        "batch": batch,
        "project": str(project),
        "name": name,
    }
    if device is not None:
        kwargs["device"] = device

    metrics = model.val(**kwargs)
    run_dir = Path(getattr(metrics, "save_dir", project / name))

    payload = {
        "run": name,
        "weights": str(weights),
        "dataset": str(data_path),
        "metrics": extract_metrics(metrics),
        "per_class_ap50_95": per_class_ap(metrics),
        "speed_ms_per_image": {
            key: float(value)
            for key, value in getattr(metrics, "speed", {}).items()
        },
        "artifacts": export_artifacts(run_dir),
    }

    out_path = Path(out_path) if out_path is not None else RESULTS_DIR / "metrics.json"
    save_metrics(payload, out_path)

    return {
        "metrics": metrics,
        "run_dir": run_dir,
        "output": out_path,
        "payload": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 detector on VOC.")
    parser.add_argument(
        "--weights",
        default="models/yolov8n_voc_best.pt",
        help="Variant key (n/s/m), checkpoint name, or path.",
    )
    parser.add_argument("--data", type=Path, default=None, help="Dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="'cpu' or a GPU index.")
    parser.add_argument("--name", default="yolov8n_voc_val")
    parser.add_argument("--out", type=Path, default=None, help="Metrics JSON path.")
    args = parser.parse_args()

    output = evaluate_model(
        weights=args.weights,
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        out_path=args.out,
    )

    print(json.dumps(output["payload"]["metrics"], indent=2))
    print("Run directory:", output["run_dir"])
    print("Metrics:", output["output"])


if __name__ == "__main__":
    main()

"""Inference for images, videos, and webcam streams with YOLO detectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import INFERENCE, RESULTS_DIR, ROOT_DIR
from model import class_names, load_model

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}

PREDICTIONS_DIR = RESULTS_DIR / "predictions"

DEFAULT_CONF = float(INFERENCE.get("conf", 0.25))
DEFAULT_IOU = float(INFERENCE.get("iou", 0.45))
DEFAULT_IMGSZ = int(INFERENCE.get("imgsz", 640))
DEFAULT_MAX_DET = int(INFERENCE.get("max_det", 300))


def _display_path(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def prediction_args(
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    device: str | None = None,
) -> dict[str, Any]:
    """Build shared Ultralytics prediction arguments."""
    args = {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "max_det": max_det,
        "verbose": False,
    }
    if device is not None:
        args["device"] = device
    return args


def detection_records(result: Any, names: list[str]) -> list[dict[str, Any]]:
    """Serialize detected boxes from one Ultralytics result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    records = []
    for box in boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id] if class_id < len(names) else str(class_id)
        records.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(box.conf[0]),
                "xyxy": [float(value) for value in box.xyxy[0]],
                "xywhn": [float(value) for value in box.xywhn[0]],
            }
        )
    return records


def infer_image(
    source: str | Path,
    weights: str | Path = "n",
    out_dir: str | Path = PREDICTIONS_DIR,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    max_det: int = DEFAULT_MAX_DET,
    device: str | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Run detection on one image and write an annotated image plus JSON."""
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(weights)
    result = model.predict(
        source=str(source),
        **prediction_args(conf, iou, imgsz, max_det, device),
    )[0]

    annotated = out_dir / f"{source.stem}_annotated{source.suffix}"
    if save:
        result.save(filename=str(annotated))

    payload = {
        "source": str(source),
        "weights": str(weights),
        "image": _display_path(annotated) if save else None,
        "detections": detection_records(result, class_names(model)),
    }
    detections_path = out_dir / f"{source.stem}.json"
    detections_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "annotated": annotated if save else None,
        "detections": detections_path,
        "payload": payload,
    }


def infer_video(
    source: str | Path,
    weights: str | Path = "n",
    out_dir: str | Path = PREDICTIONS_DIR,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    max_det: int = DEFAULT_MAX_DET,
    device: str | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Run detection on a video and write an annotated video."""
    source = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(weights)
    results = model.predict(
        source=str(source),
        save=save,
        project=str(out_dir),
        name=source.stem,
        exist_ok=True,
        stream=True,
        **prediction_args(conf, iou, imgsz, max_det, device),
    )

    save_dir = None
    frames = 0
    for result in results:
        save_dir = getattr(result, "save_dir", save_dir)
        frames += 1

    annotated = None
    if save and save_dir is not None:
        candidates = sorted(
            path
            for path in Path(save_dir).iterdir()
            if path.suffix.lower() in VIDEO_SUFFIXES
        )
        if candidates:
            annotated = candidates[0]

    payload = {
        "source": str(source),
        "weights": str(weights),
        "video": _display_path(annotated) if annotated is not None else None,
        "frames": frames,
    }
    detections_path = out_dir / f"{source.stem}.json"
    detections_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "annotated": annotated,
        "detections": detections_path,
        "payload": payload,
    }


def infer(
    source: str | int,
    weights: str | Path = "n",
    out_dir: str | Path = PREDICTIONS_DIR,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    imgsz: int = DEFAULT_IMGSZ,
    max_det: int = DEFAULT_MAX_DET,
    device: str | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Dispatch inference by source type."""
    path = Path(str(source))
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return infer_video(
            path,
            weights=weights,
            out_dir=out_dir,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
            save=save,
        )
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return infer_image(
            path,
            weights=weights,
            out_dir=out_dir,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
            save=save,
        )
    raise ValueError(f"Unsupported source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference.")
    parser.add_argument(
        "--source",
        required=True,
        help="Image path, video path, or webcam index.",
    )
    parser.add_argument(
        "--weights",
        default="n",
        help="Variant key (n/s/m), checkpoint name, or path.",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--device", default=None, help="'cpu' or a GPU index.")
    parser.add_argument("--out", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write annotated output.",
    )
    args = parser.parse_args()

    output = infer(
        args.source,
        weights=args.weights,
        out_dir=args.out,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        save=args.save,
    )

    if "detections" in output["payload"]:
        print("Detections:", len(output["payload"]["detections"]))
    else:
        print("Frames:", output["payload"]["frames"])
    print("Annotated:", output["annotated"])
    print("JSON:", output["detections"])


if __name__ == "__main__":
    main()

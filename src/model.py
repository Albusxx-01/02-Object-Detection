"""YOLOv8 model loading helpers for the Pascal VOC detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import MODELS_DIR

MODEL_VARIANTS = {
    "n": "yolov8n.pt",
    "s": "yolov8s.pt",
    "m": "yolov8m.pt",
}


def resolve_weights(weights: str | Path = "n") -> str:
    """Resolve a variant key, checkpoint name, or path to YOLO weights."""
    value = str(weights)
    if value in MODEL_VARIANTS:
        value = MODEL_VARIANTS[value]

    path = Path(value)
    if path.exists():
        return str(path)

    local = MODELS_DIR / value
    if local.exists():
        return str(local)

    return value


def load_model(weights: str | Path = "n"):
    """Load a YOLO model from a variant key, checkpoint name, or path."""
    from ultralytics import YOLO

    return YOLO(resolve_weights(weights))


def class_names(model) -> list[str]:
    """Return the model's class names ordered by class id."""
    names = model.names
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a YOLOv8 model.")
    parser.add_argument(
        "--weights",
        default="n",
        help="Variant key (n/s/m), checkpoint name, or path.",
    )
    args = parser.parse_args()

    model = load_model(args.weights)
    print("Loaded:", resolve_weights(args.weights))
    print("Task:", model.task)
    print("Classes:", ", ".join(class_names(model)))


if __name__ == "__main__":
    main()

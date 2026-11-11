"""Training wrapper for reproducible YOLO runs and checkpoint handling."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from config import (
    DATASET_YAML_PATH,
    MODELS_DIR,
    get_training_args,
    write_dataset_yaml,
)
from model import load_model


def resolve_data(data: str | Path | None = None) -> Path:
    """Return an existing dataset YAML, generating the project default if needed."""
    if data is not None:
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {path}")
        return path
    if DATASET_YAML_PATH.exists():
        return DATASET_YAML_PATH
    return write_dataset_yaml()


def train_model(
    weights: str | Path | None = None,
    data: str | Path | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Train a YOLO model and copy the best checkpoint into models/."""
    args = get_training_args()
    configured_model = args.pop("model", "yolov8n.pt")

    for key, value in overrides.items():
        if value is not None:
            args[key] = value

    data_path = resolve_data(data)
    model = load_model(weights or configured_model)
    results = model.train(data=str(data_path), **args)

    project = Path(args["project"])
    name = args["name"]
    run_dir = Path(getattr(results, "save_dir", project / name))
    best_weights = run_dir / "weights" / "best.pt"
    checkpoint = MODELS_DIR / f"{name}_best.pt"

    if best_weights.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, checkpoint)
    else:
        checkpoint = None

    return {
        "run_dir": run_dir,
        "best_weights": best_weights if best_weights.exists() else None,
        "checkpoint": checkpoint,
        "data": data_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 detector on VOC.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Variant key (n/s/m), checkpoint name, or path.",
    )
    parser.add_argument("--data", type=Path, default=None, help="Dataset YAML.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None, help="'cpu' or a GPU index.")
    parser.add_argument("--name", default=None, help="Run directory name.")
    args = parser.parse_args()

    output = train_model(
        weights=args.weights,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )

    print("Run directory:", output["run_dir"])
    print("Best weights:", output["best_weights"])
    print("Checkpoint:", output["checkpoint"])


if __name__ == "__main__":
    main()

"""End-to-end smoke test for config, artifacts, CLIs, and image inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ultralytics.utils import ASSETS

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from infer import infer_image  # noqa: E402

CLI_MODULES = (
    "dataset.py",
    "model.py",
    "train.py",
    "evaluate.py",
    "loss.py",
    "error_analysis.py",
    "infer.py",
)

METRICS_KEYS = {"metrics", "artifacts"}

ARTIFACTS = (
    "metrics.json",
    "results.csv",
    "train_config.yaml",
    "BoxPR_curve.png",
    "confusion_matrix.png",
)


def check_config() -> None:
    """Validate class names and the expected VOC label set."""
    assert NUM_CLASSES == 20, NUM_CLASSES
    assert len(CLASS_NAMES) == NUM_CLASSES
    assert "person" in CLASS_NAMES


def check_artifacts() -> None:
    """Validate committed result artifacts and the metrics schema."""
    results_dir = ROOT_DIR / "results"
    for name in ARTIFACTS:
        path = results_dir / name
        assert path.exists(), f"Missing artifact: {path}"

    payload = json.loads(
        (results_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert METRICS_KEYS <= set(payload), sorted(payload)
    assert payload["metrics"], "Empty metrics payload"


def check_clis() -> None:
    """Run --help for every CLI entry point."""
    for module in CLI_MODULES:
        subprocess.run(
            [sys.executable, str(ROOT_DIR / "src" / module), "--help"],
            check=True,
            capture_output=True,
        )


def check_inference(weights: Path) -> None:
    """Run one image through the inference pipeline when weights exist."""
    if not weights.exists():
        print(f"Skipping inference: {weights} not found.")
        return

    image_source = ASSETS / "bus.jpg"
    with tempfile.TemporaryDirectory() as temp_dir:
        output = infer_image(
            image_source,
            weights=weights,
            out_dir=Path(temp_dir),
            device="cpu",
        )
        payload = json.loads(
            output["detections"].read_text(encoding="utf-8")
        )
        assert output["annotated"].exists()
        assert isinstance(payload["detections"], list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end smoke checks.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT_DIR / "models" / "yolov8n.pt",
        help="Checkpoint used for the inference smoke check.",
    )
    args = parser.parse_args()

    check_config()
    check_artifacts()
    check_clis()
    check_inference(args.weights)

    print("End-to-end smoke test passed.")


if __name__ == "__main__":
    main()

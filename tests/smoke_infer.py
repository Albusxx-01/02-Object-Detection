"""Smoke test for the image and video inference entry points."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
from ultralytics.utils import ASSETS

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from infer import infer_image, infer_video  # noqa: E402


def _write_sample_video(image_path: Path, video_path: Path) -> None:
    frame = cv2.imread(str(image_path))
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (width, height),
    )
    try:
        for _ in range(5):
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test YOLO inference.")
    parser.add_argument(
        "--weights",
        default="n",
        help="Variant key (n/s/m), checkpoint name, or path.",
    )
    args = parser.parse_args()

    image_source = ASSETS / "bus.jpg"

    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)

        image_output = infer_image(
            image_source,
            weights=args.weights,
            out_dir=out_dir,
            device="cpu",
        )
        image_payload = json.loads(
            image_output["detections"].read_text(encoding="utf-8")
        )
        assert image_output["annotated"].exists()
        assert isinstance(image_payload["detections"], list)

        video_source = out_dir / "sample.mp4"
        _write_sample_video(image_source, video_source)
        video_output = infer_video(
            video_source,
            weights=args.weights,
            out_dir=out_dir,
            device="cpu",
        )
        video_payload = json.loads(
            video_output["detections"].read_text(encoding="utf-8")
        )
        assert video_output["annotated"] is not None
        assert video_output["annotated"].exists()
        assert video_payload["frames"] == 5

    print("Inference smoke test passed.")


if __name__ == "__main__":
    main()

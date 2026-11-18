"""Streamlit interface for the YOLOv8 Pascal VOC detector."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT / "src"))

try:
    import streamlit as st
except ImportError as error:
    raise SystemExit(
        "Streamlit is required. Install dependencies with "
        "`pip install -r requirements.txt`."
    ) from error

from config import INFERENCE, MODELS_DIR, ROOT_DIR
from model import MODEL_VARIANTS, class_names, load_model

DEFAULT_CONF = float(INFERENCE.get("conf", 0.25))
DEFAULT_IOU = float(INFERENCE.get("iou", 0.45))
DEFAULT_IMGSZ = int(INFERENCE.get("imgsz", 640))
DEFAULT_WEIGHTS = "models/yolov8n_voc_best.pt"


def available_weights() -> list[str]:
    """Return local checkpoints, falling back to YOLOv8 variant keys."""
    checkpoints = [
        path.relative_to(ROOT_DIR).as_posix()
        for path in sorted(MODELS_DIR.glob("*.pt"))
    ]
    return checkpoints or list(MODEL_VARIANTS)


@st.cache_resource(show_spinner=False)
def load_cached_model(
    weights: str,
) -> tuple[Any | None, list[str], str | None]:
    """Load a YOLO model once and cache it for the Streamlit session."""
    try:
        model = load_model(weights)
    except (ImportError, FileNotFoundError, OSError) as error:
        return None, [], str(error)
    return model, class_names(model), None


def main() -> None:
    st.set_page_config(page_title="Object Detection", layout="wide")
    st.title("Object Detection with YOLOv8")
    st.caption("Pascal VOC detector for images, camera frames, and videos.")

    options = available_weights()
    default_index = options.index(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS in options else 0
    weights = st.sidebar.selectbox("Weights", options, index=default_index)

    model, names, error = load_cached_model(weights)
    if model is None:
        st.error(f"Could not load `{weights}`: {error}")
        if "ultralytics" in str(error).lower():
            st.info("Install dependencies with `pip install -r requirements.txt`.")
        st.stop()

    st.sidebar.slider(
        "Confidence",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONF,
        step=0.01,
    )
    st.sidebar.slider(
        "NMS IoU",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_IOU,
        step=0.01,
    )
    st.sidebar.caption(f"{len(names)} classes loaded from `{weights}`.")

    image_tab, camera_tab, video_tab = st.tabs(["Image", "Camera", "Video"])
    with image_tab:
        st.info("Image detection is added in the next commit.")
    with camera_tab:
        st.info("Camera detection is added in the next commit.")
    with video_tab:
        st.info("Video detection is added in a later commit.")


if __name__ == "__main__":
    main()

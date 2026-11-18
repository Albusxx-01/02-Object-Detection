"""Streamlit interface for the YOLOv8 Pascal VOC detector."""

from __future__ import annotations

import io
import json
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

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from config import INFERENCE, MODELS_DIR, ROOT_DIR
from infer import detection_records
from model import MODEL_VARIANTS, class_names, load_model

DEFAULT_CONF = float(INFERENCE.get("conf", 0.25))
DEFAULT_IOU = float(INFERENCE.get("iou", 0.45))
DEFAULT_IMGSZ = int(INFERENCE.get("imgsz", 640))
DEFAULT_MAX_DET = int(INFERENCE.get("max_det", 300))
DEFAULT_WEIGHTS = "models/yolov8n_voc_best.pt"

IMAGE_TYPES = ["bmp", "jpeg", "jpg", "png", "tif", "tiff", "webp"]


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


def predict_image(
    model: Any,
    image: Image.Image,
    conf: float,
    iou: float,
    imgsz: int,
    class_ids: list[int],
) -> Any:
    """Run detection on a PIL image and return the first result."""
    array = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return model.predict(
        source=array,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=DEFAULT_MAX_DET,
        classes=class_ids or None,
        verbose=False,
    )[0]


def annotated_image(result: Any) -> np.ndarray:
    """Convert a plotted detection result to RGB for Streamlit."""
    return cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)


def detection_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a tidy detection table for display."""
    columns = ["class", "confidence", "x1", "y1", "x2", "y2"]
    rows = [
        {
            "class": record["class_name"],
            "confidence": round(record["confidence"], 3),
            "x1": round(record["xyxy"][0], 1),
            "y1": round(record["xyxy"][1], 1),
            "x2": round(record["xyxy"][2], 1),
            "y2": round(record["xyxy"][3], 1),
        }
        for record in records
    ]
    return pd.DataFrame(rows, columns=columns)


def detections_json(records: list[dict[str, Any]]) -> str:
    """Serialize detections for download."""
    return json.dumps(records, indent=2) + "\n"


def render_detections(records: list[dict[str, Any]], key: str) -> None:
    """Display a detection table and JSON download button."""
    st.subheader("Detections")
    if not records:
        st.info("No objects detected with the current thresholds.")
        return

    st.dataframe(
        detection_frame(records),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download detections (JSON)",
        data=detections_json(records),
        file_name=f"{key}_detections.json",
        mime="application/json",
        key=f"{key}_json",
    )


def render_image_result(
    model: Any,
    image: Image.Image,
    names: list[str],
    conf: float,
    iou: float,
    imgsz: int,
    class_ids: list[int],
    key: str,
) -> None:
    """Run detection on one image and render the result panel."""
    with st.spinner("Running detection..."):
        result = predict_image(model, image, conf, iou, imgsz, class_ids)
    records = detection_records(result, names)
    annotated = annotated_image(result)

    left, right = st.columns(2)
    left.image(image, caption="Input", use_container_width=True)
    right.image(
        annotated,
        caption=f"{len(records)} detections",
        use_container_width=True,
    )

    render_detections(records, key)

    buffer = io.BytesIO()
    Image.fromarray(annotated).save(buffer, format="PNG")
    st.download_button(
        "Download annotated image",
        data=buffer.getvalue(),
        file_name=f"{key}_annotated.png",
        mime="image/png",
        key=f"{key}_image",
    )


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

    conf = st.sidebar.slider(
        "Confidence",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_CONF,
        step=0.01,
    )
    iou = st.sidebar.slider(
        "NMS IoU",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_IOU,
        step=0.01,
    )
    st.sidebar.caption(f"{len(names)} classes loaded from `{weights}`.")

    image_tab, camera_tab, video_tab = st.tabs(["Image", "Camera", "Video"])

    with image_tab:
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=IMAGE_TYPES,
        )
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            render_image_result(
                model,
                image,
                names,
                conf,
                iou,
                DEFAULT_IMGSZ,
                [],
                key="image",
            )

    with camera_tab:
        photo = st.camera_input("Take a photo")
        if photo is not None:
            image = Image.open(photo).convert("RGB")
            render_image_result(
                model,
                image,
                names,
                conf,
                iou,
                DEFAULT_IMGSZ,
                [],
                key="camera",
            )

    with video_tab:
        st.info("Video detection is added in a later commit.")


if __name__ == "__main__":
    main()

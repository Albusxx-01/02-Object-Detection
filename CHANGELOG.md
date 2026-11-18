# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
semantic versioning.

## [1.0.0] - 2026-11-18

### Added

- Pascal VOC download, XML-to-YOLO conversion, reproducible splits, dataset
  statistics, box visualization, and anchor analysis.
- `src/train.py`, `src/model.py`, and `src/loss.py` for training runs, model
  loading, and loss-curve plotting.
- `src/evaluate.py` for mAP, precision, recall, per-class AP, PR curves, and
  confusion matrices.
- `src/error_analysis.py` for false-positive and false-negative ranking,
  class-confusion analysis, object-size recall, and confidence/IoU sweeps.
- `src/infer.py` for image, video, and webcam inference with JSON detections.
- `app.py`, a Streamlit interface with image, camera, and video detection tabs,
  class filtering, and result downloads.
- Notebooks `01-data-and-training.ipynb`, `02-evaluation.ipynb`, and
  `03-error-analysis.ipynb`.
- `tests/smoke_infer.py` and `tests/smoke_e2e.py` for inference and end-to-end
  checks.
- `docs/reproducibility.md` and troubleshooting notes in the README.

### Results

- YOLOv8n, full Pascal VOC, 50 epochs, seed 42:
  mAP@0.5 **0.826**, mAP@0.5:0.95 **0.617**, precision **0.812**, recall
  **0.749**.

### Notes

- Datasets, checkpoints, run folders, and videos stay gitignored. Reproduce them
  with the commands in the README and `docs/reproducibility.md`.
- The error-analysis report is blocked until the VOC validation split is
  prepared locally; the CLI and notebook are complete.

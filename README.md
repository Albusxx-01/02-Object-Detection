# Object Detection with YOLOv8 (Pascal VOC)

An end-to-end object detection project built with [Ultralytics YOLOv8](https://docs.ultralytics.com/)
and PyTorch. The pipeline downloads Pascal VOC, converts XML annotations to YOLO
format, explores class and box statistics, runs anchor analysis, trains a detector,
and reports validation metrics.

This is part of a hands-on CNN project series: classification → **object detection** →
segmentation → GANs → face recognition.

## Status

| Component | Status |
|-----------|--------|
| Project scaffold, config, requirements | Done |
| VOC download, XML → YOLO conversion, splits | Done |
| Dataset statistics, box visualization, anchor analysis | Done |
| YOLOv8 training + quick validation + loss/mAP curves | Implemented (`src/train.py`, notebook 01) |
| Full VOC training run | Done — 50 epochs, mAP@0.5 0.826 |
| Evaluation CLI + notebook (PR curves, per-class mAP, confusion matrix) | Implemented (`src/evaluate.py`, notebook 02) |
| Error-analysis notebook | Planned |
| Inference CLI (image / video / webcam) | Planned |
| Streamlit app (image / camera / video) | Done (`app.py`) |

## Quick Start (Google Colab)

1. Upload `notebooks/01-data-and-training.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Runtime → **Change runtime type** → GPU.
3. Runtime → **Run all**.

The notebook is self-contained and covers data prep, exploration, anchors, training,
validation, and curves. Two toggles control the run:

| Toggle | Default | Effect |
|--------|---------|--------|
| `QUICK_RUN` | `True` | VOC2007 only (~884 MB), 800 train / 200 val images, 10 epochs |
| `RUN_TRAINING` | `True` | Set to `False` to stop after dataset prep and anchors |

For the full official split, set `QUICK_RUN = False` (~2.8 GB, 16,551 train /
4,952 val images, 50 epochs).

Outputs:

- `data/images/{train,val}/` and `data/labels/{train,val}/` — YOLO-format dataset
- `results/runs/yolov8n_voc/` — training logs, curves, validation artifacts
- `models/yolov8n_voc_best.pt` — best checkpoint

## Local Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

A CPU-only install works for dataset preparation and inference. Training is much
faster on a GPU (Colab T4 is recommended).

### Prepare the dataset locally

```bash
# Quick run: VOC2007 only, 200 images per split (~884 MB download)
python src/dataset.py --quick --max-images 200

# Full official split (~2.8 GB download)
python src/dataset.py
```

This downloads the archives, converts annotations, and writes reproducible split
lists to `data/splits/`.

### Evaluate a checkpoint

```bash
python src/evaluate.py --weights models/yolov8n_voc_best.pt
python src/loss.py --csv results/results.csv
```

`src/evaluate.py` writes `results/metrics.json` and exports PR/F1/confusion
artifacts into `results/`. `src/loss.py` writes `results/loss_curves.png`.

## Project Structure

```
02-Object-Detection/
├── app.py                      # Streamlit UI for image/camera/video detection
├── config/
│   ├── detection.yaml          # dataset paths, class names, hyperparameters
│   └── classes.yaml            # class id -> name map
├── src/
│   ├── config.py               # paths, constants, dataset YAML generation
│   ├── dataset.py              # VOC download, XML->YOLO, splits, stats, visualization
│   ├── model.py                # YOLO model loader
│   ├── train.py                # training wrapper with run directories
│   ├── evaluate.py             # mAP metrics and artifact export
│   ├── loss.py                 # loss-curve plotting
│   └── infer.py                # planned: image/video/webcam inference
├── notebooks/
│   ├── 01-data-and-training.ipynb
│   └── 02-evaluation.ipynb
├── data/                       # generated, gitignored
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── splits/
├── models/                     # checkpoints, gitignored
├── results/                    # metrics, plots, sample detections
├── requirements.txt
└── README.md
```

## Pipeline

```
Pascal VOC XML annotations
        │
        ▼
YOLO labels: class x_center y_center width height (normalized)
        │
        ▼
Train / val split (official VOC split)
        │
        ▼
YOLOv8n backbone → neck → detection head
        │
        ▼
Training (640 px, batch 16, seed 42)
        │
        ▼
Validation → mAP@0.5, mAP@0.5:0.95, precision, recall
        │
        ▼
models/yolov8n_voc_best.pt
```

YOLOv8 is **anchor-free**. The notebook includes k-means anchor analysis as a
learning exercise for classic YOLO anchor design; those anchors are not passed to
the YOLOv8 trainer.

## Dataset

**Pascal VOC** — 20 classes:

```
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa,
train, tvmonitor
```

**Official split**

| Split | Source | Images |
|-------|--------|--------|
| train | VOC2012 train/val + VOC2007 train/val | 16,551 |
| val | VOC2007 test | 4,952 |
| test | aliases `images/val` | 4,952 |

**YOLO label format** — one `.txt` file per image, one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to `[0, 1]` relative to image width/height. Empty
label files are valid and represent images with no annotated objects.

## Training Configuration

| Setting | Quick run | Full run |
|---------|-----------|----------|
| Model | `yolov8n.pt` | `yolov8n.pt` |
| Train images | 800 | 16,551 |
| Val images | 200 | 4,952 |
| Epochs | 10 | 50 |
| Image size | 640 | 640 |
| Batch size | 16 | 16 |
| Seed | 42 | 42 |
| Device | GPU or CPU | GPU (Colab T4) |

The quick run is for pipeline verification. The full run targets **mAP@0.5 of
60–80%**, consistent with the project plan. Switching to `yolov8s.pt` should push
accuracy higher at the cost of speed.

## Evaluation

`src/evaluate.py` and `notebooks/02-evaluation.ipynb` report:

- mAP@0.5 and mAP@0.5:0.95
- Precision, recall, and per-class AP
- Precision-recall, F1, precision, and recall curves
- Confusion matrix
- Validation speed per image

Error analysis for false positives and false negatives is planned for
`03-error-analysis.ipynb`.

## Inference and App

Planned:

- `src/infer.py` — image, video, and webcam inference with JSON detections
- `app.py` — Streamlit UI with image, camera, and video tabs, confidence/IoU
  sliders, class filtering, and result download

## Streamlit App

```bash
streamlit run app.py
```

The sidebar selects the checkpoint, confidence, NMS IoU, and classes. The
**Image**, **Camera**, and **Video** tabs run detection, show the annotated
result and a detection table, and offer JSON downloads. The video tab also
exports the annotated MP4. Video processing runs frame by frame and is much
slower than image inference on CPU.

The app requires `streamlit` and `opencv-python`, both listed in
`requirements.txt`. If a checkpoint is missing, choose another weight or train
or download one first; the app reports the load error instead of falling back
silently.

## Results

Full official VOC split, 50 epochs, seed 42, best checkpoint at epoch 49.

| Model | Dataset | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Notes |
|-------|---------|---------|--------------|-----------|--------|-------|
| YOLOv8n | VOC quick | — | — | — | — | pipeline smoke run |
| YOLOv8n | VOC full | 0.826 | 0.617 | 0.812 | 0.749 | 50 epochs, best epoch 49 |

Training curves, PR/F1/precision/recall curves, and confusion matrices are in
`results/`; raw metrics are in `results/metrics.json`. Validation sample grids are
in `results/samples/`.

Artifacts are written to `results/`; large run folders and model weights are
gitignored.

## Workflow

- `main` is stable; `develop` is the integration branch.
- Work happens on short-lived `feat/`, `fix/`, `docs/`, `exp/`, and `chore/` branches.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat(eval): ...`, `docs(readme): ...`).
- Feature branches are squash-merged into `develop`; `develop` is merged into
  `main` for releases.

## Milestones

- [x] Milestone 0 — Setup, folder structure, requirements, config
- [x] Milestone 1 — Dataset prep (download, XML → YOLO, split, visualize)
- [x] Milestone 2 — Config + anchor analysis
- [x] Milestone 3 — Training notebook, quick validation, loss/mAP curves
- [x] Milestone 4 — Evaluation pipeline (mAP, PR curves, per-class report)
- [x] Milestone 5 — Inference CLI and Streamlit app
- [ ] Milestone 6 — Full training run, results, final documentation

## Notes

- Python 3.13 with Ultralytics 8.4 and PyTorch 2.14 (CPU) was used for local checks.
- `data/`, `models/*.pt`, `results/runs/`, `results/yolov8_results/`, and
  `results/*.zip` are gitignored.
- Full VOC download is about 2.8 GB; the quick run downloads about 884 MB.

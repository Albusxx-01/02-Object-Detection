# Reproducibility

## Environment

- Python 3.13
- Ultralytics 8.4 and PyTorch 2.14
- GPU for full training; CPU is sufficient for checks and inference

Record the exact versions when reporting new results:

```bash
python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__)"
```

## Deterministic settings

`config/detection.yaml` fixes the run configuration:

- `seed: 42`
- `deterministic: true`
- `augment: false`
- `optimizer: auto`, `lr0: 0.01`, `lrf: 0.01`
- `close_mosaic: 10`

Dataset split lists are written to `data/splits/` by `src/dataset.py`, so the
same images are used for training and validation across runs.

## Run order

```bash
python src/dataset.py
python src/train.py
python src/evaluate.py --weights models/yolov8n_voc_best.pt
python src/loss.py --csv results/results.csv
python src/error_analysis.py --weights models/yolov8n_voc_best.pt --sweep
python tests/smoke_e2e.py --weights models/yolov8n.pt
```

Set `QUICK_RUN = True` in `notebooks/01-data-and-training.ipynb` or pass
`--max-images` to `src/dataset.py` for a small pipeline check. The full VOC run
uses 50 epochs and produces the metrics recorded in `README.md`.

## Expected artifacts

- `models/yolov8n_voc_best.pt` — best checkpoint (gitignored)
- `results/metrics.json` — validation metrics and per-class AP
- `results/BoxPR_curve.png`, `results/confusion_matrix.png` — evaluation plots
- `results/loss_curves.png` — training losses from `results.csv`
- `results/error_analysis.md`, `results/error_analysis.json` — failure analysis
- `results/threshold_sweeps.png`, `results/conf_sweep.png`,
  `results/iou_sweep.png` — confidence and IoU sweep plots

## Notes

- `data/`, `models/*.pt`, `results/runs/`, and `results/yolov8_results/` are
  gitignored. Recreate them with the commands above.
- Full training is hardware-sensitive. Small mAP differences can come from GPU,
  CUDA, and Ultralytics versions even with a fixed seed.

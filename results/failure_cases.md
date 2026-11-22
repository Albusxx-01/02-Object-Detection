# Sample Detections and Failure Cases

Qualitative validation grids from the full 50-epoch YOLOv8n run (best checkpoint,
epoch 49, mAP@0.5 0.826) are stored in `results/samples/`.

| Grid | Labeled | Predicted |
|------|---------|-----------|
| Batch 0 | `val_batch0_labels.jpg` | `val_batch0_pred.jpg` |
| Batch 1 | `val_batch1_labels.jpg` | `val_batch1_pred.jpg` |
| Batch 2 | `val_batch2_labels.jpg` | `val_batch2_pred.jpg` |

Aggregate error breakdown is available in `confusion_matrix.png` and
`confusion_matrix_normalized.png`.

## Full validation error analysis

Full VOC2007 test split (4,952 images, 12,032 ground-truth boxes) with
`models/yolov8n_voc_best.pt` at conf 0.25 and IoU 0.45:

| Metric | Value |
|--------|-------|
| Predictions | 13,175 |
| True positives | 9,860 |
| False positives | 3,315 |
| False negatives | 2,172 |
| Precision | 0.748 |
| Recall | 0.819 |

Recall falls sharply for small objects: tiny 0.630, small 0.751, medium 0.842,
large 0.904.

The most frequent class confusions are chair → sofa (32), cat → dog (20),
dog → cat (19), chair → person (15), person → sofa (13), sofa → chair (13),
motorbike → person (12), and cow → dog/sheep (12 each).

High-confidence false negatives are often duplicate or heavily occluded
instances, such as three missed bottle/chair boxes in `2007_007494.jpg` at
confidence 0.957 and three missed person boxes in `2007_000014.jpg` at 0.945.

Threshold sweeps peak at mAP@0.5:0.95 0.606 for conf 0.05 and 0.617 for IoU 0.7.
The inference defaults (conf 0.25, IoU 0.45) keep fewer boxes for the app while
remaining near that operating point. Full tables are in
`results/error_analysis.md`, the machine-readable payload is in
`results/error_analysis.json`, and the sweep plots are
`results/threshold_sweeps.png`, `results/conf_sweep.png`, and
`results/iou_sweep.png`.

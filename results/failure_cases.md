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

Ranked false positives/false negatives and confidence/IoU threshold sweeps are
planned for the Day 2 error-analysis notebook and will replace this note with
concrete failure cases.

# Evaluation

Metrics for measuring object detection performance across precision, recall, and localization accuracy.

---

## mAP@0.5 (Mean Average Precision)

The primary metric for object detection. Mean of Average Precision (AP) across all classes, where a detection is correct if IoU ≥ 0.5 with a ground truth box.

```
mAP@0.5 = (1/C) Σ AP_c    (averaged over C classes)
```

| mAP Range | Interpretation |
|-----------|---------------|
| < 0.3 | Poor |
| 0.3 – 0.5 | Moderate |
| 0.5 – 0.7 | Good |
| > 0.7 | Excellent |

- **Target**: 60–80% mAP@0.5 on custom datasets

**Official Documentation:**
- [Pascal VOC: AP Calculation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.html)
- [COCO Evaluation API](https://cocodataset.org/#detection-eval)

---

## Average Precision (AP)

Area under the precision-recall curve for a single class. Captures the trade-off between precision and recall across all confidence thresholds.

```
AP = ∫₀¹ P(R) dR    (area under precision-recall curve)
```

- **Interpolation**: Precision at each recall level is interpolated as the max precision achieved at any recall level ≥ current
- AP@0.5 uses IoU threshold 0.5; AP@[.5:.95] averages across IoU thresholds (COCO standard)

**Papers:**
- [Everingham et al., 2010 — The Pascal Visual Object Classes (VOC) Challenge](https://link.springer.com/article/10.1007/s11263-009-0275-4)

---

## Precision-Recall Curve

Plots precision (y-axis) vs recall (x-axis) across all confidence thresholds.

- **Ideal**: Top-right corner (precision=1, recall=1)
- **Area under curve** = AP
- Reveals whether the model trades precision for recall or vice versa

**Official Documentation:**
- [Scikit-learn: Precision-Recall Curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)

---

## Test-Time Augmentation (TTA)

Averaging predictions across multiple augmented versions of the same input image at inference time.

| Augmentation | Purpose |
|-------------|---------|
| Horizontal flip | Capture mirrored detections |
| Multi-scale (e.g., 320, 416, 608) | Detect at multiple resolutions |
| Test-time crop | Cover different image regions |

- Improves mAP by 1–3% at the cost of N× inference time
- Use when accuracy matters more than speed

**Official Documentation:**
- [Ultralytics YOLO: TTA](https://docs.ultralytics.com/modes/predict/#tips)

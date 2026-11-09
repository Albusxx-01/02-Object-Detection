# Post-Processing

Techniques applied after model inference to produce clean, final detections.

---

## Non-Maximum Suppression (NMS)

Suppresses overlapping duplicate detections, keeping only the highest-confidence box for each object.

**Algorithm:**
1. Sort all detections by confidence score (descending)
2. Take the highest-confidence box → add to final results
3. Remove all remaining boxes with IoU ≥ threshold (typically 0.5) with it
4. Repeat until no boxes remain

```
Final Detections = NMS(Predicted Boxes, IoU Threshold=0.5)
```

- **Soft-NMS**: Decays scores instead of hard removal — better for crowded scenes

**Official Documentation:**
- [ torchvision.ops.nms](https://pytorch.org/docs/stable/torchvision.ops.html#torchvision.ops.nms)
- [TensorFlow: tf.image.combined_non_max_suppression](https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression)

**Papers:**
- [Bodla et al., 2017 — Soft-NMS](https://arxiv.org/abs/1704.04503)

---

## Confidence Threshold

Filtering out detections below a minimum confidence score before NMS.

- **Low threshold** (e.g., 0.25): More detections, more false positives
- **High threshold** (e.g., 0.7): Fewer detections, more false negatives
- Trade-off between precision and recall

**Official Documentation:**
- [Ultralytics: Confidence Threshold](https://docs.ultralytics.com/cfg/)

---

## Precision & Recall

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Of all detected objects, how many are correct? |
| **Recall** | TP / (TP + FN) | Of all actual objects, how many were found? |

- **Precision-Recall trade-off**: Higher confidence threshold → higher precision, lower recall
- Plotting P-R curve across thresholds reveals detector behavior

**Official Documentation:**
- [Scikit-learn: Precision-Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)

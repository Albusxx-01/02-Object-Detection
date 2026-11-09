# Loss & Training

Multi-component loss functions and training strategies specific to object detection.

---

## Multi-Part Loss

Object detection requires optimizing multiple objectives simultaneously:

```
L_total = λ₁ · L_classification + λ₂ · L_localization + λ₃ · L_confidence
```

| Loss Component | Purpose | Typical Loss |
|---------------|---------|-------------|
| **Classification** | Correct class per box | Cross-entropy |
| **Localization** | Precise box coordinates | Smooth L1 / CIoU |
| **Confidence/Objectness** | Object vs background | Binary cross-entropy |

- Balancing these losses via weights (λ) is critical for stable training

**Papers:**
- [Redmon et al., 2016 — YOLO: Unified Detection](https://arxiv.org/abs/1506.02640)
- [Lin et al., 2017 — Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---

## Multi-Scale Training

Randomly resizing input images during training (e.g., every 10 batches) to make the model robust to objects at different scales.

- YOLO typically trains at sizes from 320 to 608 pixels
- Improves detection across a range of input resolutions
- At inference, use a fixed resolution for speed

**Official Documentation:**
- [Ultralytics YOLO: Multi-Scale Training](https://docs.ultralytics.com/cfg/)

---

## Mosaic & Mixup Augmentation

Advanced data augmentation techniques combining multiple training images:

### Mosaic Augmentation
Combines 4 training images into one, placing them in a 2×2 grid. Forces the model to learn objects in diverse contexts and improves small object detection.

### Mixup Augmentation
Blends two images and their labels with a mixing ratio:
```
Mixed Image = α × Image_A + (1 - α) × Image_B
```

- Both are standard in YOLOv4/v5/v8 training pipelines
- Significantly improve generalization

**Papers:**
- [Bochkovskiy et al., 2020 — YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
- [Zhang et al., 2018 — mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

---

## Hard Negatives

Background/anchor samples that are challenging for the model — they have high objectness scores but are actually background.

- Cause excessive false positives if not handled
- **Hard negative mining**: Focus training on these difficult examples
- **Focal Loss** (RetinaNet): Down-weights easy negatives automatically

**Papers:**
- [Lin et al., 2017 — Focal Loss for Dense Object Detection (RetinaNet)](https://arxiv.org/abs/1708.02002)

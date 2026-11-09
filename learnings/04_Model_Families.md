# Model Families

The three major architectures for object detection, differing in speed vs accuracy trade-offs.

---

## YOLO (You Only Look Once)

One-stage detector: a single forward pass produces all bounding boxes and class predictions simultaneously.

| Version | Year | Key Feature | Speed |
|---------|------|-------------|-------|
| YOLOv1 | 2016 | Unified detection | 45 FPS |
| YOLOv3 | 2018 | Multi-scale, Darknet-53 | 20–30 FPS |
| YOLOv5 | 2020 | PyTorch, auto-anchor | 30–60 FPS |
| YOLOv8 | 2023 | Anchor-free, SOTA | 30–80 FPS |

- **Strengths**: Real-time inference (30–60 FPS), end-to-end training
- **Weakness**: Struggles with very small objects vs two-stage methods

**Papers:**
- [Redmon et al., 2016 — You Only Look Once](https://arxiv.org/abs/1506.02640)
- [Redmon & Farhadi, 2018 — YOLOv3](https://arxiv.org/abs/1804.02767)

**Official Documentation:**
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)

---

## Faster R-CNN

Two-stage detector: first generates region proposals, then classifies and refines each proposal.

```
Image → Backbone → FPN → Region Proposal Network (RPN) → ROI Pooling → Classifier + Box Refiner
```

| Stage | Function |
|-------|----------|
| **Stage 1**: RPN | Proposes candidate bounding boxes (objectness) |
| **Stage 2**: Head | Classifies proposals + refines box coordinates |

- **Strengths**: High accuracy, especially for small objects
- **Weakness**: Slower than one-stage methods (5–15 FPS)

**Papers:**
- [Ren et al., 2015 — Faster R-CNN](https://arxiv.org/abs/1506.01497)

**Official Documentation:**
- [Torchvision: Faster R-CNN](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)

---

## SSD (Single Shot MultiBox Detector)

One-stage detector using default boxes across multiple feature map levels at different resolutions.

- Predicts at multiple scales from intermediate layers
- Faster than Faster R-CNN, less accurate
- Largely superseded by YOLO and RetinaNet

**Papers:**
- [Liu et al., 2016 — SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

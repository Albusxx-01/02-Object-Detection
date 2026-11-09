# Multi-Scale Detection

Techniques for detecting objects of varying sizes by combining features from multiple resolution levels.

---

## Feature Pyramid Network (FPN)

Combines low-resolution (semantic, high-level) and high-resolution (detailed, low-level) features to create a pyramid of feature maps. Enables detection of both small and large objects.

```
Backbone (ResNet)
  C2 → P2 (high res, low semantics)  → detect small objects
  C3 → P3
  C4 → P4
  C5 → P5 (low res, high semantics)  → detect large objects
```

- **Top-down pathway**: Upsamples deeper features and merges with shallower ones via lateral connections
- **Result**: Every level has both semantic richness and spatial detail

**Papers:**
- [Lin et al., 2017 — Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

**Official Documentation:**
- [Torchvision: FPN](https://pytorch.org/vision/stable/models.html#object-detection)

---

## Backbone

The base feature extractor network that processes the input image into feature maps.

| Backbone | Used In | Key Feature |
|----------|---------|-------------|
| Darknet-53 | YOLOv3 | Custom lightweight architecture |
| CSPDarknet | YOLOv4/v5 | Cross-stage partial connections |
| ResNet-50/101 | Faster R-CNN, RetinaNet | Residual connections |
| EfficientNet | Various | Compound scaling |

- Pretrained on ImageNet; fine-tuned for detection

**Official Documentation:**
- [Torchvision: Pretrained Backbones](https://pytorch.org/vision/stable/models.html)

---

## Neck

Feature aggregation module between backbone and head. Combines multi-scale features from the backbone.

| Neck Type | Architecture |
|-----------|-------------|
| **FPN** | Top-down pyramid |
| **PANet** | FPN + additional bottom-up pathway |
| **BiFPN** | Weighted bidirectional FPN (EfficientDet) |

- Provides the head with a rich set of multi-scale features

**Papers:**
- [Liu et al., 2018 — PANet: Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534)

---

## Head

Final prediction layers that output bounding boxes, class probabilities, and confidence scores from the neck's feature maps.

- **Two-stage heads** (Faster R-CNN): Region Classification + box refinement
- **One-stage heads** (YOLO, SSD): Direct prediction of boxes + classes at each spatial position

```
Input → Backbone → Neck → Head → [boxes, classes, confidence]
```

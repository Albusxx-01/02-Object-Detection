# Anchor Boxes

Pre-defined box templates that serve as reference predictions for the detection network.

---

## Anchor Boxes

Pre-defined box templates at various scales and aspect ratios placed at every spatial position in the feature map. The network predicts offsets relative to these anchors rather than absolute box coordinates.

- **Purpose**: Provides a prior about typical object shapes, making regression easier
- **Number**: Typically 3–5 anchors per grid cell (e.g., 3 scales × 3 ratios = 9)
- **Generation**: Can be hand-picked or data-driven (anchor clustering)

**Papers:**
- [Ren et al., 2015 — Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
- [Lin et al., 2017 — Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)

---

## Anchor Clustering

Using k-means clustering on ground truth bounding box dimensions `(w, h)` from the training set to find optimal anchor shapes.

- **Why**: Generic anchors may not fit your dataset's object shapes
- **Distance metric**: Often `1 - IoU` instead of Euclidean distance
- **YOLO**: Uses anchor clustering via k-means on training boxes

**Papers:**
- [Redmon & Farhadi, 2018 — YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

---

## Positive / Negative Anchors

During training, anchors are assigned labels based on IoU with ground truth boxes:

| Condition | Label | Role |
|-----------|-------|------|
| IoU ≥ 0.7 with any GT box | **Positive** | Responsible for predicting that object |
| IoU < 0.3 with all GT boxes | **Negative (background)** | Should predict "no object" |
| 0.3 ≤ IoU < 0.7 | **Ignore** | Not used in loss computation |

- **Positive anchors**: Train the classifier and box regressor
- **Negative anchors**: Train the objectness classifier to reject background
- Class imbalance is managed via sampling (e.g., 1:3 positive-to-negative ratio)

**Official Documentation:**
- [Detectron2: Anchor Generation](https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.AnchorsBase)

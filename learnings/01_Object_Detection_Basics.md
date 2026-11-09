# Object Detection Basics

Fundamental concepts that distinguish object detection from image classification and define how objects are localized in images.

---

## Object Detection vs Classification

| Task | Output | Scope |
|------|--------|-------|
| **Classification** | Single label for entire image | What is in the image? |
| **Object Detection** | Multiple bounding boxes + labels | What, where, and how many objects? |

Detection must simultaneously **classify** and **localize** every object instance.

**Official Documentation:**
- [TensorFlow: Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [PyTorch: Torchvision Detection Models](https://pytorch.org/vision/stable/models/detection.html)

---

## Bounding Box

A rectangle `(x, y, width, height)` enclosing an detected object. Formats vary:

| Format | Description |
|--------|-------------|
| `(x, y, w, h)` | Top-left corner + dimensions (YOLO) |
| `(x_min, y_min, x_max, y_max)` | Two corners (COCO, Pascal VOC) |
| `(cx, cy, w, h)` | Center + dimensions (some frameworks) |

**Papers:**
- [Redmon et al., 2016 — You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

---

## Bounding Box Regression

Predicting offsets to adjust pre-defined or initial boxes to tightly fit objects. The network predicts `(Δx, Δy, Δw, Δh)` corrections rather than absolute coordinates.

- Regression is more stable than predicting absolute coordinates
- Applied to anchor boxes or region proposals

**Papers:**
- [Ren et al., 2015 — Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

---

## Intersection over Union (IoU)

Measures overlap between predicted and ground truth boxes:

```
IoU = Area(Overlap) / Area(Union)
```

| IoU Value | Interpretation |
|-----------|---------------|
| 1.0 | Perfect overlap |
| ≥ 0.5 | Generally considered a correct detection |
| 0.0 | No overlap |

- Used for **matching** predictions to ground truth during training
- Used for **Non-Maximum Suppression** to remove duplicates

**Official Documentation:**
- [Pascal VOC Challenge: IoU Definition](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.html)

---

## Objectness / Confidence Score

The likelihood that a predicted box contains **any** object (vs background).

- Range: [0, 1]
- High objectness = model is confident something is there
- Combined with class probabilities for final confidence score

---

## Class Probability

Conditional likelihood of each class **given** that the box contains an object.

```
Final Confidence = Objectness × Class Probability
```

- Softmax over all class scores for each box
- Only meaningful when objectness is high

**Official Documentation:**
- [CS231n: Object Detection (Stanford)](https://cs231n.github.io/detection/)

# Applications Context

Real-world domains where object detection is deployed.

---

## Autonomous Vehicles

Detecting and tracking pedestrians, vehicles, traffic signs, and lane markings in real-time.

- Requires real-time inference (30+ FPS)
- Safety-critical: high recall is essential
- Multi-class detection with strict latency requirements

**Datasets:** [KITTI](https://www.cvlibs.net/datasets/kitti/), [nuScenes](https://www.nuscenes.org/), [Waymo Open Dataset](https://waymo.com/open/)

---

## Surveillance & Security

Intrusion detection, crowd monitoring, and anomaly detection in video feeds.

- Often runs on edge devices (Jetson, Raspberry Pi)
- Requires efficient models (YOLO-nano, MobileNet-based detectors)
- Challenges: occlusion, low-light, varying camera angles

---

## Retail Inventory Management

Automated shelf monitoring, product counting, and stock-level tracking.

- Detecting specific product types and quantities
- Handling occluded objects on shelves
- Real-time alerts for out-of-stock items

---

## Sports Analytics & Player Tracking

Detecting players, balls, and equipment for performance analysis.

- Multi-object tracking across frames
- Action recognition from detected poses
- Re-identification across camera views

**Datasets:** [SportsBall Dataset](https://github.com/everstonedream/sportsball-dataset), [DeepSport](https://basecamp.inria.fr/deepsport/)

---

## Additional Applications

| Domain | Use Case |
|--------|----------|
| **Medical** | Tumor/cell detection in histopathology |
| **Agriculture** | Crop disease and pest detection |
| **Manufacturing** | Defect detection on assembly lines |
| **Construction** | Safety equipment compliance monitoring |

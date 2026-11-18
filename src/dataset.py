"""Pascal VOC download, XML-to-YOLO conversion, splitting, and visualization."""

from __future__ import annotations

import argparse
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from PIL import Image, ImageDraw

from config import CLASS_NAMES, DATA_DIR

VOC_URLS = {
    "VOCtrainval_06-Nov-2007.zip": (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        "VOCtrainval_06-Nov-2007.zip"
    ),
    "VOCtest_06-Nov-2007.zip": (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        "VOCtest_06-Nov-2007.zip"
    ),
    "VOCtrainval_11-May-2012.zip": (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        "VOCtrainval_11-May-2012.zip"
    ),
}

VOC2007_URLS = {
    name: url for name, url in VOC_URLS.items() if "2007" in name
}

OFFICIAL_SPLITS = {
    "train": [
        ("2012", "train"),
        ("2012", "val"),
        ("2007", "train"),
        ("2007", "val"),
    ],
    "val": [("2007", "test")],
}

QUICK_SPLITS = {
    "train": [("2007", "train"), ("2007", "val")],
    "val": [("2007", "test")],
}

COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
]


def _download_file(url: str, destination: Path) -> Path:
    """Download a file once, showing a simple progress percentage."""
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name} ...")

    def report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percent = downloaded / total_size * 100
        print(f"\r  {percent:5.1f}%  ({downloaded / 1e6:,.0f} MB)", end="")

    urllib.request.urlretrieve(url, destination, reporthook=report)
    print()
    return destination


def download_voc(
    data_dir: Path = DATA_DIR,
    include_2012: bool = True,
    force: bool = False,
) -> Path:
    """Download and extract Pascal VOC archives; return the VOCdevkit root."""
    data_dir = Path(data_dir)
    downloads_dir = data_dir / "downloads"
    devkit_dir = data_dir / "VOCdevkit"
    urls = VOC_URLS if include_2012 else VOC2007_URLS

    for archive_name, url in urls.items():
        archive_path = downloads_dir / archive_name
        if force and archive_path.exists():
            archive_path.unlink()
        _download_file(url, archive_path)

        marker = devkit_dir / f".{archive_name}.extracted"
        if not marker.exists():
            print(f"Extracting {archive_name} ...")
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(data_dir)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()

    return devkit_dir


def _convert_box(size: tuple[int, int], box: list[float]) -> tuple[float, ...]:
    """Convert a VOC pixel box to normalized YOLO center/width/height."""
    width, height = size
    x_min, x_max, y_min, y_max = box
    x_center = ((x_min + x_max) / 2.0 - 1.0) / width
    y_center = ((y_min + y_max) / 2.0 - 1.0) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    return x_center, y_center, box_width, box_height


def convert_annotation(
    xml_path: Path,
    label_path: Path,
    class_names: list[str] = CLASS_NAMES,
) -> int:
    """Convert one VOC XML annotation file to a YOLO label file."""
    xml_path = Path(xml_path)
    label_path = Path(label_path)

    root = ET.parse(xml_path).getroot()
    size_node = root.find("size")
    image_size = (int(size_node.find("width").text), int(size_node.find("height").text))

    lines = []
    for obj in root.iter("object"):
        class_name = obj.find("name").text
        if class_name not in class_names:
            continue

        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue

        box_node = obj.find("bndbox")
        box = [
            float(box_node.find(key).text)
            for key in ("xmin", "xmax", "ymin", "ymax")
        ]
        coordinates = _convert_box(image_size, box)
        coordinates = [min(max(value, 0.0), 1.0) for value in coordinates]

        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            continue

        class_id = class_names.index(class_name)
        values = [str(class_id)] + [f"{value:.6f}" for value in coordinates]
        lines.append(" ".join(values))

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    return len(lines)


def convert_subset(
    voc_root: Path,
    data_dir: Path,
    year: str,
    image_set: str,
    split: str,
    class_names: list[str] = CLASS_NAMES,
    max_images: int | None = None,
) -> int:
    """Convert one VOC year/image-set into a split of the project dataset."""
    voc_root = Path(voc_root)
    data_dir = Path(data_dir)
    year_dir = voc_root / f"VOC{year}"

    ids_path = year_dir / "ImageSets" / "Main" / f"{image_set}.txt"
    image_ids = ids_path.read_text(encoding="utf-8").strip().split()
    if max_images is not None:
        image_ids = image_ids[:max_images]

    images_out = data_dir / "images" / split
    labels_out = data_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_id in image_ids:
        source_image = year_dir / "JPEGImages" / f"{image_id}.jpg"
        source_xml = year_dir / "Annotations" / f"{image_id}.xml"
        if not source_image.exists():
            continue

        stem = f"{year}_{image_id}"
        shutil.copy2(source_image, images_out / f"{stem}.jpg")

        destination_label = labels_out / f"{stem}.txt"
        if source_xml.exists():
            convert_annotation(source_xml, destination_label, class_names)
        else:
            destination_label.write_text("", encoding="utf-8")

        count += 1

    return count


def _reset_split_dirs(data_dir: Path, splits: list[str]) -> None:
    """Remove previously generated images/labels for the given splits."""
    for split in splits:
        for folder in ("images", "labels"):
            directory = Path(data_dir) / folder / split
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)


def save_split_lists(data_dir: Path = DATA_DIR) -> dict[str, Path]:
    """Write image file lists per split for reproducibility."""
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for split in ("train", "val", "test"):
        images_dir = data_dir / "images" / split
        if not images_dir.exists():
            continue
        files = sorted(
            path.relative_to(data_dir).as_posix()
            for path in images_dir.glob("*.jpg")
        )
        out_path = splits_dir / f"{split}.txt"
        out_path.write_text("\n".join(files) + "\n", encoding="utf-8")
        outputs[split] = out_path

    return outputs


def prepare_voc_dataset(
    data_dir: Path = DATA_DIR,
    class_names: list[str] = CLASS_NAMES,
    max_images: int | None = None,
    include_2012: bool = True,
    force: bool = False,
    clean: bool = True,
) -> dict[str, int]:
    """Download VOC, convert labels, split images, and save split lists."""
    data_dir = Path(data_dir)
    splits = OFFICIAL_SPLITS if include_2012 else QUICK_SPLITS

    if clean:
        _reset_split_dirs(data_dir, list(splits))

    devkit_dir = download_voc(data_dir, include_2012=include_2012, force=force)

    summary = {}
    for split, subsets in splits.items():
        remaining = max_images
        total = 0
        for year, image_set in subsets:
            converted = convert_subset(
                devkit_dir,
                data_dir,
                year,
                image_set,
                split,
                class_names,
                max_images=remaining,
            )
            total += converted
            if remaining is not None:
                remaining -= converted
                if remaining <= 0:
                    break
        summary[split] = total
        print(f"{split}: {total} images")

    save_split_lists(data_dir)
    return summary


def load_label_file(label_path: Path) -> np.ndarray:
    """Load a YOLO label file as an (N, 5) array: class, xc, yc, w, h."""
    label_path = Path(label_path)
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.loadtxt(label_path, ndmin=2, dtype=np.float32)


def dataset_stats(
    data_dir: Path = DATA_DIR,
    split: str = "train",
    class_names: list[str] = CLASS_NAMES,
) -> dict[str, Any]:
    """Compute class counts, boxes-per-image, and normalized box sizes."""
    labels_dir = Path(data_dir) / "labels" / split
    label_files = sorted(labels_dir.glob("*.txt"))

    class_counts = Counter()
    boxes_per_image = []
    box_sizes = []
    images_with_objects = 0

    for label_file in label_files:
        labels = load_label_file(label_file)
        boxes_per_image.append(len(labels))
        if len(labels) > 0:
            images_with_objects += 1
        for class_id, _, _, width, height in labels:
            class_counts[class_names[int(class_id)]] += 1
            box_sizes.append((float(width), float(height)))

    sizes = np.array(box_sizes) if box_sizes else np.zeros((0, 2))
    max_boxes = int(max(boxes_per_image, default=0))
    counts = [boxes_per_image.count(value) for value in range(max_boxes + 1)]

    return {
        "split": split,
        "images": len(label_files),
        "images_with_objects": images_with_objects,
        "boxes": int(sum(boxes_per_image)),
        "boxes_per_image_mean": (
            float(np.mean(boxes_per_image)) if boxes_per_image else 0.0
        ),
        "boxes_per_image_max": max_boxes,
        "boxes_per_image_histogram": counts,
        "class_counts": {
            name: int(class_counts.get(name, 0)) for name in class_names
        },
        "box_sizes": sizes,
    }


def draw_boxes(
    image_path: Path,
    label_path: Path,
    class_names: list[str] = CLASS_NAMES,
) -> Image.Image:
    """Draw YOLO boxes and class names on an image."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)

    for (
        class_id,
        x_center,
        y_center,
        box_width,
        box_height,
    ) in load_label_file(label_path):
        class_id = int(class_id)
        x1 = (x_center - box_width / 2.0) * width
        y1 = (y_center - box_height / 2.0) * height
        x2 = (x_center + box_width / 2.0) * width
        y2 = (y_center + box_height / 2.0) * height

        color = COLORS[class_id % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = class_names[class_id]
        text_box = draw.textbbox((x1, y1), label)
        draw.rectangle(
            [text_box[0], text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
            fill=color,
        )
        draw.text((x1, y1 - 12), label, fill="white")

    return image


def visualize_sample(
    image_path: Path,
    label_path: Path,
    class_names: list[str] = CLASS_NAMES,
    ax: Axes | None = None,
) -> Axes:
    """Plot one image with its boxes, optionally on an existing axis."""
    image = draw_boxes(image_path, label_path, class_names)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.imshow(image)
    ax.set_title(Path(image_path).name)
    ax.axis("off")
    return ax


def preview_samples(
    data_dir: Path = DATA_DIR,
    split: str = "train",
    num_samples: int = 6,
    class_names: list[str] = CLASS_NAMES,
    seed: int = 42,
) -> plt.Figure:
    """Plot a grid of random samples with boxes overlaid."""
    data_dir = Path(data_dir)
    images = sorted((data_dir / "images" / split).glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found in {data_dir / 'images' / split}")

    rng = random.Random(seed)
    selected = rng.sample(images, min(num_samples, len(images)))

    cols = 3
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axes = np.atleast_1d(axes).ravel()

    for ax, image_path in zip(axes, selected):
        label_path = data_dir / "labels" / split / f"{image_path.stem}.txt"
        visualize_sample(image_path, label_path, class_names, ax=ax)

    for ax in axes[len(selected):]:
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_dataset_stats(
    stats: dict,
    class_names: list[str] = CLASS_NAMES,
) -> plt.Figure:
    """Plot class distribution and normalized box-size scatter."""
    fig, (ax_counts, ax_sizes) = plt.subplots(1, 2, figsize=(14, 5))

    names = list(class_names)
    counts = [stats["class_counts"][name] for name in names]
    ax_counts.barh(names, counts, color="#3b82f6")
    ax_counts.invert_yaxis()
    ax_counts.set_xlabel("Instances")
    ax_counts.set_title(f"Class distribution ({stats['split']})")
    ax_counts.grid(axis="x", alpha=0.3)

    sizes = stats["box_sizes"]
    if len(sizes) > 0:
        ax_sizes.scatter(sizes[:, 0], sizes[:, 1], s=8, alpha=0.25, color="#ef4444")
    ax_sizes.set_xlabel("Normalized width")
    ax_sizes.set_ylabel("Normalized height")
    ax_sizes.set_title("Box sizes")
    ax_sizes.set_xlim(0, 1)
    ax_sizes.set_ylim(0, 1)
    ax_sizes.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Pascal VOC dataset.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap images per split for a quick run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only VOC2007 and the quick split.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download archives.")
    args = parser.parse_args()

    summary = prepare_voc_dataset(
        data_dir=args.data_dir,
        max_images=args.max_images,
        include_2012=not args.quick,
        force=args.force,
    )
    print("Done:", summary)


if __name__ == "__main__":
    main()

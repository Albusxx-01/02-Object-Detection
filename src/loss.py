"""Plot training losses from an Ultralytics results.csv file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import RESULTS_DIR

LOSS_GROUPS = {
    "box loss": ("train/box_loss", "val/box_loss"),
    "cls loss": ("train/cls_loss", "val/cls_loss"),
    "dfl loss": ("train/dfl_loss", "val/dfl_loss"),
}


def load_history(csv_path: str | Path) -> pd.DataFrame:
    """Load an Ultralytics results.csv and normalise its column names."""
    history = pd.read_csv(csv_path)
    history.columns = [column.strip() for column in history.columns]
    return history


def plot_losses(
    csv_path: str | Path = RESULTS_DIR / "results.csv",
    out_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Plot train/val box, classification, and DFL losses side by side."""
    csv_path = Path(csv_path)
    history = load_history(csv_path)

    out_path = (
        Path(out_path) if out_path is not None else RESULTS_DIR / "loss_curves.png"
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for ax, (group, columns) in zip(axes, LOSS_GROUPS.items()):
        for column in columns:
            if column not in history.columns:
                continue
            split = column.split("/", 1)[0]
            ax.plot(history["epoch"], history[column], label=split)
        ax.set_title(group.title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(title or csv_path.stem)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot YOLO training losses.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_DIR / "results.csv",
        help="Ultralytics results.csv path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: results/loss_curves.png).",
    )
    args = parser.parse_args()

    output = plot_losses(args.csv, args.out)
    print("Saved:", output)


if __name__ == "__main__":
    main()

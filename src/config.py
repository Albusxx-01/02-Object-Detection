"""Global configuration for the Object Detection project."""

from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

DETECTION_CONFIG_PATH = CONFIG_DIR / "detection.yaml"

with open(DETECTION_CONFIG_PATH, "r", encoding="utf-8") as config_file:
    DETECTION = yaml.safe_load(config_file)

_names = DETECTION["names"]
CLASS_NAMES = [str(_names[index]) for index in sorted(_names)]
NUM_CLASSES = len(CLASS_NAMES)

TRAINING = DETECTION.get("training", {})
INFERENCE = DETECTION.get("inference", {})
SEED = int(TRAINING.get("seed", 42))

DATASET_YAML_PATH = DATA_DIR / "dataset.yaml"


def make_dirs() -> None:
    """Create output directories if they do not exist."""
    for directory in (
        DATA_DIR / "images" / "train",
        DATA_DIR / "images" / "val",
        DATA_DIR / "labels" / "train",
        DATA_DIR / "labels" / "val",
        DATA_DIR / "splits",
        MODELS_DIR,
        RESULTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(out_path: Path = DATASET_YAML_PATH) -> Path:
    """Write an Ultralytics-compatible dataset YAML with an absolute path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "path": str(DATA_DIR.resolve()),
        "train": DETECTION["train"],
        "val": DETECTION["val"],
        "test": DETECTION.get("test", DETECTION["val"]),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }

    with open(out_path, "w", encoding="utf-8") as out_file:
        yaml.safe_dump(payload, out_file, sort_keys=False)

    return out_path


def get_training_args() -> dict:
    """Return the training block with project/name resolved to absolute paths."""
    args = dict(TRAINING)
    project = Path(args.get("project", "results/runs"))
    if not project.is_absolute():
        project = ROOT_DIR / project
    args["project"] = str(project.resolve())
    args["name"] = args.get("name", "yolov8n_voc")
    return args

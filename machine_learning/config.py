from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    labels_csv_path: Path = Path("learning_data/20260127_202651/labels.csv")
    images_dir: Path = Path("learning_data/20260127_202651")
    drive_state_default: str = "READY"
    csv_path: Path = Path("machine_learning/data/dataset_k1.csv")
    # Servo 3-class labels (order matters for class id).
    servo_class_us: tuple[int, ...] = (1100, 1500, 1900)
    servo_class_names: tuple[str, ...] = ("LEFT", "STRAIGHT", "RIGHT")
    drive_state_order: tuple[str, ...] = (
        "READY",
        "MOVING",
        "BLOCKED_FRONT",
        "BLOCKED_REAR",
        "BLOCKED_BOTH",
        "EMERGENCY_STOP",
    )
    k: int = 1
    val_fraction: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 8
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_move: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    log_every: int = 20
    checkpoint_dir: Path = Path("machine_learning/checkpoints/ver_02")


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()

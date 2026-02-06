from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    dataset_repo_id: str = "ShonanGarage/Automation_Car_Dataset"
    dataset_revision: str = "main"
    dataset_local_dir: Path = Path("machine_learning/data/hf_cache")
    csv_path: Path = Path("machine_learning/data/dataset_0206_k3.csv")
    # Servo 3-class labels (order matters for class id).
    servo_class_us: tuple[int, ...] = (1100, 1500, 1900)
    servo_class_names: tuple[str, ...] = ("LEFT", "STRAIGHT", "RIGHT")
    k: int = 3
    val_fraction: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lambda_move: float = 1.0
    num_workers: int = 6
    device: str = "auto"
    log_every: int = 20
    checkpoint_dir: Path = Path("machine_learning/checkpoints/ver_15_k3")


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()

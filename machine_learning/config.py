from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    jsonl_path: Path = Path("AutomationCar_Orge/learning_data/log.jsonl")
    images_dir: Path = Path("AutomationCar_Orge/learning_data/images")
    csv_path: Path = Path("machine_learning/data/dataset_k1.csv")
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
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_move: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    log_every: int = 20
    checkpoint_path: Path = Path("machine_learning/checkpoints/last.pt")


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()

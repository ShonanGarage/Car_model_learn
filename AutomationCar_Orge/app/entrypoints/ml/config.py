from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MlRunConfig:
    checkpoint_path: Path = Path("machine_learning/checkpoints/ver_01/best.pt")
    data_csv_path: Path = Path("machine_learning/data/dataset_k1.csv")
    device: str = "auto"
    update_hz: int = 10

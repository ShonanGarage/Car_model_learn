from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MlRunConfig:
    checkpoint_path: Path = Path("../machine_learning/checkpoints/ver_18_k5/best.pt")
    throttle_low_us: int = 1300
    throttle_high_us: int = 1700
    device: str = "auto"
    update_hz: int = 10

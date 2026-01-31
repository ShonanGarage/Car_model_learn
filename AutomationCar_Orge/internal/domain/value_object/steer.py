from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config.settings import Settings

@dataclass(frozen=True)
class Steer:
    pulse_width_us: int

    def __post_init__(self):
        # 一般的なサーボの範囲（1000-2000）を想定
        if self.pulse_width_us < 1000 or self.pulse_width_us > 2000:
            raise ValueError(f"Invalid steer pulse width: {self.pulse_width_us}")

    @classmethod
    def set_neutral(cls) -> "Steer":
        return cls(1500)

    @classmethod
    def from_us(cls, value: int, settings: "Settings") -> "Steer":
        lo = settings.servo.min_us
        hi = settings.servo.max_us
        clamped = max(lo, min(hi, value))
        return cls(clamped)

    @property
    def value(self) -> int:
        return self.pulse_width_us

    def is_center(self) -> bool:
        return self.pulse_width_us == 1500

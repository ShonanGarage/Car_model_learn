from dataclasses import dataclass

@dataclass(frozen=True)
class Steer:
    pulse_width_us: int

    def __post_init__(self):
        # 一般的なサーボの範囲（1000-2000）を想定
        if self.pulse_width_us < 1000 or self.pulse_width_us > 2000:
            raise ValueError(f"Invalid steer pulse width: {self.pulse_width_us}")

    @property
    def value(self) -> int:
        return self.pulse_width_us

    def is_center(self) -> bool:
        return self.pulse_width_us == 1500

from dataclasses import dataclass

@dataclass(frozen=True)
class Throttle:
    pulse_width_us: int
    
    def __post_init__(self):
        # 0は停止として許可し、それ以外は一般的なESCの範囲（1000-2000）を想定
        # ただし、0以外の正の値のみバリデーション
        if self.pulse_width_us != 0 and (self.pulse_width_us < 1000 or self.pulse_width_us > 2000):
            raise ValueError(f"Invalid throttle pulse width: {self.pulse_width_us}")

    @property
    def value(self) -> int:
        return self.pulse_width_us

    def is_forward(self) -> bool:
        return self.pulse_width_us > 1500

    def is_backward(self) -> bool:
        return 0 < self.pulse_width_us < 1500

    def is_stop(self) -> bool:
        return self.pulse_width_us == 0 or self.pulse_width_us == 1500

    @classmethod
    def stop(cls) -> "Throttle":
        return cls(0)

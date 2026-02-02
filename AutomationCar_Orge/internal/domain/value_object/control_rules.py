from dataclasses import dataclass


@dataclass(frozen=True)
class ControlRules:
    steer_full_time_s: float
    steer_step_us: int
    steer_accel_step_per_s: int
    steer_max_step: int
    steer_center_us: int
    steer_min_us: int
    steer_max_us: int

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyRules:
    emergency_stop_threshold_m: float
    blocked_threshold_m: float

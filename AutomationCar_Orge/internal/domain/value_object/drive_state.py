from enum import Enum, auto

from .safety_rules import SafetyRules
from .sonar_frame import SonarFrame

class DriveState(Enum):
    READY = auto()
    MOVING = auto()
    BLOCKED_FRONT = auto()
    BLOCKED_REAR = auto()
    BLOCKED_BOTH = auto()
    EMERGENCY_STOP = auto()

    @classmethod
    def from_sonar_frame(
        cls,
        frame: SonarFrame,
        current_state: "DriveState",
        rules: SafetyRules,
    ) -> "DriveState":
        """Determine the safe state based on current distances.
        """
        if frame.is_empty():
            return current_state

        front_emergency = any(
            0 < d < rules.emergency_stop_threshold_m for d in frame.front
        )
        if front_emergency:
            return cls.EMERGENCY_STOP

        front_blocked = any(
            0 < d < rules.blocked_threshold_m for d in frame.front
        )
        rear_blocked = any(
            0 < d < rules.blocked_threshold_m for d in frame.rear
        )

        if front_blocked and rear_blocked:
            return cls.BLOCKED_BOTH
        elif front_blocked:
            return cls.BLOCKED_FRONT
        elif rear_blocked:
            return cls.BLOCKED_REAR
        
        # 危険がない場合、BLOCKED系からはREADYに戻る
        if current_state.is_blocked():
            return cls.READY
            
        return current_state

    def is_emergency_stop(self) -> bool:
        return self == self.EMERGENCY_STOP

    def is_ready(self) -> bool:
        return self == self.READY

    def is_moving(self) -> bool:
        return self == self.MOVING

    def is_blocked_front(self) -> bool:
        return self == self.BLOCKED_FRONT

    def is_blocked_rear(self) -> bool:
        return self == self.BLOCKED_REAR

    def is_blocked_both(self) -> bool:
        return self == self.BLOCKED_BOTH

    def is_blocked(self) -> bool:
        return self in (self.BLOCKED_FRONT, self.BLOCKED_REAR, self.BLOCKED_BOTH)

    @classmethod
    def ready(cls) -> "DriveState":
        return cls.READY

    @classmethod
    def moving(cls) -> "DriveState":
        return cls.MOVING

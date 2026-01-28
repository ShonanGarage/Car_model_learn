from enum import Enum, auto
from typing import List

from app.config.settings import SETTINGS

class DriveState(Enum):
    READY = auto()
    MOVING = auto()
    BLOCKED_FRONT = auto()
    BLOCKED_REAR = auto()
    BLOCKED_BOTH = auto()
    EMERGENCY_STOP = auto()

    @classmethod
    def from_distances(
        cls,
        distances: List[float],
        current_state: "DriveState",
    ) -> "DriveState":
        """Determine the safe state based on current distances.
        Indices: 0: center, 1: front_left, 2: front_right, 3: rear_left, 4: rear_right
        """
        if not distances or len(distances) < 5:
            return current_state

        front_indices = [0, 1, 2]
        rear_indices = [3, 4]
        
        front_emergency = any(
            0 < distances[i] < SETTINGS.emergency_stop_threshold_m for i in front_indices
        )
        if front_emergency:
            return cls.EMERGENCY_STOP

        front_blocked = any(
            0 < distances[i] < SETTINGS.blocked_threshold_m for i in front_indices
        )
        rear_blocked = any(
            0 < distances[i] < SETTINGS.blocked_threshold_m for i in rear_indices
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

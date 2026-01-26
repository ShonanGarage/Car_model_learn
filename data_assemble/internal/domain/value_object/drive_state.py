from enum import Enum, auto
from typing import List

class DriveState(Enum):
    READY = auto()
    MOVING = auto()
    BLOCKED_FRONT = auto()
    BLOCKED_REAR = auto()
    BLOCKED_BOTH = auto()
    EMERGENCY_STOP = auto()

    @classmethod
    def from_distances(cls, distances: List[float], current_state: "DriveState") -> "DriveState":
        """Determine the safe state based on current distances.
        Indices: 0: center, 1: front_left, 2: front_right, 3: rear_left, 4: rear_right
        """
        if not distances or len(distances) < 5:
            return current_state

        front_indices = [0, 1, 2]
        rear_indices = [3, 4]
        
        front_blocked = any(0 < distances[i] < 0.2 for i in front_indices)
        rear_blocked = any(0 < distances[i] < 0.2 for i in rear_indices)

        if front_blocked and rear_blocked:
            return cls.BLOCKED_BOTH
        elif front_blocked:
            return cls.BLOCKED_FRONT
        elif rear_blocked:
            return cls.BLOCKED_REAR
        
        # 危険がない場合、BLOCKED系からはREADYに戻る
        if current_state in [cls.BLOCKED_FRONT, cls.BLOCKED_REAR, cls.BLOCKED_BOTH]:
            return cls.READY
            
        return current_state

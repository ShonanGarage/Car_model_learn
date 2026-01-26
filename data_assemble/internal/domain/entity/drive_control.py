from dataclasses import dataclass
from typing import List
from ..value_object.throttle import Throttle
from ..value_object.steer import Steer
from ..value_object.drive_state import DriveState

@dataclass
class DriveControl:
    throttle: Throttle
    steer: Steer
    state: DriveState = DriveState.READY

    @classmethod
    def create_default(cls, steer_us: int = 1500) -> "DriveControl":
        return cls(
            throttle=Throttle.stop(),
            steer=Steer(steer_us),
            state=DriveState.READY
        )

    def update_state(self, distances: List[float]) -> None:
        """Update state based on sensor data."""
        self.state = DriveState.from_distances(distances, self.state)
        # 進行方向がブロックされたなら止める
        if self.state == DriveState.BLOCKED_BOTH:
            self.stop()
        elif self.state == DriveState.BLOCKED_FRONT and self.throttle.is_forward():
            self.stop()
        elif self.state == DriveState.BLOCKED_REAR and self.throttle.is_backward():
            self.stop()

    def update_throttle(self, us: int, distances: List[float]) -> None:
        """Update throttle with validation against current state.
        Allows movement if the direction is clear.
        """
        # 走行前に障害物チェック
        self.update_state(distances)
        
        new_throttle = Throttle(us)
        
        # ブロック状態に応じたチェック
        if self.state == DriveState.BLOCKED_BOTH and not new_throttle.is_stop():
            self.stop()
            return
        
        if self.state == DriveState.BLOCKED_FRONT and new_throttle.is_forward():
            self.stop()
            return
            
        if self.state == DriveState.BLOCKED_REAR and new_throttle.is_backward():
            self.stop()
            return

        self.throttle = new_throttle
        if not self.throttle.is_stop():
            self.state = DriveState.MOVING
        elif self.state not in [DriveState.BLOCKED_FRONT, DriveState.BLOCKED_REAR, DriveState.BLOCKED_BOTH]:
            self.state = DriveState.READY

    def update_steer(self, us: int) -> None:
        self.steer = Steer(us)

    def stop(self) -> None:
        self.throttle = Throttle.stop()
        if self.state not in [DriveState.BLOCKED_FRONT, DriveState.BLOCKED_REAR, DriveState.BLOCKED_BOTH]:
            self.state = DriveState.READY

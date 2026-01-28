from dataclasses import dataclass, replace
from typing import List

from ..value_object.throttle import Throttle
from ..value_object.steer import Steer
from ..value_object.drive_state import DriveState

# 初期化用ヘルパー関数
def boot_vehicle_motion_ready(
    throttle: Throttle | None = None,
    steer: Steer | None = None,
) -> "VehicleMotion":
    if throttle is None:
        throttle = Throttle.stop()
    if steer is None:
        steer = Steer.set_neutral()
    return VehicleMotion(
        throttle=throttle,
        steer=steer,
        state=DriveState.ready(),
    )

@dataclass
class VehicleMotion:
    throttle: Throttle
    steer: Steer
    state: DriveState = DriveState.ready()

    def apply(
        self,
        distances: List[float],
        throttle_us: int,
        steer_us: int,
        allow_reverse: bool = True,
    ) -> "VehicleMotion":
        """Return a new VehicleMotion with updated state and control inputs."""
        new_state = DriveState.from_distances(
            distances,
            self.state,
        )
        new_steer = Steer(steer_us)
        new_throttle = Throttle(throttle_us)

        if new_state.is_emergency_stop():
            if allow_reverse and new_throttle.is_backward():
                return replace(
                    self, state=DriveState.moving(), steer=new_steer, throttle=new_throttle
                )
            return replace(
                self, state=new_state, steer=new_steer, throttle=Throttle.stop()
            )

        # ブロック状態に応じたチェック
        if new_state.is_blocked_front() and new_throttle.is_forward():
            return replace(
                self, state=new_state, steer=new_steer, throttle=Throttle.stop()
            )

        if new_state.is_blocked_rear() and new_throttle.is_backward():
            return replace(
                self, state=new_state, steer=new_steer, throttle=Throttle.stop()
            )

        if not new_throttle.is_stop():
            next_state = DriveState.moving()
        elif not new_state.is_blocked():
            next_state = DriveState.ready()
        else:
            next_state = new_state
        return replace(self, state=next_state, steer=new_steer, throttle=new_throttle)


    def set_stop(self) -> "VehicleMotion":
        return replace(self, throttle=Throttle.stop())

    def set_ready(self) -> "VehicleMotion":
        return replace(self, state=DriveState.ready())

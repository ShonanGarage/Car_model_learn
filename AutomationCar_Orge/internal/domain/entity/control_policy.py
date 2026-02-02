from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import time
from typing import Dict

from internal.domain.value_object.control_input import ControlInput
from internal.domain.value_object.control_rules import ControlRules

class ThrottleAction(Enum):
    STOP = auto()
    FORWARD = auto()
    BACKWARD = auto()


class SteerAction(Enum):
    RESET = auto()
    LEFT = auto()
    RIGHT = auto()


@dataclass(frozen=True)
class ControlDecision:
    throttle_action: ThrottleAction | None
    steer_action: SteerAction | None
    steer_step: int | None

    def is_stop(self) -> bool:
        return self.throttle_action == ThrottleAction.STOP

    def is_forward(self) -> bool:
        return self.throttle_action == ThrottleAction.FORWARD

    def is_backward(self) -> bool:
        return self.throttle_action == ThrottleAction.BACKWARD

    def is_reset_steer(self) -> bool:
        return self.steer_action == SteerAction.RESET

    def is_steer_left(self) -> bool:
        return self.steer_action == SteerAction.LEFT

    def is_steer_right(self) -> bool:
        return self.steer_action == SteerAction.RIGHT


@dataclass
class ControlPolicy:
    rules: ControlRules
    hold_start: Dict[str, float] = field(default_factory=dict)

    def evaluate(
        self,
        input_state: ControlInput,
        current_steer_us: int,
    ) -> ControlDecision:
        hold_keys = {
            "a": input_state.left,
            "d": input_state.right,
        }

        hold_s = self._get_span_keep_hold(hold_keys)
        throttle_action = self._build_throttle_action(input_state)
        steer_action, steer_step = self._build_steer_action(input_state, hold_s, current_steer_us)

        return ControlDecision(
            throttle_action=throttle_action,
            steer_action=steer_action,
            steer_step=steer_step,
        )

    def _build_throttle_action(self, input_state: ControlInput) -> ThrottleAction | None:
        throttle_action: ThrottleAction = ThrottleAction.STOP
        if input_state.forward and not input_state.backward:
            throttle_action = ThrottleAction.FORWARD
        elif input_state.backward and not input_state.forward:
            throttle_action = ThrottleAction.BACKWARD

        return throttle_action

    def _build_steer_action(
        self, input_state: ControlInput, hold_s: Dict[str, float], current_steer_us: int
    ) -> tuple[SteerAction, int]:
        steer_action: SteerAction = SteerAction.RESET
        steer_step: int = 0

        left_hold_s = hold_s["a"]
        right_hold_s = hold_s["d"]
        steer_full_time_s = self.rules.steer_full_time_s
        steer_step_us = self.rules.steer_step_us
        steer_accel_step_per_s = self.rules.steer_accel_step_per_s
        steer_max_step = self.rules.steer_max_step
        steer_center_us = self.rules.steer_center_us
        steer_min_us = self.rules.steer_min_us
        steer_max_us = self.rules.steer_max_us

        if input_state.left:
            ratio = min(1.0, left_hold_s / steer_full_time_s)
            target_us = int(round(steer_center_us + (steer_min_us - steer_center_us) * ratio))
            step = int(steer_step_us + left_hold_s * steer_accel_step_per_s)
            step = min(steer_max_step, max(steer_step_us, step))
            delta = min(step, max(1, abs(current_steer_us - target_us)))
            steer_action = SteerAction.LEFT
            steer_step = delta

        elif input_state.right:
            ratio = min(1.0, right_hold_s / steer_full_time_s)
            target_us = int(round(steer_center_us + (steer_max_us - steer_center_us) * ratio))
            step = int(steer_step_us + right_hold_s * steer_accel_step_per_s)
            step = min(steer_max_step, max(steer_step_us, step))
            delta = min(step, max(1, abs(current_steer_us - target_us)))
            steer_action = SteerAction.RIGHT
            steer_step = delta

        return steer_action, steer_step

    def _update_hold(self, key: str, active: bool, now: float) -> None:
        if active:
            # 元の時刻を保持（更新されない）
            self.hold_start.setdefault(key, now)
        else:
            # 解除されたら削除
            self.hold_start.pop(key, None)

    def _get_span_keep_hold(self, keys: Dict[str, bool]) -> Dict[str, float]:
        now = time.time()
        hold_s: Dict[str, float] = {}
        for key, active in keys.items():
            self._update_hold(key, active, now)
            if active:
                hold_s[key] = now - self.hold_start.get(key, now)
            else:
                hold_s[key] = 0.0
        return hold_s

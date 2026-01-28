from dataclasses import dataclass


@dataclass(frozen=True)
class ControlInput:
    forward: bool
    backward: bool
    left: bool
    right: bool
    stop: bool
    center: bool

    @classmethod
    def from_raw(
        cls,
        forward: bool,
        backward: bool,
        left: bool,
        right: bool,
        stop: bool,
        center: bool,
    ) -> "ControlInput":
        if left and right:
            left = False
            right = False
            center = True
        if not left and not right:
            center = True
        return cls(
            forward=forward,
            backward=backward,
            left=left,
            right=right,
            stop=stop,
            center=center,
        )

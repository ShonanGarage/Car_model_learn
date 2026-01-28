import sys
from dataclasses import dataclass
from typing import List

class TerminalUI:
    @dataclass(frozen=True)
    class ControlStatus:
        state: str
        distances: List[float]
        steer_us: int
        throttle_us: int
        is_forward: bool
        camera_ok: bool

    @classmethod
    def display_status(cls, status: "TerminalUI.ControlStatus") -> None:
        dist_str = " / ".join([f"{d:.2f}m" if d > 0 else "ERR" for d in status.distances])
        cam_str = "OK" if status.camera_ok else "NG"
        forward_str = "FWD" if status.is_forward else "NO"
        sys.stdout.write(
            f"\rStatus: {status.state} | "
            f"Steer: {status.steer_us}us | "
            f"Throttle: {status.throttle_us}us | "
            f"FWD: {forward_str} | Camera: {cam_str} | Sonars: {dist_str}    "
        )
        sys.stdout.flush()

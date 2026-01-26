from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Telemetry:
    timestamp: float
    course_id: str
    distances: List[float]
    drive_state: str
    steer_us: int
    throttle_us: int
    image_filename: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "course_id": self.course_id,
            "distances": self.distances,
            "drive_state": self.drive_state,
            "steer_us": self.steer_us,
            "throttle_us": self.throttle_us,
            "image_filename": self.image_filename
        }

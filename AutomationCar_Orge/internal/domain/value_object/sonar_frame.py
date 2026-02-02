from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SonarFrame:
    front: List[float]
    rear: List[float]

    def is_empty(self) -> bool:
        return not self.front and not self.rear

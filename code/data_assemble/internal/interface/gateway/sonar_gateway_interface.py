from abc import ABC, abstractmethod
from typing import List

class SonarGatewayInterface(ABC):
    @abstractmethod
    def read_distances_m(self) -> List[float]:
        """Read distances from all sonar sensors in meters. Return -1.0 for error."""
        pass
    @abstractmethod
    def stop(self) -> None:
        """Stop background measurement thread if any."""
        pass

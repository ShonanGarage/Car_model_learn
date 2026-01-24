from abc import ABC, abstractmethod
from typing import Any
from internal.domain.value_object.telemetry import Telemetry

class DataRepositoryInterface(ABC):
    @abstractmethod
    def save(self, telemetry: Telemetry, image: Any) -> None:
        """Save telemetry data and corresponding image."""
        pass

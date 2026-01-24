from abc import ABC, abstractmethod
from internal.domain.value_object.throttle import Throttle

class DCGatewayInterface(ABC):
    @abstractmethod
    def set_throttle(self, throttle: Throttle) -> None:
        """Set throttle output using Throttle value object."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close connection to the hardware."""
        pass

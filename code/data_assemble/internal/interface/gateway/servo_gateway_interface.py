from abc import ABC, abstractmethod
from internal.domain.value_object.steer import Steer

class ServoGatewayInterface(ABC):
    @abstractmethod
    def set_steer(self, steer: Steer) -> None:
        """Set steering angle using Steer value object."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the gateway."""
        pass

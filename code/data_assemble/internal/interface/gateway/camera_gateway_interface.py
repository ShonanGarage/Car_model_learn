from abc import ABC, abstractmethod
from typing import Optional, Tuple

class CameraGatewayInterface(ABC):
    @abstractmethod
    def capture_frame(self) -> Tuple[bool, Optional[object]]:
        """Capture a single frame from the camera."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release the camera resources."""
        pass

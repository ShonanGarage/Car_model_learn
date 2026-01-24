import time
from typing import List, Optional
from app.config.settings import Settings
from internal.interface.gateway.dc_gateway_interface import DCGatewayInterface
from internal.interface.gateway.servo_gateway_interface import ServoGatewayInterface
from internal.interface.gateway.camera_gateway_interface import CameraGatewayInterface
from internal.interface.gateway.sonar_gateway_interface import SonarGatewayInterface
from internal.interface.repository.data_repository_interface import DataRepositoryInterface
from internal.domain.value_object.drive_state import DriveState
from internal.domain.entity.drive_control import DriveControl
from internal.domain.value_object.steer import Steer
from internal.domain.value_object.throttle import Throttle
from internal.domain.value_object.telemetry import Telemetry

class DriveService:
    def __init__(
        self, 
        dc_gateway: DCGatewayInterface, 
        servo_gateway: ServoGatewayInterface, 
        sonar_gateway: SonarGatewayInterface, 
        camera_gateway: CameraGatewayInterface,
        data_repository: DataRepositoryInterface,
        settings: Settings
    ):
        self.dc_gateway = dc_gateway
        self.servo_gateway = servo_gateway
        self.sonar_gateway = sonar_gateway
        self.camera_gateway = camera_gateway
        self.data_repository = data_repository
        self.settings = settings
        
        # DriveControl entity manages the driving state and control inputs
        self.control = DriveControl.create_default(steer_us=settings.servo.center_us)
        self.distances: List[float] = []
        self.frame: Optional[object] = None
        self.ok: bool = False
        self.course_id: str = "default_course"

    def set_course_id(self, course_id: str) -> None:
        self.course_id = course_id

    @property
    def state(self) -> DriveState:
        return self.control.state

    @property
    def current_steer_us(self) -> int:
        return self.control.steer.value

    def update(self) -> None:
        """Periodic update: read sonars, check collision, capture camera, update state."""
        self.distances = self.sonar_gateway.read_distances_m()
        self.ok, self.frame = self.camera_gateway.capture_frame()
        # Domain Entity updates its internal state based on sensor data
        self.control.update_state(self.distances)
        
        # ゲートウェイへの通知が必要な場合（例：BLOCKEDで停止指令を送るなど）
        # DriveControl.update_state内ですでにstop()が呼ばれているため、スロットルを更新
        self.dc_gateway.set_throttle(self.control.throttle)

    def _log_telemetry(self) -> None:
        t = time.time()
        img_filename = f"{int(t*1000)}.jpg"
        
        telemetry = Telemetry(
            timestamp=t,
            course_id=self.course_id,
            distances=self.distances,
            drive_state=self.state.name,
            steer_us=self.current_steer_us,
            throttle_us=self.control.throttle.value,
            image_filename=img_filename
        )
        
        self.data_repository.save(
            telemetry, 
            self.frame if self.ok else None
        )

    def steer_left(self, step: int = 10) -> None:
        new_us = self._clamp(
            self.control.steer.value - step,
            self.settings.servo.min_us,
            self.settings.servo.max_us
        )
        self.control.update_steer(new_us)
        self.servo_gateway.set_steer(self.control.steer)
        self._log_telemetry()

    def steer_right(self, step: int = 10) -> None:
        new_us = self._clamp(
            self.control.steer.value + step,
            self.settings.servo.min_us,
            self.settings.servo.max_us
        )
        self.control.update_steer(new_us)
        self.servo_gateway.set_steer(self.control.steer)
        self._log_telemetry()
        
    def reset_steer(self) -> None:
        self.control.update_steer(self.settings.servo.center_us)
        self.servo_gateway.set_steer(self.control.steer)
        self._log_telemetry()

    def move_forward(self) -> None:
        self.control.update_throttle(self.settings.throttle.fixed_us, self.distances)
        self.dc_gateway.set_throttle(self.control.throttle)
        self._log_telemetry()

    def move_backward(self) -> None:
        self.control.update_throttle(self.settings.throttle.back_us, self.distances)
        self.dc_gateway.set_throttle(self.control.throttle)
        self._log_telemetry()

    def stop(self) -> None:
        self.control.stop()
        self.dc_gateway.set_throttle(self.control.throttle)
        self.sonar_gateway.stop() # Ensure background thread stops
        self._log_telemetry()

    def _clamp(self, v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

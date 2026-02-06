import time
import threading
from dataclasses import dataclass
from typing import List, Optional
from app.config.settings import Settings, build_control_rules, build_safety_rules
from internal.interface.gateway.dc_gateway_interface import DCGatewayInterface
from internal.interface.gateway.servo_gateway_interface import ServoGatewayInterface
from internal.interface.gateway.camera_gateway_interface import CameraGatewayInterface
from internal.interface.gateway.sonar_gateway_interface import SonarGatewayInterface
from internal.interface.repository.data_repository_interface import DataRepositoryInterface
from internal.domain.entity.vehicle_motion import boot_vehicle_motion_ready
from internal.domain.entity.control_policy import ControlDecision, ControlPolicy
from internal.domain.value_object.control_input import ControlInput
from internal.domain.value_object.telemetry import Telemetry
from internal.domain.value_object.steer import Steer
from internal.domain.value_object.sonar_frame import SonarFrame


@dataclass(frozen=True)
class DriveInfra:
    dc_gateway: DCGatewayInterface
    servo_gateway: ServoGatewayInterface
    sonar_gateway: SonarGatewayInterface
    camera_gateway: CameraGatewayInterface
    data_repository: DataRepositoryInterface

class DriveService:
    def __init__(
        self,
        infra: DriveInfra,
        settings: Settings
    ):
        self.infra = infra
        self.settings = settings
        
        # VehicleMotion entity manages the driving state and control inputs
        self.motion_vehicle = boot_vehicle_motion_ready()
        self.safety_rules = build_safety_rules(settings)
        self.control_policy = ControlPolicy(build_control_rules(settings))
        
        self.distances: List[float] = []
        self.frame: Optional[object] = None
        self.ok: bool = False
        self._sensor_lock = threading.Lock()
        self._last_log_time: float = 0.0

    def set_course_id(self, course_id: str) -> None:
        self.settings.course_id = course_id

    @property
    def state(self):
        return self.motion_vehicle.state

    @property
    def current_steer_us(self) -> int:
        return self.motion_vehicle.steer.value

    @property
    def current(self):
        return self.motion_vehicle

    def get_sensor_snapshot(self) -> tuple[List[float], bool, Optional[object]]:
        """Return a consistent snapshot of distances/ok/frame."""
        with self._sensor_lock:
            return (list(self.distances), self.ok, self.frame)

    def _get_distances(self) -> List[float]:
        with self._sensor_lock:
            return list(self.distances)

    def _build_sonar_frame(self, distances: List[float]) -> SonarFrame:
        # Map the sensor list to logical front/rear groups in the app layer.
        sensor_names = list(self.settings.sonar.keys())
        front_names = {"center", "front_left", "front_right"}
        rear_names = {"rear_left", "rear_right"}
        front: List[float] = []
        rear: List[float] = []
        for name, dist in zip(sensor_names, distances):
            if name in front_names:
                front.append(dist)
            elif name in rear_names:
                rear.append(dist)
        return SonarFrame(front=front, rear=rear)
    
    # 入力（押されているボタン）に対する車体の動作制御ロジック
    def _apply_throttle(self, decision: "ControlDecision") -> None:
        if decision.is_stop():
            self._stop()
        elif decision.is_forward():
            self._move_forward()
        elif decision.is_backward():
            self._move_backward()

    def _apply_steer(self, decision: "ControlDecision") -> None:
        if decision.is_reset_steer():
            self._reset_steer()
        elif decision.is_steer_left():
            self._steer_left(step=decision.steer_step)
        elif decision.is_steer_right():
            self._steer_right(step=decision.steer_step)

    # Public control APIs (used by entrypoints)
    def move_forward(self) -> None:
        self._move_forward()

    def move_backward(self) -> None:
        self._move_backward()

    def stop(self) -> None:
        self._stop()

    def steer_left(self, step: int | None = None) -> None:
        self._steer_left(step=step)

    def steer_right(self, step: int | None = None) -> None:
        self._steer_right(step=step)

    def reset_steer(self) -> None:
        self._reset_steer()

    def set_steer_us(self, steer_us: int) -> None:
        new_steer = Steer.from_us(steer_us, self.settings)
        distances = self._get_distances()
        sonar_frame = self._build_sonar_frame(distances)
        self.motion_vehicle = self.motion_vehicle.apply(
            sonar_frame,
            throttle_us=self.motion_vehicle.throttle.value,
            steer_us=new_steer.value,
            safety_rules=self.safety_rules,
        )
        self.infra.servo_gateway.set_steer(self.motion_vehicle.steer)

    def set_throttle_us(self, throttle_us: int) -> None:
        distances = self._get_distances()
        sonar_frame = self._build_sonar_frame(distances)
        self.motion_vehicle = self.motion_vehicle.apply(
            sonar_frame,
            throttle_us=throttle_us,
            steer_us=self.motion_vehicle.steer.value,
            safety_rules=self.safety_rules,
            allow_reverse=True,
        )
        self.infra.dc_gateway.set_throttle(self.motion_vehicle.throttle)
        self._log_telemetry()

    def apply_control_input(self, input_state: ControlInput, now: float | None = None) -> None:
        decision = self.control_policy.evaluate(
            input_state=input_state,
            current_steer_us=self.current_steer_us
        )
        self._apply_throttle(decision)
        self._apply_steer(decision)

    
    # メインループで定期的に呼び出される更新処理（主要なデータを収集する）
    # 必要に応じて停止も行う
    # VehicleMotionドメイン 
    def update(self) -> None:
        # NOTE: 超音波データ・画像を収集
        distances = self.infra.sonar_gateway.read_distances_m()
        ok, frame = self.infra.camera_gateway.capture_frame()
        with self._sensor_lock:
            self.distances = distances
            self.ok = ok
            self.frame = frame
        sonar_frame = self._build_sonar_frame(distances)
        # NOTE: 距離をもとに、現在の状況を把握してVehicleMotionを更新
        self.motion_vehicle = self.motion_vehicle.apply(
            sonar_frame,
            throttle_us=self.motion_vehicle.throttle.value,
            steer_us=self.motion_vehicle.steer.value,
            safety_rules=self.safety_rules,
        )
        # NOTE: VehicleMotion.apply内で停止が反映されるため（例：BLOCKEDで停止指令を送るなど）、スロットルを更新
        self.infra.dc_gateway.set_throttle(self.motion_vehicle.throttle)



    def _log_telemetry(self) -> None:
        t = time.time()
        img_filename = f"{int(t*1000)}.jpg"
        distances, ok, frame = self.get_sensor_snapshot()
        
        telemetry = Telemetry(
            timestamp=t,
            course_id=self.settings.course_id,
            distances=distances,
            drive_state=self.state.name,
            steer_us=self.current_steer_us,
            throttle_us=self.motion_vehicle.throttle.value,
            image_filename=img_filename
        )
        
        self.infra.data_repository.save(
            telemetry, 
            frame if ok else None
        )

    def _steer_left(self, step: int | None = None) -> None:
        if step is None:
            step = self.settings.servo.step_us
        new_steer = Steer.from_us(
            self.motion_vehicle.steer.value - step,
            self.settings,
        )
        distances = self._get_distances()
        sonar_frame = self._build_sonar_frame(distances)
        self.motion_vehicle = self.motion_vehicle.apply(
            sonar_frame,
            throttle_us=self.motion_vehicle.throttle.value,
            steer_us=new_steer.value,
            safety_rules=self.safety_rules,
        )
        self.infra.servo_gateway.set_steer(self.motion_vehicle.steer)

    def _steer_right(self, step: int | None = None) -> None:
        if step is None:
            step = self.settings.servo.step_us
        new_steer = Steer.from_us(
            self.motion_vehicle.steer.value + step,
            self.settings,
        )
        distances = self._get_distances()
        sonar_frame = self._build_sonar_frame(distances)
        self.motion_vehicle = self.motion_vehicle.apply(
            sonar_frame,
            throttle_us=self.motion_vehicle.throttle.value,
            steer_us=new_steer.value,
            safety_rules=self.safety_rules,
        )
        self.infra.servo_gateway.set_steer(self.motion_vehicle.steer)
        
    def _reset_steer(self) -> None:
        distances = self._get_distances()
        sonar_frame = self._build_sonar_frame(distances)
        self.motion_vehicle = self.motion_vehicle.apply(
            sonar_frame,
            throttle_us=self.motion_vehicle.throttle.value,
            steer_us=self.settings.servo.center_us,
            safety_rules=self.safety_rules,
        )
        self.infra.servo_gateway.set_steer(self.motion_vehicle.steer)

    def _move_forward(self) -> None:
        distances = self._get_distances()
        sonar_frame = self._build_sonar_frame(distances)
        self.motion_vehicle = self.motion_vehicle.apply(
            sonar_frame,
            throttle_us=self.settings.throttle.fixed_us,
            steer_us=self.motion_vehicle.steer.value,
            safety_rules=self.safety_rules,
        )
        self.infra.dc_gateway.set_throttle(self.motion_vehicle.throttle)
        self._log_telemetry()

    def _move_backward(self) -> None:
        distances = self._get_distances()
        sonar_frame = self._build_sonar_frame(distances)
        self.motion_vehicle = self.motion_vehicle.apply(
            sonar_frame,
            throttle_us=self.settings.throttle.back_us,
            steer_us=self.motion_vehicle.steer.value,
            safety_rules=self.safety_rules,
            allow_reverse=True,
        )
        self.infra.dc_gateway.set_throttle(self.motion_vehicle.throttle)
        self._log_telemetry()

    def _stop(self) -> None:
        self.motion_vehicle = self.motion_vehicle.set_stop()
        if not self.motion_vehicle.state.is_blocked() and not self.motion_vehicle.state.is_emergency_stop():
            self.motion_vehicle = self.motion_vehicle.set_ready()
        self.infra.dc_gateway.set_throttle(self.motion_vehicle.throttle)

    def shutdown(self) -> None:
        """Stop motion and terminate background resources."""
        self.motion_vehicle = self.motion_vehicle.set_stop()
        if not self.motion_vehicle.state.is_blocked() and not self.motion_vehicle.state.is_emergency_stop():
            self.motion_vehicle = self.motion_vehicle.set_ready()
        self.infra.dc_gateway.set_throttle(self.motion_vehicle.throttle)
        self.infra.sonar_gateway.stop() # Ensure background thread stops

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, get_type_hints, cast
import yaml

from internal.domain.value_object.safety_rules import SafetyRules
from internal.domain.value_object.control_rules import ControlRules
# NOTE: 任意の型
T = TypeVar("T")

@dataclass
class CameraModule:
    device_number: int | str = 0
    show_ui: bool | None = None
    use_picamera2: bool = True
    img_w: int = 320
    img_h: int = 240
    fps: int = 10 # 10fps~20fpsがおススメ
    jpeg_quality: int = 85

@dataclass
class SonarModule:
    trig_gpio: int
    echo_gpio: int

@dataclass
class ServoModule:
    gpio: int = 18
    center_us: int = 1500
    min_us: int = 1000
    max_us: int = 2000
    frequency: int = 50
    step_us: int = 20

@dataclass
class DCModule:
    gpio: int = 13
    fixed_us: int = 1600
    back_us: int = 1300
    frequency: int = 50
    ramp_forward_ratio: float = 0.35
    ramp_backward_ratio: float = 0.35
    ramp_stop_ratio: float = 0.4

@dataclass
class TerminalControl:
    hold_window_s: float = 0.25
    steer_accel_step_per_s: int = 600
    steer_max_step: int = 500
    steer_full_time_s: float = 0.6
    throttle_full_time_s: float = 0.6
    throttle_grace_s: float = 0.5

@dataclass
class Settings:
    camera: CameraModule = field(default_factory=CameraModule)
    servo: ServoModule = field(default_factory=ServoModule)
    sonar: dict[str, SonarModule] = field(default_factory=lambda: {
        "center": SonarModule(23, 24),
        "front_left": SonarModule(25, 8),
        "front_right": SonarModule(7, 1),
        # FIXME: 要確認
        "rear_left": SonarModule(25, 8),
        "rear_right": SonarModule(7, 1),
    })
    throttle: DCModule = field(default_factory=DCModule)
    terminal: TerminalControl = field(default_factory=TerminalControl)
    sonar_timeout_s: float = 0.03
    sonar_inter_gap_s: float = 0.008
    sonar_sweep_sleep_s: float = 0.01
    brake_start_threshold_m: float = 0.4
    emergency_stop_threshold_m: float = 0.1
    blocked_threshold_m: float = 0.2
    control_loop_sleep_s: float = 0.02
    log_hz: float = 20.0
    out_dir: str = field(default_factory=lambda: str((Path(__file__).resolve().parents[3].parent.parent / "learning_data").resolve()))
    course_id: str = "default_course"


def from_dict(data_class: Type[T], data: dict[str, Any]) -> T:
    """Recursively convert a dictionary to a dataclass."""
    if not is_dataclass(data_class):
        return cast(T, data)

    try:
        # 
        field_types = get_type_hints(data_class)
    except Exception:
        field_types = {}
        
    kwargs = {}

    for f_name, f_value in data.items():
        if f_name not in field_types:
            continue

        f_type = field_types[f_name]
        
        # Handle dict[str, Dataclass]
        origin = getattr(f_type, "__origin__", None)
        if origin is dict:
            args = getattr(f_type, "__args__", [])
            if len(args) >= 2 and isinstance(args[1], type) and is_dataclass(args[1]):
                value_type = args[1]
                kwargs[f_name] = {k: from_dict(value_type, v) for k, v in f_value.items()}
            else:
                kwargs[f_name] = f_value
        # Handle Dataclass
        elif isinstance(f_type, type) and is_dataclass(f_type):
            kwargs[f_name] = from_dict(cast(Any, f_type), f_value)
        else:
            kwargs[f_name] = f_value

    return cast(T, data_class(**kwargs))


def load_settings(yaml_path: Path | str | None = None) -> Settings:
    """Load settings from a YAML file or use defaults."""
    if yaml_path is None:
        yaml_path = Path(__file__).parent / "config.yaml"
    else:
        yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        return Settings()

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return Settings()

    # yamlがnullだった際、out_dirをキーとしたからのリストが返されてしまう。
    if data.get("out_dir") is None:
        data.pop("out_dir", None)

    return from_dict(Settings, data)

SETTINGS = load_settings()


def build_safety_rules(settings: Settings) -> SafetyRules:
    return SafetyRules(
        emergency_stop_threshold_m=settings.emergency_stop_threshold_m,
        blocked_threshold_m=settings.blocked_threshold_m,
    )


def build_control_rules(settings: Settings) -> ControlRules:
    return ControlRules(
        steer_full_time_s=settings.terminal.steer_full_time_s,
        steer_step_us=settings.servo.step_us,
        steer_accel_step_per_s=settings.terminal.steer_accel_step_per_s,
        steer_max_step=settings.terminal.steer_max_step,
        steer_center_us=settings.servo.center_us,
        steer_min_us=settings.servo.min_us,
        steer_max_us=settings.servo.max_us,
    )

import time

from app.container import Container
from internal.domain.value_object.control_input import ControlInput
from presentation.terminal_ui import TerminalUI

# keyboardライブラリのroot権限チェックを無効化
import keyboard as keyboard_lib
try:
    import keyboard._nixcommon as _nixcommon
    _nixcommon.ensure_root = lambda: None
    import keyboard._nixkeyboard as _nixkeyboard
    _nixkeyboard.ensure_root = lambda: None
    def _build_device_no_root():
        if _nixkeyboard.device:
            return
        _nixkeyboard.device = _nixkeyboard.aggregate_devices("kbd")
    _nixkeyboard.build_device = _build_device_no_root
    def _build_tables_no_root():
        if _nixkeyboard.to_name and _nixkeyboard.from_name:
            return
        minimal_map = {
            16: "q",
            17: "w",
            30: "a",
            31: "s",
            32: "d",
            45: "x",
            46: "c",
            57: "space",
        }
        for scan_code, name in minimal_map.items():
            _nixkeyboard.register_key((scan_code, ()), name)
    _nixkeyboard.build_tables = _build_tables_no_root
except Exception:
    pass


def main() -> None:
    container = Container()
    course_id = container.settings.course_id
    container.drive_service.set_course_id(course_id)
    print(f"RC Car Terminal Control Started. Course ID: {course_id}")
    print("Control: W (Forward), S (Backward), X/Space (Stop), A (Left), D (Right), C (Center), Q (Quit)")

    try:
        while True:
            # 1. Update service (sonar readings, camera, state transitions, LOGGING)
            container.drive_service.update()

            # 2. Handle user input
            now = time.time()
            if keyboard_lib.is_pressed("q"):
                break

            container.drive_service.apply_control_input(
                input_state=ControlInput.from_raw(
                    forward=keyboard_lib.is_pressed("w"),
                    backward=keyboard_lib.is_pressed("s"),
                    left=keyboard_lib.is_pressed("a"),
                    right=keyboard_lib.is_pressed("d"),
                    stop=keyboard_lib.is_pressed("x"),
                    center=keyboard_lib.is_pressed("c"),
                ),
                now=now,
            )

            # 3. Display status (Skipping status update if too fast could be an option, but 50Hz is fine)
            container.camera_view.display(
                container.drive_service.frame,
                container.drive_service.current,
            )

            TerminalUI.display_status(
                TerminalUI.ControlStatus(
                state=container.drive_service.state.name,
                distances=container.drive_service.distances,
                steer_us=container.drive_service.current_steer_us,
                throttle_us=container.drive_service.current.throttle.value,
                is_forward=container.drive_service.current.throttle.is_forward(),
                camera_ok=container.drive_service.ok,
                )
            )
            
            time.sleep(container.settings.control_loop_sleep_s) # ~50Hz loop (much smoother)
    finally:
        container.drive_service.shutdown()
        container.camera_gateway.release()
        container.camera_view.stop()
        container.data_repository.stop()
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()

import time
from typing import Tuple

from app.container import Container

def handle_user_input(
    container: Container,
    last_cmd_time: float,
    timeout_s: float = 0.3,
) -> Tuple[float, bool]:
    cmd = container.terminal_ui.get_command()
    should_quit = False

    if cmd:
        if cmd == 'q':
            should_quit = True
        elif cmd == 'w':
            container.drive_service.move_forward()
            last_cmd_time = time.time()
        elif cmd == 's':
            container.drive_service.move_backward()
            last_cmd_time = time.time()
        elif cmd == 'x':
            container.drive_service.stop()
            last_cmd_time = 0 # Force stop
        elif cmd == 'a':
            container.drive_service.steer_left(step=100)
            # Steering doesn't reset the movement timeout if we want "steer while moving"
            # But if we want it to move only while steered, we'd update last_cmd_time here too.
            # For now, let's keep steering persistent and throttle timed.
        elif cmd == 'd':
            container.drive_service.steer_right(step=100)

    if time.time() - last_cmd_time > timeout_s:
        container.drive_service.stop()

    return last_cmd_time, should_quit

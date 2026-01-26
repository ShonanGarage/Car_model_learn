import time
import sys
from app.container import Container
from internal.domain.value_object.telemetry import Telemetry

def main():
    container = Container()
    course_id = container.settings.course_id
    container.drive_service.set_course_id(course_id)
    print(f"RC Car Terminal Control Started. Course ID: {course_id}")
    
    steer_step = 20
    steer_interval_s = 0.05
    last_steer_time = 0.0
    last_move_state = "stop"
    last_steer_state = "center"
    udp_move_state = "stop"
    udp_move_until = 0.0
    udp_steer_state = "center"
    udp_steer_until = 0.0
    
    try:
        while True:
            # 1. Update service (sonar readings, camera, state transitions, LOGGING)
            container.drive_service.update()
            
            # 2. Handle user input
            should_quit = False
            now = time.time()

            for key, event_type in container.terminal_ui.get_key_events():
                if key == 'q' and event_type == 'down':
                    should_quit = True
                elif key in ('x', ' ') and event_type == 'down':
                    last_move_state = "stop"
                    udp_move_state = "stop"
                    udp_move_until = 0.0
                    container.drive_service.stop()
                elif key == 'c' and event_type == 'down':
                    last_steer_state = "center"
                    udp_steer_state = "center"
                    udp_steer_until = 0.0
                    container.drive_service.reset_steer()
            if should_quit:
                break

            for cmd in container.terminal_ui.get_commands():
                if cmd == 'q':
                    should_quit = True
                    break
                elif cmd == 'w':
                    udp_move_state = "forward"
                    udp_move_until = now + 0.4
                elif cmd == 's':
                    udp_move_state = "backward"
                    udp_move_until = now + 0.4
                elif cmd == 'x':
                    udp_move_state = "stop"
                    udp_move_until = 0.0
                    container.drive_service.stop()
                elif cmd == 'a':
                    udp_steer_state = "left"
                    udp_steer_until = now + 0.4
                elif cmd == 'd':
                    udp_steer_state = "right"
                    udp_steer_until = now + 0.4
                elif cmd == 'c':
                    udp_steer_state = "center"
                    udp_steer_until = 0.0
                    container.drive_service.reset_steer()
            if should_quit:
                break

            pressed = container.terminal_ui.get_pressed_keys()
            if 'w' in pressed and 's' not in pressed:
                desired_move = "forward"
            elif 's' in pressed and 'w' not in pressed:
                desired_move = "backward"
            elif now < udp_move_until:
                desired_move = udp_move_state
            else:
                desired_move = "stop"

            if desired_move != last_move_state:
                if desired_move == "forward":
                    container.drive_service.move_forward()
                elif desired_move == "backward":
                    container.drive_service.move_backward()
                else:
                    container.drive_service.stop()
                last_move_state = desired_move

            if 'a' in pressed and 'd' not in pressed:
                desired_steer = "left"
            elif 'd' in pressed and 'a' not in pressed:
                desired_steer = "right"
            elif now < udp_steer_until:
                desired_steer = udp_steer_state
            else:
                desired_steer = "center"

            if desired_steer == "center":
                if last_steer_state != "center":
                    container.drive_service.reset_steer()
                    last_steer_state = "center"
            elif now - last_steer_time >= steer_interval_s:
                if desired_steer == "left":
                    container.drive_service.steer_left(step=steer_step)
                elif desired_steer == "right":
                    container.drive_service.steer_right(step=steer_step)
                last_steer_state = desired_steer
                last_steer_time = now
            
            # 3. Display status (Skipping status update if too fast could be an option, but 50Hz is fine)
            container.camera_view.display(
                container.drive_service.frame,
                container.drive_service.control
            )
            container.terminal_ui.display_status(
                state=container.drive_service.state.name,
                distances=container.drive_service.distances,
                steer_us=container.drive_service.current_steer_us
            )
            
            time.sleep(0.02) # ~50Hz loop (much smoother)
    finally:
        container.drive_service.stop()
        container.camera_gateway.release()
        container.camera_view.stop()
        container.data_repository.stop()
        container.terminal_ui.stop()
        print("\nShutdown complete.")

if __name__ == "__main__":
    main()

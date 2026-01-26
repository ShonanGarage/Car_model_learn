import time
from app.container import Container
from app.handler import handle_user_input

def main():
    container = Container()
    course_id = container.settings.course_id
    container.drive_service.set_course_id(course_id)
    print(f"RC Car Terminal Control Started. Course ID: {course_id}")
    
    last_cmd_time = time.time()
    
    try:
        while True:
            # 1. Update service (sonar readings, camera, state transitions, LOGGING)
            container.drive_service.update()
            
            # 2. Handle user input
            last_cmd_time, should_quit = handle_user_input(
                container,
                last_cmd_time,
            )
            if should_quit:
                break

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

import cv2
from typing import Optional
from internal.domain.entity.vehicle_motion import VehicleMotion

class CameraView:
    def __init__(self, show_ui: bool = False):
        self.window_name = "RC Car Camera"
        self.show_ui = show_ui

    def display(self, frame: Optional[object], control: VehicleMotion):
        """Display the camera frame with control status overlay."""
        if frame is None or not self.show_ui:
            return

        # Simple overlay for visual feedback
        status_text = f"State: {control.state.name} | T: {control.throttle.value} | S: {control.steer.value}"
        cv2.putText(
            frame, 
            status_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )
        
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def stop(self):
        if self.show_ui:
            cv2.destroyAllWindows()

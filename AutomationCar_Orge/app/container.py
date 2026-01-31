from app.config import load_settings
from infrastructure.gateway.dc.lgpio_dc_gateway import LgpDCGateway
from infrastructure.gateway.sonar.lgp_sonar_gateway import LgpSonarGateway
from infrastructure.gateway.servo.lgpio_servo_gateway import LgpServoGateway
from infrastructure.gateway.camera.lgp_camera_gateway import LgpCameraGateway
from infrastructure.repository.local_data_repository import LocalDataRepository
from internal.service.drive_service import DriveInfra, DriveService
from presentation.camera_view import CameraView
from presentation.terminal_ui import TerminalUI

class Container:
    def __init__(self):
        self.settings = load_settings()

        dc_gateway = LgpDCGateway(self.settings)
        self.infra = DriveInfra(
            dc_gateway=dc_gateway,
            servo_gateway=LgpServoGateway(self.settings, dc_gateway.handle),
            sonar_gateway=LgpSonarGateway(self.settings, dc_gateway.handle),
            camera_gateway=LgpCameraGateway(self.settings),
            data_repository=LocalDataRepository(self.settings),
        )

        self.drive_service = DriveService(self.infra, self.settings)
        self.camera_gateway = self.infra.camera_gateway
        self.data_repository = self.infra.data_repository
        self.camera_view = CameraView(show_ui=bool(self.settings.camera.show_ui))
        self.terminal_ui = TerminalUI()

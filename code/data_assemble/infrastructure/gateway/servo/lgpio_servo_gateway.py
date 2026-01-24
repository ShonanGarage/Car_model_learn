import lgpio
from internal.interface.gateway.servo_gateway_interface import ServoGatewayInterface
from internal.domain.value_object.steer import Steer
from app.config.settings import Settings

class LgpServoGateway(ServoGatewayInterface):
    def __init__(self, settings: Settings, handle: int = -1):
        self.settings = settings
        self.handle = handle
        self._own_handle = False

        if self.handle < 0:
            self.handle = lgpio.gpiochip_open(0)
            self._own_handle = True
            if self.handle < 0:
                raise RuntimeError(f"gpiochip_open(0) failed: {self.handle}")

        # Initialize Servo GPIO
        self._claim_output(self.settings.servo.gpio)
        # Set initial position to center
        lgpio.tx_servo(self.handle, self.settings.servo.gpio, self.settings.servo.center_us, servo_frequency=self.settings.servo.frequency)

    def _claim_output(self, gpio: int):
        lgpio.gpio_claim_output(self.handle, gpio)

    def set_steer(self, steer: Steer) -> None:
        # User defined frequency (e.g., 50Hz or more)
        lgpio.tx_servo(self.handle, self.settings.servo.gpio, steer.value, servo_frequency=self.settings.servo.frequency)

    def close(self):
        if self._own_handle and self.handle >= 0:
            lgpio.gpiochip_close(self.handle)
            self.handle = -1

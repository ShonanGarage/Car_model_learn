import lgpio
from internal.interface.gateway.dc_gateway_interface import DCGatewayInterface
from internal.domain.value_object.throttle import Throttle
from app.config.settings import Settings

class LgpDCGateway(DCGatewayInterface):
    def __init__(self, settings: Settings):
        self.settings = settings
        self._last_sent_us: int | None = None
        # ハードウェアの初期化
        self.handle = lgpio.gpiochip_open(0)
        if self.handle < 0:
            raise RuntimeError(f"gpiochip_open(0) failed: {self.handle}")

        # DCモータの初期化（GPIOピンを出力モードに設定）
        self._claim_output(self.settings.throttle.gpio)
        
    def _claim_output(self, gpio: int):
        lgpio.gpio_claim_output(self.handle, gpio)

    # DCモータの制御
    def set_throttle(self, throttle: Throttle) -> None:
        # ESC(電子スピードコントローラー)はサーボ信号と同じ形式で制御されるため、
        # デューティ比ではなくパルス幅(us)を指定できる tx_servo を使用します。
        # 0を設定すると「信号なし」となりエラーになる環境があるため、中立(1500us)を設定します。
        us = throttle.value if throttle.value > 0 else 1500
        if self._last_sent_us == us:
            return

        lgpio.tx_servo(
            self.handle,
            self.settings.throttle.gpio,
            us,
            servo_frequency=self.settings.throttle.frequency,
        )
        self._last_sent_us = us

    def close(self):
        if self.handle >= 0:
            lgpio.gpiochip_close(self.handle)
            self.handle = -1

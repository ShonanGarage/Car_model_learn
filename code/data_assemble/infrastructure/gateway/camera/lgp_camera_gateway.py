import cv2
import threading
from typing import Optional, Tuple
from internal.interface.gateway.camera_gateway_interface import CameraGatewayInterface
from app.config.settings import Settings

try:
    from picamera2 import Picamera2 # type: ignore
except ImportError:
    Picamera2 = None

class LgpCameraGateway(CameraGatewayInterface):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.camera = None
        self.picam2 = None
        
        self._last_frame = None
        self._ok = False
        self._stop_event = threading.Event()
        self._thread = None
        
        self._initialize()

    def _initialize(self):
        initialized = False
        if Picamera2:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": (self.settings.camera.img_w, self.settings.camera.img_h), "format": "RGB888"},
                    controls={"FrameRate": self.settings.camera.fps}
                )
                self.picam2.configure(config)
                self.picam2.start()
                initialized = True
            except Exception:
                self.picam2 = None

        if not initialized:
            self.camera = cv2.VideoCapture(self.settings.camera.device_number)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera.img_w)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera.img_h)
            self.camera.set(cv2.CAP_PROP_FPS, self.settings.camera.fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Start capture thread
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop_event.is_set():
            if self.picam2:
                try:
                    rgb = self.picam2.capture_array()
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    self._last_frame = bgr
                    self._ok = True
                except Exception:
                    self._ok = False
            elif self.camera:
                ok, frame = self.camera.read()
                if ok:
                    self._last_frame = frame
                    self._ok = ok
                else:
                    time.sleep(0.01)
            else:
                break

    def capture_frame(self) -> Tuple[bool, Optional[object]]:
        return self._ok, self._last_frame

    def release(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.picam2:
            self.picam2.stop()
        if self.camera:
            self.camera.release()

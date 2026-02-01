import cv2
import threading
from typing import Optional, Tuple
from internal.interface.gateway.camera_gateway_interface import CameraGatewayInterface
from app.config.settings import Settings
import time
_picamera2_import_error = None
try:
    from picamera2 import Picamera2 # type: ignore
except ImportError as e:
    _picamera2_import_error = e
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
        self._read_fail_count = 0
        self._first_frame_logged = False
        
        self._initialize()

    def _initialize(self):
        initialized = False
        if self.settings.camera.use_picamera2 and Picamera2:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": (self.settings.camera.img_w, self.settings.camera.img_h), "format": "RGB888"},
                    controls={"FrameRate": self.settings.camera.fps}
                )
                self.picam2.configure(config)
                self.picam2.start()
                initialized = True
                print("Camera init: Picamera2 OK")
            except Exception as e:
                print(f"Camera init: Picamera2 FAILED ({e})")
                self.picam2 = None
        elif self.settings.camera.use_picamera2 and not Picamera2:
            detail = f" ({_picamera2_import_error})" if _picamera2_import_error else ""
            print(f"Camera init: Picamera2 not available (ImportError){detail}")

        if not initialized:
            if self.settings.camera.use_picamera2:
                print("Camera init: fallback to OpenCV")
            self.camera = cv2.VideoCapture(self.settings.camera.device_number)
            if not self.camera.isOpened():
                print(f"Camera init: OpenCV FAILED (device {self.settings.camera.device_number})")
            else:
                print(f"Camera init: OpenCV OK (device {self.settings.camera.device_number})")
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
                    if not self._first_frame_logged:
                        print("Camera read: first frame OK")
                        self._first_frame_logged = True
                    self._read_fail_count = 0
                else:
                    self._read_fail_count += 1
                    if self._read_fail_count in (1, 10, 50, 200):
                        print(f"Camera read: FAILED x{self._read_fail_count}")
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

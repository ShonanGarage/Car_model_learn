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
        self._reopen_attempt_count = 0
        self._last_reopen_time = 0.0
        self._reopen_after_fails = 15
        self._reopen_cooldown_s = 1.0
        self._open_retry_interval_s = 1.0
        self._last_open_retry_time = 0.0
        
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
            while not self._stop_event.is_set():
                opened = self._open_opencv_camera(log_prefix="Camera init")
                if opened:
                    break
                self._reopen_attempt_count += 1
                print(f"Camera init: waiting for camera open (retry {self._reopen_attempt_count})")
                time.sleep(self._open_retry_interval_s)

        # Start capture thread
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _open_opencv_camera(self, log_prefix: str) -> bool:
        device = self.settings.camera.device_number
        camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
        backend = "CAP_V4L2"

        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(device)
            backend = "default"

        if not camera.isOpened():
            camera.release()
            self.camera = None
            print(f"{log_prefix}: OpenCV FAILED (device {device})")
            return False

        self.camera = camera
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera.img_w)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera.img_h)
        camera.set(cv2.CAP_PROP_FPS, self.settings.camera.fps)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"{log_prefix}: OpenCV OK (device {device}, backend {backend})")
        return True

    def _maybe_reopen_camera(self):
        now = time.monotonic()
        if (now - self._last_reopen_time) < self._reopen_cooldown_s:
            return
        self._last_reopen_time = now

        self._reopen_attempt_count += 1
        print(
            "Camera read: reopening OpenCV "
            f"(attempt {self._reopen_attempt_count}, fail_count {self._read_fail_count})"
        )
        if self.camera:
            self.camera.release()

        opened = self._open_opencv_camera(log_prefix="Camera reopen")
        self._ok = False
        if opened:
            self._read_fail_count = 0
            self._first_frame_logged = False

    def _maybe_retry_open_camera(self):
        now = time.monotonic()
        if (now - self._last_open_retry_time) < self._open_retry_interval_s:
            return
        self._last_open_retry_time = now

        self._reopen_attempt_count += 1
        print(f"Camera init retry: attempt {self._reopen_attempt_count}")
        opened = self._open_opencv_camera(log_prefix="Camera init retry")
        self._ok = False
        if opened:
            self._read_fail_count = 0
            self._first_frame_logged = False

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
                    if self._read_fail_count >= self._reopen_after_fails:
                        self._maybe_reopen_camera()
                    time.sleep(0.01)
            else:
                self._maybe_retry_open_camera()
                time.sleep(0.05)

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

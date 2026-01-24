import os
import time
import csv
import threading
import queue
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import lgpio

# Picamera2 は Raspberry Pi カメラ（libcamera）用。入っていない環境では無視してOK。
try:
    from picamera2 import Picamera2  # type: ignore
except Exception:
    Picamera2 = None  # type: ignore

# =========================
# ユーザー設定（ここだけ編集）
# =========================

# カメラ
# Pi環境では /dev/video* が複数出ることがあります。
# まずは 0 を試し、ダメなら "/dev/video0" のように指定してください。
CAM_DEVICE = 0  # int(index) または str("/dev/videoX")
# Pi Camera（CSI, libcamera）なら Picamera2 が安定しやすいので優先する
USE_PICAMERA2 = True
# GUI表示（imshow/waitKey）を使うか。未設定なら環境変数から自動判定。
# ヘッドレス環境（DISPLAY/WAYLAND_DISPLAY無し）では自動でOFFになります。
SHOW_UI: Optional[bool] = None
IMG_W, IMG_H = 320, 240
FPS = 10  # 10〜20がおすすめ
JPEG_QUALITY = 85

# サーボ（GPIO直結でソフトウェアタイミングのサーボパルス）
SERVO_GPIO = 18          # 例: GPIO18
SERVO_US_CENTER = 1500   # センター（要調整）
SERVO_US_MIN = 1000
SERVO_US_MAX = 2000

# スロットルは「固定」とするならログだけ固定値を入れる（不要なら消してOK）
THROTTLE_FIXED_US = 1600  # 一定前進。可変にするなら関数化してください

# HC-SR04 の (TRIG, ECHO) GPIO 番号
# 画像の配線表に合わせた設定:
# - sonar1: Front (中央)      Trig GPIO23 / Echo GPIO24
# - sonar2: Front-Left       Trig GPIO25 / Echo GPIO8
# - sonar3: Front-Right      Trig GPIO7  / Echo GPIO1
SONARS: List[Tuple[int, int]] = [
    (23, 24),  # sonar1 (center)
    (25, 8),   # sonar2 (front-left)
    (7,  1),   # sonar3 (front-right)
    (12, 16),  # sonar4 (rear-left)
    (20, 21)   # sonar5 (rear-right)
]

# 測距の最大待ち時間（秒）
ECHO_TIMEOUT_S = 0.03  # 3cm〜5mくらいの想定なら0.03〜0.05でOK
# 測距間隔（クロストーク対策：1本測ったら少し待つ）
INTER_SONAR_GAP_S = 0.008  # 8ms〜20msくらいで調整

# データ保存先
# 実行場所(cwd)に依存しないように、デフォルトは `code/dataset/` に保存する
# 例: /home/fuki/Garage/learn/code/data_assemble/main.py の場合
#     OUT_DIR = /home/fuki/Garage/learn/code/dataset
OUT_DIR = str((Path(__file__).resolve().parents[1] / "dataset").resolve())

# course_id 切替：キーボードで変更（OpenCVウィンドウがアクティブな状態で）
# 0キー → course_id=0, 1キー → course_id=1, ... 9キーまで
# qキー → 終了

# =========================
# HC-SR04 測距クラス（lgpio）
# =========================

class HCSR04:
    def __init__(self, handle: int, trig_gpio: int, echo_gpio: int,
                 timeout_s: float = 0.03):
        self.h = handle
        self.trig = trig_gpio
        self.echo = echo_gpio
        self.timeout_s = timeout_s

        try:
            _claim_output_with_recover(self.h, self.trig, 0)
        except lgpio.error as e:
            raise RuntimeError(f"Failed to claim TRIG GPIO{self.trig}: {e}") from e
        try:
            _claim_input_with_recover(self.h, self.echo)
        except lgpio.error as e:
            raise RuntimeError(f"Failed to claim ECHO GPIO{self.echo}: {e}") from e
        lgpio.gpio_write(self.h, self.trig, 0)

    def read_distance_m(self) -> float:
        """
        距離をmで返す。失敗時は -1.0
        """
        # ECHOを安定させる
        lgpio.gpio_write(self.h, self.trig, 0)
        time.sleep(0.0002)

        # 10usトリガ
        rc = lgpio.tx_pulse(self.h, self.trig, 10, 1000, pulse_cycles=1)
        if rc < 0:
            # fallback（環境によってはtx_pulseが使えない場合がある）
            lgpio.gpio_write(self.h, self.trig, 1)
            time.sleep(0.00002)  # 20us
            lgpio.gpio_write(self.h, self.trig, 0)

        t0 = time.time()

        # ECHO立ち上がり待ち
        while lgpio.gpio_read(self.h, self.echo) == 0:
            if time.time() - t0 > self.timeout_s:
                return -1.0

        start = time.time()

        # ECHO立ち下がり待ち
        while lgpio.gpio_read(self.h, self.echo) == 1:
            if time.time() - start > self.timeout_s:
                return -1.0

        end = time.time()
        pulse = end - start  # 秒

        # 音速 343m/s を使用：距離 = (時間 * 343) / 2
        dist_m = (pulse * 343.0) / 2.0

        # 明らかな外れ値を弾く（必要に応じて調整）
        if dist_m < 0.02 or dist_m > 5.0:
            return -1.0
        return dist_m

# =========================
# メイン
# =========================

def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _open_picamera2(width: int, height: int, fps: int):
    """
    Picamera2 を起動して capture_array() でフレーム取得できる状態にする。
    戻り値は Picamera2 インスタンス（finallyで stop() すること）。
    """
    if Picamera2 is None:
        raise RuntimeError("Picamera2 is not available.")

    picam2 = Picamera2()
    main_cfg = {"size": (width, height), "format": "RGB888"}
    controls = {"FrameRate": fps}

    # バージョン差分の吸収
    if hasattr(picam2, "create_video_configuration"):
        cfg = picam2.create_video_configuration(main=main_cfg, controls=controls)
    else:
        cfg = picam2.create_preview_configuration(main=main_cfg, controls=controls)

    picam2.configure(cfg)
    picam2.start()
    # 起動直後は黒/不安定なことがあるので少し待つ
    time.sleep(0.2)
    return picam2

def _open_camera(device, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """
    Raspberry Pi では OpenCV がデフォルトで GStreamer を選び、
    パイプライン作成に失敗して read() が常に False になることがある。
    まず V4L2 を優先して open し、だめなら他の候補も試す。
    """
    backends = [cv2.CAP_V4L2, cv2.CAP_GSTREAMER]

    sources = []
    sources.append(device)
    if isinstance(device, int):
        # 複数の video node がある環境向けに軽く探索
        for i in range(0, 8):
            if i != device:
                sources.append(i)

    for src in sources:
        for backend in backends:
            cap = cv2.VideoCapture(src, backend)
            if not cap.isOpened():
                cap.release()
                continue

            # バッファを小さくしてメモリ/遅延を抑える
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # MJPGを優先（軽い＆Piで安定しやすい）
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()

    raise RuntimeError(
        "Camera open failed. CAM_DEVICE を '/dev/video0' などに変更して試してください。"
    )

def _camera_read(camera) -> tuple[bool, Optional[object]]:
    """
    camera が cv2.VideoCapture でも Picamera2 でも同じ形で読めるようにする。
    戻り値: (ok, frame_bgr)
    """
    if Picamera2 is not None and isinstance(camera, Picamera2):
        try:
            rgb = camera.capture_array()
            if rgb is None:
                return False, None
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return True, bgr
        except Exception:
            return False, None

    ok, frame = camera.read()
    return ok, frame

class _StdinKeyReader:
    """
    headless環境向け: stdinから1行ずつ読み、コマンドとしてキューに入れる。
    - 0..9: course_id
    - a/d : steer調整
    - q   : 終了
    """
    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line is None:
                continue
            s = line.strip()
            if not s:
                continue
            self._q.put(s)

    def poll(self) -> Optional[str]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

def _claim_output_with_recover(handle: int, gpio: int, level: int = 0) -> None:
    """
    lgpioのGPIO claimが 'GPIO busy' で落ちることがあるため、
    free→再claim を一度だけ試す。
    """
    try:
        lgpio.gpio_claim_output(handle, gpio, level)
        return
    except lgpio.error as e:
        if "GPIO busy" not in str(e):
            raise
        # 既存のclaimを解放できるなら解放して再試行
        try:
            lgpio.gpio_free(handle, gpio)
        except Exception:
            pass
        lgpio.gpio_claim_output(handle, gpio, level)

def _claim_input_with_recover(handle: int, gpio: int) -> None:
    try:
        lgpio.gpio_claim_input(handle, gpio)
        return
    except lgpio.error as e:
        if "GPIO busy" not in str(e):
            raise
        try:
            lgpio.gpio_free(handle, gpio)
        except Exception:
            pass
        lgpio.gpio_claim_input(handle, gpio)

def main():
    out = Path(OUT_DIR)
    (out / "images").mkdir(parents=True, exist_ok=True)
    log_path = out / "log.csv"
    is_new = not log_path.exists()

    h: Optional[int] = None
    cap: Optional[cv2.VideoCapture] = None
    picam2 = None
    steer_us = SERVO_US_CENTER

    try:
        # lgpio: gpiochip をオープン
        # ほとんどの環境では /dev/gpiochip0 を使います
        h = lgpio.gpiochip_open(0)
        if h < 0:
            raise RuntimeError(
                f"gpiochip_open(0) failed: {h}. "
                "権限を確認してください（例: sudoで実行、または/dev/gpiochip*へのアクセス権付与）。"
            )

        # サーボ初期化
        try:
            _claim_output_with_recover(h, SERVO_GPIO, 0)
        except lgpio.error as e:
            raise RuntimeError(
                f"Failed to claim SERVO GPIO{SERVO_GPIO}: {e}\n"
                "別プロセスがGPIOを掴んでいる可能性があります。\n"
                "対処: 他の制御スクリプト/サービスを止める、または再起動後に実行してください。"
            ) from e

        rc = lgpio.tx_servo(h, SERVO_GPIO, steer_us, servo_frequency=50)
        if rc < 0:
            raise RuntimeError(f"Failed to start servo on GPIO{SERVO_GPIO}: {rc}")

        # 超音波初期化
        sonars = [HCSR04(h, trig, echo, timeout_s=ECHO_TIMEOUT_S) for trig, echo in SONARS]

        # カメラ初期化
        if USE_PICAMERA2:
            try:
                picam2 = _open_picamera2(IMG_W, IMG_H, FPS)
            except Exception as e:
                print(f"Picamera2 open failed; fallback to OpenCV: {e}")
                picam2 = None
                cap = _open_camera(CAM_DEVICE, IMG_W, IMG_H, FPS)
        else:
            cap = _open_camera(CAM_DEVICE, IMG_W, IMG_H, FPS)

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]

        # UI用
        course_id = 0
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        show_ui = has_display if SHOW_UI is None else bool(SHOW_UI)
        stdin_reader = _StdinKeyReader() if not show_ui else None

        print("Recording started.")
        if show_ui:
            print("キー操作: 0-9でcourse_id変更 / a,dでステア微調整 / qで終了")
            print("OpenCVウィンドウをアクティブにしてキー入力してください。")
        else:
            print("Headless mode: GUIなしで収集します（Enter付きで入力）")
            print("入力: 0-9(course_id) / a(left) / d(right) / q(quit)")

        period = 1.0 / FPS
        next_t = time.time()

        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow([
                    "t", "image_path",
                    "sonar1_m", "sonar2_m", "sonar3_m",
                    "steer_us", "throttle_us",
                    "course_id"
                ])

            while True:
                now = time.time()
                if now < next_t:
                    time.sleep(max(0.0, next_t - now))
                t = time.time()
                next_t = t + period

                # 画像取得
                ok, frame = _camera_read(picam2 if picam2 is not None else cap)
                if not ok:
                    print("Camera read failed; skipping frame.")
                    continue

                # 超音波（順次測距で干渉低減）
                dists = []
                for s in sonars:
                    d = s.read_distance_m()
                    dists.append(d)
                    time.sleep(INTER_SONAR_GAP_S)

                # 一つでも正常なら [Normal]、全てダメなら [Error]
                if any(v > 0 for v in dists):
                    s_str = " / ".join(f"{v:.2f}m" if v > 0 else "ERR" for v in dists)
                    print(f"Sonars: {s_str} [Normal]")
                else:
                    print(f"Sonars: {dists} [Error: All sensors failed]")

                # 保存
                img_name = f"{int(t*1000)}.jpg"
                img_rel = f"images/{img_name}"
                img_path = out / img_rel
                cv2.imencode(".jpg", frame, encode_params)[1].tofile(str(img_path))

                # ログ
                w.writerow([t, img_rel, *dists, steer_us, THROTTLE_FIXED_US, course_id])

                # 表示と入力
                key_cmd: Optional[str] = None
                if show_ui:
                    disp = frame.copy()
                    cv2.putText(disp, f"course_id={course_id} steer_us={steer_us}",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    try:
                        cv2.imshow("collect", disp)
                    except Exception:
                        # GUIが途中で使えない環境だった場合は headless に落とす
                        show_ui = False
                        stdin_reader = _StdinKeyReader()
                        print("GUI unavailable; switched to headless mode.")

                    if show_ui:
                        key = cv2.waitKey(1) & 0xFF
                        if key != 255:
                            key_cmd = chr(key)
                else:
                    # headless: stdinコマンドをポーリング
                    assert stdin_reader is not None
                    key_cmd = stdin_reader.poll()

                if not key_cmd:
                    continue

                # 複数文字入力（例: "1" や "a"）にも対応
                c = key_cmd.strip()[0]
                if c == 'q':
                    break
                if '0' <= c <= '9':
                    course_id = ord(c) - ord('0')
                if c == 'a':
                    steer_us = clamp(steer_us - 10, SERVO_US_MIN, SERVO_US_MAX)
                    lgpio.tx_servo(h, SERVO_GPIO, steer_us, servo_frequency=50)
                if c == 'd':
                    steer_us = clamp(steer_us + 10, SERVO_US_MIN, SERVO_US_MAX)
                    lgpio.tx_servo(h, SERVO_GPIO, steer_us, servo_frequency=50)

    finally:
        if picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # サーボ停止（安全のため）
        if h is not None:
            try:
                lgpio.tx_servo(h, SERVO_GPIO, 0, servo_frequency=50)
            except Exception:
                pass
            try:
                lgpio.gpiochip_close(h)
            except Exception:
                pass
        print("Recording stopped.")

if __name__ == "__main__":
    main()

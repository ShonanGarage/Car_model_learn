from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.container import Container

# Ensure repo root is on sys.path so machine_learning can be imported
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning.config import Config as MlConfig  # noqa: E402
from machine_learning.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    prepare_data_from_csv,
    steer_to_class,
)
from machine_learning.model import DrivingModel  # noqa: E402
from app.entrypoints.ml.config import MlRunConfig  # noqa: E402

def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_numeric(
    *,
    steer_us: float,
    throttle_us: float,
    class_values: tuple[int, ...],
) -> np.ndarray:
    steer_cls = steer_to_class(steer_us, class_values)

    steer_one_hot = np.zeros(len(class_values), dtype=np.float32)
    steer_one_hot[steer_cls] = 1.0

    numeric = np.concatenate(
        [
            steer_one_hot,
            np.array([throttle_us], dtype=np.float32),
        ]
    ).astype(np.float32)
    return numeric


def _to_tensor(frame: object, transform: transforms.Compose) -> torch.Tensor:
    import cv2

    if frame is None:
        raise ValueError("frame is None")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return transform(img)


def main() -> None:
    cfg = MlRunConfig()
    device = _resolve_device(cfg.device)

    prepared = prepare_data_from_csv(cfg.data_csv_path)
    numeric_dim = len(prepared.numeric_columns)

    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    class_names = tuple(ckpt.get("class_names", MlConfig().data.servo_class_names))
    class_values = tuple(ckpt.get("class_values", MlConfig().data.servo_class_us))

    model = DrivingModel(numeric_dim=numeric_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    container = Container()
    container.drive_service.set_course_id(container.settings.course_id)

    interval = 1.0 / max(1, cfg.update_hz)
    print("[ml] autodrive started")
    print(f"[ml] device={device} update_hz={cfg.update_hz}")
    print(f"[ml] checkpoint={cfg.checkpoint_path}")

    try:
        while True:
            loop_start = time.time()
            container.drive_service.update()

            _, ok, frame = container.drive_service.get_sensor_snapshot()
            if not ok or frame is None:
                container.drive_service.stop()
                print("[ml][warn] camera not ok -> stop")
                time.sleep(interval)
                continue

            try:
                image = _to_tensor(frame, image_transform)
                numeric = _build_numeric(
                    steer_us=float(container.drive_service.current_steer_us),
                    throttle_us=float(container.drive_service.current.throttle.value),
                    class_values=class_values,
                )

                with torch.no_grad():
                    steer_logits, throttle_pred = model(
                        image.unsqueeze(0).to(device),
                        torch.from_numpy(numeric).unsqueeze(0).to(device),
                    )

                steer_idx = int(steer_logits.argmax(dim=1).item())
                steer_value = class_values[steer_idx]
                steer_label = class_names[steer_idx] if steer_idx < len(class_names) else str(steer_idx)
                throttle_value = float(throttle_pred.item())
                if throttle_value < 1000:
                    throttle_us = 0
                else:
                    throttle_us = int(np.clip(throttle_value, 1000, 2000))

                container.drive_service.set_throttle_us(throttle_us)
                container.drive_service.set_steer_us(int(steer_value))
                print(f"[ml] steer={steer_label} ({steer_value}) throttle={throttle_us}")
            except Exception as exc:
                container.drive_service.stop()
                print(f"[ml][error] inference failed -> stop: {exc}")

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, interval - elapsed))
    finally:
        container.drive_service.shutdown()
        container.camera_gateway.release()
        container.camera_view.stop()
        container.data_repository.stop()
        print("[ml] shutdown complete")


if __name__ == "__main__":
    main()

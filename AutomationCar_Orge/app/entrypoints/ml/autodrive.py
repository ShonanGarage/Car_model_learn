from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.container import Container

# Ensure workspace root is on sys.path so machine_learning can be imported.
def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "machine_learning").is_dir():
            return p
    raise RuntimeError("could not locate repo root containing machine_learning/")


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from machine_learning.config import Config as MlConfig  # noqa: E402
from machine_learning.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    throttle_to_class,
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
    throttle_threshold_us: float,
    throttle_class_count: int,
) -> np.ndarray:
    steer_cls = steer_to_class(steer_us, class_values)
    throttle_cls = throttle_to_class(throttle_us, throttle_threshold_us)

    steer_one_hot = np.zeros(len(class_values), dtype=np.float32)
    steer_one_hot[steer_cls] = 1.0

    throttle_one_hot = np.zeros(throttle_class_count, dtype=np.float32)
    throttle_one_hot[throttle_cls] = 1.0

    numeric = np.concatenate(
        [
            steer_one_hot,
            throttle_one_hot,
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

    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    ml_cfg = MlConfig().data
    class_names = tuple(ckpt.get("class_names", ml_cfg.servo_class_names))
    class_values = tuple(ckpt.get("class_values", ml_cfg.servo_class_us))
    throttle_class_names = tuple(ml_cfg.throttle_class_names)
    throttle_threshold_us = float(ml_cfg.throttle_class_threshold_us)

    numeric_dim = int(ckpt["numeric_dim"])
    expected_numeric_dim = len(class_values) + len(throttle_class_names)
    if numeric_dim != expected_numeric_dim:
        raise ValueError(
            "numeric_dim mismatch: "
            f"checkpoint={numeric_dim}, expected={expected_numeric_dim} "
            f"(len(servo_classes)+len(throttle_classes))"
        )

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
                    throttle_threshold_us=throttle_threshold_us,
                    throttle_class_count=len(throttle_class_names),
                )

                with torch.no_grad():
                    steer_logits, throttle_logits = model(
                        image.unsqueeze(0).to(device),
                        torch.from_numpy(numeric).unsqueeze(0).to(device),
                    )

                steer_idx = int(steer_logits.argmax(dim=1).item())
                steer_value = class_values[steer_idx]
                steer_label = class_names[steer_idx] if steer_idx < len(class_names) else str(steer_idx)
                throttle_idx = int(throttle_logits.argmax(dim=1).item())
                throttle_label = (
                    throttle_class_names[throttle_idx]
                    if throttle_idx < len(throttle_class_names)
                    else str(throttle_idx)
                )
                throttle_us = cfg.throttle_low_us if throttle_idx == 0 else cfg.throttle_high_us

                container.drive_service.set_throttle_us(throttle_us)
                container.drive_service.set_steer_us(int(steer_value))
                print(
                    f"[ml] steer={steer_label} ({steer_value}) "
                    f"throttle={throttle_label} ({throttle_us})"
                )
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

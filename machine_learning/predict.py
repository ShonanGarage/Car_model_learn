from __future__ import annotations

from pathlib import Path

import torch

from .config import Config
from .dataset import DrivingDataset, prepare_data_from_csv
from .model import DrivingModel


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def predict_one(cfg: Config) -> None:
    device = _resolve_device(cfg.train.device)
    print(f"device: {device}")

    prepared = prepare_data_from_csv(cfg.data.csv_path)
    numeric_dim = len(prepared.numeric_columns)

    ckpt_path = cfg.train.checkpoint_dir / "best.pt"
    ckpt = _load_checkpoint(ckpt_path, device=device)
    model = DrivingModel(numeric_dim=numeric_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class_names = tuple(ckpt.get("class_names", cfg.data.servo_class_names))
    class_values = tuple(ckpt.get("class_values", cfg.data.servo_class_us))

    # まずは val の先頭サンプルで推論する
    ds = DrivingDataset(prepared.val)
    image, numeric, steer_cls_t, throttle_us_t = ds[0]

    with torch.no_grad():
        steer_logits, throttle_pred = model(
            image.unsqueeze(0).to(device),
            numeric.unsqueeze(0).to(device),
        )

    steer_idx = int(steer_logits.argmax(dim=1).item())
    steer_label = class_names[steer_idx] if steer_idx < len(class_names) else str(steer_idx)
    steer_value = class_values[steer_idx] if steer_idx < len(class_values) else steer_idx

    target_idx = int(steer_cls_t.item())
    target_label = class_names[target_idx] if target_idx < len(class_names) else str(target_idx)
    target_value = class_values[target_idx] if target_idx < len(class_values) else target_idx

    throttle_mean = float(prepared.stats.throttle_target_mean)
    throttle_std = float(prepared.stats.throttle_target_std)
    throttle_pred_us = float(throttle_pred.item()) * throttle_std + throttle_mean
    throttle_target_us = float(throttle_us_t.item()) * throttle_std + throttle_mean

    print(f"steer_pred: {steer_label} ({steer_value})")
    print(f"steer_target: {target_label} ({target_value})")
    print(f"throttle_pred(us): {throttle_pred_us:.1f}")
    print(f"throttle_target(us): {throttle_target_us:.1f}")


def main() -> None:  # pragma: no cover - script entry
    predict_one(Config())


if __name__ == "__main__":  # pragma: no cover - script mode
    main()

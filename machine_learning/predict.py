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

    ckpt = _load_checkpoint(cfg.train.checkpoint_path, device=device)
    model = DrivingModel(numeric_dim=numeric_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class_names = tuple(ckpt.get("class_names", cfg.data.servo_class_names))
    class_values = tuple(ckpt.get("class_values", cfg.data.servo_class_us))

    # まずは val の先頭サンプルで推論する
    ds = DrivingDataset(prepared.val)
    image, numeric, _steer_t, move_t = ds[0]

    with torch.no_grad():
        move_logits = model(
            image.unsqueeze(0).to(device),
            numeric.unsqueeze(0).to(device),
        )

    move_idx = int(move_logits.argmax(dim=1).item())
    move_label = class_names[move_idx] if move_idx < len(class_names) else str(move_idx)
    move_value = class_values[move_idx] if move_idx < len(class_values) else move_idx

    target_idx = int(move_t.item())
    target_label = class_names[target_idx] if target_idx < len(class_names) else str(target_idx)
    target_value = class_values[target_idx] if target_idx < len(class_values) else target_idx

    print(f"move_pred: {move_label} ({move_value})")
    print(f"move_target: {target_label} ({target_value})")


def main() -> None:  # pragma: no cover - script entry
    predict_one(Config())


if __name__ == "__main__":  # pragma: no cover - script mode
    main()

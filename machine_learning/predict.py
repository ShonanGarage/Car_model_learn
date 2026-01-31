from __future__ import annotations

from pathlib import Path

import torch

from .config import Config
from .dataset import DrivingDataset, prepare_data_from_csv
from .model import DrivingModel

MOVE_LABELS = {0: "STOP", 1: "FORWARD", 2: "BACKWARD"}


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

    # まずは val の先頭サンプルで推論する
    ds = DrivingDataset(prepared.val)
    image, numeric, steer_t, move_t = ds[0]

    with torch.no_grad():
        steer_pred, move_logits = model(
            image.unsqueeze(0).to(device),
            numeric.unsqueeze(0).to(device),
        )

    move_idx = int(move_logits.argmax(dim=1).item())
    move_label = MOVE_LABELS.get(move_idx, str(move_idx))

    print(f"steer_pred: {float(steer_pred.item()):.2f} (target: {float(steer_t.item()):.2f})")
    print(f"move_pred: {move_label} (target: {MOVE_LABELS.get(int(move_t.item()), int(move_t.item()))})")


def main() -> None:  # pragma: no cover - script entry
    predict_one(Config())


if __name__ == "__main__":  # pragma: no cover - script mode
    main()

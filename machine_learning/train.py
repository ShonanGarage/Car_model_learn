from __future__ import annotations

from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config, DataConfig, TrainConfig
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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_metrics_plot(
    checkpoint_dir: Path,
    *,
    epochs: list[int],
    train_loss: list[float],
    val_loss: list[float],
    val_acc: list[float],
) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax_loss.plot(epochs, train_loss, label="train_loss")
    ax_loss.plot(epochs, val_loss, label="val_loss")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, val_acc, label="val_acc")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = checkpoint_dir / "metrics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_config_json(checkpoint_dir: Path, cfg: Config) -> None:
    data = {
        "train": {
            "batch_size": cfg.train.batch_size,
            "epochs": cfg.train.epochs,
            "lr": cfg.train.lr,
            "weight_decay": cfg.train.weight_decay,
            "lambda_move": cfg.train.lambda_move,
            "num_workers": cfg.train.num_workers,
            "device": cfg.train.device,
            "log_every": cfg.train.log_every,
            "checkpoint_dir": str(cfg.train.checkpoint_dir),
        },
        "data": {
            "dataset_repo_id": cfg.data.dataset_repo_id,
            "dataset_revision": cfg.data.dataset_revision,
            "dataset_local_dir": str(cfg.data.dataset_local_dir),
            "csv_path": str(cfg.data.csv_path),
            "servo_class_us": list(cfg.data.servo_class_us),
            "servo_class_names": list(cfg.data.servo_class_names),
            "k": cfg.data.k,
            "val_fraction": cfg.data.val_fraction,
            "seed": cfg.data.seed,
        },
    }
    out_path = checkpoint_dir / "config.json"
    out_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def _make_loaders(data_cfg: DataConfig, train_cfg: TrainConfig) -> tuple[DataLoader, DataLoader, int]:
    prepared = prepare_data_from_csv(data_cfg.csv_path)

    train_ds = DrivingDataset(
        prepared.train,
    )
    val_ds = DrivingDataset(
        prepared.val,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, len(prepared.numeric_columns)


def _step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
    loss_fn: nn.Module,
    throttle_loss_fn: nn.Module,
    lambda_throttle: float,
) -> torch.Tensor:
    image, numeric, steer_cls_t, throttle_us_t = batch
    image = image.to(device, non_blocking=True)
    numeric = numeric.to(device, non_blocking=True)
    steer_cls_t = steer_cls_t.to(device, non_blocking=True)
    throttle_us_t = throttle_us_t.to(device, non_blocking=True)

    steer_logits, throttle_pred = model(image, numeric)
    steer_loss = loss_fn(steer_logits, steer_cls_t)
    throttle_loss = throttle_loss_fn(throttle_pred, throttle_us_t)
    loss = steer_loss + lambda_throttle * throttle_loss
    return loss


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    throttle_loss_fn: nn.Module,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_throttle_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            image, numeric, steer_cls_t, throttle_us_t = batch
            image = image.to(device, non_blocking=True)
            numeric = numeric.to(device, non_blocking=True)
            steer_cls_t = steer_cls_t.to(device, non_blocking=True)
            throttle_us_t = throttle_us_t.to(device, non_blocking=True)

            steer_logits, throttle_pred = model(image, numeric)
            loss = loss_fn(steer_logits, steer_cls_t)
            throttle_loss = throttle_loss_fn(throttle_pred, throttle_us_t)

            total_loss += float(loss.item()) * len(steer_cls_t)
            total_throttle_loss += float(throttle_loss.item()) * len(steer_cls_t)

            preds = steer_logits.argmax(dim=1)
            total_correct += int((preds == steer_cls_t).sum().item())
            total_count += int(len(steer_cls_t))

    return {
        "loss": total_loss / total_count,
        "steer_cls_acc": total_correct / total_count,
        "throttle_mse": total_throttle_loss / total_count,
    }


def train(cfg: Config) -> None:
    device = _resolve_device(cfg.train.device)
    print(f"device: {device}")

    train_loader, val_loader, numeric_dim = _make_loaders(cfg.data, cfg.train)
    print(f"numeric_dim: {numeric_dim}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    model = DrivingModel(numeric_dim=numeric_dim).to(device)

    loss_fn = nn.CrossEntropyLoss()
    throttle_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    best_val = float("inf")
    checkpoint_dir = cfg.train.checkpoint_dir
    _ensure_dir(checkpoint_dir)
    best_path = checkpoint_dir / "best.pt"
    _save_config_json(checkpoint_dir, cfg)

    epoch_history: list[int] = []
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_acc_history: list[float] = []

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()

        running_loss = 0.0
        seen = 0

        train_iter = tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False)

        for step_idx, batch in enumerate(train_iter, start=1):
            optimizer.zero_grad(set_to_none=True)

            loss = _step(
                model,
                batch,
                device=device,
                loss_fn=loss_fn,
                throttle_loss_fn=throttle_loss_fn,
                lambda_throttle=cfg.train.lambda_move,
            )
            loss.backward()
            optimizer.step()

            image, numeric, steer_cls_t, throttle_us_t = batch
            bsz = len(steer_cls_t)
            running_loss += float(loss.item()) * bsz
            seen += bsz

            train_iter.set_postfix(
                loss=f"{running_loss/seen:.4f}",
            )

            if step_idx % cfg.train.log_every == 0:
                print(
                    f"epoch {epoch:02d} step {step_idx:04d} "
                    f"loss={running_loss/seen:.4f} "
                )

        val_metrics = _evaluate(
            model,
            val_loader,
            device=device,
            loss_fn=loss_fn,
            throttle_loss_fn=throttle_loss_fn,
        )
        print(
            f"epoch {epoch:02d} val "
            f"loss={val_metrics['loss']:.4f} "
            f"steer_cls_acc={val_metrics['steer_cls_acc']:.3f} "
            f"throttle_mse={val_metrics['throttle_mse']:.2f}"
        )

        epoch_history.append(epoch)
        train_loss_history.append(running_loss / seen)
        val_loss_history.append(val_metrics["loss"])
        val_acc_history.append(val_metrics["steer_cls_acc"])
        _save_metrics_plot(
            checkpoint_dir,
            epochs=epoch_history,
            train_loss=train_loss_history,
            val_loss=val_loss_history,
            val_acc=val_acc_history,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "numeric_dim": numeric_dim,
                    "class_names": cfg.data.servo_class_names,
                    "class_values": cfg.data.servo_class_us,
                    "config": cfg,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
            print(f"saved checkpoint: {best_path}")


def main() -> None:  # pragma: no cover - entry point
    train(Config())


if __name__ == "__main__":  # pragma: no cover - script mode
    main()

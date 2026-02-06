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
    train_total_loss: list[float],
    val_total_loss: list[float],
    train_steer_ce: list[float],
    val_steer_ce: list[float],
    train_throttle_mse: list[float],
    val_throttle_mse: list[float],
    val_acc: list[float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_total = axes[0, 0]
    ax_ce = axes[0, 1]
    ax_mse = axes[1, 0]
    ax_acc = axes[1, 1]

    ax_total.plot(epochs, train_total_loss, label="train_total_loss")
    ax_total.plot(epochs, val_total_loss, label="val_total_loss")
    ax_total.set_ylabel("loss")
    ax_total.set_title("Total Loss")
    ax_total.legend()
    ax_total.grid(True, alpha=0.3)

    ax_ce.plot(epochs, train_steer_ce, label="train_steer_ce")
    ax_ce.plot(epochs, val_steer_ce, label="val_steer_ce")
    ax_ce.set_ylabel("ce")
    ax_ce.set_title("Steer CE")
    ax_ce.legend()
    ax_ce.grid(True, alpha=0.3)

    ax_mse.plot(epochs, train_throttle_mse, label="train_throttle_mse")
    ax_mse.plot(epochs, val_throttle_mse, label="val_throttle_mse")
    ax_mse.set_ylabel("mse")
    ax_mse.set_title("Throttle MSE (norm)")
    ax_mse.legend()
    ax_mse.grid(True, alpha=0.3)

    ax_acc.plot(epochs, val_acc, label="val_steer_acc")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Steer Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    ax_mse.set_xlabel("epoch")
    ax_acc.set_xlabel("epoch")

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


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    throttle_loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    lambda_throttle: float,
    log_every: int,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_total_loss = 0.0
    total_steer_ce = 0.0
    total_throttle_mse = 0.0
    total_count = 0

    train_iter = tqdm(loader, desc=f"epoch {epoch:02d}", leave=False)
    for step_idx, batch in enumerate(train_iter, start=1):
        optimizer.zero_grad(set_to_none=True)

        image, numeric, steer_cls_t, throttle_us_t = batch
        image = image.to(device, non_blocking=True)
        numeric = numeric.to(device, non_blocking=True)
        steer_cls_t = steer_cls_t.to(device, non_blocking=True)
        throttle_us_t = throttle_us_t.to(device, non_blocking=True)

        steer_logits, throttle_pred = model(image, numeric)
        steer_ce = loss_fn(steer_logits, steer_cls_t)
        throttle_mse = throttle_loss_fn(throttle_pred, throttle_us_t)
        total_loss = steer_ce + lambda_throttle * throttle_mse

        total_loss.backward()
        optimizer.step()

        bsz = len(steer_cls_t)
        total_count += bsz
        total_total_loss += float(total_loss.item()) * bsz
        total_steer_ce += float(steer_ce.item()) * bsz
        total_throttle_mse += float(throttle_mse.item()) * bsz

        train_iter.set_postfix(loss=f"{total_total_loss/total_count:.4f}")
        if step_idx % log_every == 0:
            print(
                f"epoch {epoch:02d} step {step_idx:04d} "
                f"total_loss={total_total_loss/total_count:.4f} "
                f"steer_ce={total_steer_ce/total_count:.4f} "
                f"throttle_mse_norm={total_throttle_mse/total_count:.4f}"
            )

    return {
        "total_loss": total_total_loss / total_count,
        "steer_ce": total_steer_ce / total_count,
        "throttle_mse": total_throttle_mse / total_count,
    }


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_fn: nn.Module,
    throttle_loss_fn: nn.Module,
    lambda_throttle: float,
) -> dict[str, float]:
    model.eval()

    total_total_loss = 0.0
    total_steer_ce = 0.0
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
            steer_ce = loss_fn(steer_logits, steer_cls_t)
            throttle_loss = throttle_loss_fn(throttle_pred, throttle_us_t)
            total_loss = steer_ce + lambda_throttle * throttle_loss

            total_total_loss += float(total_loss.item()) * len(steer_cls_t)
            total_steer_ce += float(steer_ce.item()) * len(steer_cls_t)
            total_throttle_loss += float(throttle_loss.item()) * len(steer_cls_t)

            preds = steer_logits.argmax(dim=1)
            total_correct += int((preds == steer_cls_t).sum().item())
            total_count += int(len(steer_cls_t))

    return {
        "total_loss": total_total_loss / total_count,
        "steer_ce": total_steer_ce / total_count,
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
    train_total_loss_history: list[float] = []
    val_total_loss_history: list[float] = []
    train_steer_ce_history: list[float] = []
    val_steer_ce_history: list[float] = []
    train_throttle_mse_history: list[float] = []
    val_throttle_mse_history: list[float] = []
    val_acc_history: list[float] = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = _train_epoch(
            model,
            train_loader,
            device=device,
            loss_fn=loss_fn,
            throttle_loss_fn=throttle_loss_fn,
            optimizer=optimizer,
            lambda_throttle=cfg.train.lambda_move,
            log_every=cfg.train.log_every,
            epoch=epoch,
        )

        val_metrics = _evaluate(
            model,
            val_loader,
            device=device,
            loss_fn=loss_fn,
            throttle_loss_fn=throttle_loss_fn,
            lambda_throttle=cfg.train.lambda_move,
        )
        print(
            f"epoch {epoch:02d} val "
            f"total_loss={val_metrics['total_loss']:.4f} "
            f"steer_ce={val_metrics['steer_ce']:.4f} "
            f"steer_cls_acc={val_metrics['steer_cls_acc']:.3f} "
            f"throttle_mse_norm={val_metrics['throttle_mse']:.4f}"
        )

        epoch_history.append(epoch)
        train_total_loss_history.append(train_metrics["total_loss"])
        val_total_loss_history.append(val_metrics["total_loss"])
        train_steer_ce_history.append(train_metrics["steer_ce"])
        val_steer_ce_history.append(val_metrics["steer_ce"])
        train_throttle_mse_history.append(train_metrics["throttle_mse"])
        val_throttle_mse_history.append(val_metrics["throttle_mse"])
        val_acc_history.append(val_metrics["steer_cls_acc"])
        _save_metrics_plot(
            checkpoint_dir,
            epochs=epoch_history,
            train_total_loss=train_total_loss_history,
            val_total_loss=val_total_loss_history,
            train_steer_ce=train_steer_ce_history,
            val_steer_ce=val_steer_ce_history,
            train_throttle_mse=train_throttle_mse_history,
            val_throttle_mse=val_throttle_mse_history,
            val_acc=val_acc_history,
        )

        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
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

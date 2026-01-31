from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore[assignment]

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


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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
    move_loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    image, numeric, _steer_t, move_t = batch
    image = image.to(device, non_blocking=True)
    numeric = numeric.to(device, non_blocking=True)
    move_t = move_t.to(device, non_blocking=True)

    move_logits = model(image, numeric)
    move_loss = move_loss_fn(move_logits, move_t)
    loss = move_loss
    return loss, move_loss.detach()


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    move_loss_fn: nn.Module,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_move = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            image, numeric, _steer_t, move_t = batch
            image = image.to(device, non_blocking=True)
            numeric = numeric.to(device, non_blocking=True)
            move_t = move_t.to(device, non_blocking=True)

            move_logits = model(image, numeric)
            move_loss = move_loss_fn(move_logits, move_t)
            loss = move_loss

            total_loss += float(loss.item()) * len(move_t)
            total_move += float(move_loss.item()) * len(move_t)

            preds = move_logits.argmax(dim=1)
            total_correct += int((preds == move_t).sum().item())
            total_count += int(len(move_t))

    return {
        "loss": total_loss / total_count,
        "move_ce": total_move / total_count,
        "move_acc": total_correct / total_count,
    }


def train(cfg: Config) -> None:
    device = _resolve_device(cfg.train.device)
    print(f"device: {device}")

    train_loader, val_loader, numeric_dim = _make_loaders(cfg.data, cfg.train)
    print(f"numeric_dim: {numeric_dim}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    model = DrivingModel(numeric_dim=numeric_dim).to(device)

    move_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    best_val = float("inf")
    _ensure_parent(cfg.train.checkpoint_path)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()

        running_loss = 0.0
        running_move = 0.0
        seen = 0

        if tqdm is None:
            train_iter = train_loader
        else:
            train_iter = tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False)

        for step_idx, batch in enumerate(train_iter, start=1):
            optimizer.zero_grad(set_to_none=True)

            loss, steer_loss, move_loss = _step(
                model,
                batch,
                device=device,
                move_loss_fn=move_loss_fn,
            )
            loss.backward()
            optimizer.step()

            image, numeric, _steer_t, move_t = batch
            bsz = len(move_t)
            running_loss += float(loss.item()) * bsz
            running_move += float(move_loss.item()) * bsz
            seen += bsz

            if tqdm is not None:
                train_iter.set_postfix(
                    loss=f"{running_loss/seen:.4f}",
                    move=f"{running_move/seen:.4f}",
                )

            if step_idx % cfg.train.log_every == 0:
                print(
                    f"epoch {epoch:02d} step {step_idx:04d} "
                    f"loss={running_loss/seen:.4f} "
                    f"move={running_move/seen:.4f}"
                )

        val_metrics = _evaluate(
            model,
            val_loader,
            device=device,
            move_loss_fn=move_loss_fn,
        )
        print(
            f"epoch {epoch:02d} val "
            f"loss={val_metrics['loss']:.4f} "
            f"move_ce={val_metrics['move_ce']:.4f} "
            f"move_acc={val_metrics['move_acc']:.3f}"
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
                cfg.train.checkpoint_path,
            )
            print(f"saved checkpoint: {cfg.train.checkpoint_path}")


def main() -> None:  # pragma: no cover - entry point
    train(Config())


if __name__ == "__main__":  # pragma: no cover - script mode
    main()

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import Config, DataConfig

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - optional dependency
    snapshot_download = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CSV_COLUMNS = [
    "timestamp_t",
    "image_path_t",
    "steer_cls_t",
    "throttle_us_t",
    "timestamp_tk",
    "steer_cls_tk",
    "throttle_us_tk",
    "k",
    "split",
]


@dataclass(frozen=True)
class NormalizationStats:
    """数値特徴の標準化に使う統計量。"""

    mean: np.ndarray
    std: np.ndarray
    numeric_columns: list[str]


@dataclass(frozen=True)
class PreparedSplit:
    image_paths: list[str]
    numeric: np.ndarray
    steer_cls_target: np.ndarray
    throttle_cls_target: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    """学習に必要な前処理済みデータ。"""

    train: PreparedSplit
    val: PreparedSplit
    numeric_columns: list[str]
    stats: NormalizationStats


def steer_to_class(steer_us: float, class_values: Sequence[int]) -> int:
    """steer_us を最も近いサーボ離散値のクラスIDに変換する。"""
    if not class_values:
        raise ValueError("class_values が空です。")
    diffs = [abs(steer_us - v) for v in class_values]
    return int(diffs.index(min(diffs)))


def throttle_to_class(throttle_us: float, threshold_us: float) -> int:
    return int(throttle_us >= threshold_us)


def _build_base_arrays_from_log_csv(
    csv_path: Path,
    images_dir: Path,
) -> dict[str, np.ndarray | list[str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSVが空です: {csv_path}")

    rows = [r for r in rows if r.get("drive_state") != "BLOCKED_FRONT"]
    if not rows:
        raise ValueError(f"BLOCKED_FRONT を除外した結果、CSVが空になりました: {csv_path}")

    rows = sorted(rows, key=lambda r: float(r["timestamp"]))

    timestamps = np.array([float(r["timestamp"]) for r in rows], dtype=np.float64)
    steer_us = np.array([float(r["steer_us"]) for r in rows], dtype=np.float32)
    throttle_us = np.array([float(r["throttle_us"]) for r in rows], dtype=np.float32)

    repo_root = images_dir.parent
    image_paths = [str(repo_root / str(r["image_filename"])) for r in rows]

    return {
        "timestamp": timestamps,
        "steer_us": steer_us,
        "throttle_us": throttle_us,
        "image_path": image_paths,
    }


def _assign_split(n: int, val_fraction: float, seed: int) -> np.ndarray:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction は 0 と 1 の間である必要があります。")
    rng = np.random.default_rng(seed)
    return rng.random(n) < val_fraction  # True が val


def _resolve_dataset_paths(data_cfg: DataConfig) -> tuple[Path, Path]:
    if snapshot_download is None:
        raise RuntimeError(
            "dataset_repo_id を使うには huggingface_hub が必要です。"
            " `uv add huggingface_hub` を実行してください。"
        )

    local_dir = data_cfg.dataset_local_dir / data_cfg.dataset_revision
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=data_cfg.dataset_repo_id,
        repo_type="dataset",
        revision=data_cfg.dataset_revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    repo_root = Path(local_dir)
    log_csv_path = repo_root / "log.csv"
    images_dir = repo_root / "images"
    return log_csv_path, images_dir


def build_shifted_rows_from_log_csv(
    log_csv_path: str | Path,
    images_dir: str | Path,
    *,
    k: int,
    class_values: Sequence[int],
) -> list[dict[str, str | float | int]]:
    """log.csv から t -> t+k の行データを作る（split未設定）。"""
    if k < 1:
        raise ValueError("k は 1 以上である必要があります。")

    log_csv_path = Path(log_csv_path)
    images_dir = Path(images_dir)

    base = _build_base_arrays_from_log_csv(
        log_csv_path,
        images_dir,
    )

    n = len(base["steer_us"])
    m = n - k
    if m <= 1:
        raise ValueError("データ数が少なすぎます。k を小さくしてください。")

    rows: list[dict[str, str | float | int]] = []
    for i in range(m):
        j = i + k
        steer_t = float(base["steer_us"][i])
        row = {
            "timestamp_t": float(base["timestamp"][i]),
            "image_path_t": str(base["image_path"][i]),
            "steer_cls_t": steer_to_class(steer_t, class_values),
            "throttle_us_t": float(base["throttle_us"][i]),
            "timestamp_tk": float(base["timestamp"][j]),
            "steer_cls_tk": steer_to_class(float(base["steer_us"][j]), class_values),
            "throttle_us_tk": float(base["throttle_us"][j]),
            "k": int(k),
            "split": "",  # 後で埋める
        }
        rows.append(row)
    return rows


def export_csv_from_log(
    log_csv_path: str | Path,
    images_dir: str | Path,
    csv_path: str | Path,
    *,
    k: int = 1,
    val_fraction: float = 0.2,
    seed: int = 42,
    class_values: Sequence[int],
) -> Path:
    """log.csv からCSVを作る（split列もここで確定させる）。"""
    rows = build_shifted_rows_from_log_csv(
        log_csv_path,
        images_dir,
        k=k,
        class_values=class_values,
    )

    val_mask = _assign_split(len(rows), val_fraction=val_fraction, seed=seed)
    for row, is_val in zip(rows, val_mask):
        row["split"] = "val" if bool(is_val) else "train"

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSVが空です: {csv_path}")
    return rows


def _sort_rows_by_timestamp(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda r: float(r["timestamp_t"]))

def prepare_data_from_csv(csv_path: str | Path) -> PreparedData:
    """CSVから学習用の配列と正規化統計量を作る。"""
    cfg = Config()
    throttle_class_names = tuple(cfg.data.throttle_class_names)
    if len(throttle_class_names) != 2:
        raise ValueError("throttle_class_names は2クラスで定義してください。")
    csv_path = Path(csv_path)
    rows = _load_csv_rows(csv_path)
    rows = _sort_rows_by_timestamp(rows)

    image_paths = [r["image_path_t"] for r in rows]
    steer_cls_t = np.array([int(float(r["steer_cls_t"])) for r in rows], dtype=np.int64)
    throttle_us_t = np.array([float(r["throttle_us_t"]) for r in rows], dtype=np.float32)

    steer_cls_tk = np.array([int(float(r["steer_cls_tk"])) for r in rows], dtype=np.int64)
    throttle_us_tk = np.array([float(r["throttle_us_tk"]) for r in rows], dtype=np.float32)

    split = np.array([r["split"].strip() for r in rows])

    # brake/invalid side-effectsを避けるため throttle=0 を除外する
    keep_mask = (throttle_us_t > 0.0) & (throttle_us_tk > 0.0)
    image_paths = [p for p, keep in zip(image_paths, keep_mask) if bool(keep)]
    steer_cls_t = steer_cls_t[keep_mask]
    throttle_us_t = throttle_us_t[keep_mask]
    steer_cls_tk = steer_cls_tk[keep_mask]
    throttle_us_tk = throttle_us_tk[keep_mask]
    split = split[keep_mask]

    train_mask = split == "train"
    val_mask = split == "val"

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError("split列に train/val が十分に含まれていません。CSVを作り直してください。")

    steer_class_count = len(cfg.data.servo_class_us)
    steer_one_hot = np.zeros((len(steer_cls_t), steer_class_count), dtype=np.float32)
    steer_one_hot[np.arange(len(steer_cls_t)), steer_cls_t] = 1.0
    threshold_us = float(cfg.data.throttle_class_threshold_us)
    throttle_cls_t = np.array([throttle_to_class(v, threshold_us) for v in throttle_us_t], dtype=np.int64)
    throttle_cls_tk = np.array([throttle_to_class(v, threshold_us) for v in throttle_us_tk], dtype=np.int64)
    throttle_class_count = len(throttle_class_names)
    throttle_one_hot = np.zeros((len(throttle_cls_t), throttle_class_count), dtype=np.float32)
    throttle_one_hot[np.arange(len(throttle_cls_t)), throttle_cls_t] = 1.0

    numeric = np.concatenate(
        [
            steer_one_hot,
            throttle_one_hot,
        ],
        axis=1,
    ).astype(np.float32)

    numeric_columns = (
        [f"steer_cls__{i}" for i in range(steer_class_count)] +
        [f"throttle_cls__{i}" for i in range(throttle_class_count)]
    )

    mean = np.zeros(numeric.shape[1], dtype=np.float32)
    std = np.ones(numeric.shape[1], dtype=np.float32)

    train = PreparedSplit(
        image_paths=[p for p, keep in zip(image_paths, train_mask) if keep],
        numeric=numeric[train_mask],
        steer_cls_target=steer_cls_tk[train_mask],
        throttle_cls_target=throttle_cls_tk[train_mask],
    )
    val = PreparedSplit(
        image_paths=[p for p, keep in zip(image_paths, val_mask) if keep],
        numeric=numeric[val_mask],
        steer_cls_target=steer_cls_tk[val_mask],
        throttle_cls_target=throttle_cls_tk[val_mask],
    )

    stats = NormalizationStats(mean=mean, std=std, numeric_columns=numeric_columns)
    return PreparedData(
        train=train,
        val=val,
        numeric_columns=numeric_columns,
        stats=stats,
    )


class DrivingDataset(Dataset):
    """画像 + 数値特徴 -> steer_cls のDataset。"""

    def __init__(
        self,
        split: PreparedSplit,
        *,
        image_transform: transforms.Compose | None = None,
    ) -> None:
        self.image_paths = split.image_paths
        self.numeric = split.numeric
        self.steer_cls_target = split.steer_cls_target
        self.throttle_cls_target = split.throttle_cls_target

        if image_transform is None:
            image_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        self.image_transform = image_transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.image_paths)

    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.image_transform(img)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self._load_image(self.image_paths[idx])
        numeric = torch.from_numpy(self.numeric[idx])
        steer_cls_target = torch.tensor(self.steer_cls_target[idx], dtype=torch.long)
        throttle_cls_target = torch.tensor(self.throttle_cls_target[idx], dtype=torch.long)
        return image, numeric, steer_cls_target, throttle_cls_target


def main() -> None:  # pragma: no cover - CLI entry
    cfg = Config()
    log_csv_path, images_dir = _resolve_dataset_paths(cfg.data)
    csv_path = export_csv_from_log(
        log_csv_path,
        images_dir,
        cfg.data.csv_path,
        k=cfg.data.k,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.data.seed,
        class_values=cfg.data.servo_class_us,
    )
    print(f"wrote csv: {csv_path}")
    print("next: python3 -m machine_learning.train")


if __name__ == "__main__":  # pragma: no cover - script mode
    main()

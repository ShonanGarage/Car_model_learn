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

from .config import Config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CSV_COLUMNS = [
    "timestamp_t",
    "image_path_t",
    "drive_state_t",
    "sonar_0_t",
    "sonar_1_t",
    "sonar_2_t",
    "sonar_3_t",
    "sonar_4_t",
    "steer_us_t",
    "throttle_us_t",
    "timestamp_tk",
    "steer_us_tk",
    "throttle_us_tk",
    "move_tk",
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
    steer_target: np.ndarray
    move_target: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    """学習に必要な前処理済みデータ。"""

    train: PreparedSplit
    val: PreparedSplit
    numeric_columns: list[str]
    drive_states: list[str]
    stats: NormalizationStats


def throttle_to_move(throttle_us: float) -> int:
    """throttle_us から 3クラス move を作る。"""
    if throttle_us == 0:
        return 0  # STOP
    if throttle_us > 1500:
        return 1  # FORWARD
    return 2  # BACKWARD


def _forward_fill_sonar(distances: np.ndarray) -> np.ndarray:
    """ソナーの -1.0 を欠損扱いにして前方補完する。"""
    arr = distances.copy()
    missing = arr == -1.0

    # 先頭欠損を埋めるために列ごとの平均を先に計算する
    col_mean = np.zeros(arr.shape[1], dtype=np.float32)
    for col in range(arr.shape[1]):
        valid = arr[~missing[:, col], col]
        col_mean[col] = float(valid.mean()) if len(valid) else 0.0

    for col in range(arr.shape[1]):
        last = col_mean[col]
        for i in range(arr.shape[0]):
            if arr[i, col] == -1.0:
                arr[i, col] = last
            else:
                last = arr[i, col]
    return arr


def _build_base_arrays_from_labels_csv(
    csv_path: Path,
    images_dir: Path,
    *,
    drive_state_default: str,
) -> dict[str, np.ndarray | list[str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSVが空です: {csv_path}")

    rows = sorted(rows, key=lambda r: float(r["timestamp_ms"]))

    timestamps = np.array([float(r["timestamp_ms"]) for r in rows], dtype=np.float64)
    drive_state = [drive_state_default for _ in rows]
    steer_us = np.array([float(r["steer_us"]) for r in rows], dtype=np.float32)
    throttle_us = np.array([float(r["thr_us"]) for r in rows], dtype=np.float32)
    distances = np.array(
        [
            [
                float(r["sonar_front_m"]),
                float(r["sonar_front_left_m"]),
                float(r["sonar_front_right_m"]),
                float(r["sonar_left_m"]),
                float(r["sonar_right_m"]),
            ]
            for r in rows
        ],
        dtype=np.float32,
    )
    distances = _forward_fill_sonar(distances)

    image_paths = [str(images_dir / str(r["filename"])) for r in rows]

    return {
        "timestamp": timestamps,
        "drive_state": drive_state,
        "steer_us": steer_us,
        "throttle_us": throttle_us,
        "distances": distances,
        "image_path": image_paths,
    }


def _assign_split(n: int, val_fraction: float, seed: int) -> np.ndarray:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction は 0 と 1 の間である必要があります。")
    rng = np.random.default_rng(seed)
    return rng.random(n) < val_fraction  # True が val


def build_shifted_rows_from_labels_csv(
    labels_csv_path: str | Path,
    images_dir: str | Path,
    *,
    k: int,
    drive_state_default: str,
) -> list[dict[str, str | float | int]]:
    """labels.csv から t -> t+k の行データを作る（split未設定）。"""
    if k < 1:
        raise ValueError("k は 1 以上である必要があります。")

    labels_csv_path = Path(labels_csv_path)
    images_dir = Path(images_dir)

    base = _build_base_arrays_from_labels_csv(
        labels_csv_path,
        images_dir,
        drive_state_default=drive_state_default,
    )

    n = len(base["steer_us"])
    m = n - k
    if m <= 1:
        raise ValueError("データ数が少なすぎます。k を小さくしてください。")

    rows: list[dict[str, str | float | int]] = []
    for i in range(m):
        j = i + k
        sonar = base["distances"][i]
        throttle_tk = float(base["throttle_us"][j])
        row = {
            "timestamp_t": float(base["timestamp"][i]),
            "image_path_t": str(base["image_path"][i]),
            "drive_state_t": str(base["drive_state"][i]),
            "sonar_0_t": float(sonar[0]),
            "sonar_1_t": float(sonar[1]),
            "sonar_2_t": float(sonar[2]),
            "sonar_3_t": float(sonar[3]),
            "sonar_4_t": float(sonar[4]),
            "steer_us_t": float(base["steer_us"][i]),
            "throttle_us_t": float(base["throttle_us"][i]),
            "timestamp_tk": float(base["timestamp"][j]),
            "steer_us_tk": float(base["steer_us"][j]),
            "throttle_us_tk": throttle_tk,
            "move_tk": int(throttle_to_move(throttle_tk)),
            "k": int(k),
            "split": "",  # 後で埋める
        }
        rows.append(row)
    return rows


def export_csv_from_labels(
    labels_csv_path: str | Path,
    images_dir: str | Path,
    csv_path: str | Path,
    *,
    k: int = 1,
    val_fraction: float = 0.2,
    seed: int = 42,
    drive_state_default: str = "READY",
) -> Path:
    """labels.csv からCSVを作る（split列もここで確定させる）。"""
    rows = build_shifted_rows_from_labels_csv(
        labels_csv_path,
        images_dir,
        k=k,
        drive_state_default=drive_state_default,
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


def _maybe_ffill_sonar_in_rows(rows: list[dict[str, str]]) -> None:
    """CSVに -1.0 が残っている場合に備えて前方補完する。"""
    sonar_cols = [f"sonar_{i}_t" for i in range(5)]
    last: dict[str, float | None] = {c: None for c in sonar_cols}

    # 先頭欠損を埋めるために平均を計算
    mean: dict[str, float] = {}
    for col in sonar_cols:
        vals = [float(r[col]) for r in rows if float(r[col]) != -1.0]
        mean[col] = float(np.mean(vals)) if vals else 0.0

    for row in rows:
        for col in sonar_cols:
            v = float(row[col])
            if v == -1.0:
                fill = last[col] if last[col] is not None else mean[col]
                row[col] = f"{fill}"
            else:
                last[col] = v


def _one_hot_with_fixed_order(
    states: Sequence[str],
    *,
    known_order: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    drive_states = list(known_order)
    index = {s: i for i, s in enumerate(drive_states)}

    unknown = sorted({s for s in states if s not in index})
    if unknown:
        print(f"[dataset] unknown drive_state values: {unknown}")

    mat = np.zeros((len(states), len(drive_states)), dtype=np.float32)
    for row, s in enumerate(states):
        idx = index.get(s)
        if idx is not None:
            mat[row, idx] = 1.0
    return mat, drive_states


def prepare_data_from_csv(csv_path: str | Path) -> PreparedData:
    """CSVから学習用の配列と正規化統計量を作る。"""
    cfg = Config()
    csv_path = Path(csv_path)
    rows = _load_csv_rows(csv_path)
    rows = _sort_rows_by_timestamp(rows)
    _maybe_ffill_sonar_in_rows(rows)

    image_paths = [r["image_path_t"] for r in rows]
    drive_state_t = [r["drive_state_t"] for r in rows]

    distances_t = np.array(
        [[float(r[f"sonar_{i}_t"]) for i in range(5)] for r in rows],
        dtype=np.float32,
    )
    steer_t = np.array([float(r["steer_us_t"]) for r in rows], dtype=np.float32)
    throttle_t = np.array([float(r["throttle_us_t"]) for r in rows], dtype=np.float32)

    steer_tk = np.array([float(r["steer_us_tk"]) for r in rows], dtype=np.float32)
    move_tk = np.array([int(float(r["move_tk"])) for r in rows], dtype=np.int64)

    split = np.array([r["split"].strip() for r in rows])
    train_mask = split == "train"
    val_mask = split == "val"

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError("split列に train/val が十分に含まれていません。CSVを作り直してください。")

    drive_one_hot, drive_states = _one_hot_with_fixed_order(
        drive_state_t,
        known_order=cfg.data.drive_state_order,
    )

    numeric = np.concatenate(
        [
            distances_t,
            steer_t[:, None],
            throttle_t[:, None],
            drive_one_hot,
        ],
        axis=1,
    ).astype(np.float32)

    numeric_columns = [
        "sonar_0_t",
        "sonar_1_t",
        "sonar_2_t",
        "sonar_3_t",
        "sonar_4_t",
        "steer_us_t",
        "throttle_us_t",
    ] + [f"drive_state__{s}" for s in drive_states]

    train_numeric = numeric[train_mask]
    mean = np.zeros(numeric.shape[1], dtype=np.float32)
    std = np.ones(numeric.shape[1], dtype=np.float32)

    # 連続値だけ標準化し、one-hot はそのまま使う
    cont_dim = 7
    cont_mean = train_numeric[:, :cont_dim].mean(axis=0)
    cont_std = train_numeric[:, :cont_dim].std(axis=0)
    cont_std = np.where(cont_std == 0.0, 1.0, cont_std)

    mean[:cont_dim] = cont_mean
    std[:cont_dim] = cont_std
    numeric[:, :cont_dim] = (numeric[:, :cont_dim] - cont_mean) / cont_std

    train = PreparedSplit(
        image_paths=[p for p, keep in zip(image_paths, train_mask) if keep],
        numeric=numeric[train_mask],
        steer_target=steer_tk[train_mask],
        move_target=move_tk[train_mask],
    )
    val = PreparedSplit(
        image_paths=[p for p, keep in zip(image_paths, val_mask) if keep],
        numeric=numeric[val_mask],
        steer_target=steer_tk[val_mask],
        move_target=move_tk[val_mask],
    )

    stats = NormalizationStats(mean=mean, std=std, numeric_columns=numeric_columns)
    return PreparedData(
        train=train,
        val=val,
        numeric_columns=numeric_columns,
        drive_states=drive_states,
        stats=stats,
    )


class DrivingDataset(Dataset):
    """画像 + 数値特徴 -> (steer, move) のDataset。"""

    def __init__(
        self,
        split: PreparedSplit,
        *,
        image_transform: transforms.Compose | None = None,
    ) -> None:
        self.image_paths = split.image_paths
        self.numeric = split.numeric
        self.steer_target = split.steer_target
        self.move_target = split.move_target

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
        steer_target = torch.tensor(self.steer_target[idx], dtype=torch.float32)
        move_target = torch.tensor(self.move_target[idx], dtype=torch.long)
        return image, numeric, steer_target, move_target


def main() -> None:  # pragma: no cover - CLI entry
    cfg = Config()
    csv_path = export_csv_from_labels(
        cfg.data.labels_csv_path,
        cfg.data.images_dir,
        cfg.data.csv_path,
        k=cfg.data.k,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.data.seed,
        drive_state_default=cfg.data.drive_state_default,
    )
    print(f"wrote csv: {csv_path}")
    print("next: python3 -m machine_learning.train")


if __name__ == "__main__":  # pragma: no cover - script mode
    main()

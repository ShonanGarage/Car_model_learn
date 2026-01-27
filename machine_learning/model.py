from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class DrivingModel(nn.Module):
    """マルチモーダルな運転モデル。

    - 画像枝: MobileNetV3-Small -> 128次元埋め込み
    - 数値枝: MLP -> 64次元埋め込み
    - 融合: 連結 -> 共有MLP
    - 出力ヘッド:
      - steer: 回帰
      - move: 3クラス分類
    """

    def __init__(self, numeric_dim: int) -> None:
        super().__init__()

        # 画像枝: 可能なら事前学習済みのMobileNetV3-Smallを使う。
        # ネットワークが無い環境では重みダウンロードに失敗するためフォールバックする。
        try:
            weights = tvm.MobileNet_V3_Small_Weights.DEFAULT
            backbone = tvm.mobilenet_v3_small(weights=weights)
        except Exception:
            backbone = tvm.mobilenet_v3_small(weights=None)
        self.image_encoder = backbone.features
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        # MobileNetV3-Smallの特徴量出力は576チャネル。
        self.image_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(),
        )

        # 数値枝。
        self.numeric_mlp = nn.Sequential(
            nn.Linear(numeric_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # 融合MLP。
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # 出力ヘッド。
        self.steer_head = nn.Linear(64, 1)  # 回帰
        self.move_head = nn.Linear(64, 3)   # 3クラス: STOP/FORWARD/BACKWARD

    def forward(self, image: torch.Tensor, numeric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """順伝播。

        Args:
            image: 形状 (B, 3, H, W) のテンソル
            numeric: 形状 (B, numeric_dim) のテンソル

        Returns:
            steer_pred: 形状 (B,) の回帰出力
            move_logits: 形状 (B, 3) の分類ロジット
        """
        x_img = self.image_encoder(image)
        x_img = self.image_pool(x_img)
        x_img = self.image_proj(x_img)

        x_num = self.numeric_mlp(numeric)

        x = torch.cat([x_img, x_num], dim=1)
        x = self.fusion(x)

        steer_pred = self.steer_head(x).squeeze(1)
        move_logits = self.move_head(x)
        return steer_pred, move_logits

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
      - steer: 3クラス分類（サーボ離散値）
      - throttle: 2クラス分類（throttle_us < 1500 / >= 1500）
    """

    def __init__(self, numeric_dim: int) -> None:
        super().__init__()

        # 画像枝: 軽量CNN。
        def conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.image_encoder = nn.Sequential(
            conv_block(3, 32, 2),
            conv_block(32, 32, 1),
            conv_block(32, 64, 2),
            conv_block(64, 64, 1),
            conv_block(64, 128, 2),
            conv_block(128, 128, 1),
        )
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
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
        self.steer_head = nn.Linear(64, 3)
        self.throttle_head = nn.Linear(64, 2)

    def forward(self, image: torch.Tensor, numeric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """順伝播。

        Args:
            image: 形状 (B, 3, H, W) のテンソル
            numeric: 形状 (B, numeric_dim) のテンソル

        Returns:
            steer_logits: 形状 (B, 3) の分類ロジット
            throttle_logits: 形状 (B, 2) の分類ロジット
        """
        x_img = self.image_encoder(image)
        x_img = self.image_pool(x_img)
        x_img = self.image_proj(x_img)

        x_num = self.numeric_mlp(numeric)

        x = torch.cat([x_img, x_num], dim=1)
        x = self.fusion(x)

        steer_logits = self.steer_head(x)
        throttle_logits = self.throttle_head(x)
        return steer_logits, throttle_logits

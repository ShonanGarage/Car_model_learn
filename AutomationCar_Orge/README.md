# RC Car Control System (Data Assemble)

このプロジェクトは、RCカーの走行制御とデータ収集（テレメトリおよび画像）を行うためのシステムです。
クリーンアーキテクチャおよびドメイン駆動設計（DDD）の原則に基づいてリファクタリングされており、ハードウェア制御ロジックと走行制御ロジックが明確に分離されています。

## プロジェクト構成

- `app/`: アプリケーション層（エントリーポイント、DIコンテナ、設定）
- `internal/`: ビジネスロジック
  - `domain/`: ドメインモデル（走行状態、エンティティ、値オブジェクト）
  - `service/`: ドメインサービス（走行・ロギングのオーケストレーション）
  - `interface/`: 全レイヤーで使用されるインターフェース（Gateway, Repository）
- `infrastructure/`: 外部への接続（GPIO制御、カメラ、データ保存）
- `presentation/`: ユーザーインターフェース（ターミナル操作、カメラ映像表示）
- `tests/`: ユニットテスト

## 環境構築

以下のライブラリが必要です。

```bash
pip install opencv-python lgpio
```

※ 実機以外で実行する場合、`lgpio` や `picamera2` の代わりにダミー実装や標準ビデオ入力が使用されます。

## 実行方法（エントリポイント別）

1. **ターミナル操作の起動**

   プロジェクトルートディレクトリ（`code/data_assemble/`）で実行します。

   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   python3 app/entrypoints/terminal.py
   ```

   - 実行すると、操作方法（W/A/S/Dなど）がターミナルに表示されます。

# 操作方法

- **W**: 前進 (Forward)
- **S**: 後退 (Backward)
- **A**: 左転舵
- **D**: 右転舵
- **X**: 停止
- **Q**: プログラム終了


2. **WebSocket操作の起動**

   ```bash
   cd AutomationCar_Orge
   uv run python -m app.entrypoints.websocket
   ```

   - WebSocketエンドポイント: `ws://<host>:8000/ws`
   - 受信コマンドは `action` に enum 文字列（例: `MOVE_FORWARD`）を指定します。
   - 送信は `telemetry`（状態/距離/操舵/スロットル/画像 base64）を定期配信します。
   - 実行には `fastapi` と `uvicorn` が必要です。

## データ収集

   走行中に自動的に `dataset/` ディレクトリ（設定による）へ以下のデータが保存されます。
   - `telemetry.jsonl`: 各フレームのタイムスタンプ、スロットル、ステアリング、距離データ
   - `*.jpg`: カメラからキャプチャされた画像

3. **sonorテスト**

   ```bash
   cd AutomationCar_Orge
   uv run python -m app.entrypoints.sonar
   ```

4. **ML自動運転の起動**

   ```bash
   cd AutomationCar_Orge
   uv run python -m app.entrypoints.ml.autodrive
   ```

   - `app/entrypoints/ml/config.py` で `checkpoint_path` と `data_csv_path` を指定します。
   - 10Hzで推論し、推論失敗やカメラNG時は停止します。

## ユニットテスト

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 -m unittest discover tests
```

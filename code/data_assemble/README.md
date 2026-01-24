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

## 実行方法

1. **メインアプリケーションの起動**

   プロジェクトルートディレクトリ（`code/data_assemble/`）で実行します。

   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   python3 app/main.py [コースID]
   ```

   - コースIDは任意です（デフォルト: `default_course`）。
   - 実行すると、操作方法（W/A/S/Dなど）がターミナルに表示されます。

2. **データ収集**

   走行中に自動的に `dataset/` ディレクトリ（設定による）へ以下のデータが保存されます。
   - `telemetry.jsonl`: 各フレームのタイムスタンプ、スロットル、ステアリング、距離データ
   - `*.jpg`: カメラからキャプチャされた画像

## 操作方法

- **W**: 前進 (Forward)
- **S**: 後退 (Backward)
- **A**: 左転舵
- **D**: 右転舵
- **X**: 停止
- **Q**: プログラム終了

## ユニットテスト

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 -m unittest discover tests
```

# 機械学習要件（シンプル版）

## 目的
現在（t）の観測から、次（t+1）のコマンドを予測する。

#### 使うデータ
- ログ: `AutomationCar_Orge/learning_data/log.jsonl`
- 画像: `AutomationCar_Orge/learning_data/images/*.jpg`

#### 入力（X）
すべて t の値を使う。

- `drive_state(t)`
- `distances(t)`（ソナー5本）
- `image(t)`
- `steer_us(t)`
- `throttle_us(t)`

#### 出力（y）
次の時刻 t+1 のコマンド。

- `steer_us(t+1)`
- `move(t+1)`（`throttle_us(t+1)` から作る）

#### move の定義
- `STOP`: `throttle_us(t+1) == 0`
- `FORWARD`: `throttle_us(t+1) > 1500`
- `BACKWARD`: `0 < throttle_us(t+1) < 1500`

#### データ整形方針
- `k` ステップ先コマンドを予測できるようにする（まずは `k=1`）
- いろいろな `k` や前処理で比較できるように、まずCSVを作る


#### CSVカラム案（t -> t+k）

- `timestamp_t`
- `image_path_t`
- `drive_state_t`
- `sonar_0_t`
- `sonar_1_t`
- `sonar_2_t`
- `sonar_3_t`
- `sonar_4_t`
- `steer_us_t`
- `throttle_us_t`
- `timestamp_tk`
- `steer_us_tk`
- `throttle_us_tk`
- `move_tk`
- `k`
- `split`（`train` / `val`）

メモ:
- `*_t` が入力側、`*_tk` が教師側


## モデル

#### 画像特徴
- 画像サイズ: `320x240 -> 160x120` にダウンサンプリング
- バックボーン: MobileNetV3-Small
- 画像埋め込み次元: 128
- 事前学習重み: 使う
- 凍結（freeze）: しない（全体を微調整する）

#### 数値特徴
- MLPで数値特徴をエンコード
- 数値埋め込み次元: 64
- Dropout: 0.1〜0.2

#### 融合と出力
- 画像128 + 数値64 を連結
- 共有MLPのあとに2ヘッド
- steerヘッド: `steer_us(t+k)` の回帰
- moveヘッド: `move(t+k)` の3クラス分類
- Dropout: 0.1〜0.3

#### 損失関数（初期案）
- steer（回帰）: MAE（L1Loss）
- move（分類）: CrossEntropyLoss
- 合成: `L = L_steer + λ * L_move`（まずは `λ = 1.0`）

## データ整形と正規化（初期案）
「t -> t+k」のずらしと、数値のスケーリングを明示しておく。

1. `timestamp` でソート
2. `k` ステップずらす（t -> t+k）
3. 末尾行や欠損を落とす

メモ:
- `move` はデータとして存在している前提で扱う

### 数値特徴の正規化まとめ

- **ソナー値**（`sonar_0_t` 〜 `sonar_4_t`）
    - `-1.0` は欠損として扱い、前方補完（ffill）で埋める
    - 置き換えた後、標準化（mean/std）で正規化

- **操舵・スロットル**（`steer_us_t`, `throttle_us_t`）
    - それぞれ標準化（mean/std）

- **走行状態**（`drive_state_t`）
    - one-hotエンコーディング（＝正規化は不要）

---

1. まずCSVを作る

```bash
python3 -m machine_learning.dataset
```

2. CSVを使って学習する

```bash
python3 -m machine_learning.train
```

チェックポイント保存先（デフォルト）:
- `machine_learning/checkpoints/last.pt`

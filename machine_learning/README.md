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

#### 数値特徴
- MLPで数値特徴をエンコード
- 数値埋め込み次元: 64

#### 融合と出力
- 画像128 + 数値64 を連結
- 共有MLPのあとに2ヘッド
- steerヘッド: `steer_us(t+k)` の回帰
- moveヘッド: `move(t+k)` の3クラス分類

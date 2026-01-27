# 機械学習要件（シンプル版）

## 目的
現在（t）の観測から、次（t+1）のコマンドを予測する。

## 使うデータ
- ログ: `AutomationCar_Orge/learning_data/log.jsonl`
- 画像: `AutomationCar_Orge/learning_data/images/*.jpg`

## 入力（X）
すべて t の値を使う。

- `drive_state(t)`
- `distances(t)`（ソナー5本）
- `image(t)`
- `steer_us(t)`
- `throttle_us(t)`

## 出力（y）
次の時刻 t+1 のコマンド。

- `steer_us(t+1)`
- `move(t+1)`（`throttle_us(t+1)` から作る）

## move の定義
- `STOP`: `throttle_us(t+1) == 0`
- `FORWARD`: `throttle_us(t+1) > 1500`
- `BACKWARD`: `0 < throttle_us(t+1) < 1500`

## 最低限の前処理
1. `log.jsonl` を読む
2. `timestamp` でソート
3. 1ステップずらす（t -> t+1）
4. `image_filename` で画像を読む
5. 末尾行や欠損を落とす

## モデル案（まずこれ）
定番のマルチモーダル構成で始める。

1. 画像をCNNに通して埋め込みベクトルにする
2. 数値特徴（drive_state, sonar5, steer_us, throttle_us）をMLPに通す
3. 2つの埋め込みを連結してMLPに入れる
4. 出力は2ヘッドに分ける

- steerヘッド: `steer_us` の回帰
- moveヘッド: 3クラス分類（STOP / FORWARD / BACKWARD）


データ整形
+kを調整できるように
いろんなデータセットでモデル見てみる
カラム

モデル
画像
320×240 を160×120にダウンサンプリング
MobileNetV3-Small
→128

数値特徴
mlp
→64

融合
→
- steerヘッド: `steer_us` の回帰
- moveヘッド: 3クラス分類（STOP / FORWARD / BACKWARD）


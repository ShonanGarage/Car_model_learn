# 機械学習要件（シンプル版）

## 目的
現在（t）の観測から、次（t+1）のコマンドを予測する。

#### 使うデータ
- ログ: `learning_data/20260127_202651/log.csv`
- 画像: `learning_data/20260127_202651/images/*.jpg`

#### log.csv カラム（現行）
- `timestamp`（ms）
- `steer_us`
- `throttle_us`
- `image_filename`

#### 入力（X）
すべて t の値を使う。

- `image(t)`
- `steer_us(t)`
- `throttle_us(t)`

#### 出力（y）
次の時刻 t+1 の操舵クラスとスロットル。

- `steer_cls(t+1)`（`steer_us(t+1)` を 3クラスに離散化）
- `throttle_us(t+1)`（連続値）

#### データ整形方針
- `k` ステップ先コマンドを予測できるようにする（まずは `k=1`）
- いろいろな `k` や前処理で比較できるように、まずCSVを作る


#### CSVカラム案（t -> t+k）

- `timestamp_t`
- `image_path_t`
- `steer_cls_t`
- `throttle_us_t`
- `timestamp_tk`
- `steer_cls_tk`
- `throttle_us_tk`
- `k`
- `split`（`train` / `val`）

メモ:
- `*_t` が入力側、`*_tk` が教師側


## モデル

#### 画像特徴
- バックボーン: 軽量CNN（畳み込みブロック + GAP）
- 画像埋め込み次元: 128

#### 数値特徴
- MLPで数値特徴をエンコード
- 数値埋め込み次元: 64
- Dropout: 0.1〜0.2

#### 融合と出力
- 画像128 + 数値64 を連結
- 共有MLPのあとに2ヘッド
- steerヘッド: `steer_cls(t+k)` の3クラス分類
- throttleヘッド: `throttle_us(t+k)` の回帰
- Dropout: 0.1〜0.3

#### 損失関数
- steer（分類）: CrossEntropyLoss
- throttle（回帰）: MSELoss
- 合計: `steer_loss + lambda_move * throttle_loss`

## データ整形と正規化（初期案）
「t -> t+k」のずらしと、数値のスケーリングを明示しておく。

1. `timestamp` でソート
2. `k` ステップずらす（t -> t+k）
3. 末尾行や欠損を落とす

メモ:
- `move` はデータとして存在している前提で扱う

### 数値特徴の正規化まとめ

- **操舵クラス**（`steer_cls_t`）
    - `steer_cls_t` は one-hot 化して入力
- **スロットル**（`throttle_us_t`）
    - 連続値のまま入力

---

1. まずCSVを作る

```bash
uv run python -m machine_learning.dataset
```

2. CSVを使って学習する

```bash
uv run python -m machine_learning.train
```

3. 学習済みモデルで推論する

```bash
uv run python -m machine_learning.predict
```

チェックポイント保存先（デフォルト）:
- `machine_learning/checkpoints/ver_11_k1/best.pt`

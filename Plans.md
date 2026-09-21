# CIFAR-10 Flow Matching モデル改善タスクボード

## 目的

`main_cifar.py`のモデルアーキテクチャを一つずつ変更し、同じ実験条件で比較する。
改善が確認できたモデルだけを`main`へ取り込む。

このファイルのチェックボックスは、実験の進捗を記録するために使う。

- `[ ]` 未着手
- `[-]` 実施中または一部確認済み
- `[x]` 完了
- `[!]` 問題あり、または再検証が必要

## 現在の状態

- [x] `main_cifar.py`の現行モデルを確認
- [x] 現行モデルの構造を記録
  - 3段U-Net
  - チャンネル数`64 → 128 → 256 → 512`
  - 各解像度への時刻埋め込み
  - 4×4ボトルネックのTransformer
  - パラメータ数: 約10,227,715
- [x] `.venv`のCUDA実行環境を確認
  - PyTorch 2.11.0+cu128
  - NVIDIA GeForce RTX 3090
- [x] 現行モデルの小さなforwardを確認
  - 入力: `(4, 3, 32, 32)`
  - 出力: `(4, 3, 32, 32)`
  - batch 4のforward: 約1.22 ms（実行時の目安）
- [x] 現行baselineを標準条件で学習し、比較用Runを完成させる
- [ ] 最初のモデル改善案を実装する

## 固定する標準実験条件

モデル構造以外は、候補モデル間で変更しない。

- [ ] seedを固定する
  - Python
  - NumPy
  - PyTorch
  - CUDA
  - DataLoader
- [ ] データセットを固定する
  - CIFAR-10
  - `ToTensor`
  - `Normalize((0.5,)*3, (0.5,)*3)`
  - `RandomHorizontalFlip`
- [ ] `batch_size=128`を固定する
- [ ] `num_epochs=20`を固定する
- [ ] `learning_rate=2e-4`を固定する
- [ ] Flow Matchingの補間を固定する
  - `x_t = (1 - t) * x_0 + t * x_1`
  - `target_velocity = x_1 - x_0`
  - velocity MSE
- [ ] samplerをHeunに固定する
- [ ] `sampling_steps=50`を固定する
- [ ] GPU、精度設定、DataLoader設定を固定する
- [ ] 実験ごとにGit commitを作成してから長時間学習する

## 評価基盤

モデル変更とは分離して、比較結果を判断できる状態を作る。

- [x] MLflow Experimentが`CIFAR10-UNet`になっていることを確認する
- [x] Run名にfeature名を使う
- [-] 以下のパラメータをMLflowへ記録する
  - [x] モデル名と変更内容
  - [x] seed
  - [x] batch size、epoch数、learning rate
  - [x] sampler、sampling steps
  - [x] パラメータ数
  - [x] device
- [x] epochごとの`train_loss`を記録する（20 epoch分）
- [x] 学習済みモデルをartifactとして保存する
- [x] 生成画像をartifactとして保存する
- [x] loss曲線をartifactとして保存する
- [x] 生成用noiseとsampling条件を固定する
- [x] validation用velocity lossを記録する
- [x] 各Runが正常終了したことを確認する
- [x] loss、生成画像、artifactが揃ったRunであることを確認する

## 実験フロー

### Phase 0: baselineの確立

- [x] `main`の状態を確認する
- [x] baseline用のcommitを特定する（評価基盤込み: `93e760e`）
- [x] 現行baselineを標準条件で20 epoch学習する
- [x] baselineのMLflow Runを確認する
- [x] baselineのtrain loss履歴を確認する
- [x] baselineの生成画像を確認する
- [x] baselineのcheckpointを確認する
- [x] baselineの学習完了状態を確認する
- [x] baselineのパラメータ数、学習時間、生成時間を記録する
- [x] baselineの評価結果をこのファイルに記録する

Baseline記録:

```text
Run名: baseline
Run ID: 988df7ecbf3c4ef0a0f8b874b0e2d2
Git commit: 93e760e
Seed: 42
Train loss: epoch 1 = 0.3480 / epoch 20 = 0.1931
Validation loss: epoch 1 = 0.2610 / epoch 20 = 0.1931
生成画像: `generated.png`（584×584、artifact登録済み）
loss曲線: `loss_curve.png`（artifact登録済み）
checkpoint: `unet_model_20.pt`（約40.9 MB、artifact登録済み）
パラメータ数: 10,227,715
学習時間: 270.385秒
生成時間: 1.515秒
所見: FINISHED。loss、validation loss、曲線、checkpoint、生成画像、時間を確認済み。
```

### Phase 1: 短時間スクリーニング

各候補を同じ短縮条件で比較し、明らかに劣る候補を除外する。

- [x] 短縮条件を決めて記録する（5 epoch、seed 42、他条件は標準条件）
- [x] baselineを短縮条件で実行する
- [x] `resblock-film`を実行する
- [x] `no-bottleneck-attention`を実行する
- [ ] `upsample-conv`を実行する
- [x] `attention-8x8`を実行する
- [ ] `deeper-unet`を実行する
- [ ] 各Runのloss曲線を比較する
- [ ] 各Runの生成画像を比較する
- [ ] 各Runの計算時間とGPUメモリを比較する
- [ ] 有望な候補を最大3つに絞る

### Phase 2: 標準条件での比較

- [x] Phase 1で選んだ候補を20 epochで実行する（E1、E3）
- [x] baselineと同じseedで比較する
- [x] train lossを比較する
- [x] validation lossを比較する
- [x] 生成画像を比較する
- [x] checkpointとartifactの存在を確認する
- [x] 各候補の改善点と悪化点を記録する
- [-] `main`へマージする候補を1つ選ぶ（seed再検証待ち）

### Phase 3: seedを変えた再検証

- [ ] 選定候補をseed `42`で再実行する
- [ ] 選定候補をseed `123`で実行する
- [ ] 選定候補をseed `456`で実行する
- [ ] baselineも同じseedで実行する
- [ ] seed間で改善が安定しているか確認する
- [ ] 1回だけの偶然の改善ではないことを確認する
- [ ] 改善が安定していれば`main`へマージする
- [ ] 改善が不安定なら採用せず、結果を記録する

## 実験カード

### E0: baseline

- [x] ブランチ: `main`
- [x] Run名: `baseline`
- [x] 変更: なし
- [x] 比較結果を記録

結果:

```text
Run: `5be835b50aec4d9497c093178a813a81` / FINISHED
Train loss: 0.3487130824565887 → 0.1926929071187973（20 epoch）
Validation loss: 未記録
生成画像: `generated.png`
checkpoint: `unet_model_20.pt`
所見: 比較用baselineとして利用可能。validation loss、loss曲線artifact、時間計測は未確認。
採用判断: baselineのため採用判断は不要

短縮スクリーニング（5 epoch）:

```text
Train loss: 0.3483 → 0.2156
Validation loss: 0.2614 → 0.2125
所見: E1との同条件比較用。正常終了、曲線・checkpoint・生成画像あり。
```
```

### E1: Residual Block + FiLM

- [x] `main`から`feature/resblock-film`を作成
- [x] 畳み込みブロックをResidual Blockへ変更
- [x] 時刻埋め込みをFiLMのscale/shiftとして注入
- [x] チャンネル数、データ、損失、samplerは変更しない
- [x] 構文チェックを実行する
- [x] 小さなforwardを実行する
- [x] パラメータ数を記録する
- [x] 変更をcommitする
- [x] Phase 1の短縮実験を実行する
- [x] Phase 2の標準実験を実行する
- [x] baselineと比較する
- [-] 採用または不採用を判断する（seed再検証待ち）

結果:

```text
Run: `9ea91ce179d648f682047b0cdbea3912` / FINISHED
Train loss: 0.3252 → 0.1909（20 epoch）
Validation loss: 0.2470 → 0.1909（20 epoch）
生成画像: `generated.png`、loss曲線、checkpointをartifact登録済み
パラメータ数: 15,362,313
学習時間: 306.405秒（20 epoch）
生成時間: 1.767秒
所見: baseline（validation 0.1931、train 0.1931）よりvalidation lossが約1.15%低い。学習時間は約13%、パラメータ数は約50%増加。
採用判断: 改善あり。seed 123/456で再検証してから採用判断。
```

### E2: Upsample + Conv

- [x] `main`から`feature/upsample-conv`を作成
- [x] `ConvTranspose2d`をUpsample + Conv2dへ変更
- [x] U-Netの解像度とskip接続を維持する
- [x] 構文チェックを実行する
- [x] 小さなforwardを実行する
- [x] 変更をcommitする
- [x] Phase 1の短縮実験を実行する
- [ ] 生成画像のアーティファクトを確認する
- [ ] Phase 2へ進めるか判断する

結果:

```text
Run: 最新MLflow Run / `upsample-conv` / FINISHED
Train loss: 0.3561 → 0.2156（5 epoch）
Validation loss: 0.2614 → 0.2131（5 epoch）
生成画像: `generated.png`、loss曲線、checkpointをartifact登録済み
所見: baseline（validation 0.2125）より0.3%悪化。明確な改善なし。
採用判断: 不採用。標準20 epochには進めない。
```

### E3: Transformerなし

- [x] `main`から`feature/no-bottleneck-attention`を作成
- [x] 4×4 Transformerだけを削除する
- [x] それ以外の構造を維持する
- [x] 構文チェックを実行する
- [x] 小さなforwardを実行する
- [x] 変更をcommitする
- [x] Phase 1の短縮実験を実行する
- [x] baselineと比較する
- [x] Attentionが有効か判断する

結果:

```text
Run: `ee7dc09352964ed49bbb62f89d2885c5` / FINISHED
Train loss: 0.3667 → 0.1926（20 epoch）
Validation loss: 0.2632 → 0.1925（20 epoch）
生成画像: `generated.png`、loss曲線、checkpointをartifact登録済み
パラメータ数: 7,067,139
学習時間: 252.258秒、生成時間: 0.723秒
所見: baseline（validation 0.1931）より約0.3%改善。E1（0.1909）より悪いが、パラメータ数は約31%、生成時間は約52%少ない。
採用判断: 軽量候補としてseed再検証対象に残す。単独での最終採用は保留。
```

### E4: 8×8 Attention

- [x] `main`から`feature/attention-8x8`を作成
- [x] 8×8・256チャンネルにAttentionを追加する
- [x] 4×4 Transformerとの違いを記録する
- [x] 構文チェックを実行する
- [x] 小さなforwardを実行する
- [x] 変更をcommitする
- [-] Phase 1の短縮実験を実行する
- [ ] baselineおよびE3と比較する
- [ ] Phase 2へ進めるか判断する

結果:

```text
Run:
Train loss:
Validation loss:
生成画像:
パラメータ数:
所見:
採用判断:
```

### E5: deeper U-Net

- [ ] `main`から`feature/deeper-unet`を作成
- [ ] 各解像度のResidual Block数を増やす
- [ ] FiLMを同時に追加する場合は別実験として分離する
- [ ] 構文チェックを実行する
- [ ] 小さなforwardを実行する
- [ ] 変更をcommitする
- [ ] Phase 1の短縮実験を実行する
- [ ] 計算時間とGPUメモリを確認する
- [ ] Phase 2へ進めるか判断する

結果:

```text
Run:
Train loss:
Validation loss:
生成画像:
パラメータ数:
学習時間:
所見:
採用判断:
```

## 採用判断の基準

- [ ] 学習が正常終了している
- [ ] MLflowに必要なパラメータが記録されている
- [ ] loss曲線が存在する
- [ ] checkpointが存在する
- [ ] 生成画像が存在する
- [ ] validation lossまたは生成品質がbaselineより改善している
- [ ] 計算量とGPUメモリが許容範囲に収まっている
- [ ] 複数seedで改善が再現している
- [ ] 改善した候補だけを`main`へマージする

## 採用済みモデルの記録

```text
モデル名:
採用日:
マージ元ブランチ:
Git commit:
変更内容:
パラメータ数:
比較したbaseline:
改善結果:
未解決の問題:
```

## 注意事項

- baselineと候補モデルを同じコード内のフラグで切り替えない。
- 1つのfeatureブランチには、原則として1つのモデル変更だけを入れる。
- 長時間学習の前に必ずcommitする。
- 実験生成物、checkpoint、生成画像、MLflowローカルデータはcommitしない。
- lossだけで採用を決めず、生成画像とRunの完了状態も確認する。

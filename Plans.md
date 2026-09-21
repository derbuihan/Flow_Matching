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
- [x] 最初のモデル改善案を実装する

## 固定する標準実験条件

モデル構造以外は、候補モデル間で変更しない。

- [x] seedを固定する
  - Python
  - NumPy
  - PyTorch
  - CUDA
  - DataLoader
- [x] データセットを固定する
  - CIFAR-10
  - `ToTensor`
  - `Normalize((0.5,)*3, (0.5,)*3)`
  - `RandomHorizontalFlip`
- [x] `batch_size=128`を固定する
- [x] `num_epochs=20`を固定する
- [x] `learning_rate=2e-4`を固定する
- [x] Flow Matchingの補間を固定する
  - `x_t = (1 - t) * x_0 + t * x_1`
  - `target_velocity = x_1 - x_0`
  - velocity MSE
- [x] samplerをHeunに固定する
- [x] `sampling_steps=50`を固定する
- [x] GPU、精度設定、DataLoader設定を固定する
- [x] 実験ごとにGit commitを作成してから長時間学習する

## 評価基盤

モデル変更とは分離して、比較結果を判断できる状態を作る。

- [x] MLflow Experimentが`CIFAR10-UNet`になっていることを確認する
- [x] Run名にfeature名を使う
- [x] 以下のパラメータをMLflowへ記録する
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
- [x] `upsample-conv`を実行する
- [x] `attention-8x8`を実行する
- [x] `deeper-unet`を実行する
- [x] 各Runのloss曲線を比較する
- [x] 各Runの生成画像を比較する
- [x] 各Runの計算時間とGPUメモリを比較する
- [x] 有望な候補を最大3つに絞る（E1、E3、baseline）

### Phase 2: 標準条件での比較

- [x] Phase 1で選んだ候補を20 epochで実行する（E1、E3）
- [x] baselineと同じseedで比較する
- [x] train lossを比較する
- [x] validation lossを比較する
- [x] 生成画像を比較する
- [x] checkpointとartifactの存在を確認する
- [x] 各候補の改善点と悪化点を記録する
- [x] `main`へマージする候補を1つ選ぶ（E1）

### Phase 3: seedを変えた再検証

- [x] 選定候補をseed `42`で再実行する
- [x] 選定候補をseed `123`で実行する
- [x] 選定候補をseed `456`で実行する
- [x] baselineも同じseedで実行する
- [x] seed間で改善が安定しているか確認する
- [x] 1回だけの偶然の改善ではないことを確認する
- [x] 改善が安定していれば`main`へマージする
- [x] 改善が不安定なら採用せず、結果を記録する（該当なし）

### Phase 4: 5 epoch棄却仮説の再検証

5 epochの短縮結果だけでは、収束の遅いモデルを誤って棄却する可能性がある。そのため、前回5 epochで棄却または保留した候補を20 epochで再実行し、最終validation lossと計算コストで再判定する。

- [x] E2 `upsample-conv`を20 epochで再実行する
- [x] E4 `attention-8x8`を20 epochで再実行する
- [x] E5 `deeper-unet`を20 epochで再実行する
- [x] 各候補をbaselineとE1に比較する
- [x] 5 epoch時点の判断が妥当だったか候補ごとに記録する
- [x] 20 epochで改善した候補を追加seedで検証する
- [x] 改善が再現した候補だけ採用判断を更新する

### Phase 5: 採用済みモデルへの8×8 Attention追加

E1 `resblock-film`を基準に、8×8・256チャンネルのSelf-Attentionを追加する。E4単独ではbaselineに対して改善したため、E1との組み合わせ効果を独立実験として検証する。

- [x] E6 `resblock-film + attention-8x8`をfeatureブランチで実装する
- [x] 構文チェックと小さなforwardを実行する
- [x] パラメータ数と計算時間を記録する
- [x] 5 epochでbaseline E1とスクリーニング比較する
- [x] 20 epochでE1と正式比較する
- [x] 生成画像、loss曲線、checkpoint、MLflow Runを確認する
- [x] 必要ならseed 123/456で再検証する
- [x] E1より改善が安定した場合だけmainへマージする

### Phase 6: E6の100 epoch再比較

20 epochではE6のseed間改善が安定しなかったため、学習後半で逆転する可能性を確認する。E6と採用済みE1を同じseed `42`、batch size、learning rate、DataLoader条件で100 epochまで実行し、epoch 100のvalidation loss、生成画像、loss曲線、学習時間を比較する。

- [ ] E6をseed `42`・100 epochで実行する
- [ ] E1をseed `42`・100 epochで実行する
- [ ] epoch 100のvalidation lossを比較する
- [ ] 途中epochのloss曲線も比較する
- [ ] 生成画像とcheckpointを確認する
- [ ] 計算時間とパラメータ数を比較する
- [ ] E6を採用または不採用と判断する

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
- [x] 採用または不採用を判断する

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
採用判断: 改善あり。seed 42/123/456でbaselineを上回ったため採用。

seed再検証:

```text
seed 42: E1 0.1909 / baseline 0.1931 / 改善 1.15%
seed 123: E1 0.1905 / baseline 0.1929 / 改善 1.24%
seed 456: E1 0.1916 / baseline 0.1926 / 改善 0.48%
E1 Run IDs: 9ea91ce179d648f682047b0cdbea3912, 62888d7a7a3f41ac87dcab2761505c3b, 8576ddd8f51a4793a03af0536f9f28d2
baseline Run IDs: 988df7ecbf3c4ef0a0f8b874b0e2d2, 6d0fb40829474499b852d51f124c2fb8, ce140079ed89406e9a93ab226c519d63
所見: 3 seedすべてでvalidation lossが改善し、artifactも全Runで確認済み。
```
```

### E2: Upsample + Conv

- [x] `main`から`feature/upsample-conv`を作成
- [x] `ConvTranspose2d`をUpsample + Conv2dへ変更
- [x] U-Netの解像度とskip接続を維持する
- [x] 構文チェックを実行する
- [x] 小さなforwardを実行する
- [x] 変更をcommitする
- [x] Phase 1の短縮実験を実行する
- [x] 生成画像のアーティファクトを確認する
- [x] Phase 2へ進めるか判断する（20 epochで改善、追加seed検証済み）

結果:

```text
Run: `e5b26f83fd5c4537a7a9ada15da7759b` / FINISHED
Train loss: 0.3561 → 0.1931（20 epoch）
Validation loss: 0.2614 → 0.1926（20 epoch）
生成画像: `generated.png`、loss曲線、checkpointをartifact登録済み
パラメータ数: 10,227,715
学習時間: 287.451秒
所見: baseline（seed 42: validation 0.1931）より約0.3%改善。5 epochでは0.2131で悪化していたため、早期スクリーニングによる棄却は不適切だった。
追加seed結果:
```text
seed 42: 0.1926 / baseline 0.1931
seed 123: 0.1927 / baseline 0.1929
seed 456: 0.1930 / baseline 0.1926
Run IDs: e5b26f83fd5c4537a7a9ada15da7759b, 7a891860946643df99aa27bfd9355df9, 585cc39b900d4e5e86903e435b621ea8
```
採用判断: seed 456で悪化し改善が安定しないため不採用。5 epochだけで棄却する判断は不適切だったが、追加seedで採用には至らなかった。
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
- [x] Phase 1の短縮実験を実行する
- [x] baselineおよびE3と比較する
- [x] Phase 2へ進めるか判断する（20 epochで改善、追加seed検証済み）

結果:

```text
Run: `b34c59f2244c45a68c5760ee73b14d24` / FINISHED
Train loss: 0.3664 → 0.1919（20 epoch）
Validation loss: 0.2676 → 0.1919（20 epoch）
生成画像: `generated.png`、loss曲線、checkpointをartifact登録済み
 パラメータ数: 11,033,859
学習時間: 286.066秒
所見: baseline（seed 42: validation 0.1931）より約0.65%改善。5 epochでは0.2126で同等以下だったため、早期スクリーニングによる棄却は不適切だった。
追加seed結果:
```text
seed 42: 0.1919 / baseline 0.1931
seed 123: 0.1915 / baseline 0.1929
seed 456: 0.1916 / baseline 0.1926
Run IDs: b34c59f2244c45a68c5760ee73b14d24, 4e8b2caafdcb4cfcb07e87a9dfa4b306, 3bcf897ee0c941dcab523ee9d4b31895
```
採用判断: 3 seedで改善を再現。ただし採用済みE1よりvalidation lossが高く、追加Attentionの計算量もあるためmainには採用しない。5 epochだけで棄却する判断は不適切だった。
```

### E5: deeper U-Net

- [x] `main`から`feature/deeper-unet`を作成
- [x] 各解像度の畳み込みブロック数を増やす（baselineはResidual Block未使用）
- [x] FiLMを同時に追加する場合は別実験として分離する
- [x] 構文チェックを実行する
- [x] 小さなforwardを実行する
- [x] 変更をcommitする
- [x] Phase 1の短縮実験を実行する
- [x] 計算時間とGPUメモリを確認する
- [x] Phase 2へ進めるか判断する（20 epochでも不採用）

結果:

```text
Run: `d8a2da95598b4c599a53c6ad3dad5afe` / FINISHED
Train loss: 0.3984 → 0.1949（20 epoch）
Validation loss: 0.2856 → 0.1944（20 epoch）
生成画像: `generated.png`、loss曲線、checkpointをartifact登録済み
パラメータ数: 13,329,667
学習時間: 377.362秒
所見: baseline（seed 42: 0.1931）より約0.66%悪化。20 epochでも改善せず、5 epochの棄却判断は妥当。
採用判断: 不採用。追加seed検証は行わない。
```

### E6: Residual Block + FiLM + 8×8 Attention

- [x] `main`から`feature/resblock-film-attention-8x8`を作成
- [x] E1のResidual Block + FiLMを維持する
- [x] 8×8・256チャンネルにSelf-Attentionを追加する
- [x] 4×4 Transformerを維持する
- [x] 構文チェックを実行する
- [x] 小さなforwardを実行する
- [x] パラメータ数を記録する
- [x] 変更をcommitする
- [x] 5 epochスクリーニングを実行する
- [x] 20 epoch標準実験を実行する
- [x] E1とvalidation loss、生成画像、計算量を比較する
- [x] 20 epoch時点の採用判断を保留し、100 epoch再検証へ進める
- [x] 100 epochでE1と同一seed・同一条件で比較する
- [x] 100 epochのvalidation loss、loss曲線、生成画像、checkpoint、学習時間を確認する
- [x] 採用または不採用を判断する

結果:

```text
Run: `022ecf42c8a3403096322c69ba96328c`, `22f51d8fc6624b668f2cb0e7cecc50cd`, `85d6d6c0f3a44aa896971d8837463ce4` / FINISHED
Train loss: seed 42 = 0.1897、seed 123 = 0.1903、seed 456 = 0.1908（20 epoch）
Validation loss: seed 42 = 0.1901、seed 123 = 0.1919、seed 456 = 0.1917（20 epoch）
生成画像: 全Runで`generated.png`、loss曲線、checkpointをartifact登録済み
パラメータ数: 16,168,457
学習時間: 349.509〜548.629秒
所見: seed 42ではE1を改善したが、seed 123/456ではE1を改善しなかった。改善が安定せず、パラメータ数と計算時間も増加。
採用判断: 20 epochでは不採用とした。100 epoch再検証の結果、E6のvalidation lossは0.180823、E1は0.182780でE6が0.001957（約1.07%）低かった。ただし100 epochの追加seed検証は未実施で、20 epochでは3 seed中1 seedのみ改善だったため、改善の再現性は未確認。mainにはマージせず、採用済みE1を維持する。
```

### Phase 6: E6の100 epoch再比較

- [x] E6をseed `42`・100 epochで実行する
- [x] E1をseed `42`・100 epochで実行する
- [x] epoch 100のvalidation lossを比較する
- [x] 途中epochのloss曲線も比較する
- [x] 生成画像とcheckpointを確認する
- [x] 計算時間とパラメータ数を比較する
- [x] E6を採用または不採用と判断する

結果:

```text
条件: CIFAR-10、seed 42、batch size 128、learning rate 2e-4、Heun法、sampling steps 50

E6 Run: `157666ed28494b2f96e84f35783865cf` / FINISHED
- epoch 100 train loss: 0.179799
- epoch 100 validation loss: 0.180823
- 最良validation loss: 約0.1779（epoch 93）
- パラメータ数: 16,168,457
- 学習時間: 約1,635.274秒
- artifacts: `generated.png`、`loss_curve.png`、`unet_model_100.pt`

E1 Run: `fd4d15fce2ff42d9bdc6bc2832b50f1b` / FINISHED
- epoch 100 train loss: 0.180892
- epoch 100 validation loss: 0.182780
- 途中の最良validation loss: 約0.1785（epoch 93）
- パラメータ数: 15,362,313
- 学習時間: 1,565.720秒
- artifacts: `generated.png`、`loss_curve.png`、`unet_model_100.pt`

判断: E6はepoch 100のvalidation lossでE1より0.001957（約1.07%）改善し、学習後半で逆転した。ただし20 epochの3 seed検証ではE6の改善がseed 42に限定され、seed 123/456ではE1に負けている。100 epochでseed 42以外を再検証していないため、安定した精度向上とは判断しない。E6はmainへマージせず、E1を採用モデルとして維持する。
```

## 採用判断の基準

- [x] 学習が正常終了している
- [x] MLflowに必要なパラメータが記録されている
- [x] loss曲線が存在する
- [x] checkpointが存在する
- [x] 生成画像が存在する
- [x] validation lossまたは生成品質がbaselineより改善している
- [x] 計算量とGPUメモリが許容範囲に収まっている
- [x] 複数seedで改善が再現している
- [x] 改善した候補だけを`main`へマージする

## 採用済みモデルの記録

```text
モデル名: UNet-resblock-film
採用日: 2026-09-21
マージ元ブランチ: feature/resblock-film
Git commit: 5d88f0a
変更内容: 畳み込みブロックをResidual Blockへ変更し、時刻埋め込みをFiLMのscale/shiftで注入
パラメータ数: 15,362,313
比較したbaseline: UNet-baseline、10,227,715 parameters
改善結果: seed 42/123/456のvalidation lossすべてで改善（0.48〜1.24%）
未解決の問題: baselineより学習時間約13%、パラメータ数約50%増加。生成品質の定量評価は未実装。
```

## 注意事項

- baselineと候補モデルを同じコード内のフラグで切り替えない。
- 1つのfeatureブランチには、原則として1つのモデル変更だけを入れる。
- 長時間学習の前に必ずcommitする。
- 実験生成物、checkpoint、生成画像、MLflowローカルデータはcommitしない。
- lossだけで採用を決めず、生成画像とRunの完了状態も確認する。

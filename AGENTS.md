# 実験ルール

このリポジトリでは、モデルの変更を一つずつ検証し、結果が確認できたものだけを`main`へ取り込む。

## Gitブランチ

- `main`には、現在採用しているモデルと再現可能な実験コードを置く。
- 新しいモデル変更は、必ず`main`から`feature/<変更名>`ブランチを作って行う。
- 1つのfeatureブランチでは、原則として1つのモデル変更だけを扱う。
- baselineと候補モデルを同じコード内の切り替えフラグで管理しない。比較対象はブランチとして分ける。
- 長い学習を始める前に、変更をcommitしておく。MLflowに記録されるGit commitと学習内容を対応させる。
- 実験結果を確認し、改善が認められた場合だけ`main`へマージする。
- 実験中の生成物やMLflowのローカルデータをcommitしない。

## 実験条件

baselineと候補モデルを比較するときは、モデル構造以外をそろえる。

- データセット、前処理、データ拡張を同じにする。
- seedを固定し、Python、NumPy、PyTorch、CUDA、DataLoaderの乱数を記録する。
- `batch_size=128`、`num_epochs=20`、`learning_rate=2e-4`を標準条件とする。
- Flow Matchingの補間と損失の定義を変更しない。
- 生成時はHeun法、`sampling_steps=50`を標準条件とする。
- 比較の途中でGPU、精度設定、データローダー設定を変更しない。
- 条件を変更する場合は、モデル改善とは別の実験として扱い、変更内容をRunのパラメータに記録する。

## MLflow

- Experiment名は`CIFAR10-UNet`を使う。
- MLflowのtracking URIをコードで明示的に変更しない。MLflow 3.7以降のデフォルトであるプロジェクト直下の`mlflow.db`を使う。
- UIは必要に応じて次で起動する。

  ```bash
  .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```

- Run名にはfeature名を使う。例：`baseline`、`film`、`resblock-film`。
- 少なくとも次のパラメータを記録する。
  - モデル名と変更内容
  - seed
  - batch size、epoch数、learning rate
  - サンプラーとsampling steps
  - パラメータ数とdevice
- epochごとの`train_loss`を記録する。
- 学習済みモデルと生成画像をartifactとして保存する。
- lossだけでなく、loss曲線、生成画像、学習完了状態を確認してから結果を判断する。
- lossや生成画像が欠けているRunは、比較結果の根拠にしない。

## 実験の進め方

1. `main`の状態を確認し、`feature/<変更名>`ブランチを作る。
2. モデル変更を実装し、構文チェックと小さなforward確認を行う。
3. 実験条件をMLflowへ記録できることを確認する。
4. 変更をcommitしてから、標準条件で学習を実行する。
5. MLflowのloss、artifact、生成画像、Runの完了状態を確認する。
6. `main`のbaselineと比較し、改善が確認できた場合だけマージする。
7. 次の変更は、更新された`main`から新しいfeatureブランチを作る。

MLflowの実験履歴をリセットする場合は、`mlflow.db`と関連artifactを削除する破壊的操作になるため、対象を明示してから行う。

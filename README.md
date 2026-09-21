# Flow Matching

PyTorch で Flow Matching を試すための小さなサンプルです。

## サンプル

- `main_spiral.py`: 2 次元のスパイラル分布を学習し、ノイズからサンプルを生成します。
- `main_cifar.py`: CIFAR-10 を使った小さな U-Net で画像生成を試します。

## セットアップ

Python 3.10 以上を推奨します。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib seaborn
```

CUDA が使える環境では自動的に GPU を使用します。CPU でも実行できますが、CIFAR-10 の学習には時間がかかります。

## 実行

```bash
python main_spiral.py
python main_cifar.py
```

実行すると、生成画像が `images/generated.png` に保存されます。

`main_cifar.py` は初回実行時に CIFAR-10 を `data/` へダウンロードし、学習済みモデルを `models/` に保存します。これらの生成物は Git 管理対象外です。

## 参考

両方のサンプルで、ノイズ `x_0` とデータ `x_1` の間を線形補間し、速度場 `x_1 - x_0` をニューラルネットワークに学習させています。

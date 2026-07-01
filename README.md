# HeadPoseBench-RS

**PyTorchで実装した6種類の頭部姿勢推定(Head Pose Estimation)モデルを、Intel RealSenseカメラと統合して動作させ、比較・ベンチマークするための実験用リポジトリ。**

RealSenseは頭部姿勢推定の精度向上には利用しておらず、**深度情報からカメラ〜頭部間の距離を推定する目的のみ**で使用しています。姿勢推定自体はRGB画像 + 自作PyTorchモデルによって行われます。

---

## 📌 特徴 (Features)

- 🧠 **自作PyTorchモデル 6種類** を切り替えて実行・比較可能
- 📷 **Intel RealSense** (D400シリーズ等) のRGB-Dストリームに対応
- 📏 深度情報を用いた **カメラ〜頭部間の距離推定**
- 📊 複数モデルの推論結果を並べて比較できるベンチマーク機構
- 🎥 リアルタイム推論・可視化(Yaw / Pitch / Roll のオーバーレイ表示)

---

## 🏗️ 構成 (Architecture)

```
RealSenseカメラ
 ├── RGBストリーム ──▶ 顔検出 ──▶ [Head Pose Model ×6] ──▶ Yaw/Pitch/Roll
 └── Depthストリーム ────────────▶ 距離推定 (頭部までの距離)
                                          │
                                          ▼
                          結果の統合・可視化・ベンチマークログ出力
```

- 姿勢推定パイプラインと距離推定パイプラインは独立しており、距離情報は姿勢推定の入力には使用されません。
- モデルはコマンドライン引数や設定ファイルで切り替え可能な設計を想定しています。

---

## 📁 ディレクトリ構成 (Directory Structure)

> ⚠️ 実際の構成に合わせて書き換えてください(以下は一例です)

```
HeadPoseBench-RS/
├── models/
│   ├── model_a.py
│   ├── model_b.py
│   ├── model_c.py
│   ├── model_d.py
│   ├── model_e.py
│   └── model_f.py
├── realsense/
│   ├── capture.py          # RGB-D取得
│   └── distance_estimator.py
├── benchmark/
│   ├── run_benchmark.py    # 6モデル一括評価
│   └── metrics.py
├── weights/                # 学習済み重み (.pth) ※Git管理外推奨
├── configs/
│   └── model_config.yaml
├── requirements.txt
└── README.md
```

---

## 🔧 必要環境 (Requirements)

- Python 3.9+
- PyTorch (CUDA対応推奨)
- Intel RealSense SDK 2.0 (`pyrealsense2`)
- OpenCV
- NumPy

```bash
pip install -r requirements.txt
```

> `pyrealsense2` はOS・Pythonバージョンによって導入方法が異なるため、[公式ドキュメント](https://github.com/IntelRealSense/librealsense) も参照してください。

---

## 🚀 使い方 (Usage)

### 1. RealSenseカメラで単一モデルを実行

```bash
python run.py --model model_a --show-distance
```

### 2. 6モデルを一括でベンチマーク

```bash
python benchmark/run_benchmark.py --models all --output results/benchmark.csv
```

### 3. オプション例

| オプション | 説明 |
|---|---|
| `--model` | 使用するモデル名 (`model_a`〜`model_f`) |
| `--show-distance` | 深度情報から推定した距離を画面に表示 |
| `--save-video` | 推論結果を動画として保存 |
| `--device` | `cpu` / `cuda` |

---

## 🧪 モデル一覧 (Models)

> ⚠️ 各モデルの詳細(アーキテクチャ・学習データ・入力サイズ等)に書き換えてください

| モデル名 | 概要 | パラメータ数 | 備考 |
|---|---|---|---|
| model_a | | | |
| model_b | | | |
| model_c | | | |
| model_d | | | |
| model_e | | | |
| model_f | | | |

---

## 📊 ベンチマーク結果 (Benchmark)

> ⚠️ 実験結果に置き換えてください

| モデル | MAE (Yaw) | MAE (Pitch) | MAE (Roll) | FPS (RealSense入力) |
|---|---|---|---|---|
| model_a | - | - | - | - |
| model_b | - | - | - | - |
| ... | | | | |

---

## 📝 今後の予定 (Roadmap)

- [ ] 各モデルの学習コード公開
- [ ] 深度情報を用いた姿勢推定精度の検証実験
- [ ] Dockerfileの追加
- [ ] ONNX/TensorRTへのエクスポート対応

---

## 📄 ライセンス (License)

このリポジトリは [MIT License](LICENSE) のもとで公開されています。(必要に応じて変更してください)

---

## 🙏 引用 (Citation)

研究等で利用する場合は以下のように引用してください(内容は適宜書き換えてください):

```bibtex
@misc{headposebench_rs,
  author = {Your Name},
  title  = {HeadPoseBench-RS: Multi-Model Head Pose Estimation Benchmark with RealSense},
  year   = {2026},
  url    = {https://github.com/yourname/HeadPoseBench-RS}
}
```

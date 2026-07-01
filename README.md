# HeadPoseBench-RS

**Intel RealSenseカメラを用いて、複数の頭部姿勢推定(Head Pose Estimation)手法を実際に動かし、統合・検証した際のスクリプト集 兼 環境構築ノウハウ集。**

世の中に存在する頭部姿勢推定の実装は、論文はあってもREADMEが不十分だったり、依存関係が古くて動かなかったりすることが多いです。本リポジトリでは実際に**複数の手法を試し、スクリプトとして動かせたものを記録**しています。

> ⚠️ 各手法の精度を比較する「ベンチマーク」ではなく、**「RealSense環境下でどう動かすか」の実践的ノウハウ集**です。

RealSenseは深度情報から**カメラ〜頭部間の距離推定**に使用しており、姿勢推定の精度そのものには関与していません(距離推定と姿勢推定は独立したパイプラインです)。

---

## 📌 検証結果一覧

| 手法 | 言語/FW | RealSense対応 | 状態 | コメント |
|---|---|---|---|---|
| [6DRepNet](#1-6drepnet) | Python / PyTorch | スクリプト内で対応 | ✅ 動作 | pipで導入可、CPU動作可(`gpu_id=-1`) |
| [Head-Pose-ncnn-Raspberry-Pi-4](#2-head-pose-ncnn-raspberry-pi-4) | C++ / ncnn | ネイティブ対応 (`realsense.cpp`) | ✅ 動作 | Ubuntu 20.04/22.04限定、24.04はncnnビルド不可 |
| [Lightweight-Head-Pose-Estimation](#3-lightweight-head-pose-estimation) | Python / PyTorch | スクリプト内で対応 | ✅ 動作 | 1箇所コード修正でCPU動作可 |
| [FSA-Net](#4-fsa-net) | Python / TensorFlow(Keras) | スクリプト内で対応 | ⚠️ 動作(重い) | 2箇所コード修正が必要、ノートPCでは処理が重い |
| [headpose-fsanet-pytorch](#5-headpose-fsanet-pytorch) | Python / PyTorch, ONNX Runtime | スクリプト内で対応 | ✅ 動作 | FSA-NetのPyTorch移植版 |

---

## 📁 ディレクトリ構成

```
HeadPoseBench-RS/
├── 6DRepNet/
│   ├── LICENSE
│   └── realsense.py
├── Head-Pose-ncnn-Raspberry-Pi-4/
│   ├── LICENSE
│   └── realsense.cpp
├── Lightweight-Head-Pose-Estimation/
│   ├── LICENSE
│   └── realsense.py
├── FSA-Net/
│   ├── LICENSE
│   └── realsense.py
├── headpose-fsanet-pytorch/
│   ├── LICENSE
│   └── realsense.py
├── LICENSE          # 本リポジトリ自体のライセンス
└── README.md
```

各手法のフォルダは独立しており、`realsense.py`(Head-Pose-ncnn-Raspberry-Pi-4のみC++の`realsense.cpp`)を実行すればその場でRealSense統合デモが動く構成になっています。


---

## 🔧 共通の必要環境

- Ubuntu 20.04 / 22.04 推奨(一部手法は24.04でビルド不可)
- Python 3.9+
- Intel RealSense SDK 2.0 (`pyrealsense2` / `librealsense2`)
- OpenCV
- 手法ごとの追加依存(下記参照)

---

## 各手法の詳細

### 1. 6DRepNet

- GitHub: https://github.com/thohemp/6DRepNet
- arXiv: https://doi.org/10.48550/arXiv.2202.12555

最も手軽に動かせた手法。pipで導入可能で、CPUでも動作する。

```bash
uv pip install sixdrepnet
```

顔検出はHaar Cascade(OpenCV付属)を使用し、検出した顔領域を`SixDRepNet.predict()`に渡すことでYaw/Pitch/Rollを取得。`6DRepNet/realsense.py` を実行すれば、カメラ映像上に姿勢軸とFPSがオーバーレイ表示される。

```bash
python 6DRepNet/realsense.py
```

---

### 2. Head-Pose-ncnn-Raspberry-Pi-4

- GitHub: https://github.com/Qengineering/Head-Pose-ncnn-Raspberry-Pi-4

Raspberry Pi 4でも動作するように、[ncnn](https://github.com/Tencent/ncnn)(Tencent製の軽量推論フレームワーク)で実装された手法。**RealSenseとの統合が最も明示的**で、`solvePnP`による姿勢推定と、深度フレームからの距離推定(`get_distance` + `rs2_deproject_pixel_to_point`)を両方行っている。

C++実装のため、ncnn自体をソースからビルドする必要がある。

#### ncnnのビルド

```bash
sudo apt update
sudo apt install -y build-essential git cmake libopencv-dev wget unzip

mkdir -p ~/build_ncnn && cd ~/build_ncnn
git clone https://github.com/Tencent/ncnn.git
cd ncnn && git submodule update --init

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=OFF \
      -DNCNN_SYSTEM_GLSLANG=OFF \
      -DNCNN_BUILD_EXAMPLES=OFF ..
make -j$(nproc)
sudo make install
```

> ⚠️ **Ubuntu 24.04ではncnnのビルドに失敗する。** Ubuntu 20.04 / 22.04を推奨。

#### 本体のビルド

`CMakeLists.txt`内の`ncnn_DIR`は環境に応じて書き換える(`sudo make install`先に依存)。

```cmake
set(ncnn_DIR /build_ncnn/ncnn/build/install/lib/cmake/ncnn)
```

```bash
git clone https://github.com/Qengineering/Head-Pose-ncnn-Raspberry-Pi-4.git
mkdir build && cd build
cmake ..
make -j$(nproc)
./HeadPose
```

> ⚠️ `FaceDetector.cpp` 44行目の `set_num_threads` 呼び出しは、参照先のncnnに存在しないためコメントアウトが必要。

---

### 3. Lightweight-Head-Pose-Estimation

- GitHub: https://github.com/Shaw-git/Lightweight-Head-Pose-Estimation
- IEEE: https://doi.org/10.1109/TMM.2022.3144893

```bash
pip install numpy opencv-python torch torchvision
python Lightweight-Head-Pose-Estimation/realsense.py
```

> ⚠️ CPUで動かす場合は `realsense.py` 内の該当箇所を以下のように書き換える:
> ```python
> pose_estimator = pose_estimator.to('cpu').eval()
> ```
> 顔検出を使いたい場合は追加で MTCNN + TensorFlow が必要。

---

### 4. FSA-Net

- GitHub: https://github.com/shamangary/FSA-Net
- IEEE: https://doi.org/10.1109/CVPR.2019.00118

```bash
git clone https://github.com/shamangary/FSA-Net.git
pip install numpy opencv-python "tensorflow<2.16"
```

以下2箇所の修正が必要(Apache License 2.0では改変ファイルへの変更通知が必要なため、各ファイルの先頭にも変更内容を明記すること):

- `lib/loupe_keras.py` 15行目
  ```python
  # Modified by [your name], 2026: replaced deprecated tensorflow.contrib with tf_slim
  # import tensorflow.contrib.slim as slim
  import tf_slim as slim
  ```
- `lib/capsulelayers.py` 195行目
  ```python
  # Modified by [your name], 2026: dim= -> axis= for newer TensorFlow API
  # c = tf.nn.softmax(b, dim=1)
  c = tf.nn.softmax(b, axis=1)
  ```

```bash
python FSA-Net/realsense.py
```

> ⚠️ ノートPC環境では処理がカクつく(重い)。

---

### 5. headpose-fsanet-pytorch

- GitHub: https://github.com/ohtlab/headpose-fsanet-pytorch
- IEEE: https://doi.org/10.1109/CVPR.2019.00118 (FSA-NetのPyTorch移植版)

```bash
git clone https://github.com/ohtlab/headpose-fsanet-pytorch.git
pip install numpy opencv-python onnxruntime
python headpose-fsanet-pytorch/realsense.py
```

FSA-Netと同じ精度・軽さを、TensorFlow環境構築なしで得られるのが利点。

---

## 📝 今後の予定 (Roadmap)

- [ ] 各手法の推論速度(FPS)を同一環境で計測・記録
- [ ] RealSenseの距離推定を全手法で統一的に使えるようラッパー化
- [ ] Ubuntu 24.04でのncnnビルド問題の解消

---

## 📄 ライセンス

本リポジトリのそれぞれのフォルダは、元となったリポジトリのライセンスに従います。**各フォルダに元リポジトリの`LICENSE`ファイルをそのまま同梱**しています(著作権表示・ライセンス文を保持する義務があるため)。

| 手法 | オリジナル | ライセンス | 備考 |
|---|---|---|---|
| 6DRepNet | [thohemp/6DRepNet](https://github.com/thohemp/6DRepNet) | MIT | |
| Head-Pose-ncnn-Raspberry-Pi-4 | [Qengineering/Head-Pose-ncnn-Raspberry-Pi-4](https://github.com/Qengineering/Head-Pose-ncnn-Raspberry-Pi-4) | BSD-3-Clause | 著作権者名を宣伝・推奨目的で使用することは不可 |
| Lightweight-Head-Pose-Estimation | [Shaw-git/Lightweight-Head-Pose-Estimation](https://github.com/Shaw-git/Lightweight-Head-Pose-Estimation) | MIT | |
| FSA-Net | [shamangary/FSA-Net](https://github.com/shamangary/FSA-Net) | Apache License 2.0 | **本リポジトリで2箇所のコード改変あり。改変ファイルには変更内容を明記済み**([該当箇所](#6-fsa-net)を参照) |
| headpose-fsanet-pytorch | [ohtlab/headpose-fsanet-pytorch](https://github.com/ohtlab/headpose-fsanet-pytorch) | MIT | Copyright (c) 2020, Omar Hassan |

いずれも比較的制約の緩い(permissive)ライセンスのため、著作権表示とライセンス文を保持していれば再配布・改変は問題ありません。

本リポジトリ自体(統合スクリプト・検証ノウハウ部分)は [MIT License](LICENSE) のもとで公開します(必要に応じて変更してください)。

---

## 📚 引用 (Citation)

各手法を使う際は、ライセンス上の義務ではありませんが、開発者への敬意として論文を引用することを推奨します。

**6DRepNet**
> Thorsten Hempel, Ahmed A. Abdelrahman, Ayoub Al-Hamadi. "6D Rotation Representation for Unconstrained Head Pose Estimation." ICIP 2022.

**Lightweight-Head-Pose-Estimation**
> "Accurate Head Pose Estimation Using Image Rectification and Lightweight Convolutional Neural Network." IEEE Transactions on Multimedia (TMM), 2022.

**FSA-Net / headpose-fsanet-pytorch**
> Tsun-Yi Yang, Yi-Ting Chen, Yen-Yu Lin, Yung-Yu Chuang. "FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image." CVPR 2019.

**Head-Pose-ncnn-Raspberry-Pi-4**
> 論文はなく、[Tencent/ncnn](https://github.com/Tencent/ncnn)・[Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)・[Face-Detector-1MB-with-landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark) を基にしたエンジニアリング実装のため、これらのリポジトリへのクレジットを記載してください。

# AI Vision Visualizer

AI の「環世界（Umwelt）」を可視化するツール。動画の各フレームに対して **Grad-CAM ヒートマップ**、**物体検出**、**セマンティックセグメンテーション**、**深度推定** を実行し、AI が何を見ているかを直感的に理解できる映像を生成する。

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

> [English README](README.en.md)

## Vision Modes

| モード | モデル | 説明 |
|--------|--------|------|
| `gradcam` | ResNet50 (Grad-CAM) | CNN の注目領域をヒートマップで可視化 |
| `detect` | YOLOv8 | 物体検出 + 5種 BBox スタイル |
| `segment` | DeepLabV3 ResNet101 | セマンティックセグメンテーション + グリッチエフェクト |
| `depth` | MiDaS DPT-Large | 単眼深度推定 + 4種深度表示 |
| `all` | 上記4モデル同時 | 2x2 グリッドで全モードを同時表示 |

## Demo

### sidebyside

左: 元映像、右: Grad-CAM ヒートマップオーバーレイ

![sidebyside](assets/sidebyside.png)

### triple

左: 元映像、中: ヒートマップ単体、右: オーバーレイ

![triple](assets/triple.png)

## Requirements

- Python 3.9+
- GPU 推奨 (CUDA)、CPU でも動作可

## Setup

```bash
# コア依存 (必須)
pip install torch torchvision opencv-python numpy tqdm timm

# 物体検出モード (--mode detect)
pip install ultralytics

# Depth Anything V2 (--mode depth --depth-model depth_anything)
pip install transformers
```

## Usage

```bash
# Grad-CAM (デフォルト、後方互換)
python src/umwelt.py input.mp4
python src/umwelt.py input.mp4 --layout triple --alpha 0.6 --colormap turbo

# 物体検出
python src/umwelt.py input.mp4 --mode detect --bbox-style cyber
python src/umwelt.py input.mp4 --mode detect --bbox-style hud --conf-threshold 0.5

# セマンティックセグメンテーション
python src/umwelt.py input.mp4 --mode segment --glitch-style mixed --glitch-intensity 0.8

# 深度推定
python src/umwelt.py input.mp4 --mode depth --depth-style fog
python src/umwelt.py input.mp4 --mode depth --depth-style 3d

# 全モード同時表示 (2x2 グリッド)
python src/umwelt.py input.mp4 --mode all
```

### Options

#### 共通

| Option | Default | Description |
|--------|---------|-------------|
| `input` | *(required)* | 入力動画ファイル |
| `-o, --output` | `{input}_{mode}.mp4` | 出力ファイルパス |
| `--mode` | `gradcam` | `gradcam` / `detect` / `segment` / `depth` / `all` |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--no-audio` | `False` | 音声を含めない |

#### Grad-CAM

| Option | Default | Description |
|--------|---------|-------------|
| `--layout` | `overlay` | `overlay` / `sidebyside` / `triple` |
| `--alpha` | `0.5` | ヒートマップ透明度 (0.0-1.0) |
| `--colormap` | `jet` | カラーマップ (`jet`, `hot`, `inferno`, `turbo` 等) |
| `--target-class` | auto | 対象クラス ID (0-999) |
| `--top-k` | `3` | 表示する予測クラス数 |

#### Detection

| Option | Default | Description |
|--------|---------|-------------|
| `--bbox-style` | `default` | `default` / `corners` / `cyber` / `minimal` / `hud` |
| `--yolo-model` | `yolov8n.pt` | YOLOv8 モデル名 |
| `--conf-threshold` | `0.25` | 検出確信度の閾値 |

#### Segmentation

| Option | Default | Description |
|--------|---------|-------------|
| `--glitch-style` | `mixed` | `rgb_shift` / `pixel_sort` / `scanline` / `displacement` / `mixed` |
| `--glitch-intensity` | `0.5` | グリッチ強度 (0.0-1.0) |
| `--seg-alpha` | `0.5` | セグメンテーションマスクの透明度 |

#### Depth

| Option | Default | Description |
|--------|---------|-------------|
| `--depth-style` | `colormap` | `colormap` / `fog` / `contour` / `3d` |
| `--depth-model` | `midas` | `midas` / `depth_anything` |

## Architecture

```
src/
├── umwelt.py              CLI・モード分岐・メインループ
├── gradcam.py             Grad-CAM エンジン (ResNet50 layer4)
├── detector.py            物体検出エンジン (YOLOv8)
├── segmentor.py           セマンティックセグメンテーション (DeepLabV3)
├── depth.py               深度推定エンジン (MiDaS / Depth Anything V2)
├── renderer.py            描画・レイアウト合成
├── video_io.py            動画 I/O (OpenCV)
├── utils/
│   ├── colors.py          カラーパレット・カラーマップ定義
│   └── labels.py          クラスラベル辞書 (ImageNet/COCO/VOC)
└── effects/
    ├── bbox_styles.py     5種 BBox 描画スタイル
    ├── glitch.py          5種グリッチエフェクト
    └── depth_styles.py    4種深度表示スタイル
```

## Performance

| Mode | GPU (RTX 3060) | CPU (Core i7) |
|------|---------------|---------------|
| gradcam | ~25 fps | ~2-5 fps |
| detect | ~45 fps | ~5-10 fps |
| segment | ~9 fps | ~1-2 fps |
| depth | ~12 fps | ~1-3 fps |
| all | ~4 fps | <1 fps |

## Notes

- `--mode` 未指定時は `gradcam` がデフォルト（v1 との後方互換性を維持）
- `ultralytics`, `transformers` は該当モード使用時のみインポート。未インストール時は `pip install` を案内
- `ffmpeg` がインストール済みの場合、元動画の音声を自動的にコピーする
- `--mode all` は 4 モデルを同時ロード。VRAM が不足した場合は該当エンジンをスキップ

## References

- Selvaraju, R.R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Redmon, J. et al. — YOLOv8 (Ultralytics)
- Chen, L.-C. et al. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.* ECCV 2018.
- Ranftl, R. et al. (2021). *Vision Transformers for Dense Prediction.* ICCV 2021. (MiDaS)

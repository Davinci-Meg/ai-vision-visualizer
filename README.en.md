# AI Vision Visualizer

A tool for visualizing an AI's "Umwelt" (perceptual world). It runs **Grad-CAM heatmaps**, **object detection**, **semantic segmentation**, and **monocular depth estimation** on each frame of a video, generating an intuitive visualization of what the AI sees.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

> [日本語版 README はこちら](README.md)

## Vision Modes

| Mode | Model | Description |
|------|-------|-------------|
| `gradcam` | ResNet50 (Grad-CAM) | Visualize CNN attention regions as heatmaps |
| `detect` | YOLOv8 | Object detection with 5 bounding box styles |
| `segment` | DeepLabV3 ResNet101 | Semantic segmentation with glitch effects |
| `depth` | MiDaS DPT-Large | Monocular depth estimation with 4 display styles |
| `all` | All 4 models | 2x2 grid showing all modes simultaneously |

## Demo

### sidebyside

Left: original, Right: Grad-CAM heatmap overlay

![sidebyside](assets/sidebyside.png)

### triple

Left: original, Center: raw heatmap, Right: overlay

![triple](assets/triple.png)

## Requirements

- Python 3.9+
- GPU recommended (CUDA), also runs on CPU

## Setup

```bash
# Core dependencies (required)
pip install torch torchvision opencv-python numpy tqdm timm

# Object detection mode (--mode detect)
pip install ultralytics

# Depth Anything V2 (--mode depth --depth-model depth_anything)
pip install transformers
```

## Usage

```bash
# Grad-CAM (default, backward compatible)
python src/umwelt.py input.mp4
python src/umwelt.py input.mp4 --layout triple --alpha 0.6 --colormap turbo

# Object detection
python src/umwelt.py input.mp4 --mode detect --bbox-style cyber
python src/umwelt.py input.mp4 --mode detect --bbox-style hud --conf-threshold 0.5

# Semantic segmentation
python src/umwelt.py input.mp4 --mode segment --glitch-style mixed --glitch-intensity 0.8

# Depth estimation
python src/umwelt.py input.mp4 --mode depth --depth-style fog
python src/umwelt.py input.mp4 --mode depth --depth-style 3d

# All modes simultaneously (2x2 grid)
python src/umwelt.py input.mp4 --mode all
```

### Options

#### Common

| Option | Default | Description |
|--------|---------|-------------|
| `input` | *(required)* | Input video file |
| `-o, --output` | `{input}_{mode}.mp4` | Output file path |
| `--mode` | `gradcam` | `gradcam` / `detect` / `segment` / `depth` / `all` |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--no-audio` | `False` | Exclude audio from output |

#### Grad-CAM

| Option | Default | Description |
|--------|---------|-------------|
| `--layout` | `overlay` | `overlay` / `sidebyside` / `triple` |
| `--alpha` | `0.5` | Heatmap opacity (0.0-1.0) |
| `--colormap` | `jet` | Colormap (`jet`, `hot`, `inferno`, `turbo`, etc.) |
| `--target-class` | auto | Target class ID (0-999) |
| `--top-k` | `3` | Number of top predictions to display |

#### Detection

| Option | Default | Description |
|--------|---------|-------------|
| `--bbox-style` | `default` | `default` / `corners` / `cyber` / `minimal` / `hud` |
| `--yolo-model` | `yolov8n.pt` | YOLOv8 model name |
| `--conf-threshold` | `0.25` | Detection confidence threshold |

#### Segmentation

| Option | Default | Description |
|--------|---------|-------------|
| `--glitch-style` | `mixed` | `rgb_shift` / `pixel_sort` / `scanline` / `displacement` / `mixed` |
| `--glitch-intensity` | `0.5` | Glitch effect intensity (0.0-1.0) |
| `--seg-alpha` | `0.5` | Segmentation mask opacity |

#### Depth

| Option | Default | Description |
|--------|---------|-------------|
| `--depth-style` | `colormap` | `colormap` / `fog` / `contour` / `3d` |
| `--depth-model` | `midas` | `midas` / `depth_anything` |

## Architecture

```
src/
├── umwelt.py              CLI, mode dispatch & main loop
├── gradcam.py             Grad-CAM engine (ResNet50 layer4)
├── detector.py            Object detection engine (YOLOv8)
├── segmentor.py           Semantic segmentation (DeepLabV3)
├── depth.py               Depth estimation engine (MiDaS / Depth Anything V2)
├── renderer.py            Rendering & layout compositing
├── video_io.py            Video I/O (OpenCV)
├── utils/
│   ├── colors.py          Color palettes & colormap definitions
│   └── labels.py          Class label dictionaries (ImageNet/COCO/VOC)
└── effects/
    ├── bbox_styles.py     5 bounding box drawing styles
    ├── glitch.py          5 glitch effects
    └── depth_styles.py    4 depth visualization styles
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

- When `--mode` is omitted, defaults to `gradcam` (backward compatible with v1)
- `ultralytics` and `transformers` are only imported when their respective modes are used. If not installed, a helpful `pip install` message is shown
- If `ffmpeg` is installed, audio from the original video is automatically copied to the output
- `--mode all` loads 4 models simultaneously. If VRAM is insufficient, unavailable engines are gracefully skipped

## References

- Selvaraju, R.R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Redmon, J. et al. — YOLOv8 (Ultralytics)
- Chen, L.-C. et al. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.* ECCV 2018.
- Ranftl, R. et al. (2021). *Vision Transformers for Dense Prediction.* ICCV 2021. (MiDaS)

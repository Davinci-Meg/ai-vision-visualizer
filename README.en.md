# AI Vision Visualizer

A tool that overlays **Grad-CAM heatmaps** on each frame of a video, visualizing what a CNN "sees" in every frame.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

> [日本語版 README はこちら](README.md)

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
pip install torch torchvision opencv-python numpy tqdm
```

## Usage

```bash
# Basic
python src/umwelt.py input.mp4

# Customize layout, colormap, and alpha
python src/umwelt.py input.mp4 -o output.mp4 \
    --layout triple \
    --alpha 0.6 \
    --colormap turbo \
    --top-k 5

# Visualize attention for a specific class (e.g. ImageNet "cat" = 281)
python src/umwelt.py input.mp4 --target-class 281 --layout sidebyside

# Force CPU
python src/umwelt.py input.mp4 --device cpu
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `input` | *(required)* | Input video file |
| `-o, --output` | `{input}_umwelt.mp4` | Output file path |
| `--layout` | `overlay` | `overlay` / `sidebyside` / `triple` |
| `--alpha` | `0.5` | Heatmap opacity (0.0-1.0) |
| `--colormap` | `jet` | Colormap (`jet`, `hot`, `inferno`, `turbo`, etc.) |
| `--target-class` | auto | Target class ID (0-999). Uses top prediction if omitted |
| `--top-k` | `3` | Number of top predictions to display |
| `--show-label` | `True` | Show class names and confidence scores |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--no-audio` | `False` | Exclude audio from output |

## Architecture

```
src/
├── umwelt.py      CLI & main loop
├── gradcam.py     Grad-CAM engine (ResNet50 layer4)
├── renderer.py    Heatmap rendering & layout compositing
└── video_io.py    Video I/O (OpenCV)
```

**Processing pipeline:**

```
Input video → Read frames → Preprocess (224x224) → ResNet50 inference
    → Grad-CAM heatmap → Apply colormap → Blend with original frame
    → Draw predictions → Write to output video
```

## Performance

| Environment | Speed |
|-------------|-------|
| GPU (RTX 3060) | ~20-30 fps |
| GPU (RTX 4090) | ~50-80 fps |
| CPU (Core i7) | ~2-10 fps |

Inference runs at a fixed 224x224, so processing speed is largely independent of input resolution.

## Notes

- Model is **ResNet50** (ImageNet pretrained)
- Grad-CAM targets **layer4** (final conv block, 7x7 feature map)
- If `ffmpeg` is installed, audio from the original video is automatically copied to the output
- Output codec is MP4 (mp4v)

## References

- Selvaraju, R.R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.

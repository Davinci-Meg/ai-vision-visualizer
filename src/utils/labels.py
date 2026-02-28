"""ラベル辞書 — ImageNet / COCO / VOC クラス名"""

import json
import urllib.request

# ── ImageNet ラベル（初回のみダウンロードしてキャッシュ） ──────────────────

_IMAGENET_CACHE: list[str] | None = None
_IMAGENET_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels"
    "/master/imagenet-simple-labels.json"
)


def get_imagenet_labels() -> list[str]:
    """ImageNet 1000 クラスのラベルリストを取得する。"""
    global _IMAGENET_CACHE
    if _IMAGENET_CACHE is not None:
        return _IMAGENET_CACHE
    try:
        with urllib.request.urlopen(_IMAGENET_URL, timeout=10) as resp:
            _IMAGENET_CACHE = json.loads(resp.read().decode())
    except Exception:
        # フォールバック: クラスIDをそのまま文字列に
        _IMAGENET_CACHE = [f"class_{i}" for i in range(1000)]
    return _IMAGENET_CACHE


# ── COCO 80 クラス名 ─────────────────────────────────────────────────────

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ── Pascal VOC 21 クラス名（background 含む） ────────────────────────────

VOC_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train", "tv/monitor",
]

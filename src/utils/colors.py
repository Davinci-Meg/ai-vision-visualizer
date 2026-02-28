"""カラーユーティリティ — カラーマップテーブル、VOC/COCO色定義、クラス色割り当て"""

import cv2
import numpy as np

# OpenCV カラーマップの対応表（renderer.py から移動）
COLORMAP_TABLE = {
    "jet": cv2.COLORMAP_JET,
    "hot": cv2.COLORMAP_HOT,
    "inferno": cv2.COLORMAP_INFERNO,
    "turbo": cv2.COLORMAP_TURBO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "bone": cv2.COLORMAP_BONE,
    "ocean": cv2.COLORMAP_OCEAN,
}

# Pascal VOC 21 クラス（background 含む）の標準色パレット (BGR)
VOC_COLORS = [
    (0, 0, 0),        # background
    (0, 0, 128),      # aeroplane
    (0, 128, 0),      # bicycle
    (0, 128, 128),    # bird
    (128, 0, 0),      # boat
    (128, 0, 128),    # bottle
    (128, 128, 0),    # bus
    (128, 128, 128),  # car
    (0, 0, 64),       # cat
    (0, 0, 192),      # chair
    (0, 128, 64),     # cow
    (0, 128, 192),    # dining table
    (128, 0, 64),     # dog
    (128, 0, 192),    # horse
    (128, 128, 64),   # motorbike
    (128, 128, 192),  # person
    (0, 64, 0),       # potted plant
    (0, 64, 128),     # sheep
    (0, 192, 0),      # sofa
    (0, 192, 128),    # train
    (128, 64, 0),     # tv/monitor
]


def _generate_distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """HSV 空間で均等に配置した n 個の区別しやすい BGR 色を生成する。"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color_hsv = np.uint8([[[hue, 200, 230]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))
    return colors


# COCO 80 クラス用カラーパレット (BGR)
COCO_COLORS = _generate_distinct_colors(80)


def get_class_color(class_id: int, palette: str = "coco") -> tuple[int, int, int]:
    """クラス ID に対応する BGR 色を返す。

    Parameters
    ----------
    class_id : int
        クラスインデックス。
    palette : str
        カラーパレット名 ('voc' or 'coco')。

    Returns
    -------
    color : tuple[int, int, int]
        BGR 色タプル。
    """
    if palette == "voc":
        return VOC_COLORS[class_id % len(VOC_COLORS)]
    return COCO_COLORS[class_id % len(COCO_COLORS)]

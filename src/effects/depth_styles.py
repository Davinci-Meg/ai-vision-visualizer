"""深度表示スタイル — 4種の深度マップ可視化"""

import cv2
import numpy as np


def depth_colormap(
    frame: np.ndarray, depth: np.ndarray, **kwargs,
) -> np.ndarray:
    """カラーマップ: 深度値を疑似カラーで表示。"""
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
    colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
    return colored


def depth_fog(
    frame: np.ndarray, depth: np.ndarray, **kwargs,
) -> np.ndarray:
    """フォグ: 深度に基づいて霧を重ねる（遠方ほど霧が濃い）。"""
    h, w = frame.shape[:2]
    depth_resized = cv2.resize(depth, (w, h))

    # 正規化（0=近い, 1=遠い）
    d_min, d_max = depth_resized.min(), depth_resized.max()
    fog_density = (depth_resized - d_min) / (d_max - d_min + 1e-8)

    # MiDaS は近い=大きい値なので反転
    fog_density = 1.0 - fog_density

    # 霧の色（白）
    fog_color = np.full_like(frame, 220, dtype=np.float32)

    fog_alpha = fog_density[:, :, np.newaxis].astype(np.float32) * 0.85
    result = (frame.astype(np.float32) * (1.0 - fog_alpha) + fog_color * fog_alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def depth_contour(
    frame: np.ndarray, depth: np.ndarray, **kwargs,
) -> np.ndarray:
    """コンター: 等深度線をフレームに重ねる。"""
    h, w = frame.shape[:2]
    depth_resized = cv2.resize(depth, (w, h))

    # 正規化してコンターレベルを生成
    d_min, d_max = depth_resized.min(), depth_resized.max()
    depth_norm = ((depth_resized - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)

    result = frame.copy()
    n_levels = 12

    for level in range(1, n_levels):
        threshold = int(255 * level / n_levels)
        _, binary = cv2.threshold(depth_norm, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 深度に応じた色相
        hue = int(180 * level / n_levels)
        color_hsv = np.uint8([[[hue, 180, 220]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        color = tuple(int(c) for c in color_bgr)

        cv2.drawContours(result, contours, -1, color, 1, cv2.LINE_AA)

    return result


def depth_3d(
    frame: np.ndarray, depth: np.ndarray, **kwargs,
) -> np.ndarray:
    """疑似 3D: 深度に基づいてピクセルを水平方向にシフト（ステレオ風）。"""
    h, w = frame.shape[:2]
    depth_resized = cv2.resize(depth, (w, h))

    # 正規化
    d_min, d_max = depth_resized.min(), depth_resized.max()
    depth_norm = (depth_resized - d_min) / (d_max - d_min + 1e-8)

    # 最大シフト量
    max_shift = int(w * 0.03)

    # X 方向のリマッピング
    shift_map = (depth_norm * max_shift).astype(np.float32)
    map_x = np.arange(w, dtype=np.float32)[np.newaxis, :] + shift_map
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.arange(h, dtype=np.float32)[:, np.newaxis] * np.ones(w, dtype=np.float32)

    result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    # 深度マップのオーバーレイ（薄い擬似カラー）
    depth_color = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO,
    )
    result = cv2.addWeighted(result, 0.7, depth_color, 0.3, 0)

    return result


# ── スタイルディスパッチ ─────────────────────────────────────────────────

DEPTH_STYLES = {
    "colormap": depth_colormap,
    "fog": depth_fog,
    "contour": depth_contour,
    "3d": depth_3d,
}


def apply_depth_style(
    frame: np.ndarray,
    depth: np.ndarray,
    style: str = "colormap",
) -> np.ndarray:
    """深度マップを指定スタイルで可視化する。

    Parameters
    ----------
    frame : np.ndarray
        元フレーム (BGR)。
    depth : np.ndarray
        深度マップ (H×W, float32)。
    style : str
        深度表示スタイル名。

    Returns
    -------
    result : np.ndarray
        可視化済みフレーム。
    """
    fn = DEPTH_STYLES.get(style, depth_colormap)
    return fn(frame, depth)

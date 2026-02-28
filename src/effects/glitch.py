"""グリッチエフェクト — セグメンテーションマスクに基づく5種の視覚効果"""

import cv2
import numpy as np


def glitch_rgb_shift(
    frame: np.ndarray, mask: np.ndarray, intensity: float = 0.5,
) -> np.ndarray:
    """RGB チャンネルシフト: マスク領域の RGB チャンネルをずらす。"""
    result = frame.copy()
    h, w = frame.shape[:2]
    shift = int(w * 0.02 * intensity)
    if shift < 1:
        return result

    mask_bool = mask > 0

    # B チャンネルを左にシフト
    shifted_b = np.roll(frame[:, :, 0], -shift, axis=1)
    result[:, :, 0] = np.where(mask_bool, shifted_b, frame[:, :, 0])

    # R チャンネルを右にシフト
    shifted_r = np.roll(frame[:, :, 2], shift, axis=1)
    result[:, :, 2] = np.where(mask_bool, shifted_r, frame[:, :, 2])

    return result


def glitch_pixel_sort(
    frame: np.ndarray, mask: np.ndarray, intensity: float = 0.5,
) -> np.ndarray:
    """ピクセルソート: マスク領域の行を輝度順にソートする。"""
    result = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask_bool = mask > 0

    # ソートする行のサンプリング率を intensity で制御
    step = max(int(4 / intensity), 1)
    for y in range(0, frame.shape[0], step):
        row_mask = mask_bool[y]
        if not row_mask.any():
            continue
        # マスク領域のピクセルを輝度でソート
        indices = np.where(row_mask)[0]
        if len(indices) < 2:
            continue
        start, end = indices[0], indices[-1] + 1
        segment = frame[y, start:end].copy()
        brightness = gray[y, start:end]
        order = np.argsort(brightness)
        result[y, start:end] = segment[order]

    return result


def glitch_scanline(
    frame: np.ndarray, mask: np.ndarray, intensity: float = 0.5,
) -> np.ndarray:
    """スキャンライン: マスク領域に CRT 風の水平走査線を追加。"""
    result = frame.copy()
    h = frame.shape[0]
    gap = max(int(6 / intensity), 2)
    darkness = 0.3 + 0.4 * intensity  # 0.3〜0.7

    mask_bool = mask > 0

    for y in range(0, h, gap):
        row_mask = mask_bool[y]
        if row_mask.any():
            result[y, row_mask] = (
                result[y, row_mask].astype(np.float32) * (1.0 - darkness)
            ).astype(np.uint8)

    # ランダムな水平シフトライン
    n_glitch = int(5 * intensity)
    rng = np.random.default_rng(42)
    for _ in range(n_glitch):
        gy = rng.integers(0, h)
        shift = rng.integers(-int(20 * intensity), int(20 * intensity))
        if mask_bool[gy].any():
            result[gy] = np.roll(result[gy], shift, axis=0)

    return result


def glitch_displacement(
    frame: np.ndarray, mask: np.ndarray, intensity: float = 0.5,
) -> np.ndarray:
    """ディスプレースメント: マスクのエッジに基づいてピクセルを変位。"""
    result = frame.copy()
    h, w = frame.shape[:2]

    # マスクのエッジ検出
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    edges = cv2.Canny(mask_u8, 50, 150)
    edges = cv2.dilate(edges, None, iterations=int(2 * intensity))

    # エッジ近辺のピクセルを変位
    displacement = int(10 * intensity)
    if displacement < 1:
        return result

    edge_coords = np.where(edges > 0)
    for y, x in zip(edge_coords[0], edge_coords[1]):
        src_x = min(max(x + displacement, 0), w - 1)
        result[y, x] = frame[y, src_x]

    return result


def glitch_mixed(
    frame: np.ndarray, mask: np.ndarray, intensity: float = 0.5,
) -> np.ndarray:
    """ミックス: RGB シフト + スキャンライン + ディスプレースメントの組み合わせ。"""
    result = glitch_rgb_shift(frame, mask, intensity * 0.6)
    result = glitch_scanline(result, mask, intensity * 0.5)
    result = glitch_displacement(result, mask, intensity * 0.4)
    return result


# ── スタイルディスパッチ ─────────────────────────────────────────────────

GLITCH_STYLES = {
    "rgb_shift": glitch_rgb_shift,
    "pixel_sort": glitch_pixel_sort,
    "scanline": glitch_scanline,
    "displacement": glitch_displacement,
    "mixed": glitch_mixed,
}


def apply_glitch(
    frame: np.ndarray,
    mask: np.ndarray,
    style: str = "mixed",
    intensity: float = 0.5,
) -> np.ndarray:
    """セグメンテーションマスクに基づくグリッチエフェクトを適用する。

    Parameters
    ----------
    frame : np.ndarray
        元フレーム (BGR)。
    mask : np.ndarray
        セグメンテーションマスク (H×W, int)。0 = 背景。
    style : str
        グリッチスタイル名。
    intensity : float
        エフェクト強度 (0.0〜1.0)。

    Returns
    -------
    result : np.ndarray
        エフェクト適用済みフレーム。
    """
    fn = GLITCH_STYLES.get(style, glitch_mixed)
    return fn(frame, mask, intensity)

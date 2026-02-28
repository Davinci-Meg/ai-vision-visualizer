"""BBox 描画スタイル — 5種の物体検出バウンディングボックス表現"""

import cv2
import numpy as np


def draw_bbox_default(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str, color: tuple[int, int, int], conf: float,
) -> None:
    """標準スタイル: 矩形 + ラベル背景帯。"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
    )
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, cv2.FILLED)
    cv2.putText(
        frame, text, (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_bbox_corners(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str, color: tuple[int, int, int], conf: float,
) -> None:
    """コーナースタイル: 四隅の L 字型マーカーのみ描画。"""
    w, h = x2 - x1, y2 - y1
    corner_len = max(int(min(w, h) * 0.2), 8)
    t = 2

    # 左上
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, t)
    # 右上
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, t)
    # 左下
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, t)
    # 右下
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, t)

    # ラベル（左上の外側）
    text = f"{label} {conf:.0%}"
    cv2.putText(
        frame, text, (x1, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
    )


def draw_bbox_cyber(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str, color: tuple[int, int, int], conf: float,
) -> None:
    """サイバーパンクスタイル: ネオングロー + スキャンライン。"""
    overlay = frame.copy()

    # グロー効果（太い半透明ライン）
    cv2.rectangle(overlay, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, 4)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # メインライン
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # コーナーアクセント（太い短線）
    w, h = x2 - x1, y2 - y1
    accent_len = max(int(min(w, h) * 0.15), 6)
    for (cx, cy), (dx, dy) in [
        ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
        ((x1, y2), (1, -1)), ((x2, y2), (-1, -1)),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * accent_len, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy * accent_len), color, 3)

    # スキャンライン（ボックス内に薄い水平線）
    for sy in range(y1 + 4, y2, 4):
        cv2.line(frame, (x1, sy), (x2, sy), color, 1)
        cv2.addWeighted(frame[sy:sy+1, x1:x2], 0.85, overlay[sy:sy+1, x1:x2], 0.15, 0,
                        frame[sy:sy+1, x1:x2])

    # ラベル（サイバー風フォント）
    text = f"[ {label.upper()} | {conf:.0%} ]"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    label_y = max(y1 - 8, th + 4)
    cv2.putText(
        frame, text, (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
    )


def draw_bbox_minimal(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str, color: tuple[int, int, int], conf: float,
) -> None:
    """ミニマルスタイル: 細線 + 小さなラベルドット。"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # 小さな円形インジケータ + ラベル
    cx = x1 + 4
    cy = y1 - 8 if y1 > 20 else y2 + 16
    cv2.circle(frame, (cx, cy), 3, color, cv2.FILLED)
    cv2.putText(
        frame, f"{label}", (cx + 8, cy + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
    )


def draw_bbox_hud(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str, color: tuple[int, int, int], conf: float,
) -> None:
    """HUD スタイル: ヘッドアップディスプレイ風ブラケット + データ表示。"""
    w, h = x2 - x1, y2 - y1
    bracket_len = max(int(min(w, h) * 0.25), 10)
    t = 2

    # 上部ブラケット [ ]
    cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), color, t)
    cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), color, t)

    # 下部ブラケット
    cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), color, t)
    cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), color, t)

    # 中央十字線
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cross = max(int(min(w, h) * 0.05), 4)
    cv2.line(frame, (cx - cross, cy), (cx + cross, cy), color, 1)
    cv2.line(frame, (cx, cy - cross), (cx, cy + cross), color, 1)

    # データパネル（右上外側）
    texts = [label.upper(), f"CONF: {conf:.0%}", f"SIZE: {w}x{h}"]
    for i, text in enumerate(texts):
        ty = y1 - 6 - (len(texts) - 1 - i) * 14
        if ty < 10:
            ty = y2 + 14 + i * 14
        cv2.putText(
            frame, text, (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )


# ── スタイルディスパッチ ─────────────────────────────────────────────────

BBOX_STYLES = {
    "default": draw_bbox_default,
    "corners": draw_bbox_corners,
    "cyber": draw_bbox_cyber,
    "minimal": draw_bbox_minimal,
    "hud": draw_bbox_hud,
}


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    style: str = "default",
) -> np.ndarray:
    """検出結果をフレームに描画する。

    Parameters
    ----------
    frame : np.ndarray
        描画対象の BGR フレーム（インプレース変更）。
    detections : list of dict
        各検出は {'bbox': (x1,y1,x2,y2), 'label': str, 'conf': float, 'color': (B,G,R)}。
    style : str
        BBox スタイル名。

    Returns
    -------
    frame : np.ndarray
        描画済みフレーム。
    """
    draw_fn = BBOX_STYLES.get(style, draw_bbox_default)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw_fn(frame, x1, y1, x2, y2, det["label"], det["color"], det["conf"])
    return frame

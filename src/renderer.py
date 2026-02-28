"""レンダリング — カラーマップ変換・テキスト描画・レイアウト合成"""

import cv2
import numpy as np

from utils.colors import COLORMAP_TABLE


class HeatmapRenderer:
    """ヒートマップのカラーマップ変換とフレームへの合成を行う。

    Parameters
    ----------
    alpha : float
        ヒートマップの透明度 (0.0〜1.0)。
    colormap : str
        カラーマップ名。
    """

    def __init__(self, alpha: float = 0.5, colormap: str = "jet"):
        self.alpha = alpha
        self.colormap_id = COLORMAP_TABLE.get(colormap, cv2.COLORMAP_JET)

    def colorize(self, heatmap: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """ヒートマップをカラー画像に変換しリサイズする。

        Parameters
        ----------
        heatmap : np.ndarray
            正規化されたヒートマップ (H×W, float32, 0.0〜1.0)。
        target_size : (width, height)
            出力サイズ。

        Returns
        -------
        colored : np.ndarray
            カラーマップ適用済み BGR 画像。
        """
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, self.colormap_id)
        colored = cv2.resize(colored, target_size, interpolation=cv2.INTER_LINEAR)
        return colored

    def overlay(self, frame: np.ndarray, heatmap_colored: np.ndarray) -> np.ndarray:
        """元フレームにヒートマップをアルファブレンドする。"""
        blended = cv2.addWeighted(heatmap_colored, self.alpha, frame, 1.0 - self.alpha, 0)
        return blended


class LayoutComposer:
    """レイアウトモードに応じたフレーム合成を行う。

    Parameters
    ----------
    layout : str
        レイアウトモード ('overlay', 'sidebyside', 'triple')。
    """

    def __init__(self, layout: str = "overlay"):
        self.layout = layout

    def compose(
        self,
        original: np.ndarray,
        heatmap_colored: np.ndarray,
        overlay_frame: np.ndarray,
    ) -> np.ndarray:
        """レイアウトに応じてフレームを合成する。

        Parameters
        ----------
        original : np.ndarray
            元フレーム (BGR)。
        heatmap_colored : np.ndarray
            カラーマップ適用済みヒートマップ (BGR)。
        overlay_frame : np.ndarray
            オーバーレイ済みフレーム (BGR)。

        Returns
        -------
        composed : np.ndarray
            合成後のフレーム。
        """
        if self.layout == "sidebyside":
            return np.hstack([original, overlay_frame])
        elif self.layout == "triple":
            return self._compose_labeled(
                [original, heatmap_colored, overlay_frame],
                ["Human", "AI Raw", "Overlay"],
            )
        else:
            # overlay (デフォルト)
            return overlay_frame

    @staticmethod
    def _compose_labeled(
        panels: list[np.ndarray], labels: list[str],
    ) -> np.ndarray:
        """パネルにラベル帯を付けて横に並べる。"""
        h = panels[0].shape[0]
        label_h = max(int(h * 0.04), 16)
        labeled_panels = []

        for panel, label in zip(panels, labels):
            p = panel.copy()
            font_scale = max(h * 0.001, 0.4)
            thickness = max(int(h * 0.003), 1)
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness,
            )[0]
            text_x = (panel.shape[1] - text_size[0]) // 2
            text_y = h - label_h // 2 + text_size[1] // 2
            # 半透明の黒帯
            cv2.rectangle(p, (0, h - label_h), (panel.shape[1], h), (0, 0, 0), cv2.FILLED)
            cv2.putText(
                p, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA,
            )
            labeled_panels.append(p)

        return np.hstack(labeled_panels)

    @staticmethod
    def compose_panels(
        panels: list[np.ndarray], labels: list[str] | None = None,
    ) -> np.ndarray:
        """可変数のパネルを横に並べる汎用レイアウト。

        Parameters
        ----------
        panels : list of np.ndarray
            並べるパネル (すべて同じ高さを想定)。
        labels : list of str, optional
            各パネルのラベル。None の場合ラベルなし。

        Returns
        -------
        composed : np.ndarray
            合成後のフレーム。
        """
        if labels:
            return LayoutComposer._compose_labeled(panels, labels)
        return np.hstack(panels)

    @staticmethod
    def compose_grid(
        panels: list[np.ndarray],
        labels: list[str] | None = None,
        cols: int = 2,
    ) -> np.ndarray:
        """パネルをグリッド状に配置する。

        Parameters
        ----------
        panels : list of np.ndarray
            配置するパネル。
        labels : list of str, optional
            各パネルのラベル。
        cols : int
            列数 (default: 2)。

        Returns
        -------
        grid : np.ndarray
            グリッド合成後のフレーム。
        """
        if not panels:
            raise ValueError("panels must not be empty")

        # 全パネルを同一サイズにリサイズ
        target_h, target_w = panels[0].shape[:2]
        resized = []
        for p in panels:
            if p.shape[:2] != (target_h, target_w):
                p = cv2.resize(p, (target_w, target_h))
            resized.append(p)

        # ラベル描画
        if labels:
            label_h = max(int(target_h * 0.05), 18)
            labeled = []
            for panel, label in zip(resized, labels):
                p = panel.copy()
                font_scale = max(target_h * 0.0008, 0.35)
                thickness = max(int(target_h * 0.002), 1)
                text_size = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness,
                )[0]
                text_x = (target_w - text_size[0]) // 2
                text_y = target_h - label_h // 2 + text_size[1] // 2
                # 半透明の黒帯
                overlay = p.copy()
                cv2.rectangle(
                    overlay, (0, target_h - label_h), (target_w, target_h),
                    (0, 0, 0), cv2.FILLED,
                )
                cv2.addWeighted(overlay, 0.6, p, 0.4, 0, p)
                cv2.putText(
                    p, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA,
                )
                labeled.append(p)
            resized = labeled

        # グリッド構築
        rows_needed = (len(resized) + cols - 1) // cols

        # パネルが足りない場合は黒パネルで埋める
        while len(resized) < rows_needed * cols:
            resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

        grid_rows = []
        for r in range(rows_needed):
            row_panels = resized[r * cols:(r + 1) * cols]
            grid_rows.append(np.hstack(row_panels))

        return np.vstack(grid_rows)


def draw_predictions(
    frame: np.ndarray,
    predictions: list[tuple[int, str, float]],
    show_label: bool = True,
) -> np.ndarray:
    """フレームの左上に予測クラス名と確信度を描画する。

    Parameters
    ----------
    frame : np.ndarray
        描画対象のフレーム (BGR)。変更はインプレースで行う。
    predictions : list of (class_idx, label, confidence)
        上位予測リスト。
    show_label : bool
        False の場合、描画をスキップする。

    Returns
    -------
    frame : np.ndarray
        テキスト描画済みフレーム。
    """
    if not show_label or not predictions:
        return frame

    h, w = frame.shape[:2]
    font_scale = max(h * 0.0012, 0.157)
    thickness = max(int(h * 0.0017), 1)
    outline_thickness = thickness + max(int(h * 0.0014), 1)
    line_height = int(h * 0.0525)
    padding = int(h * 0.027)

    # 背景帯の領域を計算
    max_text_width = 0
    texts = []
    for rank, (idx, label, conf) in enumerate(predictions, 1):
        text = f"[{rank}] {label} ({conf * 100:.1f}%)"
        texts.append(text)
        font = cv2.FONT_HERSHEY_DUPLEX if rank == 1 else cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        max_text_width = max(max_text_width, text_size[0])

    bg_h = padding * 2 + line_height * len(predictions)
    bg_w = padding * 2 + max_text_width

    # 半透明の黒帯
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (bg_w, bg_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # テキスト描画（黒アウトライン + 白文字）
    for rank, text in enumerate(texts, 1):
        x = padding
        y = padding + line_height * rank - int(line_height * 0.25)
        font = cv2.FONT_HERSHEY_DUPLEX if rank == 1 else cv2.FONT_HERSHEY_SIMPLEX
        # アウトライン
        cv2.putText(
            frame, text, (x, y),
            font, font_scale, (0, 0, 0), outline_thickness, cv2.LINE_AA,
        )
        # 白文字
        cv2.putText(
            frame, text, (x, y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return frame


def draw_info_text(
    frame: np.ndarray,
    text: str,
    position: str = "top-left",
) -> np.ndarray:
    """フレームに情報テキストを描画する（汎用）。

    Parameters
    ----------
    frame : np.ndarray
        描画対象の BGR フレーム。
    text : str
        描画するテキスト。
    position : str
        テキスト位置 ('top-left', 'top-right', 'bottom-left', 'bottom-right')。

    Returns
    -------
    frame : np.ndarray
        描画済みフレーム。
    """
    h, w = frame.shape[:2]
    font_scale = max(h * 0.001, 0.35)
    thickness = max(int(h * 0.002), 1)
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness,
    )
    pad = 8

    if position == "top-right":
        x, y = w - tw - pad, th + pad
    elif position == "bottom-left":
        x, y = pad, h - pad
    elif position == "bottom-right":
        x, y = w - tw - pad, h - pad
    else:  # top-left
        x, y = pad, th + pad

    # アウトライン + 白文字
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    return frame

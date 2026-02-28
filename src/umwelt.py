"""AI の環世界（Umwelt）可視化システム — エントリーポイント

ResNet50 + Grad-CAM / YOLOv8 物体検出 / DeepLabV3 セグメンテーション /
MiDaS 深度推定 の各ビジョンモードを動画にオーバーレイし、
AIが「何を見ているか」を可視化する。
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Windows での Unicode 出力を有効化
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
from tqdm import tqdm

from renderer import LayoutComposer, draw_predictions, draw_info_text
from video_io import VideoReader, VideoWriter


# ═══════════════════════════════════════════════════════════════════════
#  引数パーサー
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Vision Visualizer — 環世界の可視化",
    )

    # ── 共通引数 ──
    parser.add_argument("input", help="入力動画ファイルのパス")
    parser.add_argument("-o", "--output", default=None, help="出力ファイルパス")
    parser.add_argument(
        "--mode", default="gradcam",
        choices=["gradcam", "detect", "segment", "depth", "all"],
        help="可視化モード (default: gradcam)",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推論デバイス (default: auto)",
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="音声を含めない",
    )

    # ── Grad-CAM 固有引数 ──
    gradcam_group = parser.add_argument_group("Grad-CAM options")
    gradcam_group.add_argument(
        "--target-class", type=int, default=None,
        help="Grad-CAM の対象クラス ID (0〜999)",
    )
    gradcam_group.add_argument(
        "--alpha", type=float, default=0.5,
        help="ヒートマップの透明度 (0.0〜1.0, default: 0.5)",
    )
    gradcam_group.add_argument(
        "--colormap", default="jet",
        help="ヒートマップのカラーマップ (jet, hot, inferno, turbo 等, default: jet)",
    )
    gradcam_group.add_argument(
        "--show-label", type=bool, default=True,
        help="予測クラス名と確信度の表示 (default: True)",
    )
    gradcam_group.add_argument(
        "--top-k", type=int, default=3,
        help="表示する上位予測クラス数 (default: 3)",
    )
    gradcam_group.add_argument(
        "--layout", default="overlay",
        choices=["overlay", "sidebyside", "triple"],
        help="表示レイアウト (default: overlay)",
    )

    # ── 物体検出固有引数 ──
    detect_group = parser.add_argument_group("Detection options")
    detect_group.add_argument(
        "--bbox-style", default="default",
        choices=["default", "corners", "cyber", "minimal", "hud"],
        help="BBox描画スタイル (default: default)",
    )
    detect_group.add_argument(
        "--yolo-model", default="yolov8n.pt",
        help="YOLOv8 モデル名 (default: yolov8n.pt)",
    )
    detect_group.add_argument(
        "--conf-threshold", type=float, default=0.25,
        help="検出確信度の閾値 (default: 0.25)",
    )

    # ── セグメンテーション固有引数 ──
    segment_group = parser.add_argument_group("Segmentation options")
    segment_group.add_argument(
        "--glitch-style", default="mixed",
        choices=["rgb_shift", "pixel_sort", "scanline", "displacement", "mixed"],
        help="グリッチエフェクトスタイル (default: mixed)",
    )
    segment_group.add_argument(
        "--glitch-intensity", type=float, default=0.5,
        help="グリッチ強度 (0.0〜1.0, default: 0.5)",
    )
    segment_group.add_argument(
        "--seg-alpha", type=float, default=0.5,
        help="セグメンテーションマスクの透明度 (default: 0.5)",
    )

    # ── 深度推定固有引数 ──
    depth_group = parser.add_argument_group("Depth estimation options")
    depth_group.add_argument(
        "--depth-style", default="colormap",
        choices=["colormap", "fog", "contour", "3d"],
        help="深度表示スタイル (default: colormap)",
    )
    depth_group.add_argument(
        "--depth-model", default="midas",
        choices=["midas", "depth_anything"],
        help="深度推定モデル (default: midas)",
    )

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  ヘルパー関数
# ═══════════════════════════════════════════════════════════════════════

def resolve_device(device_arg: str) -> torch.device:
    """デバイス引数を解決する。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA が利用不可能です。CPU にフォールバックします。")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_output_path(input_path: str, output_arg: str | None, mode: str) -> str:
    """出力パスを決定する。"""
    if output_arg:
        return output_arg
    base, _ext = os.path.splitext(input_path)
    return f"{base}_{mode}.mp4"


def merge_audio(video_path: str, original_path: str, output_path: str) -> bool:
    """ffmpeg で元動画の音声を出力動画にマージする。"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    try:
        cmd = [
            ffmpeg, "-y",
            "-i", video_path,
            "-i", original_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def compute_output_size(
    reader: VideoReader, mode: str, layout: str,
) -> tuple[int, int]:
    """モードとレイアウトに応じた出力フレームサイズを返す。"""
    w, h = reader.width, reader.height
    if mode == "all":
        return (w * 2, h * 2)
    if mode == "gradcam":
        if layout == "sidebyside":
            return (w * 2, h)
        elif layout == "triple":
            return (w * 3, h)
    return (w, h)


def setup_writer(
    args, reader: VideoReader, output_path: str, out_w: int, out_h: int,
) -> tuple["VideoWriter", str | None, bool]:
    """VideoWriter と音声マージ用の一時ファイルをセットアップする。"""
    use_audio = not args.no_audio and shutil.which("ffmpeg") is not None
    if use_audio:
        tmp_fd, tmp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        writer = VideoWriter(tmp_video_path, reader.fps, out_w, out_h)
        return writer, tmp_video_path, True
    else:
        if not args.no_audio and shutil.which("ffmpeg") is None:
            print("  WARNING: ffmpeg が見つかりません。音声なしで出力します。")
        writer = VideoWriter(output_path, reader.fps, out_w, out_h)
        return writer, None, False


def finalize_output(
    use_audio: bool, tmp_video_path: str | None,
    input_path: str, output_path: str,
):
    """音声マージと一時ファイルの後処理を行う。"""
    if use_audio and tmp_video_path:
        print("Merging audio...", end=" ", flush=True)
        success = merge_audio(tmp_video_path, input_path, output_path)
        if success:
            os.remove(tmp_video_path)
            print("done.")
        else:
            os.replace(tmp_video_path, output_path)
            print("failed (output without audio).")


def print_banner(mode: str, model_info: str, device, reader, output_path: str, args):
    """コンソールバナーを表示する。"""
    print()
    print("\U0001f310 AI Vision Visualizer")
    print("\u2501" * 42)
    print(f"  Mode     : {mode}")
    print(f"  Model    : {model_info}")
    print(f"  Device   : {device}")
    print(f"  Input    : {args.input} ({reader.width}x{reader.height}, "
          f"{reader.fps:.0f}fps, {reader.frame_count} frames)")
    print(f"  Output   : {output_path}")
    if mode == "gradcam":
        print(f"  Layout   : {args.layout}")
        print(f"  Alpha    : {args.alpha}")
        print(f"  Colormap : {args.colormap}")
    elif mode == "detect":
        print(f"  BBox     : {args.bbox_style}")
        print(f"  Conf     : {args.conf_threshold}")
    elif mode == "segment":
        print(f"  Glitch   : {args.glitch_style} (intensity={args.glitch_intensity})")
    elif mode == "depth":
        print(f"  Style    : {args.depth_style}")
    print("\u2501" * 42)
    print()


def print_done(output_path: str, total_frames: int, elapsed: float):
    """完了メッセージを表示する。"""
    fps_speed = total_frames / elapsed if elapsed > 0 else 0
    print()
    print(f"\u2705 Done! Output saved to: {output_path}")
    print(f"   Processing speed: {fps_speed:.2f} fps")
    print(f"   Total time: {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════════════
#  モード別処理関数
# ═══════════════════════════════════════════════════════════════════════

def process_gradcam(args, device, reader, writer):
    """Grad-CAM モードのフレーム処理ループ。"""
    from gradcam import GradCAM
    from renderer import HeatmapRenderer

    print("Loading model...", end=" ", flush=True)
    cam_engine = GradCAM(device)
    print("done.")

    heatmap_renderer = HeatmapRenderer(alpha=args.alpha, colormap=args.colormap)
    layout_composer = LayoutComposer(layout=args.layout)

    with tqdm(total=reader.frame_count, desc="Processing frames",
              unit="frame", dynamic_ncols=True) as pbar:
        for frame in reader:
            input_tensor = cam_engine.preprocess(frame)
            heatmap, class_idx, conf, top_k_preds = cam_engine.generate(
                input_tensor, target_class=args.target_class, top_k=args.top_k,
            )
            heatmap_colored = heatmap_renderer.colorize(
                heatmap, (reader.width, reader.height),
            )
            overlay_frame = heatmap_renderer.overlay(frame, heatmap_colored)
            composed = layout_composer.compose(frame, heatmap_colored, overlay_frame)
            composed = draw_predictions(composed, top_k_preds, show_label=args.show_label)
            writer.write(composed)
            pbar.update(1)


def process_detect(args, device, reader, writer):
    """物体検出モードのフレーム処理ループ。"""
    from detector import YOLODetector
    from effects.bbox_styles import draw_detections
    from utils.colors import get_class_color

    print("Loading model...", end=" ", flush=True)
    device_str = "cpu" if device.type == "cpu" else str(device)
    detector = YOLODetector(
        device=device_str,
        model_name=args.yolo_model,
        conf_threshold=args.conf_threshold,
    )
    print("done.")

    with tqdm(total=reader.frame_count, desc="Processing frames",
              unit="frame", dynamic_ncols=True) as pbar:
        for frame in reader:
            detections = detector.detect(frame)
            # 色を付与
            for det in detections:
                det["color"] = get_class_color(det["class_id"])
            result = draw_detections(frame, detections, style=args.bbox_style)
            # 検出数の表示
            result = draw_info_text(
                result, f"Detected: {len(detections)} objects", position="top-left",
            )
            writer.write(result)
            pbar.update(1)


def process_segment(args, device, reader, writer):
    """セグメンテーションモードのフレーム処理ループ。"""
    from segmentor import SemanticSegmentor
    from effects.glitch import apply_glitch

    print("Loading model...", end=" ", flush=True)
    segmentor = SemanticSegmentor(device)
    print("done.")

    with tqdm(total=reader.frame_count, desc="Processing frames",
              unit="frame", dynamic_ncols=True) as pbar:
        for frame in reader:
            mask, colored_mask = segmentor.segment(frame)
            # セグメンテーションオーバーレイ
            overlay = segmentor.overlay(frame, colored_mask, alpha=args.seg_alpha)
            # グリッチエフェクト
            result = apply_glitch(
                overlay, mask,
                style=args.glitch_style,
                intensity=args.glitch_intensity,
            )
            writer.write(result)
            pbar.update(1)


def process_depth(args, device, reader, writer):
    """深度推定モードのフレーム処理ループ。"""
    from depth import DepthEstimator
    from effects.depth_styles import apply_depth_style

    print("Loading model...", end=" ", flush=True)
    estimator = DepthEstimator(device, model_type=args.depth_model)
    print("done.")

    with tqdm(total=reader.frame_count, desc="Processing frames",
              unit="frame", dynamic_ncols=True) as pbar:
        for frame in reader:
            depth_map = estimator.estimate(frame)
            result = apply_depth_style(frame, depth_map, style=args.depth_style)
            writer.write(result)
            pbar.update(1)


def process_all(args, device, reader, writer):
    """全モード同時表示 (2x2 グリッド) のフレーム処理ループ。"""
    from gradcam import GradCAM
    from renderer import HeatmapRenderer
    from effects.bbox_styles import draw_detections
    from effects.glitch import apply_glitch
    from effects.depth_styles import apply_depth_style
    from utils.colors import get_class_color

    # ── モデルロード（OOM 時は逐次フォールバック） ──
    print("Loading models...", flush=True)
    engines = {}

    print("  [1/4] Grad-CAM...", end=" ", flush=True)
    cam_engine = GradCAM(device)
    heatmap_renderer = HeatmapRenderer(alpha=args.alpha, colormap=args.colormap)
    engines["gradcam"] = True
    print("done.")

    try:
        print("  [2/4] YOLO...", end=" ", flush=True)
        from detector import YOLODetector
        device_str = "cpu" if device.type == "cpu" else str(device)
        detector = YOLODetector(device=device_str)
        engines["detect"] = True
        print("done.")
    except (ImportError, Exception) as e:
        print(f"skipped ({e})")
        detector = None

    try:
        print("  [3/4] DeepLabV3...", end=" ", flush=True)
        from segmentor import SemanticSegmentor
        segmentor = SemanticSegmentor(device)
        engines["segment"] = True
        print("done.")
    except (ImportError, Exception) as e:
        print(f"skipped ({e})")
        segmentor = None

    try:
        print("  [4/4] MiDaS...", end=" ", flush=True)
        from depth import DepthEstimator
        estimator = DepthEstimator(device)
        engines["depth"] = True
        print("done.")
    except (ImportError, Exception) as e:
        print(f"skipped ({e})")
        estimator = None

    loaded = sum(1 for v in engines.values() if v)
    print(f"  {loaded}/4 models loaded.")

    with tqdm(total=reader.frame_count, desc="Processing frames",
              unit="frame", dynamic_ncols=True) as pbar:
        for frame in reader:
            panels = []
            labels = []
            w, h = reader.width, reader.height

            # ── Panel 1: Grad-CAM ──
            input_tensor = cam_engine.preprocess(frame)
            heatmap, _, _, top_k_preds = cam_engine.generate(
                input_tensor, top_k=args.top_k,
            )
            heatmap_colored = heatmap_renderer.colorize(heatmap, (w, h))
            gradcam_frame = heatmap_renderer.overlay(frame, heatmap_colored)
            gradcam_frame = draw_predictions(
                gradcam_frame, top_k_preds, show_label=True,
            )
            panels.append(gradcam_frame)
            labels.append("Grad-CAM")

            # ── Panel 2: Detection ──
            if detector:
                det_frame = frame.copy()
                detections = detector.detect(frame)
                for det in detections:
                    det["color"] = get_class_color(det["class_id"])
                det_frame = draw_detections(det_frame, detections)
                panels.append(det_frame)
            else:
                panels.append(frame.copy())
            labels.append("Detection")

            # ── Panel 3: Segmentation ──
            if segmentor:
                mask, colored_mask = segmentor.segment(frame)
                seg_frame = segmentor.overlay(frame, colored_mask)
                seg_frame = apply_glitch(seg_frame, mask, intensity=0.3)
                panels.append(seg_frame)
            else:
                panels.append(frame.copy())
            labels.append("Segmentation")

            # ── Panel 4: Depth ──
            if estimator:
                depth_map = estimator.estimate(frame)
                depth_frame = apply_depth_style(frame, depth_map, style="colormap")
                panels.append(depth_frame)
            else:
                panels.append(frame.copy())
            labels.append("Depth")

            # ── 2x2 グリッド合成 ──
            composed = LayoutComposer.compose_grid(panels, labels, cols=2)
            writer.write(composed)
            pbar.update(1)


# ═══════════════════════════════════════════════════════════════════════
#  メインエントリーポイント
# ═══════════════════════════════════════════════════════════════════════

# モード → (処理関数, モデル情報文字列)
MODE_DISPATCH = {
    "gradcam": (process_gradcam, "ResNet50 (Grad-CAM)"),
    "detect": (process_detect, "YOLOv8"),
    "segment": (process_segment, "DeepLabV3 ResNet101"),
    "depth": (process_depth, "MiDaS DPT-Large"),
    "all": (process_all, "All (Grad-CAM + YOLO + DeepLab + MiDaS)"),
}


def main():
    args = parse_args()

    # --- 入力検証 ---
    if not os.path.isfile(args.input):
        print(f"ERROR: 入力ファイルが見つかりません: {args.input}")
        sys.exit(1)

    # --- デバイス・出力パス ---
    device = resolve_device(args.device)
    output_path = resolve_output_path(args.input, args.output, args.mode)

    # 出力先ディレクトリの作成
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # --- 動画読み込み ---
    try:
        reader = VideoReader(args.input)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # --- モード解決 ---
    process_fn, model_info = MODE_DISPATCH[args.mode]
    out_w, out_h = compute_output_size(reader, args.mode, args.layout)

    # --- バナー表示 ---
    print_banner(args.mode, model_info, device, reader, output_path, args)

    # --- Writer セットアップ ---
    total_frames = reader.frame_count
    writer, tmp_video_path, use_audio = setup_writer(
        args, reader, output_path, out_w, out_h,
    )

    # --- フレーム処理 ---
    start_time = time.time()
    process_fn(args, device, reader, writer)
    writer.release()
    elapsed = time.time() - start_time

    # --- 音声マージ・完了表示 ---
    finalize_output(use_audio, tmp_video_path, args.input, output_path)
    print_done(output_path, total_frames, elapsed)


if __name__ == "__main__":
    main()

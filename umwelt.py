"""AI の環世界（Umwelt）可視化システム — エントリーポイント

ResNet50 + Grad-CAM によるヒートマップを動画の各フレームにオーバーレイし、
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

from gradcam import GradCAM
from renderer import HeatmapRenderer, LayoutComposer, draw_predictions
from video_io import VideoReader, VideoWriter


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Vision Visualizer — Grad-CAM ヒートマップによる環世界の可視化",
    )
    parser.add_argument("input", help="入力動画ファイルのパス")
    parser.add_argument("-o", "--output", default=None, help="出力ファイルパス")
    parser.add_argument(
        "--target-class", type=int, default=None,
        help="Grad-CAM の対象クラス ID (0〜999)。未指定時は最大確信度クラス",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="ヒートマップの透明度 (0.0〜1.0, default: 0.5)",
    )
    parser.add_argument(
        "--colormap", default="jet",
        help="ヒートマップのカラーマップ (jet, hot, inferno, turbo 等, default: jet)",
    )
    parser.add_argument(
        "--show-label", type=bool, default=True,
        help="予測クラス名と確信度の表示 (default: True)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="表示する上位予測クラス数 (default: 3)",
    )
    parser.add_argument(
        "--layout", default="overlay",
        choices=["overlay", "sidebyside", "triple"],
        help="表示レイアウト (default: overlay)",
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
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """デバイス引数を解決する。"""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA が利用不可能です。CPU にフォールバックします。")
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_output_path(input_path: str, output_arg: str | None) -> str:
    """出力パスを決定する。"""
    if output_arg:
        return output_arg
    base, ext = os.path.splitext(input_path)
    return f"{base}_umwelt.mp4"


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


def compute_output_size(reader: VideoReader, layout: str) -> tuple[int, int]:
    """レイアウトに応じた出力フレームサイズを返す。"""
    w, h = reader.width, reader.height
    if layout == "sidebyside":
        return (w * 2, h)
    elif layout == "triple":
        return (w * 3, h)
    return (w, h)


def main():
    args = parse_args()

    # --- 入力検証 ---
    if not os.path.isfile(args.input):
        print(f"ERROR: 入力ファイルが見つかりません: {args.input}")
        sys.exit(1)

    # --- デバイス・出力パス ---
    device = resolve_device(args.device)
    output_path = resolve_output_path(args.input, args.output)

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

    # --- コンソール表示 ---
    print()
    print("\U0001f310 AI Vision Visualizer")
    print("\u2501" * 42)
    print(f"  Model    : ResNet50 (ImageNet pretrained)")
    print(f"  Target   : layer4 (Grad-CAM)")
    print(f"  Device   : {device}")
    print(f"  Input    : {args.input} ({reader.width}x{reader.height}, "
          f"{reader.fps:.0f}fps, {reader.frame_count} frames)")
    print(f"  Output   : {output_path}")
    print(f"  Layout   : {args.layout}")
    print(f"  Alpha    : {args.alpha}")
    print(f"  Colormap : {args.colormap}")
    print("\u2501" * 42)
    print()

    # --- モデル・レンダラー初期化 ---
    print("Loading model...", end=" ", flush=True)
    cam_engine = GradCAM(device)
    print("done.")

    heatmap_renderer = HeatmapRenderer(alpha=args.alpha, colormap=args.colormap)
    layout_composer = LayoutComposer(layout=args.layout)
    out_w, out_h = compute_output_size(reader, args.layout)

    # --- 音声マージ用の一時ファイル判定 ---
    use_audio = not args.no_audio and shutil.which("ffmpeg") is not None
    if use_audio:
        tmp_fd, tmp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        writer = VideoWriter(tmp_video_path, reader.fps, out_w, out_h)
    else:
        if not args.no_audio and shutil.which("ffmpeg") is None:
            print("  WARNING: ffmpeg が見つかりません。音声なしで出力します。")
        writer = VideoWriter(output_path, reader.fps, out_w, out_h)

    # --- フレーム処理 ---
    total_frames = reader.frame_count
    start_time = time.time()

    with tqdm(total=total_frames, desc="Processing frames",
              unit="frame", dynamic_ncols=True) as pbar:
        for frame in reader:
            # 前処理
            input_tensor = cam_engine.preprocess(frame)

            # Grad-CAM ヒートマップ生成
            heatmap, class_idx, conf, top_k_preds = cam_engine.generate(
                input_tensor,
                target_class=args.target_class,
                top_k=args.top_k,
            )

            # ヒートマップのカラーマップ変換
            heatmap_colored = heatmap_renderer.colorize(
                heatmap, (reader.width, reader.height),
            )

            # オーバーレイ合成
            overlay_frame = heatmap_renderer.overlay(frame, heatmap_colored)

            # レイアウト合成
            composed = layout_composer.compose(frame, heatmap_colored, overlay_frame)

            # テキスト描画
            composed = draw_predictions(composed, top_k_preds, show_label=args.show_label)

            # 書き出し
            writer.write(composed)
            pbar.update(1)

    writer.release()
    elapsed = time.time() - start_time
    fps_speed = total_frames / elapsed if elapsed > 0 else 0

    # --- 音声マージ ---
    if use_audio:
        print("Merging audio...", end=" ", flush=True)
        success = merge_audio(tmp_video_path, args.input, output_path)
        if success:
            os.remove(tmp_video_path)
            print("done.")
        else:
            # マージ失敗時は音声なし動画をそのまま使用
            os.replace(tmp_video_path, output_path)
            print("failed (output without audio).")

    # --- 完了表示 ---
    print()
    print(f"\u2705 Done! Output saved to: {output_path}")
    print(f"   Processing speed: {fps_speed:.2f} fps")
    print(f"   Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

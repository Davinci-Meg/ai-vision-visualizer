# AIの環世界（Umwelt）可視化システム — 仕様書

## 1. プロジェクト概要

### 1.1 コンセプト

生物学者ヤーコプ・フォン・ユクスキュルが提唱した「環世界（Umwelt）」の概念をAIに適用する。
環世界とは、同じ環境にいても生物種ごとに知覚・認識する世界が異なるという考え方である。

本システムは、**画像分類CNN（ResNet50）が動画の各フレームで「何を見ているか」**を
Grad-CAMヒートマップとしてリアルタイムにオーバーレイし、動画として出力する。
これにより、人間が見ている映像とAIが見ている映像の違い＝**AIの環世界**を可視化する。

### 1.2 ユースケース

- アート・インスタレーション向けの映像素材生成
- CNN の判断根拠の教育的デモンストレーション
- モデルの注目領域分析ツール

---

## 2. システム要件

### 2.1 動作環境

- Python 3.9 以上
- GPU 推奨（CUDA対応）、CPU でも動作可能
- OS: Linux / macOS / Windows

### 2.2 依存ライブラリ

| ライブラリ | バージョン | 用途 |
|-----------|----------|------|
| torch | >= 2.0 | 深層学習フレームワーク |
| torchvision | >= 0.15 | ResNet50 事前学習モデル |
| opencv-python | >= 4.8 | 動画の読み込み・書き出し・フレーム処理 |
| numpy | >= 1.24 | 数値計算 |
| tqdm | >= 4.65 | 進捗表示 |

### 2.3 インストール

```bash
pip install torch torchvision opencv-python numpy tqdm
```

---

## 3. 機能仕様

### 3.1 入力

| 項目 | 仕様 |
|------|------|
| 入力形式 | MP4, AVI, MOV 等（OpenCV が対応する形式） |
| 動画長 | 制限なし（短い動画を推奨、目安: 〜60秒） |
| 解像度 | 任意（内部で 224×224 にリサイズして推論、出力は元解像度を維持） |

### 3.2 出力

| 項目 | 仕様 |
|------|------|
| 出力形式 | MP4（H.264コーデック） |
| 出力ファイル名 | `{入力ファイル名}_umwelt.mp4`（デフォルト、`-o` で変更可） |
| 出力解像度 | 入力と同一 |
| フレームレート | 入力と同一 |

### 3.3 コマンドラインインターフェース

```bash
python umwelt.py input.mp4 [オプション]
```

#### 必須引数

| 引数 | 説明 |
|------|------|
| `input` | 入力動画ファイルのパス |

#### オプション引数

| 引数 | デフォルト | 説明 |
|------|----------|------|
| `-o, --output` | `{input}_umwelt.mp4` | 出力ファイルパス |
| `--target-class` | `None`（最大確信度クラス） | Grad-CAM の対象クラスID（ImageNet の 0〜999） |
| `--alpha` | `0.5` | ヒートマップの透明度（0.0〜1.0） |
| `--colormap` | `jet` | ヒートマップのカラーマップ（`jet`, `hot`, `inferno`, `turbo` 等） |
| `--show-label` | `True` | フレーム上に予測クラス名と確信度を表示 |
| `--top-k` | `3` | 表示する予測クラス数（上位K個） |
| `--layout` | `overlay` | 表示レイアウト（下記参照） |
| `--device` | `auto` | 推論デバイス（`auto`, `cpu`, `cuda`） |
| `--no-audio` | `False` | 音声を含めない |

#### レイアウトモード（`--layout`）

| モード | 説明 |
|--------|------|
| `overlay` | 元映像にヒートマップを半透明オーバーレイ（デフォルト） |
| `sidebyside` | 左: 元映像、右: ヒートマップオーバーレイを横並び表示 |
| `triple` | 左: 元映像、中: ヒートマップ単体、右: オーバーレイ の3列表示 |

---

## 4. アーキテクチャ

### 4.1 全体構成

```
入力動画 (MP4)
    │
    ▼
┌─────────────────┐
│  VideoReader     │  OpenCV で動画をフレーム単位で読み込み
└────────┬────────┘
         │ フレーム (BGR, H×W×3)
         ▼
┌─────────────────┐
│  Preprocessor    │  BGR→RGB変換、224×224リサイズ、正規化
└────────┬────────┘
         │ テンソル (1×3×224×224)
         ▼
┌─────────────────┐
│  ResNet50        │  ImageNet 事前学習済みモデル
│  + GradCAM Hook  │  layer4 の activation と gradient を hook で取得
└────────┬────────┘
         │ ヒートマップ (224×224) + 予測クラス
         ▼
┌─────────────────┐
│  HeatmapRenderer │  ヒートマップを元解像度にリサイズし、
│                  │  元フレームに合成。クラス名・確信度をテキスト描画
└────────┬────────┘
         │ 合成フレーム (BGR, H×W×3)
         ▼
┌─────────────────┐
│  VideoWriter     │  OpenCV で MP4 として書き出し
└─────────────────┘
```

### 4.2 モジュール構成

```
umwelt.py                  # エントリーポイント（CLI）
├── gradcam.py             # Grad-CAM エンジン
│   ├── class GradCAM      #   hook 登録・ヒートマップ生成
│   └── get_imagenet_labels()  # ImageNet ラベル辞書
├── renderer.py            # 映像合成・テキスト描画
│   ├── class HeatmapRenderer  # ヒートマップ合成
│   └── class LayoutComposer   # レイアウト制御
└── video_io.py            # 動画入出力
    ├── class VideoReader   # フレーム読み込みイテレータ
    └── class VideoWriter   # フレーム書き出し
```

> **注意**: 上記は推奨構成であるが、単一ファイル `umwelt.py` での実装も許容する。
> その場合もクラス・関数の論理的な分離は維持すること。

---

## 5. 詳細設計

### 5.1 Grad-CAM エンジン（`GradCAM` クラス）

#### 対象層

ResNet50 の **`layer4`**（最終畳み込みブロック）を対象とする。
この層は 7×7 の特徴マップを持ち、画像全体のセマンティックな情報を含む。

#### 処理フロー

```python
class GradCAM:
    def __init__(self, model, target_layer):
        # 1. model を eval モードに設定
        # 2. target_layer に forward hook と backward hook を登録
        #    - forward hook: activation（特徴マップ）を保存
        #    - backward hook: gradient を保存

    def generate(self, input_tensor, target_class=None):
        # 1. 順伝播: output = model(input_tensor)
        # 2. target_class が None の場合、argmax で最大確信度クラスを取得
        # 3. 対象クラスのスコアに対して backward() を実行
        # 4. 保存された gradient のチャンネル方向平均 → 重み α
        # 5. α × activation をチャンネル方向に加算
        # 6. ReLU 適用（負の寄与を除去）
        # 7. [0, 1] に正規化
        # 8. 224×224 にリサイズ（元の入力サイズに合わせる）
        # 返り値: (heatmap: np.ndarray, class_idx: int, confidence: float, top_k_predictions: list)
```

#### 重要な実装ポイント

- **`model.eval()`** を必ず設定（Dropout/BN の挙動が変わるため）
- **`torch.no_grad()` は使わない**（Grad-CAM は勾配計算が必要）
- **`model.zero_grad()`** を各フレームの処理前に呼ぶ（勾配の蓄積を防ぐ）
- **backward 後に hook を解除しない**（動画の全フレームで再利用するため）
- `retain_graph=True` は不要（各フレームで新しい計算グラフを構築するため）

### 5.2 前処理（Preprocessing）

ImageNet の事前学習モデルに合わせた標準的な前処理を行う。

```python
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 5.3 ヒートマップレンダリング

#### カラーマップ変換

```python
# 1. ヒートマップ (0.0〜1.0, float) → (0〜255, uint8) に変換
# 2. OpenCV の applyColorMap で着色（cv2.COLORMAP_JET 等）
# 3. 元フレームの解像度にリサイズ
# 4. 元フレームとアルファブレンド: output = α * heatmap + (1-α) * original
```

#### テキスト描画

各フレームの左上に以下の情報を表示する:

```
[1] golden retriever (87.3%)
[2] Labrador retriever (5.2%)
[3] cocker spaniel (2.1%)
```

- フォント: `cv2.FONT_HERSHEY_SIMPLEX`
- 文字サイズ: フレーム高さに応じて動的に調整（目安: 高さの 2.5%）
- 背景: 半透明の黒帯を文字の後ろに描画（可読性のため）
- 最も確信度の高いクラスは太字（`cv2.FONT_HERSHEY_DUPLEX`）
- 色: 白（`(255, 255, 255)`）

### 5.4 動画入出力

#### VideoReader

```python
class VideoReader:
    # cv2.VideoCapture でフレームを順次読み込むイテレータ
    # プロパティ: fps, width, height, frame_count, duration
    # __iter__ / __next__ でフレームを yield
```

#### VideoWriter

```python
class VideoWriter:
    # cv2.VideoWriter で MP4 (H.264) を書き出す
    # codec: cv2.VideoWriter_fourcc(*'mp4v')
    # fps, width, height は入力動画に合わせる
```

#### 音声の保持

デフォルトでは `ffmpeg` を用いて元動画の音声トラックを出力にコピーする。
`ffmpeg` が利用できない場合や `--no-audio` 指定時は音声なしで出力する。

```bash
ffmpeg -i heatmap_video.mp4 -i original.mp4 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 output.mp4
```

---

## 6. 表示レイアウト詳細

### 6.1 `overlay`（デフォルト）

```
┌──────────────────────┐
│  [1] dog (92.3%)     │
│  [2] cat (3.1%)      │
│                      │
│   元映像 + ヒートマップ │
│                      │
└──────────────────────┘
出力解像度: 入力と同一 (W × H)
```

### 6.2 `sidebyside`

```
┌────────────┬────────────┐
│            │            │
│  元映像     │ オーバーレイ │
│            │            │
└────────────┴────────────┘
出力解像度: (2W × H)
```

### 6.3 `triple`

```
┌────────────┬────────────┬────────────┐
│            │            │            │
│  元映像     │ ヒートマップ │ オーバーレイ │
│  (Human)   │ (AI Raw)   │ (AI Umwelt)│
│            │            │            │
└────────────┴────────────┴────────────┘
出力解像度: (3W × H)
各パネル下部にラベルを表示
```

---

## 7. 出力例（想定）

### CLI 実行例

```bash
# 基本的な使い方
python umwelt.py cat_video.mp4

# カスタマイズ例
python umwelt.py street.mp4 -o street_ai_vision.mp4 \
    --layout triple \
    --alpha 0.6 \
    --colormap turbo \
    --top-k 5

# 特定クラスへの注目を可視化（例: ImageNet の "cat" = 281）
python umwelt.py pet.mp4 --target-class 281 --layout sidebyside

# CPU で実行
python umwelt.py input.mp4 --device cpu
```

### コンソール出力（想定）

```
🌐 AI Umwelt Visualizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Model    : ResNet50 (ImageNet pretrained)
  Target   : layer4 (Grad-CAM)
  Device   : cuda
  Input    : cat_video.mp4 (1920x1080, 30fps, 450 frames)
  Output   : cat_video_umwelt.mp4
  Layout   : overlay
  Alpha    : 0.5
  Colormap : jet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Processing frames: 100%|████████████████████| 450/450 [00:23<00:00, 19.57 fps]

✅ Done! Output saved to: cat_video_umwelt.mp4
   Processing speed: 19.57 fps
   Total time: 23.0s
```

---

## 8. エラーハンドリング

| エラー条件 | 対応 |
|-----------|------|
| 入力ファイルが存在しない | エラーメッセージを表示して終了 |
| 入力ファイルが動画でない（OpenCV で開けない） | エラーメッセージを表示して終了 |
| CUDA が利用不可能で `--device cuda` 指定 | 警告を出して CPU にフォールバック |
| 出力先ディレクトリが存在しない | 自動で作成 |
| `ffmpeg` が利用不可能 | 警告を出して音声なしで出力 |
| 処理中のメモリ不足 | フレームを逐次処理しているため基本的に発生しないが、例外発生時はエラーメッセージを表示 |

---

## 9. パフォーマンス目安

| 環境 | 期待処理速度 |
|------|-----------|
| GPU（RTX 3060 相当） | 約 20〜30 fps |
| GPU（RTX 4090 相当） | 約 50〜80 fps |
| CPU（Core i7 相当） | 約 2〜5 fps |

> 処理速度は入力解像度にほぼ依存しない（推論は 224×224 固定のため）。
> ボトルネックは ResNet50 の推論 + backward 計算。

---

## 10. 将来の拡張案（本バージョンでは未実装）

以下は今回の実装スコープ外だが、将来の拡張として検討可能な機能:

- **モデル切り替え**: ResNet50 以外（VGG16, EfficientNet, ViT 等）への対応
- **複数モデル比較**: 異なるモデルのヒートマップを並べて表示
- **時系列分析**: フレーム間の注目領域の変化をグラフ化
- **インタラクティブモード**: GUI で動画を再生しながらリアルタイムにヒートマップを確認
- **Webカメラ入力**: リアルタイムストリーミング対応
- **Score-CAM / Grad-CAM++ 切り替え**: 複数のCAM手法の比較
- **オブジェクト検出モデル対応**: YOLOv8 等の検出モデルでバウンディングボックス＋ヒートマップ

---

## 11. 参考文献

- Selvaraju, R.R. et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.
- von Uexküll, J. (1934). "Streifzüge durch die Umwelten von Tieren und Menschen."（動物の環世界と内的世界）
- He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016.

"""Grad-CAM エンジン — ResNet50 の layer4 を対象としたヒートマップ生成"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from utils.labels import get_imagenet_labels

# ImageNet の前処理
PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class GradCAM:
    """Grad-CAM によるヒートマップ生成クラス。

    Parameters
    ----------
    device : torch.device
        推論に使用するデバイス。
    """

    def __init__(self, device: torch.device):
        self.device = device

        # ResNet50 事前学習済みモデルのロード
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = self.model.to(device)
        self.model.eval()

        # hook 用の保存領域
        self._activations = None
        self._gradients = None

        # layer4 に forward / backward hook を登録
        target_layer = self.model.layer4
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

        # ImageNet ラベル
        self.labels = get_imagenet_labels()

    def _forward_hook(self, module, input, output):
        self._activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """BGR フレームを推論用テンソルに変換する。"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = PREPROCESS(frame_rgb)
        return tensor.unsqueeze(0).to(self.device)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        top_k: int = 3,
    ) -> tuple[np.ndarray, int, float, list[tuple[int, str, float]]]:
        """Grad-CAM ヒートマップを生成する。

        Parameters
        ----------
        input_tensor : torch.Tensor
            前処理済みの入力テンソル (1×3×224×224)。
        target_class : int or None
            ヒートマップ対象のクラス ID。None の場合、最大確信度クラスを使用。
        top_k : int
            返却する上位予測数。

        Returns
        -------
        heatmap : np.ndarray
            正規化されたヒートマップ (224×224, float32, 0.0〜1.0)。
        class_idx : int
            対象クラスの ID。
        confidence : float
            対象クラスの確信度 (0.0〜1.0)。
        top_k_predictions : list of (class_idx, label, confidence)
            上位 K 個の予測。
        """
        self.model.zero_grad()

        # 順伝播
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        # 上位 K 個の予測
        top_probs, top_indices = probs.topk(top_k, dim=1)
        top_k_predictions = []
        for i in range(top_k):
            idx = top_indices[0, i].item()
            prob = top_probs[0, i].item()
            label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
            top_k_predictions.append((idx, label, prob))

        # 対象クラスの決定
        if target_class is None:
            class_idx = top_indices[0, 0].item()
        else:
            class_idx = target_class
        confidence = probs[0, class_idx].item()

        # 逆伝播
        score = output[0, class_idx]
        score.backward()

        # Grad-CAM 計算
        gradients = self._gradients  # (1, C, H, W)
        activations = self._activations  # (1, C, H, W)

        # チャンネル方向の重み α = GAP(gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # 加重和
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU
        cam = F.relu(cam)

        # 224×224 にリサイズ
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)

        # [0, 1] に正規化
        cam = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, confidence, top_k_predictions

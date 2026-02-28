"""セマンティックセグメンテーションエンジン — DeepLabV3+ (ResNet101)"""

import cv2
import numpy as np
import torch
from torchvision import models, transforms

from utils.labels import VOC_LABELS
from utils.colors import VOC_COLORS


# DeepLabV3 の前処理
_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(520),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class SemanticSegmentor:
    """DeepLabV3 ResNet101 によるセマンティックセグメンテーション。

    Parameters
    ----------
    device : torch.device
        推論デバイス。
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.model = models.segmentation.deeplabv3_resnet101(
            weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT,
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.labels = VOC_LABELS
        self.colors = VOC_COLORS

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """BGR フレームを推論用テンソルに変換する。"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = _PREPROCESS(frame_rgb)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def segment(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """フレームのセマンティックセグメンテーションを実行する。

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR フレーム。

        Returns
        -------
        mask : np.ndarray
            クラスインデックスマスク (H×W, int)。元フレームサイズにリサイズ済み。
        colored_mask : np.ndarray
            クラス色で着色されたマスク (H×W×3, uint8, BGR)。
        """
        h, w = frame_bgr.shape[:2]
        input_tensor = self.preprocess(frame_bgr)

        output = self.model(input_tensor)["out"]
        mask = output.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

        # 元フレームサイズにリサイズ
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 色付きマスク生成
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id in range(len(self.colors)):
            colored_mask[mask == cls_id] = self.colors[cls_id]

        return mask, colored_mask

    def overlay(
        self, frame: np.ndarray, colored_mask: np.ndarray, alpha: float = 0.5,
    ) -> np.ndarray:
        """元フレームにセグメンテーションマスクをオーバーレイする。"""
        return cv2.addWeighted(colored_mask, alpha, frame, 1.0 - alpha, 0)

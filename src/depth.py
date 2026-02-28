"""深度推定エンジン — MiDaS / Depth Anything V2"""

import cv2
import numpy as np
import torch


class DepthEstimator:
    """MiDaS または Depth Anything V2 による単眼深度推定。

    Parameters
    ----------
    device : torch.device
        推論デバイス。
    model_type : str
        モデルタイプ ('midas' or 'depth_anything')。
    """

    def __init__(self, device: torch.device, model_type: str = "midas"):
        self.device = device
        self.model_type = model_type

        if model_type == "depth_anything":
            self._init_depth_anything()
        else:
            self._init_midas()

    def _init_midas(self):
        """MiDaS DPT-Large モデルを初期化する。"""
        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "DPT_Large", trust_repo=True,
            )
        except Exception:
            # フォールバック: 小型モデル
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True,
            )
        self.model = self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True,
        )
        self.transform = midas_transforms.dpt_transform

    def _init_depth_anything(self):
        """Depth Anything V2 モデルを初期化する。"""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "Depth Anything V2 には transformers が必要です。\n"
                "  pip install transformers"
            )

        self._pipeline = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=self.device,
        )
        self.model = None  # pipeline が内部で保持
        self.transform = None

    @torch.no_grad()
    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """フレームの深度を推定する。

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR フレーム。

        Returns
        -------
        depth : np.ndarray
            深度マップ (H×W, float32)。値が大きいほど近い。
        """
        if self.model_type == "depth_anything":
            return self._estimate_depth_anything(frame_bgr)
        return self._estimate_midas(frame_bgr)

    def _estimate_midas(self, frame_bgr: np.ndarray) -> np.ndarray:
        """MiDaS で深度推定。"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(frame_rgb).to(self.device)

        prediction = self.model(input_tensor)

        # 元フレームサイズにリサイズ
        h, w = frame_bgr.shape[:2]
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        return prediction.cpu().numpy()

    def _estimate_depth_anything(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Depth Anything V2 で深度推定。"""
        from PIL import Image

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        result = self._pipeline(pil_image)
        depth = np.array(result["depth"], dtype=np.float32)

        # 元フレームサイズにリサイズ
        h, w = frame_bgr.shape[:2]
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth

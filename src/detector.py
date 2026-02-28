"""物体検出エンジン — YOLOv8 による物体検出"""

import numpy as np


class YOLODetector:
    """YOLOv8 を用いた物体検出クラス。

    ultralytics は初期化時に遅延インポートする。

    Parameters
    ----------
    device : str
        推論デバイス ('cpu', 'cuda', etc.)。
    model_name : str
        YOLOv8 モデル名 (default: 'yolov8n.pt')。
    conf_threshold : float
        検出確信度の閾値。
    """

    def __init__(
        self,
        device: str = "cpu",
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "物体検出モードには ultralytics が必要です。\n"
                "  pip install ultralytics"
            )

        self.model = YOLO(model_name)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(
        self, frame_bgr: np.ndarray,
    ) -> list[dict]:
        """フレーム内の物体を検出する。

        Parameters
        ----------
        frame_bgr : np.ndarray
            BGR フレーム。

        Returns
        -------
        detections : list of dict
            各検出は {'bbox': (x1,y1,x2,y2), 'class_id': int,
            'label': str, 'conf': float}。
        """
        results = self.model(frame_bgr, verbose=False, conf=self.conf_threshold)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                label = result.names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "bbox": tuple(xyxy),
                    "class_id": cls_id,
                    "label": label,
                    "conf": conf,
                })

        return detections

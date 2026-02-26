"""動画入出力 — OpenCV による動画の読み込み・書き出し"""

import cv2


class VideoReader:
    """動画ファイルからフレームを順次読み込むイテレータ。

    Parameters
    ----------
    path : str
        動画ファイルのパス。
    """

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"動画ファイルを開けません: {path}")

    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration(self) -> float:
        fps = self.fps
        if fps > 0:
            return self.frame_count / fps
        return 0.0

    def __iter__(self):
        return self

    def __next__(self) -> "cv2.Mat":
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    def release(self):
        self.cap.release()


class VideoWriter:
    """動画ファイルへフレームを書き出す。

    Parameters
    ----------
    path : str
        出力ファイルパス。
    fps : float
        フレームレート。
    width : int
        フレーム幅。
    height : int
        フレーム高さ。
    """

    def __init__(self, path: str, fps: float, width: int, height: int):
        self.path = path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"VideoWriter を開けません: {path}")

    def write(self, frame: "cv2.Mat"):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

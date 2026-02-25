import cv2
import numpy as np
from camera.camera_base import CameraSource


class WebcamSource(CameraSource):
    """OpenCV webcam implementation."""

    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720):
        self.device_index = device_index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam at device index {self.device_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print(f"[WebcamSource] Started — device {self.device_index} @ {self.width}x{self.height}")

    def read(self) -> tuple[bool, np.ndarray]:
        if self._cap is None:
            raise RuntimeError("Camera not started. Call start() first.")
        success, frame = self._cap.read()
        if success:
            frame = frame[:, ::-1, :].copy()  # flip horizontally (mirror correction), copy for contiguous memory
        return success, frame

    def stop(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
            print("[WebcamSource] Stopped.")

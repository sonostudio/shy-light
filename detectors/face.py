import numpy as np
import mediapipe as mp
from pathlib import Path
from detectors.detector_base import Detector

BaseOptions         = mp.tasks.BaseOptions
FaceDetector        = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode   = mp.tasks.vision.RunningMode

FACE_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_detector.tflite"


class FaceCoordinateDetector(Detector):
    """
    Detects the center of the primary face and returns normalized
    x, y coordinates (0.0–1.0 relative to frame size).

    Sends every frame for smooth continuous control.
    Returns (-1.0, -1.0) when no face is detected so Unreal can
    distinguish "no face" from "face at top-left corner".

    OSC output (via main.py):
        /person/face/x  float 0.0–1.0  (or -1.0 if no face)
        /person/face/y  float 0.0–1.0  (or -1.0 if no face)
    """

    def __init__(self):
        if not FACE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {FACE_MODEL_PATH}\n"
                "Run: uv run python download_models.py"
            )

        self._detector = FaceDetector.create_from_options(
            FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
                running_mode=VisionRunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
        )

    def detect(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.detections:
            return {"face_x": -1.0, "face_y": -1.0, "face_bbox": None}

        # Use the largest face (primary subject)
        best = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bb = best.bounding_box

        cx = round((bb.origin_x + bb.width  / 2) / w, 3)
        cy = round((bb.origin_y + bb.height / 2) / h, 3)

        # face_bbox in pixel coords for debug overlay
        bbox = (bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height)

        return {"face_x": cx, "face_y": cy, "face_bbox": bbox}

    def release(self) -> None:
        self._detector.close()
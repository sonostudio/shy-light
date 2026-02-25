import numpy as np
import mediapipe as mp
from pathlib import Path
from detectors.detector_base import Detector
from state.schema import Proximity

BaseOptions         = mp.tasks.BaseOptions
FaceDetector        = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode   = mp.tasks.vision.RunningMode

MODEL_PATH = Path(__file__).parent.parent / "models" / "face_detector.tflite"


class ProximityDetector(Detector):
    """
    Detects person proximity by measuring face bounding box area
    relative to total frame area.

    Returns both:
    - `proximity`: discrete enum (CLOSE / MEDIUM / FAR / NONE) for state management
    - `proximity_value`: float 0.0–1.0 normalized against close_threshold,
                         for smooth continuous control on the Unreal side.
                         0.0 = no face / furthest, 1.0 = closest possible.

    Thresholds configurable via config.yaml:
        close_threshold:  face area fraction → CLOSE  (default 0.15)
        far_threshold:    face area fraction → FAR    (default 0.04)
    """

    def __init__(self, close_threshold: float = 0.15, far_threshold: float = 0.04):
        self.close_threshold = close_threshold
        self.far_threshold   = far_threshold

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}\n"
                "Run: uv run python download_models.py"
            )

        self._detector = FaceDetector.create_from_options(
            FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=VisionRunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
        )

    def detect(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        frame_area = h * w

        rgb = frame[:, :, ::-1]  # BGR → RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._detector.detect(mp_image)

        if not results.detections:
            return {"proximity": Proximity.NONE, "proximity_value": 0.0}

        # Use the largest detected face (primary subject)
        largest_area = 0
        for detection in results.detections:
            bbox = detection.bounding_box
            face_area = bbox.width * bbox.height
            if face_area > largest_area:
                largest_area = face_area

        ratio = largest_area / frame_area

        # Discrete state
        if ratio >= self.close_threshold:
            proximity = Proximity.CLOSE
        elif ratio <= self.far_threshold:
            proximity = Proximity.FAR
        else:
            proximity = Proximity.MEDIUM

        # Continuous float: normalize ratio to 0.0–1.0 clamped at close_threshold
        proximity_value = round(min(ratio / self.close_threshold, 1.0), 3)

        return {"proximity": proximity, "proximity_value": proximity_value}

    def release(self) -> None:
        self._detector.close()

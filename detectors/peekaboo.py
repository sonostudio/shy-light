import numpy as np
import mediapipe as mp
from pathlib import Path
from detectors.detector_base import Detector
from state.schema import Gesture

BaseOptions         = mp.tasks.BaseOptions
FaceDetector        = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode   = mp.tasks.vision.RunningMode

FACE_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_detector.tflite"


class PeekabooDetector(Detector):
    """
    Detects peekaboo gesture by detecting the absence of a face.

    PEEKABOO is triggered when:
      1. A face WAS recently detected (person is present)
      2. Face has been gone for at least `face_lost_frames` consecutive frames

    Tune via config.yaml:
      face_lost_frames: consecutive frames face must be absent (default 3)
    """

    def __init__(
        self,
        face_lost_frames: int = 3,
        debug: bool = False,
    ):
        self.face_lost_frames     = face_lost_frames
        self.debug                = debug
        self._last_face_bbox      = None
        self._frames_without_face = 0

        if not FACE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {FACE_MODEL_PATH}\n"
                "Run: uv run python download_models.py"
            )

        self._face_detector = FaceDetector.create_from_options(
            FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
                running_mode=VisionRunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
        )

    def detect(self, frame: np.ndarray) -> dict:
        rgb = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = self._face_detector.detect(mp_image)
        face_detected = bool(face_result.detections)

        if face_detected:
            fb = face_result.detections[0].bounding_box
            self._last_face_bbox = (
                fb.origin_x, fb.origin_y,
                fb.origin_x + fb.width, fb.origin_y + fb.height,
            )
            self._frames_without_face = 0
        else:
            self._frames_without_face += 1

        if self.debug:
            print(
                f"[peekaboo] face={face_detected} "
                f"frames_without_face={self._frames_without_face}/{self.face_lost_frames}"
            )

        # Person must have been seen before
        if self._last_face_bbox is None:
            return {"gesture": Gesture.NONE}

        # Face must have been gone long enough
        if self._frames_without_face >= self.face_lost_frames:
            return {"gesture": Gesture.PEEKABOO}

        return {"gesture": Gesture.NONE}

    def release(self) -> None:
        self._face_detector.close()
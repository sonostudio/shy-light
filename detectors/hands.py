import numpy as np
import mediapipe as mp
from pathlib import Path
from detectors.detector_base import Detector

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

HAND_MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"

# Wrist landmark index — most stable single point for tracking hand position
WRIST = 0


class HandCoordinateDetector(Detector):
    """
    Tracks left and right hands independently, returning normalized
    wrist position (x, y) for each hand (0.0–1.0 relative to frame).

    Operates independently of face detection — hands are always tracked.
    Returns -1.0 for a hand's coordinates when that hand is not detected,
    so Unreal can distinguish "not present" from "at edge of frame".

    Note: MediaPipe handedness labels are from the model's perspective
    (mirrored relative to the person). We correct for this since the
    camera feed is already horizontally flipped in webcam.py.

    OSC output (via main.py), sent every frame:
        /person/hand/right/x  float 0.0–1.0  (or -1.0 if not detected)
        /person/hand/right/y  float 0.0–1.0  (or -1.0 if not detected)
        /person/hand/left/x   float 0.0–1.0  (or -1.0 if not detected)
        /person/hand/left/y   float 0.0–1.0  (or -1.0 if not detected)
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        if not HAND_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {HAND_MODEL_PATH}\n"
                "Run: uv run python download_models.py"
            )

        self._landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
            )
        )

    def detect(self, frame: np.ndarray) -> dict:
        rgb = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        coords = {
            "hand_right_x":    -1.0, "hand_right_y":    -1.0,
            "hand_left_x":     -1.0, "hand_left_y":     -1.0,
            "hand_right_bbox": None,
            "hand_left_bbox":  None,
        }

        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            # MediaPipe handedness is from camera perspective, but since we flip
            # the frame in webcam.py, left and right are swapped — correct here.
            raw_label = handedness[0].category_name.lower()
            label = "left" if raw_label == "right" else "right"

            wrist = landmarks[WRIST]
            x = round(wrist.x, 3)
            y = round(wrist.y, 3)

            coords[f"hand_{label}_x"] = x
            coords[f"hand_{label}_y"] = y

            # Compute pixel bbox from all landmarks for debug overlay
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            coords[f"hand_{label}_bbox"] = (min(xs), min(ys), max(xs), max(ys))

        return coords

    def release(self) -> None:
        self._landmarker.close()
import numpy as np
from detectors.detector_base import Detector
from state.schema import Expression


# Map DeepFace emotion labels → our Expression enum
_EMOTION_MAP = {
    "happy":     Expression.SMILE,
    "angry":     Expression.ANGRY,
    "sad":       Expression.SAD,
    "neutral":   Expression.NEUTRAL,
    "fear":      Expression.NEUTRAL,   # unmapped emotions fall back to NEUTRAL
    "disgust":   Expression.NEUTRAL,
    "surprise":  Expression.NEUTRAL,
}

# Minimum confidence to accept an emotion (0–100 scale from DeepFace)
_MIN_CONFIDENCE = 40.0


class ExpressionDetector(Detector):
    """
    Detects facial expressions (smile, angry, sad, neutral) using DeepFace.

    DeepFace runs a lightweight emotion classifier on top of detected faces.
    It's heavier than MediaPipe alone, so by default it skips frames to
    stay performant. Tune `analyze_every_n_frames` based on your hardware.

    Requires: pip install deepface
    """

    def __init__(self, analyze_every_n_frames: int = 3):
        self.analyze_every_n_frames = analyze_every_n_frames
        self._frame_count = 0
        self._last_result: Expression = Expression.NONE

        # Lazy import — DeepFace loads TF/models on first use
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
        except ImportError:
            raise ImportError(
                "DeepFace is not installed. Run: pip install deepface"
            )

    def detect(self, frame: np.ndarray) -> dict:
        self._frame_count += 1

        # Skip frames for performance
        if self._frame_count % self.analyze_every_n_frames != 0:
            return {"expression": self._last_result}

        try:
            analyses = self._deepface.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,  # don't raise if no face found
                silent=True,
            )

            if not analyses:
                self._last_result = Expression.NONE
                return {"expression": Expression.NONE}

            # Use the first (most prominent) face
            dominant_emotion = analyses[0].get("dominant_emotion", "neutral")
            confidence       = analyses[0].get("emotion", {}).get(dominant_emotion, 0)

            if confidence < _MIN_CONFIDENCE:
                expression = Expression.NEUTRAL
            else:
                expression = _EMOTION_MAP.get(dominant_emotion, Expression.NEUTRAL)

        except Exception:
            expression = self._last_result  # hold last known state on error

        self._last_result = expression
        return {"expression": expression}

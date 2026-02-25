from collections import deque
from state.schema import DetectionResult, Proximity, Expression, Gesture


class StateManager:
    """
    Smooths raw per-frame detections into stable states using debounce.

    A state change is only confirmed and emitted when the new value
    has been consistently detected for `debounce_frames` consecutive frames.
    This prevents OSC signal spam from flickering detections.
    """

    def __init__(self, debounce_frames: int = 5):
        self.debounce_frames = debounce_frames

        # Current confirmed (stable) state
        self._confirmed = DetectionResult()

        # Rolling history buffers — one per field
        self._history: dict[str, deque] = {
            "proximity":  deque(maxlen=debounce_frames),
            "expression": deque(maxlen=debounce_frames),
            "gesture":    deque(maxlen=debounce_frames),
        }

    def update(self, raw: DetectionResult) -> list[tuple[str, str]]:
        """
        Feed in a raw DetectionResult for the current frame.

        Returns a list of (field, value) tuples for every field
        whose confirmed state has changed. Empty list = no change.
        """
        self._history["proximity"].append(raw.proximity)
        self._history["expression"].append(raw.expression)
        self._history["gesture"].append(raw.gesture)

        changed = []

        for field, history in self._history.items():
            if len(history) < self.debounce_frames:
                continue  # not enough samples yet

            # All values in the window must agree
            candidate = history[-1]
            if all(v == candidate for v in history):
                current_confirmed = getattr(self._confirmed, field)
                if candidate != current_confirmed:
                    setattr(self._confirmed, field, candidate)
                    changed.append((field, candidate.value))

        return changed

    @property
    def current_state(self) -> DetectionResult:
        return self._confirmed

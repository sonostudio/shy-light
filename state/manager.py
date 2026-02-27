import time
from collections import deque
from state.schema import DetectionResult, Proximity, Expression, Gesture


class StateManager:
    """
    Smooths raw per-frame detections into stable states using debounce.

    A state change is only confirmed and emitted when the new value
    has been consistently detected for `debounce_frames` consecutive frames.
    This prevents OSC signal spam from flickering detections.

    Idle reset: if no face is detected for `idle_timeout_seconds`, all states
    are reset to their defaults and the reset is emitted as a state change.
    Useful for installations where the piece should return to a clean state
    between viewers.
    """

    _IDLE_DEFAULTS = {
        "proximity":  Proximity.NONE,
        "expression": Expression.NONE,
        "gesture":    Gesture.NONE,
    }

    def __init__(self, debounce_frames: int = 5, idle_timeout_seconds: float = 5.0):
        self.debounce_frames       = debounce_frames
        self.idle_timeout_seconds  = idle_timeout_seconds

        self._confirmed = DetectionResult()
        self._history: dict[str, deque] = {
            "proximity":  deque(maxlen=debounce_frames),
            "expression": deque(maxlen=debounce_frames),
            "gesture":    deque(maxlen=debounce_frames),
        }

        self._last_face_seen_at: float = time.monotonic()
        self._idle: bool = False

    def update(self, raw: DetectionResult) -> list[tuple[str, str]]:
        """
        Feed in a raw DetectionResult for the current frame.

        Returns a list of (field, value) tuples for every field
        whose confirmed state has changed. Empty list = no change.
        """
        # Track last time a face was present
        if raw.proximity != Proximity.NONE:
            self._last_face_seen_at = time.monotonic()
            self._idle = False

        # Check for idle timeout
        seconds_without_face = time.monotonic() - self._last_face_seen_at
        if not self._idle and seconds_without_face >= self.idle_timeout_seconds:
            self._idle = True
            return self._reset_to_idle()

        # If already idle, don't process further until a face returns
        if self._idle:
            return []

        # Normal debounce update
        self._history["proximity"].append(raw.proximity)
        self._history["expression"].append(raw.expression)
        self._history["gesture"].append(raw.gesture)

        changed = []
        for field, history in self._history.items():
            if len(history) < self.debounce_frames:
                continue
            candidate = history[-1]
            if all(v == candidate for v in history):
                current_confirmed = getattr(self._confirmed, field)
                if candidate != current_confirmed:
                    setattr(self._confirmed, field, candidate)
                    changed.append((field, candidate.value))

        return changed

    def _reset_to_idle(self) -> list[tuple[str, str]]:
        """Reset all confirmed states to defaults and return the changes."""
        print("[StateManager] Idle timeout — resetting to default state.")
        changed = []
        for field, default in self._IDLE_DEFAULTS.items():
            if getattr(self._confirmed, field) != default:
                setattr(self._confirmed, field, default)
                changed.append((field, default.value))

        # Also clear history buffers so debounce starts fresh
        for history in self._history.values():
            history.clear()

        return changed

    @property
    def is_idle(self) -> bool:
        return self._idle

    @property
    def current_state(self) -> DetectionResult:
        return self._confirmed
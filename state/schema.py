from enum import Enum


class Proximity(str, Enum):
    CLOSE  = "close"
    MEDIUM = "medium"
    FAR    = "far"
    NONE   = "none"   # no face detected


class Expression(str, Enum):
    SMILE   = "smile"
    ANGRY   = "angry"
    SAD     = "sad"
    NEUTRAL = "neutral"
    NONE    = "none"  # no face detected


class Gesture(str, Enum):
    PEEKABOO = "peekaboo"
    NONE     = "none"


class DetectionResult:
    """Container for all detector outputs in a single frame."""

    def __init__(
        self,
        proximity: Proximity = Proximity.NONE,
        expression: Expression = Expression.NONE,
        gesture: Gesture = Gesture.NONE,
        proximity_value: float = 0.0,
    ):
        self.proximity       = proximity
        self.expression      = expression
        self.gesture         = gesture
        self.proximity_value = proximity_value  # continuous 0.0–1.0

    def __repr__(self):
        return (
            f"DetectionResult("
            f"proximity={self.proximity.value}({self.proximity_value}), "
            f"expression={self.expression.value}, "
            f"gesture={self.gesture.value})"
        )

from abc import ABC, abstractmethod
import numpy as np


class Detector(ABC):
    """
    Base class for all detectors.
    Each detector takes a frame and returns a partial result
    that the orchestrator will merge into a DetectionResult.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> dict:
        """
        Analyze the frame and return a dict of detected values.
        Keys should match fields on DetectionResult (proximity, expression, gesture).
        Only include keys that this detector is responsible for.
        """
        ...

    def release(self) -> None:
        """Optional cleanup hook (e.g. close MediaPipe sessions)."""
        pass

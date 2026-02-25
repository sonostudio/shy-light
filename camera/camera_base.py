from abc import ABC, abstractmethod
import numpy as np


class CameraSource(ABC):
    """Abstract base class for all camera sources."""

    @abstractmethod
    def start(self) -> None:
        """Initialize and start the camera."""
        ...

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray]:
        """
        Read a frame from the camera.
        Returns (success, frame) where frame is a BGR numpy array.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Release camera resources."""
        ...

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

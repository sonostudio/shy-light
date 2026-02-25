import numpy as np
from camera.camera_base import CameraSource


class RealSenseSource(CameraSource):
    """
    Intel RealSense camera implementation.

    Requires: pip install pyrealsense2

    This implementation uses the RGB stream by default.
    The depth stream is also captured and accessible via self.last_depth_frame
    for use in future proximity detection improvements.
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline = None
        self._config = None
        self.last_depth_frame = None

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 is not installed. Run: pip install pyrealsense2"
            )

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._pipeline.start(self._config)
        print(f"[RealSenseSource] Started — {self.width}x{self.height} @ {self.fps}fps")

    def read(self) -> tuple[bool, np.ndarray]:
        if self._pipeline is None:
            raise RuntimeError("Camera not started. Call start() first.")

        import pyrealsense2 as rs

        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame:
            return False, np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.last_depth_frame = depth_frame  # store for optional external use
        color_image = np.asanyarray(color_frame.get_data())
        return True, color_image

    def stop(self) -> None:
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
            print("[RealSenseSource] Stopped.")

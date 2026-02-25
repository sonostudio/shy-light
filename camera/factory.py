from camera.camera_base import CameraSource


def create_camera(config: dict) -> CameraSource:
    """
    Factory function that returns the appropriate CameraSource based on config.
    """
    cam_cfg = config.get("camera", {})
    cam_type = cam_cfg.get("type", "webcam").lower()
    width = cam_cfg.get("width", 1280)
    height = cam_cfg.get("height", 720)

    if cam_type == "webcam":
        from camera.webcam import WebcamSource
        return WebcamSource(
            device_index=cam_cfg.get("device_index", 0),
            width=width,
            height=height,
        )

    elif cam_type == "realsense":
        from camera.realsense import RealSenseSource
        return RealSenseSource(
            width=width,
            height=height,
            fps=cam_cfg.get("fps", 30),
        )

    else:
        raise ValueError(f"Unknown camera type: '{cam_type}'. Valid options: webcam, realsense")

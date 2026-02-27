"""
Download required MediaPipe model files for shy-light.

Run once before starting the app:
    uv run python download_models.py
"""

import subprocess
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

MODELS = {
    "face_detector.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}


def download_models():
    MODELS_DIR.mkdir(exist_ok=True)
    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"[skip] {filename} already exists")
            continue
        print(f"[download] {filename} ...")
        result = subprocess.run(
            ["curl", "-L", "-o", str(dest), url],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[error] Failed to download {filename}:\n{result.stderr}")
            dest.unlink(missing_ok=True)  # remove partial file
        else:
            print(f"[done] {filename} saved to {dest}")


if __name__ == "__main__":
    download_models()
    print("\nAll models ready.")
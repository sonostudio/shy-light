"""
Visual debugger for peekaboo detection.

Draws face and hand bounding boxes on the camera feed in real time
so you can see exactly what the detectors are seeing.

    uv run python debug_peekaboo.py

Controls:
    Q — quit
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

BaseOptions           = mp.tasks.BaseOptions
FaceDetector          = mp.tasks.vision.FaceDetector
FaceDetectorOptions   = mp.tasks.vision.FaceDetectorOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

FACE_MODEL_PATH = Path(__file__).parent.parent / "models" / "face_detector.tflite"
HAND_MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"

OVERLAP_THRESHOLD = 0.3

# Colors (BGR)
COLOR_FACE        = (0, 255, 0)    # green
COLOR_FACE_CACHED = (0, 200, 255)  # yellow — cached (stale) bbox
COLOR_HAND        = (255, 100, 0)  # blue
COLOR_HAND_HIT    = (0, 0, 255)    # red — hand overlapping face
COLOR_TEXT        = (255, 255, 255)
COLOR_PEEKABOO    = (0, 0, 255)


def _overlap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / max((ax2 - ax1) * (ay2 - ay1), 1)


def _center_distance(a, b):
    return (((a[0]+a[2])/2 - (b[0]+b[2])/2)**2 + ((a[1]+a[3])/2 - (b[1]+b[3])/2)**2) ** 0.5


def draw_labeled_rect(frame, x1, y1, x2, y2, color, label):
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(frame, label, (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main():
    face_detector = FaceDetector.create_from_options(
        FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
    )
    hand_landmarker = HandLandmarker.create_from_options(
        HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    last_face_bbox = None
    MAX_FACE_DRIFT = 150

    print("Visual peekaboo debugger running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = frame[:, ::-1, :].copy()  # flip horizontally
        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = face_detector.detect(mp_image)
        hand_result = hand_landmarker.detect(mp_image)

        # Update cached face bbox
        face_bbox_is_fresh = False
        if face_result.detections:
            fb = face_result.detections[0].bounding_box
            candidate = (fb.origin_x, fb.origin_y,
                         fb.origin_x + fb.width, fb.origin_y + fb.height)
            if last_face_bbox is None:
                last_face_bbox = candidate
            drift = _center_distance(candidate, last_face_bbox)
            if drift <= MAX_FACE_DRIFT:
                last_face_bbox = candidate
                face_bbox_is_fresh = True

        # Draw face bbox
        if last_face_bbox:
            fx1, fy1, fx2, fy2 = last_face_bbox
            color = COLOR_FACE if face_bbox_is_fresh else COLOR_FACE_CACHED
            label = "face" if face_bbox_is_fresh else "face (cached)"
            draw_labeled_rect(frame, fx1, fy1, fx2, fy2, color, label)

        # Draw hand bboxes and compute overlap
        peekaboo = False
        if last_face_bbox and hand_result.hand_landmarks:
            fx1, fy1, fx2, fy2 = last_face_bbox
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                hxs = [lm.x * w for lm in hand_landmarks]
                hys = [lm.y * h for lm in hand_landmarks]
                hx1, hx2 = min(hxs), max(hxs)
                hy1, hy2 = min(hys), max(hys)
                overlap = _overlap(fx1, fy1, fx2, fy2, hx1, hy1, hx2, hy2)
                hit = overlap >= OVERLAP_THRESHOLD
                if hit:
                    peekaboo = True
                color = COLOR_HAND_HIT if hit else COLOR_HAND
                draw_labeled_rect(frame, hx1, hy1, hx2, hy2, color,
                                  f"hand {i} overlap={overlap:.2f}")

        # Peekaboo indicator
        if peekaboo:
            cv2.putText(frame, "PEEKABOO!", (w // 2 - 120, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, COLOR_PEEKABOO, 4, cv2.LINE_AA)

        # HUD
        cv2.putText(frame, f"threshold={OVERLAP_THRESHOLD}  hands={len(hand_result.hand_landmarks)}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)

        cv2.imshow("peekaboo debug", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    face_detector.close()
    hand_landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

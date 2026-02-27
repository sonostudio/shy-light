"""
light-puppet — main entry point

Reads camera frames, runs all detectors, manages state transitions,
and fires OSC messages to Unreal Engine on state changes.

Usage:
    python main.py
    python main.py --config path/to/config.yaml
"""

import argparse
import sys
import cv2
import yaml

from camera.factory      import create_camera
from detectors           import ProximityDetector, ExpressionDetector, PeekabooDetector, FaceCoordinateDetector, HandCoordinateDetector
from state.schema        import DetectionResult, Expression, Gesture
from state.manager       import StateManager
from osc.sender          import OSCSender


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_detection_result(raw_outputs: list[dict]) -> DetectionResult:
    """Merge outputs from all detectors into a single DetectionResult."""
    merged = {}
    for output in raw_outputs:
        merged.update(output)

    result = DetectionResult()
    if "proximity"       in merged: result.proximity       = merged["proximity"]
    if "proximity_value" in merged: result.proximity_value = merged["proximity_value"]
    if "expression"      in merged: result.expression      = merged["expression"]
    if "gesture"         in merged: result.gesture         = merged["gesture"]
    if "face_x"          in merged: result.face_x          = merged["face_x"]
    if "face_y"          in merged: result.face_y          = merged["face_y"]
    if "hand_right_x"    in merged: result.hand_right_x    = merged["hand_right_x"]
    if "hand_right_y"    in merged: result.hand_right_y    = merged["hand_right_y"]
    if "hand_left_x"     in merged: result.hand_left_x     = merged["hand_left_x"]
    if "hand_left_y"     in merged: result.hand_left_y     = merged["hand_left_y"]
    # Debug-only bboxes
    if "face_bbox"       in merged: result.face_bbox       = merged["face_bbox"]
    if "hand_right_bbox" in merged: result.hand_right_bbox = merged["hand_right_bbox"]
    if "hand_left_bbox"  in merged: result.hand_left_bbox  = merged["hand_left_bbox"]
    return result


def draw_debug_overlay(frame, state: DetectionResult, detection: DetectionResult) -> None:
    """Draw current state values and bboxes onto the preview frame."""
    h, w = frame.shape[:2]

    # Face bbox — green rectangle
    face_bbox = getattr(detection, "face_bbox", None)
    if face_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx = int(detection.face_x * w)
        cy = int(detection.face_y * h)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(frame, f"face ({detection.face_x:.2f}, {detection.face_y:.2f})",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Hand bboxes — right (blue), left (orange)
    hand_styles = {
        "right": (255, 100,   0),
        "left":  (  0, 165, 255),
    }
    for side, color in hand_styles.items():
        bbox = getattr(detection, f"hand_{side}_bbox", None)
        hx   = getattr(detection, f"hand_{side}_x")
        hy   = getattr(detection, f"hand_{side}_y")
        if bbox is not None:
            x1 = int(bbox[0] * w); y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w); y2 = int(bbox[3] * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Wrist dot
            wx = int(hx * w); wy = int(hy * h)
            cv2.circle(frame, (wx, wy), 6, color, -1)
            cv2.putText(frame, f"{side[0].upper()} ({hx:.2f}, {hy:.2f})",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # State text overlay (top-left)
    lines = [
        f"proximity:  {state.proximity.value} ({state.proximity_value:.2f})",
        f"expression: {state.expression.value}",
        f"gesture:    {state.gesture.value}",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 28


def main():
    parser = argparse.ArgumentParser(description="light-puppet: camera → OSC bridge")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    det_cfg   = config.get("detection", {})
    debug_cfg = config.get("debug", {})

    show_preview     = debug_cfg.get("show_preview", True)
    print_detections = debug_cfg.get("print_detections", True)

    # ── Build components ────────────────────────────────────────────────────
    camera = create_camera(config)

    proximity_cfg  = det_cfg.get("proximity", {})
    expression_cfg = det_cfg.get("expression", {})
    peekaboo_cfg   = det_cfg.get("peekaboo", {})
    state_cfg      = det_cfg.get("state", {})

    detectors = [
        ProximityDetector(
            close_threshold=proximity_cfg.get("close_threshold", 0.15),
            far_threshold=proximity_cfg.get("far_threshold", 0.04),
        ),
        ExpressionDetector(
            analyze_every_n_frames=expression_cfg.get("analyze_every_n_frames", 3),
        ),
        PeekabooDetector(
            overlap_threshold=peekaboo_cfg.get("overlap_threshold", 0.3),
            face_lost_frames=peekaboo_cfg.get("face_lost_frames", 3),
            hand_center_margin=peekaboo_cfg.get("hand_center_margin", 300),
            debug=True,
        ),
        FaceCoordinateDetector(),
        HandCoordinateDetector(
            min_detection_confidence=det_cfg.get("hands", {}).get("min_detection_confidence", 0.5),
        ),
    ]

    state_manager = StateManager(
        debounce_frames=state_cfg.get("debounce_frames", 5),
        idle_timeout_seconds=state_cfg.get("idle_timeout_seconds", 5.0),
    )

    osc_cfg = config.get("osc", {})
    osc = OSCSender(
        host=osc_cfg.get("host", "127.0.0.1"),
        port=osc_cfg.get("port", 7000),
    )

    # ── Main loop ────────────────────────────────────────────────────────────
    print("\n[main] Starting. Press Q in preview window (or Ctrl+C) to quit.\n")

    try:
        with camera:
            # Sync Unreal with initial state on startup
            osc.send_all(state_manager.current_state)

            while True:
                success, frame = camera.read()
                if not success or frame is None:
                    print("[main] Warning: empty frame, skipping.")
                    continue

                # Run all detectors and merge results
                raw_outputs = [d.detect(frame) for d in detectors]
                detection   = build_detection_result(raw_outputs)

                # Suppress expression when peekaboo is active —
                # face is covered so any expression reading is noise
                if detection.gesture == Gesture.PEEKABOO:
                    detection.expression = Expression.NONE

                if print_detections:
                    print(detection)

                # Send continuous floats every frame
                osc.send_change("proximity_value", detection.proximity_value)
                osc.send_change("face_x",          detection.face_x)
                osc.send_change("face_y",          detection.face_y)
                osc.send_change("hand_right_x",    detection.hand_right_x)
                osc.send_change("hand_right_y",    detection.hand_right_y)
                osc.send_change("hand_left_x",     detection.hand_left_x)
                osc.send_change("hand_left_y",     detection.hand_left_y)

                # State manager debounces and returns only changed fields
                changes = state_manager.update(detection)
                for field, value in changes:
                    osc.send_change(field, value)

                # Preview window
                if show_preview:
                    draw_debug_overlay(frame, state_manager.current_state, detection)
                    cv2.imshow("light-puppet preview", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[main] Q pressed — quitting.")
                        break

    except KeyboardInterrupt:
        print("\n[main] Interrupted — shutting down.")

    finally:
        for d in detectors:
            d.release()
        if show_preview:
            cv2.destroyAllWindows()
        print("[main] Done.")


if __name__ == "__main__":
    main()
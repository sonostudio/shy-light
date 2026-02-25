"""
light-puppet — main entry point

Reads camera frames, runs all detectors, manages state transitions,
and fires OSC messages to Unreal Engine on state changes.

Usage:
    python main.py
    python main.py --config path/to/config.yaml
"""

import argparse
import cv2
import yaml

from camera.factory      import create_camera
from detectors           import ProximityDetector, ExpressionDetector, PeekabooDetector
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
    return result


def draw_debug_overlay(frame, state: DetectionResult) -> None:
    """Draw current state values onto the preview frame."""
    lines = [
        f"proximity:  {state.proximity.value}",
        f"expression: {state.expression.value}",
        f"gesture:    {state.gesture.value}",
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += 30


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
    ]

    state_manager = StateManager(
        debounce_frames=state_cfg.get("debounce_frames", 5),
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

                # Send continuous proximity float every frame
                osc.send_change("proximity_value", detection.proximity_value)

                # State manager debounces and returns only changed fields
                changes = state_manager.update(detection)
                for field, value in changes:
                    osc.send_change(field, value)

                # Preview window
                if show_preview:
                    draw_debug_overlay(frame, state_manager.current_state)
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

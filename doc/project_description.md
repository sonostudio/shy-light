## shy-light: Camera-to-OSC Bridge for Interactive 3D Lighting

### Project Overview

shy-light is a real-time camera detection system that translates physical human presence and behaviour into OSC signals for Unreal Engine. It detects proximity, facial expression, and a peekaboo gesture and fires structured events that drive dynamic 3D lighting in a live scene.

The system is designed as a modular, configurable bridge between a camera and any OSC-capable environment, with Unreal Engine as the primary target.

* **Project type:** Creative technology tool / interactive system
* **Intended audience:** Creative developers, interactive installation designers, real-time 3D artists

---

### Goal & Intent

The project was built to explore how subtle, natural human behaviour could be used as input for interactive environments — without buttons, touch interfaces, or explicit interaction.

A central question guiding the work was how to create a system that is:

* **Responsive to nuance** — distinguishing not just presence but emotional state and playful gesture
* **Low-latency** — suitable for real-time use in live performance or interactive installation contexts
* **Modular and reusable** — separable detectors that can be recombined for different projects

---

### Process

#### Research & Detector Design

Development began with evaluating detection approaches for three distinct signals: proximity, expression, and gesture. MediaPipe was selected for face detection (proximity, coordinates) and hand tracking (gesture), with DeepFace layered on top for emotion classification. Each detector was built as an independent class inheriting from a common `Detector` base, enabling clean composition and independent testing.

#### Proximity Detection

Proximity is measured by computing the face bounding box area relative to total frame area and mapping it to three discrete states (CLOSE, MEDIUM, FAR) as well as a continuous float (0.0–1.0). The continuous value gives Unreal Engine smooth, interpolatable input rather than a stepped signal — suitable for driving light intensity or angle.

#### Expression Detection

DeepFace runs an emotion classifier on detected faces, mapping dominant emotions to four states: smile, angry, sad, neutral. Because DeepFace is computationally heavy, it runs every N frames (configurable) rather than every frame, with the last known result held between runs.

#### Peekaboo Detection

The peekaboo detector tracks face absence rather than presence. Once a face has been seen (establishing that a person is present), the detector counts consecutive frames without a detected face. After a configurable threshold is crossed, it fires a `PEEKABOO` state. When the peekaboo gesture is active, expression output is suppressed — since face coverage makes emotion readings unreliable noise.

#### State Management & Debouncing

Raw per-frame detections are noisy. A `StateManager` class debounces all state changes — a new state must persist for a configurable number of consecutive frames before it is confirmed and emitted as an OSC message. An idle timeout resets all states to defaults after a configured period with no face detected, ensuring the system returns to a clean state between viewers.

#### OSC Output

All signals are sent via `python-osc` to a configurable host and port. Continuous floats (proximity value, face coordinates, hand coordinates) are sent every frame. Discrete states (proximity, expression, gesture) are sent only on confirmed change. OSC addresses and value formats are defined in a central mapping file, making it straightforward to adapt to different Unreal Engine setups or other OSC receivers.

---

### Challenges & Learnings

#### Suppressing expression during peekaboo

An early issue was that DeepFace would fire noisy or incorrect emotion readings while a hand covered the face. The solution was explicit suppression: when the gesture detector reports `PEEKABOO`, the expression field is overwritten with `NONE` before OSC dispatch. This required the main loop to have visibility across detector outputs before sending — a key reason the architecture merges all detector outputs into a single `DetectionResult` object rather than dispatching independently.

#### Handedness correction

MediaPipe reports hand labels from the camera's perspective. Since the webcam feed is flipped horizontally (mirror correction), left and right are swapped relative to the person. This required an explicit label inversion in the hand detector — a subtle bug that only became apparent during live testing.

#### Balancing DeepFace performance

DeepFace adds significant per-frame overhead. Frame-skipping (`analyze_every_n_frames`) was necessary to maintain acceptable FPS, but the right value is hardware-dependent. The configurable parameter in `config.yaml` allows tuning per deployment without code changes.

#### Camera abstraction

Supporting both webcam and Intel RealSense required a clean abstraction layer. The `CameraSource` abstract base class with `start()`, `read()`, and `stop()` methods made it straightforward to add the RealSense implementation without touching the main loop — and the factory pattern in `camera/factory.py` keeps the switch to a single config change.

---

### Output

#### Final system

* Python application with modular detector pipeline (proximity, expression, peekaboo, face coordinates, hand coordinates)
* Webcam and Intel RealSense camera support
* Real-time OSC output to Unreal Engine (or any OSC receiver)
* Debug preview window with bounding box overlays and state annotations
* Configurable via a single `config.yaml` — camera type, OSC target, detection thresholds, debounce settings

#### User / viewer experience

For a person standing in front of the camera, the interaction is entirely invisible as a system. Light responds to how close they are, shifts when their expression changes, and reacts with a distinct behaviour when they cover and reveal their face. The experience is one of a space that notices and responds — without any explicit interface.

#### Media

* Video documentation: *to be added*

---

### Technical / Architecture Description

#### System overview

A camera source feeds frames into a pipeline of independent detectors. Their outputs are merged into a single `DetectionResult` per frame, passed through a state manager for debouncing, and dispatched as OSC messages. Continuous values are sent every frame; discrete states are sent only on confirmed change.

#### Data flow

1. Camera captures frame (webcam or RealSense)
2. Frame is passed to all detectors in parallel: ProximityDetector, ExpressionDetector, PeekabooDetector, FaceCoordinateDetector, HandCoordinateDetector
3. Outputs are merged into a `DetectionResult`
4. Expression suppressed if gesture == PEEKABOO
5. Continuous floats dispatched immediately via OSC
6. `StateManager` debounces discrete states; confirmed changes dispatched via OSC
7. Debug overlay rendered to preview window (optional)

```
┌─────────────────────────────────────┐
│         Camera Source               │
│  WebcamSource / RealSenseSource     │
└────────────────┬────────────────────┘
                 │  BGR frame
                 ▼
┌─────────────────────────────────────┐
│         Detector Pipeline           │
│                                     │
│  ProximityDetector  (MediaPipe)     │
│  ExpressionDetector (DeepFace)      │
│  PeekabooDetector   (MediaPipe)     │
│  FaceCoordinateDetector             │
│  HandCoordinateDetector             │
│                                     │
│  → merged DetectionResult           │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│         StateManager                │
│  Debounce · Idle timeout            │
│  → confirmed state changes only     │
└────────────────┬────────────────────┘
                 │  OSC (UDP)
                 ▼
┌─────────────────────────────────────┐
│       Unreal Engine (or any         │
│       OSC receiver)                 │
│  /person/proximity                  │
│  /person/proximity/value            │
│  /person/expression                 │
│  /person/gesture                    │
│  /person/face/x · /person/face/y    │
│  /person/hand/right/x · ...         │
└─────────────────────────────────────┘
```

**Technologies**

* Language: Python
* Detection: MediaPipe (face detection, hand landmarks), DeepFace (emotion classification)
* Camera: OpenCV, pyrealsense2 (optional)
* Communication: python-osc (UDP)
* Configuration: YAML

**GitHub**

https://github.com/sonostudio/shy-light

---

### Technology Reusability & Other Use Cases

#### Reusable components

* Modular detector architecture (each detector is independently usable)
* Camera abstraction layer (webcam / RealSense swap via config)
* Debounce state manager for any multi-state detection system
* OSC dispatch layer (address map is fully configurable)

#### Alternative applications

##### Interactive performance and stage design

The same proximity and expression signals can drive lighting, sound, or projection in live performance contexts — responding to a performer's position on stage or emotional intensity without any wearable sensors or explicit triggers.

##### Visitor experience in museums and galleries

Proximity detection can trigger content at specific distances, creating a sense of an exhibit that activates as a visitor approaches. Expression detection could be used to log emotional response to exhibits (with appropriate consent framing).

##### Accessibility interfaces

The gesture and coordinate outputs can serve as hands-free input for users who cannot use traditional input devices — mapping head position or expression state to navigation or control signals.

#### Client value

shy-light provides a ready-made perception layer for any interactive environment that wants to respond to human presence and behaviour. By abstracting detection complexity behind a clean OSC interface, it lets creative developers and designers focus on the experience rather than the computer vision infrastructure.
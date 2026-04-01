# Raspberry Pi 5 setup & mediapipe troubleshooting

## Context

**Project**: shy-light — a camera → detection → OSC bridge for interactive lighting in Unreal Engine.

**Architecture**:
- Raspberry Pi 5 (8GB RAM, Debian Trixie) — camera capture + full detection stack (MediaPipe, DeepFace, OSC sender)
- MacBook Pro M4 — runs Unreal Engine, receives OSC, outputs visuals to bare LCD panel
- Arducam (USB) — camera input on Pi

**Detection stack** (`pyproject.toml` dependencies):
- `mediapipe>=0.10` — face detection, proximity, peekaboo gesture, hand tracking
- `deepface>=0.0.90` — facial expression detection
- `opencv-python>=4.8` — camera capture
- `python-osc>=1.8` — OSC sender

---

## The core problem

MediaPipe publishes no official wheel for ARM64 Linux. The error on `uv sync`:

```
error: Distribution `mediapipe==0.10.33` can't be installed because it doesn't
have a source distribution or wheel for the current platform

hint: You're on Linux (manylinux_2_41_aarch64), but mediapipe (v0.10.33) only
has wheels for: manylinux_2_28_x86_64, macosx_11_0_arm64, win_amd64
```

piwheels (the community Pi wheel repository) also shows zero built files for mediapipe across all versions.

DeepFace has a similar issue — it depends on TensorFlow which also has no official ARM64 wheel.

---

## Alternative implementation options

Before attempting a source build, consider these alternatives depending on your priorities:

### Option A — Split architecture (recommended for production)
- **Pi** handles only camera capture + OSC sending (no detection)
- **MacBook** runs the full detection stack (mediapipe, deepface, everything)
- Pi streams camera feed to MacBook over LAN (MJPEG or similar)
- Pros: no ARM compatibility issues, faster detection on Mac, simpler Pi setup
- Cons: requires network streaming, Pi is a thin client only

### Option B — Replace mediapipe with tflite-runtime
- Use `tflite-runtime` instead of the full `mediapipe` Python package
- MediaPipe's detection code already uses `.tflite` and `.task` model files
- The shy-light detection logic (face, hands) wouldn't need major changes
- `tflite-runtime` has proper ARM64 wheels and is much lighter
- DeepFace would still need to be replaced or disabled

### Option C — Replace DeepFace with a lighter expression model
- `fer` (Facial Expression Recognition) library is lighter than DeepFace
- A direct `.tflite` emotion classification model avoids TensorFlow entirely
- Can be combined with Option B for a fully Pi-native stack

### Option D — Build mediapipe from source (this document)
- Takes 1–2 hours on Pi 5, complex, may hit further issues
- Good for: learning the build pipeline, client work knowledge, experimental prototypes
- Not recommended for production deployments

---

## Environment

| Component | Version |
|---|---|
| Hardware | Raspberry Pi 5, 8GB RAM |
| OS | Debian GNU/Linux 13 (Trixie) |
| System Python | 3.13.5 |
| Target Python | 3.11.9 (via pyenv) |
| Bazel | 6.5.0 |
| mediapipe target | v0.10.14 |

---

## Conceptual overview of what we solved

Every problem we hit came from running newer software (Trixie, glibc 2.41, Python 3.13) than what MediaPipe's build system was designed for. Each error was the build system or runtime discovering an assumption that didn't hold on our platform:

1. **Python version** — Trixie ships 3.13, MediaPipe needs 3.11. Solved with pyenv.
2. **Bazel toolchain for ninja** — `rules_foreign_cc` assumed a Python toolchain would always be available to compile ninja from source. On our non-standard setup it was `None`. Patched the Bazel rule directly to guard against `None`.
3. **TensorFlow overriding our patches** — TF pulls in its own `rules_foreign_cc` version, overriding WORKSPACE changes. Had to patch the cached file directly instead of upgrading the version.
4. **ARM NEON/carotene** — OpenCV's build enabled ARM NEON optimizations that produced symbols unavailable at runtime. Disabled via build flags in `third_party/BUILD`.
5. **Wheel version metadata** — the build produced version `dev` instead of a real version number, which uv rejected. Patched the zip metadata.
6. **Protobuf conflict** — mediapipe 0.10.14 requires `protobuf<5` but newer tf-keras requires `protobuf>=5`. Pinned tf-keras to an older compatible version.
7. **mediapipe `__init__.py`** — the tasks API namespace wasn't wired up correctly in our non-standard build. Manually reconstructed `mp.tasks` namespace.
8. **Numpy array contiguity** — BGR→RGB flip produced a non-contiguous array that MediaPipe's C++ bindings rejected. Fixed with `.copy()`.

---

## Step-by-step build guide

### 1. Clone the project

```bash
git clone https://github.com/sonostudio/shy-light.git
cd shy-light
```

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

Running `uv sync` at this point will fail with the mediapipe ARM64 error — this is expected.

### 3. Install Python 3.11 via pyenv

Debian Trixie ships Python 3.13. MediaPipe's build requires 3.11. deadsnakes PPA is Ubuntu-only and unavailable on Debian — use pyenv instead.

Install pyenv dependencies:
```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev
```

Install pyenv:
```bash
curl https://pyenv.run | bash
```

Add to shell:
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

Install Python 3.11 (compiles from source, ~10–15 min):
```bash
pyenv install 3.11.9
```

Verify:
```bash
python3.11 --version  # should print Python 3.11.9
```

### 4. Install build dependencies

```bash
sudo apt update && sudo apt install -y \
  cmake protobuf-compiler libprotobuf-dev \
  git make wget unzip pkg-config \
  libopencv-dev libopencv-contrib-dev \
  libflatbuffers-dev flatbuffers-compiler \
  ninja-build tmux
```

Note: `python3.11-dev` is not needed — pyenv-compiled Python includes headers at `~/.pyenv/versions/3.11.9/include/python3.11/`.

### 5. Install Bazel 6.5.0

MediaPipe v0.10.14 requires exactly Bazel 6.5.0. Use the ARM64 binary directly:

```bash
wget https://github.com/bazelbuild/bazel/releases/download/6.5.0/bazel-6.5.0-linux-arm64
chmod +x bazel-6.5.0-linux-arm64
sudo mv bazel-6.5.0-linux-arm64 /usr/local/bin/bazel
bazel --version  # should print bazel 6.5.0
```

### 6. Clone mediapipe source

```bash
cd ~
git clone -b v0.10.14 https://github.com/google-ai-edge/mediapipe.git
cd mediapipe
```

Pinning to `v0.10.14` rather than `0.10.33` — better documented aarch64 build reports.

### 7. Patch mediapipe for aarch64

Remove OpenCV modules unavailable on Debian:
```bash
sed -i -e "/\"imgcodecs\"/d;/\"calib3d\"/d;/\"features2d\"/d;/\"highgui\"/d;/\"video\"/d;/\"videoio\"/d" third_party/BUILD
```

Remove problematic linker flags:
```bash
sed -i -e "/-ljpeg/d;/-lpng/d;/-ltiff/d;/-lImath/d;/-lIlmImf/d;/-lHalf/d;/-lIex/d;/-lIlmThread/d;/-lrt/d;/-ldc1394/d;/-lavcodec/d;/-lavformat/d;/-lavutil/d;/-lswscale/d;/-lavresample/d" third_party/BUILD
```

Disable NEON, carotene and Tengine — critical for runtime stability on Pi:
```bash
sed -i 's/"WITH_WEBP": "OFF",/"WITH_WEBP": "OFF",\n        "ENABLE_NEON": "OFF",\n        "WITH_CAROTENE": "OFF",\n        "WITH_TENGINE": "OFF",/' third_party/BUILD
```

Verify all three flags are present:
```bash
grep -n "NEON\|CAROTENE\|TENGINE" third_party/BUILD
```

Patch `rules_foreign_cc` in WORKSPACE (upgrading the version doesn't work because TensorFlow overrides it — we patch the sha256 to force a fresh fetch):
```bash
# Get the correct sha256
wget https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.10.1.tar.gz \
  -O /tmp/rules_foreign_cc_0.10.1.tar.gz
sha256sum /tmp/rules_foreign_cc_0.10.1.tar.gz
```

Update WORKSPACE (correct sha256 as of writing: `476303bd0f1b04cc311fc258f1708a5f6ef82d3091e53fd1977fa20383425a6a`):
```bash
sed -i 's|strip_prefix = "rules_foreign_cc-0.9.0"|strip_prefix = "rules_foreign_cc-0.10.1"|' WORKSPACE
sed -i 's|url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.9.0.tar.gz"|url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.10.1.tar.gz"|' WORKSPACE
sed -i 's|sha256 = "2a4d07cd64b0719b39a7c12218a3e507672b82a97b98c6a89d38565894cf7c51"|sha256 = "476303bd0f1b04cc311fc258f1708a5f6ef82d3091e53fd1977fa20383425a6a"|' WORKSPACE
```

### 8. Set Python environment variables

**Important:** These must be re-exported in every new SSH session before running the build.

```bash
export PYTHON_BIN_PATH=~/.pyenv/versions/3.11.9/bin/python3.11
export PYTHON_LIB_PATH=~/.pyenv/versions/3.11.9/lib/python3.11/site-packages
```

Install required Python packages for the build:
```bash
~/.pyenv/versions/3.11.9/bin/pip install numpy wheel setuptools
```

### 9. Build the wheel

Run inside tmux so the build survives SSH disconnections:
```bash
tmux new -s mediapipe-build
# If disconnected: tmux attach -t mediapipe-build
```

Generate protobuf files:
```bash
cd ~/mediapipe
$PYTHON_BIN_PATH setup.py gen_protos
```

Build the wheel (1–2 hours on Pi 5):
```bash
$PYTHON_BIN_PATH setup.py bdist_wheel
```

**Important:** Do NOT run `bazel clean --expunge` before the build — it wipes the patched `ninja_build.bzl` from Bazel cache. If you must clean, re-apply the ninja patch (see errors table) before rebuilding.

The build is complete when a `.whl` file appears in `~/mediapipe/dist/`.

### 10. Patch and install the wheel

The build produces a wheel with version `dev` which uv rejects. Patch the metadata inside the zip:

```bash
python3 << 'EOF'
import zipfile, os, shutil

src = os.path.expanduser("~/mediapipe/dist/mediapipe-dev-cp311-cp311-linux_aarch64.whl")
dst = os.path.expanduser("~/mediapipe/dist/mediapipe-0.10.14-cp311-cp311-linux_aarch64.whl")

shutil.copy2(src, dst)
tmp = dst + ".tmp"

with zipfile.ZipFile(dst, 'r') as zin:
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            new_name = item.filename.replace('mediapipe-dev.dist-info', 'mediapipe-0.10.14.dist-info')
            if item.filename in ('mediapipe-dev.dist-info/METADATA', 'mediapipe-dev.dist-info/WHEEL'):
                data = data.replace(b'Version: dev', b'Version: 0.10.14')
                data = data.replace(b'Name: mediapipe-dev', b'Name: mediapipe')
            item.filename = new_name
            zout.writestr(item, data)

os.replace(tmp, dst)
print("Done:", dst)
EOF
```

### 11. Set up the shy-light environment

Update `pyproject.toml` to pin compatible versions and point uv at the local wheel:

```bash
cd ~/Desktop/shy-light

cat > pyproject.toml << 'EOF'
[project]
name = "shy-light"
version = "0.1.0"
description = "Camera → detection → OSC bridge for interactive 3D lighting in Unreal Engine"
requires-python = ">=3.11,<3.12"
dependencies = [
    "opencv-python>=4.8",
    "mediapipe>=0.10",
    "deepface>=0.0.90",
    "python-osc>=1.8",
    "pyyaml>=6.0",
    "tf-keras>=2.16,<2.20",
    "protobuf>=4.25.3,<5",
]

[project.scripts]
shy-light = "main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
include = [
    "main.py",
    "camera/**/*.py",
    "detectors/**/*.py",
    "state/**/*.py",
    "osc/**/*.py",
    "config.yaml",
]

[tool.hatch.build.targets.editable]
sources = ["."]

[tool.uv.sources]
mediapipe = { path = "/home/sono/mediapipe/dist/mediapipe-0.10.14-cp311-cp311-linux_aarch64.whl" }

[tool.uv]
environments = ["sys_platform == 'linux' and platform_machine == 'aarch64'"]
EOF
```

Create the venv and install:
```bash
uv venv --python ~/.pyenv/versions/3.11.9/bin/python3.11
uv pip install ~/mediapipe/dist/mediapipe-0.10.14-cp311-cp311-linux_aarch64.whl
uv pip install hatchling editables
uv sync --python ~/.pyenv/versions/3.11.9/bin/python3.11 --no-build-isolation
```

### 12. Patch mediapipe's __init__.py

The installed mediapipe `__init__.py` has import issues specific to our non-standard build — the tasks API namespace isn't wired up and the file gets corrupted by repeated edits. Replace it entirely:

```bash
cat > /home/sono/Desktop/shy-light/.venv/lib/python3.11/site-packages/mediapipe/__init__.py << 'EOF'
from mediapipe.python import *
import mediapipe.python.solutions as solutions
import mediapipe.tasks.python as tasks_python
import mediapipe.tasks.python.core as tasks_core

class _Tasks:
    python = tasks_python
    vision = tasks_python.vision
    audio = tasks_python.audio
    text = tasks_python.text
    BaseOptions = tasks_core.base_options.BaseOptions

tasks = _Tasks()

del framework
del gpu
del modules
del python
del mediapipe
del util
__version__ = '0.10.14'
EOF
```

### 13. Patch numpy array contiguity in detectors

MediaPipe's C++ bindings require contiguous numpy arrays. The BGR→RGB flip produces non-contiguous arrays:

```bash
cd ~/Desktop/shy-light
sed -i 's/rgb = frame\[:, :, ::-1\]/rgb = frame[:, :, ::-1].copy()/' \
  detectors/proximity.py \
  detectors/peekaboo.py \
  detectors/face.py \
  detectors/hands.py
```

### 14. Download models and run

```bash
cd ~/Desktop/shy-light
.venv/bin/python3.11 utils/download_models.py
```

Run headless (SSH only, no preview):
```bash
sed -i 's/show_preview: true/show_preview: false/' config.yaml
.venv/bin/python3.11 main.py
```

Run with preview on a monitor connected to the Pi (from SSH):
```bash
DISPLAY=:0 .venv/bin/python3.11 main.py
```

---

## Errors encountered and fixes

| Error | Cause | Fix |
|---|---|---|
| `Unable to locate package software-properties-common` | Ubuntu-only package, not on Debian | Use pyenv instead of deadsnakes PPA |
| `Unable to locate package python3.11-dev` | 3.11 not in Debian repos | Not needed — pyenv includes headers |
| `error: invalid command 'bdist_wheel'` | `wheel` package missing in pyenv Python | `pip install wheel setuptools` |
| `ModuleNotFoundError: No module named 'numpy'` | numpy missing in pyenv Python | `pip install numpy` |
| `Error in depset: got element of type NoneType` | `rules_foreign_cc` ninja_tool assumes Python toolchain always present | Patch `ninja_build.bzl` in Bazel cache to guard `py3_runtime` fields against `None` |
| `NoneType has no field 'path'` | Same root cause as above, next line in the file | Extend same patch to guard `_interp.path` too |
| `rules_foreign_cc` version not updating after WORKSPACE patch | TensorFlow loads its own version of `rules_foreign_cc`, overriding ours | Delete the cached directory: `rm -rf ~/.cache/bazel/.../external/rules_foreign_cc` — then patch the fetched file directly |
| Checksum mismatch for `rules_foreign_cc` | Incorrect sha256 in WORKSPACE patch | Download the tarball and run `sha256sum` to get the real hash |
| `undefined symbol: _ZN12carotene_o4t...` | ARM NEON/carotene optimizations built into OpenCV, unavailable at runtime | Add `ENABLE_NEON`, `WITH_CAROTENE`, `WITH_TENGINE` = OFF to `third_party/BUILD` cmake flags |
| `Invalid version: expected version to start with a number` (wheel filename) | Wheel built with version `dev` | Rename wheel file to include a real version number |
| `Failed to read mediapipe wheel` — invalid version | Version `dev` also baked into wheel zip metadata | Patch `METADATA` and `WHEEL` files inside the zip, rename `dist-info` directory |
| `pyrealsense2` has no ARM64 wheels | No ARM64 support in RealSense SDK | Remove `realsense` optional dependency from `pyproject.toml` entirely |
| `mediapipe` and `tf-keras` incompatible (protobuf conflict) | mediapipe needs `protobuf<5`, tf-keras >= 2.20 needs `protobuf>=5` | Pin `tf-keras>=2.16,<2.20` and `protobuf>=4.25.3,<5` |
| `No module named 'hatchling'` | hatchling not installed in venv | `uv pip install hatchling editables` |
| `cannot import name 'python' from 'mediapipe.tasks.python'` | `__init__.py` corrupted by repeated sed edits, and circular import issue | Replace `__init__.py` entirely with clean handwritten version |
| `AttributeError: '_Tasks' object has no attribute 'BaseOptions'` | `BaseOptions` lives in `tasks.python.core`, not exposed on `_Tasks` shim | Add `BaseOptions = tasks_core.base_options.BaseOptions` to the `_Tasks` class |
| `TypeError: incompatible constructor arguments` (numpy array) | BGR→RGB flip via `[:, :, ::-1]` produces non-contiguous array | Add `.copy()` after the flip in all four detectors |

---

## Notes

- System Python (3.13) is untouched throughout — pyenv installs 3.11 in `~/.pyenv/versions/`
- Bazel lives at `/usr/local/bin/bazel` — single binary, easily removed
- The Pi is not locked to this project — all installs are in user directories or easily reversible
- DeepFace latency on Pi is an open question — to be tested separately
- The `__init__.py` and detector patches will be lost if the venv is recreated — re-apply steps 12 and 13 after any `uv sync` that rebuilds the venv
- Environment variables (`PYTHON_BIN_PATH`, `PYTHON_LIB_PATH`) must be re-exported in each new SSH session if rebuilding mediapipe
- If rebuilding mediapipe, always verify the `ninja_build.bzl` patch is in place before starting — `bazel clean` wipes it
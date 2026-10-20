# Blight World

**Start with [the developer and agent handover](docs/HANDOVER.md).** It documents
the installed stack, fresh-checkout reconstruction, every source/tool area,
sensor/action contracts, video commands, world editing, known limitations,
lessons from the failed and corrected demos, and the current validation results.

A MIMo 2.0.0 simulation with a basic room, editable primitive worlds, and an
Unreal street scene. Binocular RGB vision and minimal
proprioception are present by default. The full biology profile enables MIMo's
proprioception, touch and vestibular sensors; audition and Unreal rendering are
optional. Scripted demonstration controllers are supplied separately from the
neutral environment. The simulation does not yet integrate a neural learning
agent, training objective or curriculum.

The simulation dependency is pinned to upstream MIMo commit
`040b0ae4914cbfb26afdf830aa81775b90922f3f` (version 2.0.0).

## Shape classification and motion network

The standalone [shape network experiment](networks/simple_shapes/README.md)
contains the continuous stereo-video renderer, retinal preprocessing, joint
classification/motion training code, recorded results and reproduction commands.
The current 128-channel motion RNN achieved **94.67% object classification**,
**8.00° mean direction error** and **1.03 cm/s mean speed error** on its six-shape
TEST dataset. It estimates object velocity in room coordinates and trains both
pathways together from scratch. See the guide for dependencies and dataset
generation; its training environment is separate from the simulation stack.

## Installed Unreal workspace

The working stack uses UE **5.5.4**, SPEAR commit
`38b6323d28c4db3e53d70b9c667770e3c2976d42`, Python **3.12.13**, MuJoCo
**3.3.0**, and Steam Audio **4.8.1**. Use `.venv-unreal` for this stack.
The engine is installed at `/home/vishnu/Unreal/UE_5.5.4`; the project is
[`ue_project/BlightWorld/BlightWorld.uproject`](ue_project/BlightWorld/BlightWorld.uproject).
[`config/spear_unreal.yaml`](config/spear_unreal.yaml) selects the A6000 and
offscreen rendering. MuJoCo owns all physical integration in metres, kilograms
and seconds. SPEAR transfers the refreshed body and eye poses into Unreal.

Reproduce the verified, supported limb-inspection sequence:

```bash
MUJOCO_GL=egl .venv-unreal/bin/python tools/export_self_examination_demo.py \
  --renderer unreal --output artifacts/unreal_self_examination_handover
```

The output is `artifacts/unreal_self_examination_handover/limb_sequence.mp4`: two eye
views, 10 seconds, 500 frames at 50 fps. It shows right hand, left hand, right
lower leg/foot, left lower leg/foot, then returns to neutral. This first video
is silent and uses the existing scripted torque controller and explicit root
support; it does not demonstrate learned balance.

Reproduce the corrected road run with recorded barefoot footsteps:

```bash
MUJOCO_GL=egl .venv-unreal/bin/python tools/export_street_run.py \
  --seconds 8 --warmup 2 --output artifacts/unreal_street_run_handover
```

This produces `street_run.mp4`, stereo listening audio, motion/contact arrays and
footfall events. It is a scripted adult-reference retarget with external
balance/height/pacing forces and adjusted demo joint damping; external vertical
assistance supplied approximately 72% of bodyweight in the corrected reference
run. The previously verified corrected video is in
`artifacts/unreal_street_run_redo`, while `artifacts/unreal_street_run` contains
rejected earlier motion. Use fresh output directories to preserve accepted media.

Ensure `ffmpeg` and `ffprobe` are on PATH (locally,
`export PATH="/home/vishnu/miniconda3/bin:$PATH"`). Generated recordings and asset
manifests remain local under ignored `artifacts/`; after cloning, regenerate the
manifests using the [asset instructions](docs/HANDOVER.md#8-vision-and-unreal-assets).
The current SPEAR config includes the working Unreal file-cache fallback, so no
temporary configuration under `/tmp` is required.

Enable the complete observation interface:

```python
from blight_world import BlightWorldEnv, AuditionConfig
from blight_world.unreal_vision import make_unreal_vision_provider

env = BlightWorldEnv(
    width=320, height=240,
    biology_profile="full",
    audition=AuditionConfig(backend="steam_audio"),
    vision_provider_factory=make_unreal_vision_provider,
)
observation, info = env.reset(seed=1401)
# observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

The torque interface has 90 actions. Each 10 ms step contains two 5 ms physics
substeps and returns two RGB arrays `(240, 320, 3)`, 621 proprioceptive values,
15,996 tactile values, six vestibular values, and stereo audio `(2, 480)` at
48 kHz. These dimensions refer to the pinned body and full profile above.
`motor_delay` and `sensory_delay` count complete 10 ms steps. The latter delays
the whole observation together. Reset clears both queues and acoustic history.

For optional muscles, pass
`actuation_model=blight_world.biology.CorrectedMuscleModel` and use that
environment's `action_space`. Setting a target advances no biological time;
muscle activation advances once per physics substep. Sensor evaluation uses a
refreshed copy of the completed state.

Steam Audio uses its reference HRTF and returns a pressure proxy, rather than
calibrated infant ear pressure. Its direct and reflected branches share the
causal flight delay at 340 m/s; the additional measured processing latency is
60 samples. Audio follows simulated time and does not depend on playback.
The original Python audio backend described below remains available.

To rebuild the small project after C++ changes:

```bash
/home/vishnu/Unreal/UE_5.5.4/Engine/Build/BatchFiles/Linux/Build.sh \
  BlightWorldEditor Linux Development \
  -Project=/home/vishnu/coding_proj/bl_1/ue_project/BlightWorld/BlightWorld.uproject \
  -WaitMutex -NoHotReloadFromIDE -MaxParallelActions=12
```

The external environment, editor Python and native libraries are already
installed. Editor Python is UE's separate Python 3.11.8. The local SPEAR Python
dependencies use SciPy 1.13.1 and OpenCV headless 4.11.0.86 to share MIMo's
scientific environment. Keep the project directory named `ue_project`: a root
folder named `unreal` interferes with SPEAR's detection of Unreal's Python API.

## Edit the room

The project opens `/Game/Maps/WorldEditor`, already populated with 14 editable
primitives and two preview lights. Open it from a terminal on the remote desktop:

```bash
UE_EDITOR=/home/vishnu/Unreal/UE_5.5.4/Engine/Binaries/Linux/UnrealEditor
UPROJECT="$PWD/ue_project/BlightWorld/BlightWorld.uproject"
"$UE_EDITOR" "$UPROJECT" -graphicsadapter=1 -ddc=InstalledNoZenLocalFallback
```

Select an object in the Outliner and press **F** to frame it. Move/rotate it with
the usual editor tools. Under **Blight**, edit its SI dimensions, mass, friction
and material name. Keep actor scale at 1. Mass zero means static; positive mass
creates a free body in MuJoCo. Duplicate a primitive to add an object and give
it a unique Physics Name. The available shapes are box, sphere, cylinder and
plane. Save and close the editor before exporting.

From the repository root, with the two variables above still set:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
BLIGHT_UNREAL_WORLD_MODE=export "$UE_EDITOR" "$UPROJECT" \
  -unattended -vulkan -renderoffscreen -graphicsadapter=1 \
  -ddc=InstalledNoZenLocalFallback \
  -ExecutePythonScript="$PWD/tools/unreal_edit_world.py"

.venv-unreal/bin/python tools/unreal_prepare_assets.py \
  --scene-template config/edited_room.xml.template \
  --content-root /Game/EditedWorld --output artifacts/unreal/edited_assets

VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
BLIGHT_UNREAL_ASSET_MANIFEST="$PWD/artifacts/unreal/edited_assets/manifest.json" \
  "$UE_EDITOR" "$UPROJECT" \
  -unattended -vulkan -renderoffscreen -graphicsadapter=1 \
  -ddc=InstalledNoZenLocalFallback \
  -ExecutePythonScript="$PWD/tools/unreal_import_assets.py"
```

For the exported room, set `BLIGHT_UNREAL_ASSET_MANIFEST` to the absolute path
of `artifacts/unreal/edited_assets/manifest.json`, and pass
`scene_template="config/edited_room.xml.template"` to the environment above.
Both its physical and acoustic geometry then come from that exported model.
The baseline video's assets stay under `/Game/BlightWorld`.

These commands have been run for the supplied room. To populate a fresh
WorldEditor map, use the export command with `BLIGHT_UNREAL_WORLD_MODE=populate`.
Population refuses to replace existing primitive edits. Actor placement needs
the real Vulkan viewport, so do not use `-nullrhi` for population.

## Older native-only setup

For a new combined Unreal checkout, use the complete
[reconstruction procedure](docs/HANDOVER.md#3-installed-stack-and-rebuilding-a-checkout)
instead. It includes the tracked SPEAR compatibility patch, both Python
environments, native extension, SDK and project build. The following original
installation is sufficient only for the native Python/MuJoCo package.

```bash
/home/vishnu/miniconda3/envs/expectation_snn/bin/python -m venv .venv
PIP_CONSTRAINT="$PWD/build-constraints.txt" .venv/bin/python -m pip install -r requirements-dev.txt
```

Use the requirements file: upstream MIMo 2.0.0's wheel omits required package
modules and assets, so this project installs the pinned commit as an editable
VCS checkout while still enforcing `MIMo==2.0.0` in project metadata.

## Test

```bash
MUJOCO_GL=egl .venv-unreal/bin/python -m pytest -q
```

The suite does not launch a full Unreal video. See the handover's
[validation section](docs/HANDOVER.md#13-validation-and-its-limits) for the actual
results and separately exercised rendering workflows.

## Instantiate

```bash
MUJOCO_GL=egl .venv/bin/python - <<'PY'
from blight_world import BlightWorldEnv

env = BlightWorldEnv(width=96, height=96)
observation, info = env.reset(seed=0)
env.close()
PY
```

## Optional binaural audition

Audition is opt-in. The default constructor retains the original three
observation keys. The two supplied profiles add a C-contiguous float32 pressure
waveform in pascals under `observation["audition"]`:

- `AuditionConfig.quality_48k()` produces `(2, 480)` per 10 ms environment step.
- `AuditionConfig.training_24k()` produces `(2, 240)` per 10 ms environment step.

Both profiles keep MuJoCo at a 5 ms substep and MIMo at two substeps per action.
Sources can be fixed in the room or attached to a named MuJoCo body or site.

```bash
MUJOCO_GL=egl .venv/bin/python - <<'PY'
import numpy as np

from blight_world import (
    AuditionConfig,
    AudioSourceSpec,
    BlightWorldEnv,
    BodyAttachment,
    ContinuousSoundConfig,
)

env = BlightWorldEnv(
    audition=AuditionConfig.quality_48k(
        continuous=ContinuousSoundConfig(),
    )
)
observation, info = env.reset(seed=0)
assert env.audition is not None
env.audition.register_source(
    AudioSourceSpec("ball_chime", "chime", BodyAttachment("ball"))
)
env.audition.play("ball_chime")
observation, reward, terminated, truncated, info = env.step(
    np.zeros(env.action_space.shape, dtype=np.float32)
)
pressure_pa = observation["audition"]
scrape_diagnostics = env.audition.scrape_diagnostics
env.close()
PY
```

`ContinuousSoundConfig()` enables persistent physics-driven scrape audio for
solved sliding contacts. It uses a separate four-stream pool, so it does not
reserve any of the eight clip/impact voices. The immutable diagnostics are
available only on `env.audition.scrape_diagnostics`; contact identities and
controls are never added to observations or `info`. Rolling, clothing, and air
sounds are outside the current scope.

The renderer uses the paired adult SADIE II H10 HRIR measurements, virtual
ears rigidly attached to MIMo's head, propagation delay and distance loss,
direct-path occlusion, six first-order room reflections, an eight-line late
tail, and material-dependent contact transients. The H10 measurements are an
adult surrogate, not an infant-anatomy claim. Impact clips are deterministic
engineering assets and are documented in
[`src/mimo_audition/assets/audio/README.md`](src/mimo_audition/assets/audio/README.md).
There is no clipping or automatic gain control, and source definitions or
positions are never inserted into observations or `info`.

The reusable interfaces are available directly from `mimo_audition`; the
common configuration and source types are also re-exported from `blight_world`.
See [`docs/research_log.md`](docs/research_log.md) for provenance, equations,
scientific grounding, and declared approximations. Reproduce the measured
live-contact results with `tools/profile_audition_scrape.py`; the generated
report is [`docs/audition_scrape_validation_profile.md`](docs/audition_scrape_validation_profile.md).

The bounded-state timing run is deliberately outside CI:

```bash
MUJOCO_GL=egl .venv/bin/python tools/benchmark_audition_streaming.py \
  --duration-24k 600 --duration-48k 60 --warmup 2 \
  --json-output /tmp/blight-audition-benchmark.json
MUJOCO_GL=egl .venv/bin/python tools/profile_audition_scrape.py \
  --benchmark-json /tmp/blight-audition-benchmark.json \
  --json-output /tmp/blight-audition-scrape-profile.json \
  --markdown-output /tmp/blight-audition-scrape-profile.md
```

The reference Python renderer is not advertised as real-time; use the reported
p99 substep latency to evaluate it on the target machine.

# Blight World developer and agent handover

This is the main handover for the workspace at `/home/vishnu/coding_proj/bl_1`.
It describes the code and installed system inspected on **10 September 2026**.
Read the quick start and capability boundaries before changing or running a demo.
The validation section records what was actually checked; installation on another
machine is a separate operation.

## Contents

1. [Purpose and current state](#1-purpose-and-current-state)
2. [Run an existing video on this machine](#2-run-an-existing-video-on-this-machine)
3. [Installed stack and rebuilding a checkout](#3-installed-stack-and-rebuilding-a-checkout)
4. [Repository map](#4-repository-map)
5. [Physics, rendering and observation timing](#5-physics-rendering-and-observation-timing)
6. [Using the environment for learning](#6-using-the-environment-for-learning)
7. [Bodies and motor controllers](#7-bodies-and-motor-controllers)
8. [Vision and Unreal assets](#8-vision-and-unreal-assets)
9. [Audio](#9-audio)
10. [Editing the world](#10-editing-the-world)
11. [Video tools and their outputs](#11-video-tools-and-their-outputs)
12. [Lessons from this conversation and implementation](#12-lessons-from-this-conversation-and-implementation)
13. [Validation and its limits](#13-validation-and-its-limits)
14. [Version-control boundaries](#14-version-control-boundaries)
15. [References and asset attribution](#15-references-and-asset-attribution)

## 1. Purpose and current state

The project is an embodied-agent **world and sensor/action interface**. The
eventual learner will consume sensory arrays and produce motor commands. There
is currently no supplied neural learning architecture, training objective,
curriculum, or demonstrated end-to-end learning run.

The current architecture is:

```text
Python learner or scripted controller
          | bounded motor command
          v
Python MuJoCo + MIMo                 sole owner of physical state and time
          | completed-state copy
          +--> MIMo proprioception, touch, vestibular processing
          +--> SPEAR --> Unreal --> left/right RGB images
          |
          +--> contact/source events --> numerical audio --> ear waveforms
          |
          v
Python observation dictionary       returned by env.step(action)
```

Unreal renders the body/world and supports primitive editing. It does not run a
second interacting body simulation. Steam Audio runs through its standalone C
library in Python; it does not depend on Unreal's audio device or an installed
Steam Audio Unreal plugin. The original native MuJoCo renderer and Python audio
backend also remain usable.

The source began as a copy of `/home/vishnu/coding_proj/blight_world`. Work in
this checkout, not the original directory. Earlier native-controller and audio
experiments remain in the tree and are distinguished below from the newer Unreal
examples.

### What has been demonstrated

| Capability | Actual evidence and boundary |
| --- | --- |
| Binocular self-inspection in Unreal | A scripted sequence looks at right hand, left hand, right lower leg/foot, left lower leg/foot, then neutral. A world weld supports the root. |
| Road with streetlights | MuJoCo body motion is transferred to Unreal; both physical eye streams and a separate spectator view are captured. |
| Corrected running demonstration | Adult-reference joint motion, contact-driven recorded footsteps and external balance/height/pacing assistance. It is not autonomous running. |
| Sensor/action API | Gymnasium-style reset/step, bounded torque actions, configurable full MIMo senses, numerical stereo audio, explicit delays. |
| World editing | Editable boxes, spheres, cylinders and planes can be exported to MJCF and reimported as visual assets. This is not a general arbitrary-mesh physical-asset importer. |
| Learning | The interfaces permit initial learning experiments. A completed learner or long-duration training qualification has not been demonstrated. |

The default observation profile is intentionally small. Selecting an Unreal
renderer does **not** enable all body senses automatically. The demo videos use
scripted controllers, and the running demo retains assistance and adjusted
passive damping. A visually plausible video alone does not establish biological
accuracy, learned balance, or realistic ground-reaction forces.

## 2. Run an existing video on this machine

Run commands from the repository root. Use the existing `.venv-unreal` for the
Unreal/Steam Audio stack. There is no reason to rebuild the engine or reinstall
dependencies before reproducing an existing example.

```bash
cd /home/vishnu/coding_proj/bl_1
export PATH="/home/vishnu/miniconda3/bin:$PATH"
export MUJOCO_GL=egl
```

`ffmpeg` and `ffprobe` must be discoverable. The installed versions are available
through the PATH above. Close an editor instance using this project before
starting a SPEAR-owned run. Preserve unrelated processes and other GPU users.

### Binocular limb inspection: the foundational example

```bash
.venv-unreal/bin/python tools/export_self_examination_demo.py \
  --renderer unreal \
  --output artifacts/unreal_self_examination_handover
```

Expected result: `limb_sequence.mp4`, ten simulated seconds, 500 frames at 50 fps,
640×272 pixels. The top strip contains labels; the remaining image contains two
320×240 eye views. This Unreal branch is **silent**. It also saves designated
hold images. The native branch has a different output/reporting path; do not
assume the Unreal branch emits every native diagnostic file.

The sequence is right hand, left hand, right lower leg/foot, left lower leg/foot,
with neutral settling/return. `--overwrite` explicitly permits replacement of an
existing output video. `--smoke` applies to the native branch; it is not an
Unreal smoke mode. Use a fresh output directory to preserve accepted examples.

### Corrected road run with footsteps

```bash
.venv-unreal/bin/python tools/export_street_run.py \
  --seconds 8 --warmup 2 \
  --output artifacts/unreal_street_run_handover
```

Expected result: `street_run.mp4`, eight seconds, 400 frames at 50 fps, 960×816
pixels, with stereo sound. The spectator is above the two actual eye streams.
The directory also contains `footsteps.wav`, `motion.npz`, `footfalls.json`, and
first/middle/last stills. Duration must be a positive multiple of 20 ms. The
default output directory is an older demo path, so always provide `--output`
when preserving earlier results. This exporter can overwrite files in that
directory; it has no separate `--overwrite` gate.

The previously verified corrected artifact is
`artifacts/unreal_street_run_redo/street_run.mp4`. Do not mistake
`artifacts/unreal_street_run/street_run.mp4` or the preview directory for that
correction: those contain rejected earlier motion.

`gait_comparison.png` and `.json` in the corrected directory were produced during
the motion analysis. They are **not automatic outputs of the street exporter**.
The runtime gait reference itself is embedded in `street_run.py`; generating a
video does not require the temporary human-data analysis files used during the
redo.

### Retrieve a result

Run this on the destination Mac, changing only the artifact directory as needed:

```bash
scp -r vishnu@100.123.25.88:/home/vishnu/coding_proj/bl_1/artifacts/unreal_street_run_handover /Users/vmoh0219/Downloads
```

An exporter exiting successfully is the start of artifact inspection. Check the
requested sequence, both eyes, the complete duration, and sound timing before
calling a newly generated example correct.

## 3. Installed stack and rebuilding a checkout

### Versions and host-specific paths

| Component | Inspected local version/location |
| --- | --- |
| External Python | 3.12.13, `.venv-unreal/bin/python` |
| MIMo | 2.0.0, commit `040b0ae4914cbfb26afdf830aa81775b90922f3f`, editable `third_party/MIMo` |
| MuJoCo | 3.3.0 |
| Gymnasium | 1.2.3 |
| NumPy / SciPy | 2.0.2 / 1.13.1 |
| OpenCV | `opencv-python-headless==4.11.0.86` |
| Matplotlib / trimesh | 3.9.4 / 4.6.2 in the external environment |
| pytest | 8.4.2 |
| Unreal Engine | 5.5.4, `/home/vishnu/Unreal/UE_5.5.4` |
| Unreal editor Python | Separate embedded Python 3.11.8 |
| SPEAR | `spear-sim` 1.1.0, commit `38b6323d28c4db3e53d70b9c667770e3c2976d42` |
| Steam Audio | Standalone SDK 4.8.1; `third_party/steam_audio/lib/linux-x64/libphonon.so` |
| Project | `ue_project/BlightWorld/BlightWorld.uproject` |
| Rendering | NVIDIA Vulkan, `graphicsadapter=1` selects the A6000 on this host |

These are the installed combination, not a recommendation to track latest
versions. `.venv` is the older native environment; `.venv-unreal` is the current
combined stack. Unreal's embedded Python is neither of these environments.

The following settings are machine-specific:

- `config/spear_unreal.yaml`: editor executable, project path, SPEAR temporary
  directory, Vulkan ICD path and graphics adapter.
- `.uproject` `AdditionalPluginDirectories`: the repository-relative path
  `../../third_party/spear/cpp/unreal_plugins`; keep that checkout layout.
- `BLIGHT_UNREAL_CONFIG`: optional alternate SPEAR YAML path. The self-inspection
  CLI instead sets this from its `--spear-config` argument.
- `BLIGHT_UNREAL_ASSET_MANIFEST`: scene geometry manifest. The street exporter
  sets its own street manifest; use the basic manifest for the basic room.
- `STEAM_AUDIO_LIBRARY` or `AuditionConfig.steam_audio_library_path`: optional
  override for the standalone shared library.

The checked-in SPEAR config uses
`ddc: InstalledNoZenLocalFallback`. This was the successful remedy for an actual
Zen derived-data-cache port recovery loop on this host. It selects an existing
Unreal file-cache graph; it is not a custom cache service. The current commands
must not depend on a YAML file under `/tmp`.

### Fresh-checkout reconstruction

The Python package metadata alone is insufficient to recreate the combined
stack: the Unreal engine, ignored third-party trees, native builds and generated
asset manifests must also exist. The reconstruction procedure is detailed below;
the installed checkout is the immediately usable path.

The following commands reconstruct the installed Linux arrangement from pinned
source, successful build logs and retained CMake caches. The entire fresh-checkout
procedure was **not** replayed during this handover. Do not run clone/install steps
over the working environment merely to reproduce a video.

1. Install the Linux UE **5.5.4** distribution from Epic and provide an NVIDIA
   driver/Vulkan runtime, Git, GCC/G++, Make, and a Python **3.12.13** interpreter.
   The inspected host is Ubuntu 22.04.3 x86-64 with GCC/G++ 11.4.0 and Make 4.3.
   The UE engine itself was not built from source. Obtain `ffmpeg`/`ffprobe`
   through the target machine's package environment; both are already installed
   on this host.
2. Create the external environment and clone the two pinned source trees:

```bash
BLIGHT_ROOT=/home/vishnu/coding_proj/bl_1
BLIGHT_UE=/home/vishnu/Unreal/UE_5.5.4
cd "$BLIGHT_ROOT"

# Substitute the target machine's Python 3.12.13 executable on a new host.
# This is the interpreter used to create the current environment:
.venv/bin/python -m venv .venv-unreal
BLIGHT_PY="$BLIGHT_ROOT/.venv-unreal/bin/python"
export PATH="$BLIGHT_ROOT/.venv-unreal/bin:$PATH"

git clone https://github.com/trieschlab/MIMo.git third_party/MIMo
git -C third_party/MIMo checkout 040b0ae4914cbfb26afdf830aa81775b90922f3f
git clone https://github.com/spear-sim/spear.git third_party/spear
git -C third_party/spear checkout 38b6323d28c4db3e53d70b9c667770e3c2976d42
git -C third_party/spear submodule update --init --recursive
git -C third_party/spear apply "$BLIGHT_ROOT/patches/spear-python-compat.patch"

"$BLIGHT_PY" -m pip install \
  "pip==25.0.1" "setuptools==80.9.0" "wheel==0.48.0" "cmake==4.1.0"
"$BLIGHT_PY" -m pip install --no-build-isolation -e "$BLIGHT_ROOT/third_party/MIMo"
"$BLIGHT_PY" -m pip install -e "$BLIGHT_ROOT[test]"
"$BLIGHT_PY" -m pip install -e "$BLIGHT_ROOT/third_party/spear/python"
```

The original `.venv` is not part of a fresh Git checkout: substitute the available
Python executable on the creation line. The pinned setuptools supplies
`pkg_resources`, which MIMo's setup imports. The explicit editable MIMo install
keeps its required modules/assets in the source tree.

`patches/spear-python-compat.patch` preserves the **only local SPEAR source
modification**: OpenCV becomes headless 4.11.0.86, and SciPy becomes 1.13.1. This
allows the pinned MIMo and SPEAR clients to share the external environment. Install
SPEAR's base package there: its `[editor]` extra requests trimesh 4.4.0, while
MIMo uses 4.6.2; its `[pipeline]` extra requests MuJoCo 3.5.0. Neither extra is
needed in the client for this world. The editor extra belongs in UE's Python.

3. Build SPEAR's native dependencies with UE's compiler/libc++:

```bash
"$BLIGHT_PY" "$BLIGHT_ROOT/third_party/spear/tools/build_third_party_libs.py" \
  --unreal-engine-dir "$BLIGHT_UE" --skip-boost
"$BLIGHT_PY" "$BLIGHT_ROOT/third_party/spear/tools/build_third_party_libs.py" \
  --unreal-engine-dir "$BLIGHT_UE" --skip-rpclib --skip-yaml-cpp
```

The split commands reflect the successful build sequence; the helper can also
build all three. The complete recursive checkout is required before Boost.
Pinned native dependencies are Boost `1bed2b0712b2119f20d66c5053def9173c8462a5`
(headers identify 1.90), rpclib `d1bb5b476bc806fbd67e125181f31ffb07b3e696`, and
yaml-cpp `edadfecdc644e2a75ef6074e7afe74a3ebb44078`. The 171 Boost child checkouts
must match the parent pins. A prior interrupted recursive download was repaired
before the successful build; do not assume a cloned Boost top directory is
complete.

The helper supplies the required position-independent code, UE compiler paths,
libc++ flags and yaml-cpp CMake compatibility settings. It creates static libraries
under Boost's `stage/lib` and rpclib/yaml-cpp `BUILD/Linux` directories. Retain
these settings instead of rebuilding the Unreal plugin against an unrelated
system C++ ABI.

4. Build `spear_ext` explicitly in the **external Python 3.12 environment**:

```bash
BLIGHT_CLANG="$BLIGHT_UE/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v23_clang-18.1.0-rockylinux8/x86_64-unknown-linux-gnu/bin/clang++"
BLIGHT_LIBCPP="$BLIGHT_UE/Engine/Source/ThirdParty/Unix/LibCxx"
BLIGHT_EXT_FLAGS="-std=c++20 -O3 -D_LIBCPP_ENABLE_EXPERIMENTAL -nostdinc++ -I'$BLIGHT_LIBCPP/include/c++/v1' -Wno-reserved-macro-identifier -stdlib=libc++ -L'$BLIGHT_LIBCPP/lib/Unix/x86_64-unknown-linux-gnu' -lc++ -lc++abi"

"$BLIGHT_PY" -m pip install -e "$BLIGHT_ROOT/third_party/spear/python_ext" \
  -C "cmake.define.Python_EXECUTABLE=$BLIGHT_PY" \
  -C "cmake.define.CMAKE_CXX_COMPILER=$BLIGHT_CLANG" \
  -C "cmake.define.CMAKE_CXX_FLAGS=$BLIGHT_EXT_FLAGS"
```

The stock `install_python_extension.py` helper sources conda and activates its
default `spear-env`; a prior attempt therefore built a **cp313** extension. The
successful corrected build was **cp312** with the interpreter selected above.
Its isolated CMake was 4.4.3; that build tool is not pinned in the upstream
extension metadata. The pinned extension uses nanobind 2.7.0 and
scikit-build-core 0.10. `_LIBCPP_ENABLE_EXPERIMENTAL` is intentional: substituting
`-fexperimental-library` tries to link a library absent from this UE distribution.

5. Install editor-side SPEAR separately and build the supplied small project:

```bash
"$BLIGHT_PY" \
  "$BLIGHT_ROOT/third_party/spear/tools/install_python_package_in_editor_env.py" \
  --unreal-engine-dir "$BLIGHT_UE"

"$BLIGHT_UE/Engine/Build/BatchFiles/Linux/Build.sh" \
  BlightWorldEditor Linux Development \
  "-Project=$BLIGHT_ROOT/ue_project/BlightWorld/BlightWorld.uproject" \
  -WaitMutex -NoHotReloadFromIDE -MaxParallelActions=12
```

The first command invokes UE's embedded Python 3.11.8 to install SPEAR's
`python[editor]` package; it does not install MIMo into the editor. The checked-in
project already enables the necessary plugins, uses the relative plugin path,
and contains the `/SpContent` cook entry. Do not rerun project scaffolding over
the current C++ files. When initially creating a *different* project, SPEAR's
`install_plugins_in_external_project.py --external-project-dir ...` supplies the
registration, but is not needed for this checked-in project.

6. Install the standalone Steam Audio library; it needs no compilation or Unreal
   audio plugin for this implementation:

```bash
curl -L --fail --retry 2 \
  -o /tmp/bl1_steamaudio_4.8.1.zip \
  https://github.com/ValveSoftware/steam-audio/releases/download/v4.8.1/steamaudio_4.8.1.zip

"$BLIGHT_PY" - <<'PY'
from pathlib import Path
from zipfile import ZipFile

destination = Path("third_party/steam_audio")
files = (
    "include/phonon.h", "include/phonon_version.h",
    "lib/linux-x64/libphonon.so", "THIRDPARTY.md",
)
with ZipFile("/tmp/bl1_steamaudio_4.8.1.zip") as archive:
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read("steamaudio/" + relative))
PY
```

These paths assume the shell is still at `BLIGHT_ROOT`. The ABI loader defaults
to that library and checks for its presence. The local archive's SHA-256 is
`4a0aa5ec1176f38f0b0993a37c2259d9e86f27e22d5e24f83ec4c3cb9a1d5449` and the local
library's is `faf7c4d69e0f104366f6553cda38d3f946f761f7db223baf12663f6fbf2387da`.
These identify the locally used files, not publisher-provided checksums.

7. Adjust the machine-specific launch YAML. Regenerate the basic/street manifests
   using section 8, and import any missing/changed visual content. The Unreal
   `.uasset`/`.umap` content is retained with the project; the generated manifests
   are not committed and still need to be prepared after cloning.
8. Run the minimal environment example, the existing checks and a complete
   requested Unreal example. Do not claim another machine is working merely
   because these installation commands were copied successfully.

## 4. Repository map

Keep this separation when making a narrowly scoped change. The tree already
separates reusable simulation code, one-off executable workflows, configuration,
tests, generated output and external dependencies; moving modules for cosmetic
reasons would invalidate imports and example scripts without improving control.

```text
README.md                    entry point and common commands
MEMORY.md                    short lessons; this guide is the detailed handover
docs/HANDOVER.md              this document
docs/*profile.md              earlier measured native-audio reports
docs/research_log.md          earlier acoustic research/assumptions
pyproject.toml               package metadata and existing pytest configuration
requirements-dev.txt         editable pinned MIMo + project/test installation
build-constraints.txt        setuptools build constraint
patches/                    required compatibility diff for ignored SPEAR source
src/blight_world/            environment, biological adapters, controllers, vision
src/blight_world/assets/     packaged MJCF templates
src/mimo_audition/           numerical auditory sensor and backends
src/mimo_audition/assets/    packaged audio and HRTF data with attribution
tools/                      executable exporters, editor scripts, older profiles
tests/                      existing unit and integration checks
config/                     SPEAR launch YAML and alternate MJCF scene templates
ue_project/BlightWorld/      Unreal project, C++ authoring actor and content
third_party/                locally installed MIMo, SPEAR and Steam Audio SDK
artifacts/                  generated media, measurements and asset manifests
```

### `src/blight_world`: core and current Unreal examples

| File | Responsibility |
| --- | --- |
| `environment.py` | `BlightWorldEnv`: materializes scene XML, initializes MIMo, selects senses/actuation/vision, owns observation copies and delay queues, attaches audition, resets/closes resources; neutral task/reward. |
| `biology.py` | `CorrectedMuscleModel`, independent `SensorContext`, corrected native acuity handling and `BiologicalVision`, plus the vision-provider protocol. |
| `motor_control.py` | Named actuator-to-joint mapping, sparse action composition, bounded bias-compensated PD helpers and coupled-hip handling. |
| `unreal_vision.py` | SPEAR lifecycle, mesh/model consistency check, pose and unit conversion, body/light/camera spawning, same-frame binocular capture, diagnostic body-ID rendering. |
| `street_run_vision.py` | Street lighting and following spectator camera in addition to the same biological eye capture. Spectator pixels are not added to the agent observation. |
| `street_run.py` | `StreetRunEnv` assistance forces and `StreetRunController` reference torque gait; human-reference anchors and declared demo damping. |
| `__init__.py` | Re-exports older environment/controller helpers and common audio types. Newer Unreal/biology/street classes are imported from their modules directly. |

### Existing native scenarios and manipulation controllers

These modules are retained because their exporters and tests use them. Their
names describe specific scripted/fixture capabilities, not a general learned
motor repertoire. Read the assistance/reset code before using their trajectories
as training demonstrations.

| File | Purpose |
| --- | --- |
| `motor_skills.py` | Supported right-hand sphere pushing; right-arm actuator definitions and controller diagnostics. |
| `locomotion_skills.py` | Static support/contact classification and bounded unsupported belly-crawl controller experiments. |
| `supported_pick_place.py` | Root-supported right-hand sphere pick/place, using contact-informed phases; the ball remains free and there is no grip weld. |
| `supported_bimanual.py` | Root-supported bimanual foam-block clamp, contact classification and phase control; a clamp demonstration, not pickup/lift. |
| `crawl_ball_fixture.py` | Dedicated free-ball fixture in the crawl hand path, with optional per-substep contact recording; no manipulation policy. |
| `three_object_scenario.py` | Older 45-second schedule: ball, block and cylinder, 15 seconds each, with approach/pick/inspect/glance/throw/settle phases and controller hooks. |
| `constraint_assisted_three_object.py` | Root travels through a mocap weld; holding uses a grip weld and collision suppression/restoration. Free object physics resumes after release. This is constraint-assisted carrying/stepping. |
| `assisted_ball_throw.py` | Recorded throw starts from a prepared, collision-suppressed mocap hold. Release disables the weld and seeds velocity before free flight; not a learned pickup or grasp. |

### `src/mimo_audition`

| File | Responsibility |
| --- | --- |
| `config.py` | Immutable sample-rate, capacity, propagation, scrape and backend settings. |
| `sensor.py` | `MIMoAudition`: environment integration, source registration/scheduling, contact capture, per-substep rendering and public observation assembly. |
| `sources.py` | Source definitions, static/body/site attachments and scheduled playback data. |
| `clips.py` | Clip loading/registration, named assets and sample-rate handling. |
| `contacts.py` | Physics contact-to-impact detection, onset/energy/material event information. |
| `materials.py` | Material-name classification, supported impact pairs and canonical pair ordering. |
| `mujoco_scene.py` | Adapter exposing listener/source poses and geometry from MuJoCo to the audio layer. |
| `renderer.py` | Reference Python binaural rendering, clip/continuous voice scheduling, spatial processing and retained tails. |
| `hrtf.py` | Adult SADIE H10 data and directional HRIR processing for the native backend. |
| `dsp.py` | Streaming numerical filters, fractional propagation and related audio utilities. |
| `room.py` | Native rectangular-room reflections and late-reverberation model. |
| `continuous.py` | Bounded continuous-emission data/interface. |
| `scrape.py` | Optional persistent sliding-contact sound synthesis and contact hysteresis. |
| `steam_audio.py` | Standalone Steam Audio scene/effect adapter, shared causal delay, dynamic geometry, synchronous 240-sample rendering. |
| `_steam_audio_sdk.py` | Minimal ctypes declarations and dynamic loader matching the pinned SDK C ABI. |
| `assets/audio/` | Native generated impact bank at 24/48 kHz, named sound clips, and four 48 kHz real barefoot footstep crops. |
| `assets/hrtf/` | Packaged adult H10 measurements and provenance for the native backend. |

### Unreal project source

`Source/BlightWorld/BlightWorldPrimitive.{h,cpp}` defines the editable
`ABlightWorldPrimitive` actor. Its preview mesh has collision and physics
disabled. Shape, SI dimensions, mass, friction and material metadata are exported
to MuJoCo; the preview itself is not the physics engine.

`BlightWorld.{h,cpp}`, `BlightWorld.Build.cs`, the game/editor target files and
`SpTargetRules.Target.cs` connect the small project to SPEAR's modules. Engine
configuration selects SPEAR's game/editor engine classes and game mode, the Room
runtime map and WorldEditor authoring map. `Content/BlightWorld`, `StreetRun` and
`EditedWorld` contain distinct imported visual namespaces; `Content/Maps` contains
the runtime and editable maps. Generated Unreal build/cache directories are not
source code.

## 5. Physics, rendering and observation timing

### One physical owner and explicit time

MuJoCo uses metres, kilograms, seconds and radians. The scene timestep is
**0.005 s**; one public action lasts two substeps, **0.010 s**. A wall-clock pause
does not imply an extra simulated interval. Do not drive scientific stepping from
Unreal's free-running tick or use playback speed as the acoustic clock.

The configured MIMo path is approximately:

```text
env.step(action)
  -> apply motor-delay queue, set requested action
  -> for each of two 5 ms substeps:
       actuation_model.substep_update()
       audition captures start-of-substep kinematics, when enabled
       mj_step() and upstream post-constraint processing
       audition processes that substep's contacts and [t, t+5 ms) sound
       refresh an independent observation-state copy
  -> publish the two audio blocks
  -> collect body senses and render the copied endpoint through both eyes
  -> apply whole-observation sensory-delay queue
  -> return observation, reward, terminated, truncated, info
```

The audio interval and endpoint image have different meanings: an observation
returned at time `t` contains the recent audio interval ending at `t`, whereas its
eye/body snapshot describes the endpoint. With sensory delay, the entire
dictionary is delivered later. Do not label every sample in an audio block with
the image endpoint time.

`SensorContext.refresh()` copies `MjData`, then calls `mj_forward()` on the copy.
This refreshes body/camera transforms and derived sensors after integration
without advancing physical time or changing the authoritative trajectory. Merely
reading `cam_xpos`, `cam_xmat`, `xpos` or `sensordata` immediately after `mj_step()`
is insufficient: those derived fields can still describe the prior evaluation.
The copied actuation evaluator also preserves activation state while computing
endpoint sensor values.

Repeated observation reads are cached/idempotent on the snapshot path: they do
not step physics, advance muscles, emit another sound interval or consume another
delay slot. The legacy native path retains its older behavior for compatibility;
use full biology, `snapshot_sensors=True`, or the custom Unreal provider when
endpoint snapshots are required.

### Unreal frame transaction

The vision provider places body meshes and both cameras from the same refreshed
copy inside SPEAR's `begin_frame` / `end_frame(single_step=True)` transaction.
Capture buffers are single-buffered and RGB arrays are copied from returned BGRA
storage. Same-frame binocular capture only helps if the supplied MuJoCo state is
itself current; the snapshot refresh is essential.

The physical frame is right-handed with +X forward, +Y left and +Z up. At the
Unreal boundary, positions are multiplied by 100 and Y is reflected. Body
rotation uses `S R S`, `S = diag(1,-1,1)`. Camera optical axes require their own
conversion; do not substitute the body rotation conversion for camera rotation.

The exporter already accounts for the Unreal legacy OBJ importer's Y reflection.
Enabling another scene/unit conversion in the importer produces mirrored or
incorrectly scaled geometry.

### Delay queues and resets

`motor_delay` and `sensory_delay` are non-negative counts of **whole public
steps**, not milliseconds. At the current rate, `2` means 20 ms. Motor history is
primed with zero commands. Sensory history is primed with reset observations.
Reset clears the queues, actuation state and acoustic history, refreshes the
initial snapshot and returns copies of the initial observations.

These are engineering timing controls, not validated developmental neural-delay
models. Do not silently convert renderer latency or GPU copying time into a
biological sensory delay.

## 6. Using the environment for learning

### Minimal native use

```bash
MUJOCO_GL=egl .venv-unreal/bin/python - <<'PY'
import numpy as np
from blight_world import BlightWorldEnv

env = BlightWorldEnv(width=96, height=96, snapshot_sensors=True)
try:
    observation, info = env.reset(seed=1401)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    print(env.action_space)
    print({name: (array.shape, str(array.dtype))
           for name, array in observation.items()})
finally:
    env.close()
PY
```

A zero-action example proves the interface, not upright balance. The neutral base
world has a free body; it does not automatically supply the support used by demo
subclasses.

### Full senses with Unreal

```python
import numpy as np
from blight_world import AuditionConfig, BlightWorldEnv
from blight_world.unreal_vision import make_unreal_vision_provider

env = BlightWorldEnv(
    width=320, height=240,
    biology_profile="full",
    audition=AuditionConfig(backend="steam_audio"),
    vision_provider_factory=make_unreal_vision_provider,
)
try:
    observation, info = env.reset(seed=1401)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
finally:
    env.close()
```

Use the basic-room asset manifest for this example. The code does not contain a
network; replacing the zero action with a network output is the connection point.
In an actual experiment, define which fields are inputs, which are supervision
targets, and the episode/loss/reward outside the neutral world.

### Observation contract

For the pinned standard body, torque actuation and the full profile above:

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `eye_left`, `eye_right` | `(240,320,3)`, `uint8` | RGB images; height/width are configurable. |
| `observation` | `(621,)`, `float64` | Full concatenated proprioception; this key is not named `proprioception`. |
| `touch` | `(15996,)`, `float64` | Flattened MIMo tactile force-vector outputs for the configured skin sampling. |
| `vestibular` | `(6,)`, `float64` | MIMo's configured head acceleration/rotation measurements. |
| `audition` | `(2,480)`, `float32` | Left/right ear waveform, 48 kHz, ten milliseconds. |

Legacy defaults return just `observation` with 93 joint-position values and the
two RGB arrays, 96×96 by default. `biology_profile="full"` enables the full MIMo
proprioception configuration, touch and vestibular processing; audio remains
opt-in. Custom modalities, age/body changes or a different actuation model can
change dimensions: use the instance's `observation_space` and `action_space`,
not copied constants in a learner.

The 93 default values are the named MIMo robot-joint positions, not a copy of the
entire world/root qpos vector. There is no package console-script entry point or
Gym registration for this world; construct `BlightWorldEnv` directly as shown.

`SimpleProprioception.sensor_outputs` and `sensor_names` expose component/name
mapping. Components are concatenated by sorted output key, not by the order of
the requested component list. Positions are always included; optional components
are velocity, torque, limits and actuation. Angles are radians, angular velocity
is radians/s and torque is N·m. Torque sensor vectors and actuator output torque
are different quantities. Do not treat the 621 values as 621 independent joints.

MIMo proprioception here directly reads simulated quantities. It is not a
biophysical muscle-spindle, tendon-organ or afferent-spike model. Likewise,
tactile/vestibular outputs and RGB/audio observations do not automatically include
a neural transduction model. Acuity and foveation are configurable existing MIMo
filters, disabled by default; selecting `full` does not turn them on.

For visual–proprioceptive association, pairs of eye images, motor commands and
body measurements can be collected directly. If a task asks a network to infer
body position, do not inadvertently feed its exact target joint positions into
the same input. Simulator truth and controller diagnostics can be useful for
supervision/evaluation without being learner observations.

The base world returns reward `0.0`, has no success/failure task and does not
truncate itself. Define episode duration and a learning objective explicitly.
No throughput or long-running training guarantee follows from the short videos.

## 7. Bodies and motor controllers

The default `SpringDamperModel` exposes **90 bounded float32 motor actions**.
These are normalized torque commands, not desired joint angles or a high-level
"look at target" command. A control of one corresponds to the actuator's gear
scale subject to its configured bounds; some bounds are asymmetric. Passive
joint springs/dampers remain part of the physical model.

Resolve motors by names such as `act:head_swivel`, `act:head_tilt`,
`act:left_eye_horizontal`, `act:right_eye_vertical` and the limb actuator names.
`motor_control.py` resolves actuator/joint IDs, qpos/dof addresses and limits.
Do not assume an actuator index is a qpos index, or that all of MIMo's coupled
joints map one-to-one to motors. The coupled hip helpers exist for a reason.

The self-inspection and street controllers translate reference poses into bounded
torques using state feedback and bias terms. The older static-support and
belly-crawl controllers instead issue open-loop normalized torque sequences.
Where a controller accesses privileged MuJoCo state, that access is part of its
scripted demonstration; it does not define a biological policy input.

### Optional muscle interface

Pass `actuation_model=CorrectedMuscleModel` from `blight_world.biology` to request
the corrected MIMo muscle model. Inspect that instance's action/observation
spaces: the pinned body has 180 opposing-muscle activation controls in `[0,1]`,
a different interface from the 90-motor torque example. Full biology alone does
not select muscles, and the existing named torque-controller helpers are not
drop-in muscle controllers.

The correction separates setting requested activation from advancing activation
dynamics. Activation advances exactly once per 5 ms physical substep. In the
upstream path, setting an action also advanced activation, giving three updates
for two physical substeps. With the cited 10 ms time constant and a unit request,
two 5 ms updates should produce 0.75 from zero, not 0.875. Existing tests check
this timing and verify sensor evaluation does not mutate the muscle trajectory.
Do not revert to the uncorrected upstream class merely because it imports.

### Limb-inspection controller

`tools/export_self_examination_demo.py` contains both `SelfExaminationEnv` and
the specific `SelfExaminationController`. A root weld holds the body while
bounded motor commands raise a limb and rotate head/eyes toward it. The camera
poses still follow physical joints. The controller's targets, hold intervals,
joint bounds and visible evidence points are defined in that tool.

To change this example, adjust the sequence/controller there. Do not add another
camera-control framework or move its gaze targets by directly overwriting the
learner's cameras.

### Corrected street controller

`StreetRunEnv` owns external forces/torques on the hip for balance, height and
forward pace; only reset stages the initial pose. `StreetRunController.action()`
drives the existing bounded limb motors. There is no continuous qpos replay.

The reference is a symmetric engineering retarget of Fukuchi et al.'s adult
RBDS001 2.5 m/s running data, scaled to MIMo and slowed to a 0.60 s stride and
approximately 1 m/s. It includes body-pitch correction and limited early-stance
ankle adjustments for the rigid sole. The original source is not measured infant
running. The embedded reference value `STANCE_FRACTION=0.396` is a source target;
it does **not** guarantee the actual contacts last 39.6% of a stride.

The corrected recorded result had:

- Continuous heel-to-sole/forefoot-to-toe loading, rather than persistent toes.
- Head/eye lateral tilt within approximately ±1.7°.
- Actual loaded stance about 49–50% of the stride, with minimal flight.
- Roughly 72% of bodyweight carried by the external vertical assistance.
- Narrow central swing clearance, about 2.2 mm at its minimum in the inspected
  contact-defined window. This is not evidence of rough-terrain robustness.

The demo changes sagittal passive damping to hip 0.40, knee 0.18 and ankle
0.06 N·m·s/rad. It uses root height acceleration support and native mass-matrix
acceleration feedforward for the six sagittal leg motors, including root-height
coupling. Original motor limits remain. Report the assistance and damping when
using these trajectories; do not call them unassisted biomechanics.

## 8. Vision and Unreal assets

The body is rendered as closed meshes built from the current physical primitive
geometry, grouped by MuJoCo body. It is not an imported production skeletal skin
with a separate Unreal animation controller. The physical eye cameras supply
their own world poses and configured optics. The provider checks the imported
manifest against body/geom count, names, sizes and local transforms before use.

The current eye output is tone-mapped 8-bit sRGB, not raw retinal irradiance.
The provider uses FXAA without temporal image history, disables motion blur and
depth of field, fixes exposure, and sets a 0.5 cm (5 mm) near clipping plane.
Configured vertical FOV is converted to Unreal's horizontal FOV using the image
aspect ratio. The room uses finite interior area lights in place of MuJoCo's
directional lights so the closed ceiling does not block illumination; the street
has its own directional/point-light profile. These are explicit rendering
approximations, not a photometric calibration between the two engines.

`tools/unreal_prepare_assets.py` runs in external Python. It compiles the basic
or selected MJCF template and writes OBJ meshes, materials, texture images and a
manifest. It currently handles the implemented primitive shapes; it is not the
general SPEAR arbitrary-world collision-export pipeline discussed in the initial
architecture research.

`tools/unreal_import_assets.py` runs **inside Unreal's editor Python**. It imports
the prepared files into the manifest's `/Game/...` namespace, creates materials,
disables generated mesh collisions and makes sure the empty Room runtime map
exists. It can replace previously imported assets in that namespace. Keep the
authoring map separate and save user edits before imports.

Runtime starts the Room map and spawns the body's movable actors, lights and
cameras. Both Unreal Chaos interaction and independent Unreal body animation
are outside this model. The street provider adds a spectator view for humans,
stored separately from the agent observations.

### Regenerate basic and street asset manifests

```bash
.venv-unreal/bin/python tools/unreal_prepare_assets.py \
  --output artifacts/unreal/assets

.venv-unreal/bin/python tools/unreal_prepare_assets.py \
  --scene-template config/street_run.xml.template \
  --content-root /Game/StreetRun \
  --output artifacts/unreal/street_assets
```

These commands prepare external files; they do not import them into Unreal.
For each prepared manifest, run the following editor command with the matching
path. Use the basic manifest first when constructing a new workspace.

```bash
BLIGHT_UE_ROOT=/home/vishnu/Unreal/UE_5.5.4
BLIGHT_PROJECT="$PWD/ue_project/BlightWorld/BlightWorld.uproject"

VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
BLIGHT_UNREAL_ASSET_MANIFEST="$PWD/artifacts/unreal/assets/manifest.json" \
  "$BLIGHT_UE_ROOT/Engine/Binaries/Linux/UnrealEditor" "$BLIGHT_PROJECT" \
  -unattended -vulkan -renderoffscreen -graphicsadapter=1 \
  -ddc=InstalledNoZenLocalFallback \
  -ExecutePythonScript="$PWD/tools/unreal_import_assets.py"
```

Repeat with `artifacts/unreal/street_assets/manifest.json` for street assets.
Manifest source-file paths are absolute. Recreate manifests after relocating a
checkout, even if imported `.uasset` files are already present. A geometry edit
requires both regeneration and import; changing only the visual asset or only
the MJCF breaks the agreement.

Changing Python controllers does not require reimporting unchanged geometry or
rebuilding C++. Changing actor C++ requires a project build. Changing only an
exported video filename requires neither.

## 9. Audio

Audio is opt-in through `AuditionConfig`. Both backends produce a C-contiguous
float32 array with channels first. At 48 kHz, each 5 ms substep has 240 samples
per ear; the public 10 ms observation has 480. The native 24 kHz profile has
120/240 respectively. Steam Audio accepts only 48 kHz in this implementation.

### Backend distinctions

| Setting | Native Python backend | Standalone Steam Audio backend |
| --- | --- | --- |
| Constructor | `AuditionConfig()` / `quality_48k()` / `training_24k()` | `AuditionConfig(backend="steam_audio")` |
| Rates | 24 or 48 kHz | 48 kHz |
| HRTF | Packaged adult SADIE II H10 | SDK built-in reference-head HRTF |
| Room processing | Implemented first-order room reflections and separate late tail | SDK geometry simulation and convolution |
| Geometry | Native acoustic scene/ray interface | Separate scene built from MuJoCo primitives; dynamic body transforms updated |
| Output interpretation | Numerical pressure convention with engineering source calibration | Source-scaled pressure proxy, not measured infant ear pressure |

The native backend's room model is the original **5×5×2.5 m shoebox**; it is not
automatically fitted to an arbitrary edited scene or the outdoor street. Use the
Steam Audio geometry path for the current street workflow.

The native backend's H10 provenance and old timing reports do not validate the
Steam Audio backend automatically. In the Steam backend, reflection processing
is enabled when either configured reflection/reverberation flag requests it;
these flags do not create the native backend's two independent room algorithms.

Sources can be fixed (`StaticAttachment`), follow a body (`BodyAttachment`) or
follow a site (`SiteAttachment`). Register a named `AudioSourceSpec`, then call
`env.audition.play(source_name)`. `sensor.py`, `sources.py` and the clip library
define the source/scheduling interface. Contact impacts are detected from
physical motion and solved contacts; they are not automatically equivalent to
one perceptual footstep per impact event.

Material classification currently uses known geometry/body names and roots:
MIMo is soft, the known ball is rubber, the known table/block are wood, the known
cylinder is plastic, and unmatched geometry defaults to hard. Selecting a visual
Material Name in the editor does not infer measured acoustic properties for a
new object. This is a small explicit engineering material model, not an automatic
recognition system for arbitrary imported art.

`ContinuousSoundConfig()` enables the existing sliding-contact scrape channel.
It is disabled by default, including in the street-running exporter. Rolling,
clothing and air sounds are not implemented. The normal limits are 64 registered
sources, 256 scheduled events and eight active clip voices, with a separate
bounded pool for continuous emissions. Capacity limits matter if an experiment
exceeds them; do not redesign the pools merely to make one demo.

### Causal sound and filter history

The Steam SDK's reflection delay is relative to direct arrival. This adapter
applies the source/listener distance flight delay once to the shared input for
**both** direct and reflection branches. At the pinned SDK speed of 340 m/s:

```text
direct arrival     = emission time + direct path length / 340
reflected arrival  = emission time + reflected path length / 340
```

The measured reference binaural filter latency is an additional 60 samples;
it is not propagation distance or a modeled neural delay. Preserve convolution
history/tails across blocks, source pauses and resumable continuous streams.
Resetting effects or discarding partial blocks to align to a control decision
changes what the agent hears.

Audio uses start-of-substep poses and contact-event timing for the rendered
interval. Endpoint contact recomputation is for observation consistency, not an
excuse to retimestamp the earlier physical impact. Repeated `_get_obs()` calls
read the current audio cache rather than synthesizing more samples.

### Footsteps in the corrected video

The four `footstep_concrete_*.wav` clips are actual barefoot running impacts from
sturmankin's CC0 concrete recording, cropped from the public high-quality MP3
preview into mono 48 kHz PCM. The source was recorded **indoors in a basement**;
it is not an outdoor or infant-foot recording. Source gain is uncalibrated.
See [FOOTSTEPS.md](../src/mimo_audition/assets/audio/FOOTSTEPS.md) for source,
license, crop times, fades and processing details.

The street exporter registers these clips explicitly. They are not automatically
selected by the general material impact bank. A per-foot air-gap latch accepts
one strongest landing onset and suppresses toe/heel contact chatter; variants
rotate between landings. The eight-second corrected recording contained 27
alternating events matched to physical contact onset.

The public pressure arrays are not automatically clipped or gain-controlled.
For the showcase WAV/MP4, a constant monitor gain scales the rendered waveform
to a listenable peak. Do not use that normalized listening file as calibrated
scientific pressure. Signal/clip/timing verification was performed, but direct
listening was unavailable during the redo; perceived timbre was not established
by listening through the assistant's tools.

## 10. Editing the world

The authoring map is `/Game/Maps/WorldEditor`. Launch it on the remote desktop:

```bash
BLIGHT_UE_ROOT=/home/vishnu/Unreal/UE_5.5.4
BLIGHT_PROJECT="$PWD/ue_project/BlightWorld/BlightWorld.uproject"
"$BLIGHT_UE_ROOT/Engine/Binaries/Linux/UnrealEditor" "$BLIGHT_PROJECT" \
  -graphicsadapter=1 -ddc=InstalledNoZenLocalFallback
```

Select an actor in the Outliner and press **F** to frame it. Use the actor's
Blight properties for physical dimensions and metadata:

- Box/plane dimensions are **half extents in metres**; cylinder height is half
  height; sphere/cylinder radius is metres. Actor location is Unreal centimetres.
- Keep actor scale at `(1,1,1)`; change the SI shape properties instead.
- Mass zero exports a static primitive; a positive mass exports a free MuJoCo
  body. The visible preview never owns dynamics.
- Give each actor a unique Physics Name. Material Name must refer to a material
  defined in the source room. Friction values must be finite and non-negative.
- Save and close the editor before an export/SPEAR process uses the project.

Export the edited metadata to the separate scene template:

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
BLIGHT_UNREAL_WORLD_MODE=export \
  "$BLIGHT_UE_ROOT/Engine/Binaries/Linux/UnrealEditor" "$BLIGHT_PROJECT" \
  -unattended -vulkan -renderoffscreen -graphicsadapter=1 \
  -ddc=InstalledNoZenLocalFallback \
  -ExecutePythonScript="$PWD/tools/unreal_edit_world.py"

.venv-unreal/bin/python tools/unreal_prepare_assets.py \
  --scene-template config/edited_room.xml.template \
  --content-root /Game/EditedWorld \
  --output artifacts/unreal/edited_assets
```

Then run the import command from section 8 with the edited manifest. Set
`BLIGHT_UNREAL_ASSET_MANIFEST` to its absolute path and pass
`scene_template="config/edited_room.xml.template"` when constructing the
environment. This supplies the same exported model to dynamics, visual mapping
and acoustic geometry.

`BLIGHT_UNREAL_WORLD_MODE=populate` creates the initial editable primitive room
on a fresh WorldEditor map. It refuses to replace existing primitive actors, and
requires the baseline materials already imported. Population needs the real
Vulkan editor viewport; `-nullrhi` was not suitable. Do not repopulate a map to
"fix" another person's edits.

The exporter retains the MIMo body and source asset definitions while replacing
the room primitives. It is deliberately limited: arbitrary skeletal meshes,
concave collision decomposition, mass/inertia inference for arbitrary art,
caregiver physical interaction and a generalized acoustic asset pipeline are
not provided by this small authoring actor.

## 11. Video tools and their outputs

All Python tools are run from the root with the appropriate environment and
`MUJOCO_GL=egl` for native MuJoCo rendering. The two `unreal_*` editor scripts are
the exception: they run through Unreal's `-ExecutePythonScript`.

| Tool | Role and status |
| --- | --- |
| `export_self_examination_demo.py` | Supported limb inspection; `--renderer unreal` is the foundational Unreal example, default renderer is native MuJoCo. |
| `export_street_run.py` | Current Unreal street sequence with two eyes, spectator, recorded contact footsteps and motion arrays. |
| `export_ball_throw.py` | Existing native assisted throwing exporter; `--smoke` provides its bounded native check. No Unreal renderer option. |
| `export_three_object_walk_pick_throw.py` | Older composed native scenario with explicit assistance/fixtures; inspect its flags and labels before claiming locomotion/manipulation. |
| `export_continuous_embodied_demo.py` | Older continuous native embodied showcase. Its implementation is not the current street renderer. |
| `export_binocular_physics_demo.py` | Native binocular physical-interaction showcase; several controlled fixtures, not a learned policy. |
| `export_audition_demo.py` | Native audio/room showcase and encoded artifact preparation. |
| `unreal_prepare_assets.py` | External Python MJCF-to-visual-asset preparation, with `--model` or `--scene-template`. |
| `unreal_import_assets.py` | Editor-only import of a prepared manifest. |
| `unreal_edit_world.py` | Editor-only `populate`/`export` of the primitive authoring map. |
| `generate_impact_bank.py` | Generates the existing engineering impact bank; not the real barefoot recordings. |
| `prepare_audition_assets.py` | Converts a supplied SOFA HRTF asset to the native packaged representation. |
| `profile_audition_world.py` | Existing native world/contact acoustic profile. |
| `profile_audition_scrape.py` | Existing live scrape fixture profile. |
| `benchmark_audition_streaming.py` | Long native-audio timing benchmark; not necessary to produce a video. |

Use a tool's `--help` to select its own output/smoke options. Do not assume all
exporters share flags or artifact schemas. Reuse the two short Unreal workflows
first; the historical profiling scripts are not prerequisite infrastructure.

Several older tools default to `/home/vishnu/neuroips_outputs/...`, outside this
repository. Pass an explicit local `--output` when using self-examination,
continuous embodied, ball throw or three-object tools. The audition and binocular
physics exporters instead use `--output-directory`. The native self-inspection
output is named `self_examination.mp4`, unlike the Unreal `limb_sequence.mp4`.
Native ball throw records five seconds after hold preparation; continuous
embodied export runs 41.8 seconds. The three-object export's controller hook is
not fully generic: its audit code currently requires the supplied
`ConstraintAssistedThreeObjectController` type. `--plan-only` inspects its plan
without claiming that it rendered a video.

### Street recording data

`motion.npz` contains actual state, not just intended reference poses:

- `time`, `qpos`, `qvel`, `ctrl`, `pelvis_wrench`, `contact_normal`: samples at
  200 Hz physics resolution, including warmup.
- `pose_time`, `body_pos`, `body_rot`, `foot_geom_pos`, `foot_geom_rot`,
  `camera_rot`: refreshed poses at 100 Hz.
- Body, joint, actuator and foot-geometry names/addresses/sizes plus
  `recording_start_time_s`: mappings and the video start offset.

Subtract the recording start offset before comparing absolute simulation times
with MP4 time. Contacts are solved-substep data; copied poses describe completed
endpoints. `footfalls.json` records accepted audio-trigger events with time,
position/energy and clip identity. The MP4 is presentation, not the full raw
training observation archive: it contains resized/composited camera pixels and
listening audio. The exporter does not save every enabled modality as a general
dataset.

## 12. Lessons from this conversation and implementation

### The user's completion criterion is the requested visible behavior

The user repeatedly rejected replacing a simple working world with more harnesses,
abstractions and reports. The initial target was a clearly visible limb-by-limb
inspection. The later target was a believable road-running example with footstep
sound. A successful build, valid arrays or an attractive room do not substitute
for those sequences.

Start with the existing example and the requested scene/action. Inspect the actual
eye streams and whole sequence. Make the smallest causally justified correction
when a failure appears. Do not turn these lessons into an instruction to build
more infrastructure or run every historical benchmark for every change.

### Why the first road run was wrong

| Observed failure | Confirmed cause / useful evidence | Correction and lesson |
| --- | --- | --- |
| Persistent right lean in body and eye streams | Root angular damping of 10 produced alternating saturated torques at the 5 ms step; late head tilt averaged roughly 39°. | Reduce the unstable damping to 2. Check angular state and applied torque, not just a spectator image. Do not level the eyes cosmetically. |
| Toes only, heels high, legs trailing | Original kinematic references demanded an ankle angle around 59° against about a 32° limit and ignored body-pitch/sole geometry; actual angles did not track the intended poses. | Replace infeasible references and compare actual hip/knee/ankle trajectories and ankle position relative to the pelvis. Toe contact itself is not abnormal; persistent toe dragging was. |
| Recovering foot caught the floor early | Pelvis height fell roughly 19–26 mm below its needed trajectory during the rapid phase; a swing toe hit early and friction stalled recovery. | Coordinate body height with actual foot geometry and reference acceleration. More horizontal pulling or a different camera would hide the symptom. |
| Large support-force spikes | Extending forced-flat ankle anchors too far into stance introduced large height acceleration; required support rose beyond its cap. | Restore the measured later-stance angles. Smooth-looking positions can still imply infeasible accelerations. |
| Remaining leg tracking lag and short heel-only stance | PD/bias compensation omitted the substantial inertial torque of the timed leg motion and root-height coupling. | Add native mass-matrix acceleration feedforward for the actual bounded motors. Keep the original torque limits and assess contacts again. |
| Pan-like impact timbre | Generic synthesized material transients were a poor footstep source; per-geom impacts could chatter. | Use an appropriate real barefoot-concrete asset and one physical landing event per foot. A generic impact detector is not already a footstep sound designer. |

The corrected video was compared to measured adult motion and inspected over its
full duration, including consecutive complete strides. Joint-state agreement
between native verification and the final Unreal recording was exact at matched
times. This verifies transfer consistency, not infant biomechanics. The high
external support, adjusted damping, different stance duty and low clearance must
remain visible in any description of the result.

### Scientific timing lessons that a good-looking video cannot settle

1. **Setting an action is not passage of biological time.** The muscle model
   must update activation once per physics substep, regardless of how often the
   public action setter or observation reader is called.
2. **Post-integration transforms need refresh.** `mj_step` does not promise all
   derived fields describe its final integrated qpos/qvel. Refresh a separate
   observation copy so sensing cannot perturb the reference trajectory.
3. **Both sound branches need absolute travel time.** Steam Audio reflections
   are relative to direct arrival. Delay the shared input once, rather than only
   the direct branch or both branches twice.
4. **Preserve streaming tails.** A substep boundary, policy pause or resumed
   scrape is not a reason to reset filters, discard samples or replay a tail.
5. **Image freshness and physical correctness are distinct.** Two mutually
   synchronized images can both depict an old state or an impossible motion.
6. **Keep diagnostics distinct from perception.** Object IDs, world positions,
   contacts and scripted-controller truth are useful evidence; they are not
   automatically sensory inputs for the learner.

### Practical setup pitfalls already encountered

- **Python environment mismatch:** use `.venv-unreal`; Unreal's editor Python is
  separate. An extension built for Python 3.11 does not become a Python 3.12
  extension by changing which interpreter imports it.
- **Dependency mismatch:** the pinned SPEAR scientific dependencies needed the
  two documented local adjustments to coexist with MIMo. Installing SPEAR's
  full pipeline extra would also request a different MuJoCo version; it is not
  needed for this primitive world.
- **Incomplete MIMo wheel:** use the pinned editable source checkout. The original
  MIMo 2.0.0 wheel omitted modules/assets required here.
- **Unreal Python module shadowing:** do not rename `ue_project` to a root folder
  named `unreal`; that interfered with SPEAR's detection of Unreal's Python API.
- **Zen cache loop:** a client timeout during repeated DDC recovery is not a
  controller bug. The working config uses the existing file-cache fallback.
- **Wrong renderer entry point:** native `--smoke` passing is not an Unreal
  capture. The self-inspection CLI defaults to native until `--renderer unreal`.
- **Stale scene manifest:** model and asset metadata must agree. Reprepare and
  import the correct scene instead of weakening the consistency checks.
- **Mirrored geometry or bad eye orientation:** respect the single metre-to-cm
  conversion, single Y reflection and distinct camera-axis mapping.
- **Editor-only operations in external Python:** `import unreal` in the editor
  scripts needs the engine's Python API. Execute those scripts through the editor.
- **Hidden assistance:** root welds, external forces, contact fixtures and grip
  constraints must be described. They cannot stand in for learned balance or
  natural infant movement.
- **Reviewing only the start:** the rejected video looked less wrong during its
  first steps. Inspect later motion and whole cycles; compare actual data while
  the body is moving, not just reset poses.
- **Claiming audio was heard:** waveform/format/timing checks do not establish
  perceived sound quality. State when direct listening is unavailable.
- **Losing accepted media:** use fresh output directories. Several older artifact
  folders are rejected/diagnostic runs, not the current reference.
- **Killing unrelated work:** this is a shared GPU machine. Stop only processes
  launched for the task. Do not kill MATLAB, other editors or training jobs to
  free memory without their owner's authorization.

### Additional regressions exposed while preparing this handover

- **Initialize subclasses through the current base class.** The older crawl
  fixture called `MIMoEnv.__init__` directly and bypassed Blight World's new
  delay/cache/snapshot fields. Adding only the first missing field would have
  exposed another. Passing its unchanged packaged scene through the existing
  base constructor restored initialization while preserving physical staging.
- **Do not freeze installation paths as scene physics.** The old crawl scene
  fingerprint contained `/home/vishnu/coding_proj/blight_world/.venv/src/mimo`.
  Moving to this checkout changed its hash despite identical scene content.
  Normalize only the three resolved MIMo asset references back to their template
  placeholders before that exact check; retain all other scene and trajectory
  comparisons.
- **Distinguish physical contact identity from an acoustic anchor.** The mirrored
  landing profile tried to identify the foot from a static sound attachment.
  The actual impact event identified the correct foot, and physical/acoustic
  left/right measurements were already correct. The profile now records the
  physical event body without changing where its sound is rendered.
- **Explain a changed fingerprint before updating it.** Earlier commit `521db47`
  moved native impact sound from a moving body attachment to the projected
  contact position. The older audition export retained the previous exact
  pressure hash/peak. Replaying only the historical attachment method reproduced
  those old values exactly; current physics/contact/scrape checks remained
  unchanged. This justified updating the expected signal, not relaxing tolerance
  or reverting the corrected runtime placement.
- **Compare asset behavior, not optional-default spelling.** The older basic
  manifest omitted zero emission. Fresh preparation writes `emission: 0.0`, and
  the importer already treats omission as zero. All regenerated meshes/textures
  matched. A metadata spelling difference was not a reason to rebuild rendering
  infrastructure or replace the imported body.

### How the team should work

The user's requested team has a lead, researcher, coder, debugger and validator.
Give each a concrete bounded task. The debugger establishes the cause of an
actual failure before a coder changes it; the validator checks the requested
result independently. Shared files need clear ownership while edits are in
flight. Avoid redundant parallel investigations, simultaneous source edits or
several renderers racing for the same project/GPU. Communication is useful when
it carries a finding, evidence, an assigned boundary or a blocking dependency.

Current user instructions take precedence over historical suggestions in
`MEMORY.md`. Do not treat "validate" as authorization to redesign the project,
and do not ask again for work the user already authorized. Stop when the requested
deliverable is complete.

## 13. Validation and its limits

The existing suite is configured by `pyproject.toml` under `tests/`. Run it in the
installed combined environment:

```bash
MUJOCO_GL=egl .venv-unreal/bin/python -m pytest -q
```

It includes environment reset/action/sensor checks, native rendering and media
helpers, supported controller fixtures, audio DSP/contact/room/scrape behavior,
muscle/snapshot/delay corrections and tests against the actual Steam Audio SDK
when that library is installed. It is not a substitute for an Unreal video run;
the usual suite does not launch and inspect a complete Unreal eye sequence.

For a newly requested video, the useful acceptance evidence is the complete
requested action sequence, physical/eye consistency, relevant actual state and
contact data, and audio interval/clip alignment. Decoding all video frames only
establishes file integrity. Inspect representative frames across the full clip
and consecutive frames through the critical transitions; listen when a playback
tool is available. Record any support, approximation or unverified modality.

### Handover validation on 10 September 2026

The full existing suite and the following independent reproduction checks
completed against the prepared working tree:

| Check | Result |
| --- | --- |
| Full existing test suite | Exit 0; 175 passed, 0 failed, 0 errors, 0 skipped in 871.70 seconds (14 minutes 31 seconds). All previously failing regression checks passed. |
| External environment `pip check` | No broken requirements. |
| Fresh Unreal self-inspection using checked-in config | Exit 0; 10.000 seconds, 500 decoded frames, 50 fps, 640×272. Both eyes show all four targets in order and return to neutral. 100 encoded samples plus four full-resolution holds inspected. Silent by design. |
| Fresh Unreal street run using checked-in config | Exit 0; 8 seconds, 400 decoded frames, 50 fps, 960×816. Actual joint state, contacts, body poses and eye rotations exactly match the verified redo. |
| Street audio reproduction | All 27 footfall events and the WAV match the verified redo exactly. Encoded audio has 384,000 stereo samples at 48 kHz, finite/unclipped and aligned with the WAV. No direct-listening claim. |
| Fresh basic asset preparation | Exit 0; 89 geoms in 60 body meshes, six textures. All 67 generated asset files match existing bytes; the only non-path manifest difference is explicit default-zero emission. |
| Fresh street asset preparation | Exit 0; 141 geoms in 66 body meshes, five textures. All 72 generated asset files match existing bytes; manifest matches after output-path normalization. |
| SPEAR compatibility patch | Matches the exact local source diff; reverse applicability checked on the patched checkout. |

The fresh videos are in `artifacts/unreal_self_examination_handover` and
`artifacts/unreal_street_run_handover`; regenerated asset checks are in
`artifacts/handover_asset_check`. Accepted earlier media remain intact. The new
renders also exercised the relative SPEAR plugin path and permanent DDC option.

The final suite ran once against the frozen source, with no exclusions, using
`MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 .venv-unreal/bin/python -m pytest -q`.
Its 15 warnings are dependency deprecations: one CacheToolsUtils/`pkg_resources`
warning and 14 Matplotlib/Pyparsing warnings. The validation fixes and their
confirmed causes are documented above; physical assertions and tolerances were
retained. Local file links in this guide, README and MEMORY resolved, and the
final whitespace check passed.

Long native-audio timing results in the older `docs/*profile.md` files describe
their dated fixtures and software combination. They are not measurements of the
full Unreal + touch + audio + learner workload. No new real-time, distributed
training, rough-terrain, long-horizon or biological-validity claim is made here.

## 14. Version-control boundaries

The organization intentionally keeps the working source paths. The small
preparation changes are a canonical guide/README entry, the preserved SPEAR
compatibility patch, a repository-relative Unreal plugin directory and the
confirmed DDC launch setting. No Python module was moved merely to rearrange
the tree.

| Include as source/input | Keep generated or external |
| --- | --- |
| `src/`, packaged WAV/HRTF assets and their attribution | `.venv/`, `.venv-unreal/`, Python caches and egg-info |
| `tests/`, existing `tools/`, package/build requirements | `/third_party/` clones, SDK files and compiled dependencies |
| `config/`, `docs/`, README/MEMORY and `patches/` | `/artifacts/`: all recordings, diagnostic arrays and generated import manifests |
| Unreal `.uproject`, `Source/`, `Config/`, `Content/` | Unreal `Binaries/`, `Intermediate/`, `Saved/`, `DerivedDataCache/` |

At inspection, generated artifacts occupied about 181 MiB and required Unreal
Content about 12 MiB. Ignoring artifacts preserves them locally and prevents
accidental staging of rejected videos, logs and machine-specific manifests.
Do not delete them as part of a commit preparation. The two `.umap` files and
all required imported `.uasset` content remain eligible for the commit. A fresh
checkout still generates its absolute-path asset manifests as described above.

The ignored SPEAR checkout contains a deliberate dependency metadata change;
`patches/spear-python-compat.patch` makes it reproducible. Apply it once to the
pinned fresh SPEAR source. Native submodule `BUILD/` products do not belong in
that patch. MIMo's pinned source checkout is unchanged.

Before committing, `git status --short` and `git diff --check` should identify
the intended source/documentation changes without new generated output. A clean
whitespace check is not a functional test. Nothing in this guide requests a
commit, stage, push, deletion or unrelated refactor automatically.

Preserve the existing package/module names and script entry points. Keep future
scene data under `config/` or packaged assets, reusable simulation code under
`src/`, executable demonstrations under `tools/`, and generated recordings under
`artifacts/`. A new controller does not need a new runner framework. Update the
guide if its actual command, observation contract, assistance or dependency
requirement changes.

This preparation does not stage, commit or push changes. The working tree already
contained the preceding implementation and user work before documentation began.
Review the complete intended diff before making the commit, including new Python
modules, tests, WAV assets, Unreal source/config/content and this guide.

## 15. References and asset attribution

These links support the installed components and the engineering choices already
made. Their presence is not a claim that every upstream example is integrated.

- [Pinned MIMo source](https://github.com/trieschlab/MIMo/tree/040b0ae4914cbfb26afdf830aa81775b90922f3f)
  supplies the body, motor interfaces and existing body-sensor processing.
- [Pinned SPEAR source and setup](https://github.com/spear-sim/spear/blob/38b6323d28c4db3e53d70b9c667770e3c2976d42/docs/getting_started.md)
  supplies the renderer transaction, shared-memory captures and native/Unreal
  integration. Its stock MuJoCo example is not this complete body adapter.
- [MuJoCo 3.3.0 simulation documentation](https://mujoco.readthedocs.io/en/3.3.0/programming/simulation.html)
  explains stepping, forward computations and derived-state handling.
- [Steam Audio C API guide](https://valvesoftware.github.io/steam-audio/doc/capi/guide.html)
  describes numerical effects and caller-owned audio buffers. The local ABI is
  pinned to 4.8.1; do not load an arbitrary newer shared library.
- [Fukuchi et al. running dataset](https://figshare.com/articles/dataset/4543435)
  and [paper](https://peerj.com/articles/3298/) provide the adult motion reference
  used in the corrected street gait (CC BY 4.0; RBDS001, 2.5 m/s, engineered
  retarget and timing changes declared above).
- [Barefoot concrete running by sturmankin](https://freesound.org/people/sturmankin/sounds/272287/)
  supplies the four CC0 footstep crops. Keep the local
  [asset attribution](../src/mimo_audition/assets/audio/FOOTSTEPS.md) with them.
- [Native audio asset notes](../src/mimo_audition/assets/audio/README.md) and
  [HRTF provenance](../src/mimo_audition/assets/hrtf/PROVENANCE.md) describe the
  original generated transients and adult H10 data separately from Steam Audio.

Unreal installation and engine content remain subject to Epic's distribution
terms; the ignored engine/SDK trees are not part of this repository. Do not infer
that arbitrary third-party art, voice or character assets are licensed for a
training corpus merely because the engine can render them. The current primitive
world and explicitly documented audio sources avoid depending on an unreviewed
asset-store collection.

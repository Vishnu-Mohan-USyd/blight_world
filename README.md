# Blight World

A deliberately small MIMo 2.0.0 world containing a static room, a low table,
and three visible collidable objects. Binocular RGB vision and minimal
proprioception are always present; optional binaural audition can be enabled
without changing the physics clock, RGB images, or proprioception. Touch and
vestibular sensing remain disabled. The package contains no agent, reward
design, target semantics, gaze automation, curriculum, or persistence.

The dependency is pinned to upstream MIMo commit
`040b0ae4914cbfb26afdf830aa81775b90922f3f` (version 2.0.0).

## Setup

```bash
/home/vishnu/miniconda3/envs/expectation_snn/bin/python -m venv .venv
PIP_CONSTRAINT="$PWD/build-constraints.txt" .venv/bin/python -m pip install -r requirements-dev.txt
```

Use the requirements file: upstream MIMo 2.0.0's wheel omits required package
modules and assets, so this project installs the pinned commit as an editable
VCS checkout while still enforcing `MIMo==2.0.0` in project metadata.

## Test

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest -q
```

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
)

env = BlightWorldEnv(audition=AuditionConfig.quality_48k())
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
env.close()
PY
```

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
scientific grounding, and declared approximations.

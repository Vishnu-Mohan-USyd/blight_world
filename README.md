# Blight World

A deliberately minimal MIMo 2.0.0 world containing only a static room, a low
table, and three visible collidable objects. The environment exposes MIMo's
binocular RGB vision and minimal proprioception; touch and vestibular sensing
are disabled. It contains no agent, reward design, target semantics, gaze
automation, curriculum, or persistence.

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

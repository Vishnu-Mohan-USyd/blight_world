"""Acceptance tests for the frozen minimal MIMo v2 world."""

from __future__ import annotations

import gc
import hashlib
from importlib import metadata
from pathlib import Path
from typing import Iterator

import mimoEnv
import mujoco
import numpy as np
import pytest
from gymnasium.spaces import Box

from blight_world import BlightWorldEnv


EYE_HEIGHT = 48
EYE_WIDTH = 64
EXPECTED_OBSERVATION_KEYS = {"observation", "eye_left", "eye_right"}


def _asset_snapshot(asset_root: Path) -> dict[str, str]:
    """Hash every installed MIMo asset without changing filesystem metadata."""

    return {
        path.relative_to(asset_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(asset_root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def environment() -> Iterator[BlightWorldEnv]:
    """Create one deterministic environment and verify read-only asset use."""

    asset_root = Path(mimoEnv.__file__).resolve().parent / "assets"
    before = _asset_snapshot(asset_root)
    env = BlightWorldEnv(width=EYE_WIDTH, height=EYE_HEIGHT)
    scene_path = env.scene_path

    assert scene_path.is_file()
    assert asset_root not in scene_path.parents
    materialized = scene_path.read_text(encoding="utf-8")
    assert "{{MIMO_TEXTURE_DIR}}" not in materialized
    assert "{{MIMO_META_XML}}" not in materialized
    assert "{{MIMO_MODEL_XML}}" not in materialized
    assert (asset_root / "mimo" / "MIMo_metav2.xml").resolve().as_posix() in materialized
    assert (asset_root / "mimo" / "MIMo_modelv2.xml").resolve().as_posix() in materialized

    try:
        yield env
    finally:
        env.close()

    assert not scene_path.exists()
    assert _asset_snapshot(asset_root) == before


def _model_id(env: BlightWorldEnv, object_type: mujoco.mjtObj, name: str) -> int:
    """Resolve a required named MuJoCo object and assert that it exists."""

    object_id = mujoco.mj_name2id(env.model, object_type, name)
    assert object_id >= 0, f"missing MuJoCo object: {name}"
    return object_id


def test_imports_and_pinned_mimo_distribution() -> None:
    """The public package imports against exactly MIMo 2.0.0."""

    assert metadata.version("MIMo") == "2.0.0"
    assert BlightWorldEnv.__module__ == "blight_world.environment"


def test_scene_contains_only_required_named_structure(environment: BlightWorldEnv) -> None:
    """The room, table, and three collidable object primitives all load."""

    for geom_name in (
        "floor",
        "ceiling",
        "wall_front",
        "wall_back",
        "wall_left",
        "wall_right",
        "table_top",
    ):
        _model_id(environment, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    for body_name in ("table", "ball", "block", "cylinder"):
        _model_id(environment, mujoco.mjtObj.mjOBJ_BODY, body_name)

    for geom_name in ("ball_geom", "block_geom", "cylinder_geom"):
        geom_id = _model_id(environment, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert environment.model.geom_contype[geom_id] != 0
        assert environment.model.geom_conaffinity[geom_id] != 0


def test_table_object_contact_pairs_compile_with_local_parameters(
    environment: BlightWorldEnv,
) -> None:
    """Only each object/table pair receives its selected contact physics."""

    expected = {
        frozenset(("ball_geom", "table_top")): {
            "condim": 3,
            "solref": np.asarray((0.015, 0.40)),
            "friction": np.asarray((0.55, 0.55, 0.01, 0.002, 0.002)),
        },
        frozenset(("block_geom", "table_top")): {
            "condim": 3,
            "solref": np.asarray((0.02, 1.0)),
            "friction": np.asarray((0.30, 0.30, 0.02, 0.001, 0.001)),
        },
        frozenset(("cylinder_geom", "table_top")): {
            "condim": 6,
            "solref": np.asarray((0.02, 1.0)),
            "friction": np.asarray((0.50, 0.50, 0.02, 0.025, 0.025)),
        },
    }
    observed: dict[frozenset[str], dict[str, int | np.ndarray]] = {}
    for pair_id in range(environment.model.npair):
        geom_names = frozenset(
            (
                environment.model.geom(int(environment.model.pair_geom1[pair_id])).name,
                environment.model.geom(int(environment.model.pair_geom2[pair_id])).name,
            )
        )
        if geom_names in expected:
            observed[geom_names] = {
                "condim": int(environment.model.pair_dim[pair_id]),
                "solref": environment.model.pair_solref[pair_id].copy(),
                "friction": environment.model.pair_friction[pair_id].copy(),
            }

    assert observed.keys() == expected.keys()
    for geom_names, parameters in expected.items():
        assert observed[geom_names]["condim"] == parameters["condim"]
        np.testing.assert_array_equal(
            observed[geom_names]["solref"],
            parameters["solref"],
        )
        np.testing.assert_array_equal(
            observed[geom_names]["friction"],
            parameters["friction"],
        )


def test_scene_objects_have_binocular_field_of_view_margin(
    environment: BlightWorldEnv,
) -> None:
    """Every object center remains at least five degrees inside both eye views."""

    margin = np.deg2rad(5.0)
    aspect_ratio = EYE_WIDTH / EYE_HEIGHT

    for eye_name in ("eye_left", "eye_right"):
        camera_id = _model_id(environment, mujoco.mjtObj.mjOBJ_CAMERA, eye_name)
        vertical_half_fov = np.deg2rad(environment.model.cam_fovy[camera_id]) / 2
        horizontal_half_fov = np.arctan(
            np.tan(vertical_half_fov) * aspect_ratio
        )
        camera_rotation = environment.data.cam_xmat[camera_id].reshape(3, 3)

        for body_name in ("ball", "block", "cylinder"):
            offset_world = (
                environment.data.body(body_name).xpos
                - environment.data.cam_xpos[camera_id]
            )
            offset_camera = camera_rotation.T @ offset_world
            forward = -offset_camera[2]
            assert forward > 0
            horizontal_angle = abs(np.arctan2(offset_camera[0], forward))
            vertical_angle = abs(np.arctan2(offset_camera[1], forward))
            assert horizontal_angle <= horizontal_half_fov - margin
            assert vertical_angle <= vertical_half_fov - margin


def test_reset_observation_contract(environment: BlightWorldEnv) -> None:
    """Reset returns exactly minimal proprioception and two RGB eye images."""

    result = environment.reset(seed=0)
    assert isinstance(result, tuple)
    assert len(result) == 2
    observation, info = result

    assert isinstance(info, dict)
    assert set(observation) == EXPECTED_OBSERVATION_KEYS
    assert environment.observation_space.contains(observation)

    proprioception = observation["observation"]
    assert proprioception.shape == (93,)
    assert proprioception.dtype == np.float64

    left = observation["eye_left"]
    right = observation["eye_right"]
    for eye in (left, right):
        assert eye.shape == (EYE_HEIGHT, EYE_WIDTH, 3)
        assert eye.dtype == np.uint8
        assert np.unique(eye.reshape(-1, 3), axis=0).shape[0] > 1
    assert not np.shares_memory(left, right)
    assert not np.array_equal(left, right)

    assert environment.touch_params is None
    assert environment.touch is None
    assert environment.vestibular_params is None
    assert environment.vestibular is None


def test_action_and_single_step_contract(environment: BlightWorldEnv) -> None:
    """A zero action produces one conformant neutral Gymnasium transition."""

    environment.reset(seed=0)
    assert isinstance(environment.action_space, Box)
    assert environment.action_space.shape == (90,)
    assert environment.action_space.dtype == np.float32

    action = np.zeros((90,), dtype=np.float32)
    assert environment.action_space.contains(action)

    result = environment.step(action)
    assert isinstance(result, tuple)
    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert set(observation) == EXPECTED_OBSERVATION_KEYS
    assert environment.observation_space.contains(observation)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert isinstance(info, dict)

    assert observation["observation"].shape == (93,)
    assert observation["observation"].dtype == np.float64
    for key in ("eye_left", "eye_right"):
        assert observation[key].shape == (EYE_HEIGHT, EYE_WIDTH, 3)
        assert observation[key].dtype == np.uint8


def test_closed_environment_finalizer_cannot_blacken_new_renderer() -> None:
    """Late collection of an old vision cycle cannot free a new GL context."""

    gc.collect()
    collection_was_enabled = gc.isenabled()
    gc.disable()
    first: BlightWorldEnv | None = BlightWorldEnv(width=32, height=24)
    second: BlightWorldEnv | None = None
    try:
        first_observation, _ = first.reset(seed=79)
        for eye_name in ("eye_left", "eye_right"):
            assert np.any(first_observation[eye_name])
        assert not np.array_equal(
            first_observation["eye_left"],
            first_observation["eye_right"],
        )
        first.close()
        del first_observation
        first = None

        second = BlightWorldEnv(width=32, height=24)
        before_collection, _ = second.reset(seed=83)
        before_rgb = {
            eye_name: before_collection[eye_name].copy()
            for eye_name in ("eye_left", "eye_right")
        }
        for eye in before_rgb.values():
            assert np.any(eye)
        assert not np.array_equal(before_rgb["eye_left"], before_rgb["eye_right"])

        gc.collect()
        after_collection, _ = second.reset(seed=83)
        for eye_name, expected in before_rgb.items():
            assert np.any(after_collection[eye_name])
            np.testing.assert_array_equal(after_collection[eye_name], expected)
        assert not np.array_equal(
            after_collection["eye_left"],
            after_collection["eye_right"],
        )
    finally:
        if first is not None:
            first.close()
        if second is not None:
            second.close()
            second.close()
        if collection_was_enabled:
            gc.enable()

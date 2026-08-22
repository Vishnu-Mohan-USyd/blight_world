"""Live MIMo integration tests for the optional auditory observation."""

from __future__ import annotations

from collections.abc import Iterator
import pickle

import numpy as np
import pytest
import mujoco

from blight_world import (
    AuditionConfig,
    AudioSourceSpec,
    BlightWorldEnv,
    BodyAttachment,
    SiteAttachment,
    StaticAttachment,
)
from mimo_audition.mujoco_scene import LEFT_EAR_LOCAL_M, RIGHT_EAR_LOCAL_M


@pytest.fixture(scope="module")
def auditory_environment() -> Iterator[BlightWorldEnv]:
    """Create one full-quality auditory environment for integration checks."""

    environment = BlightWorldEnv(
        width=32,
        height=24,
        audition=AuditionConfig.quality_48k(),
    )
    try:
        yield environment
    finally:
        environment.close()


def test_enabled_reset_exposes_silent_conformant_pressure(
    auditory_environment: BlightWorldEnv,
) -> None:
    """Space inference and reset expose silence without advancing audio time."""

    observation, info = auditory_environment.reset(seed=17)
    pressure = observation["audition"]

    assert info == {}
    assert set(observation) == {
        "observation",
        "eye_left",
        "eye_right",
        "audition",
    }
    assert pressure.shape == (2, 480)
    assert pressure.dtype == np.float32
    assert pressure.flags.c_contiguous
    assert np.count_nonzero(pressure) == 0
    assert auditory_environment.observation_space.contains(observation)
    assert auditory_environment.audition is not None
    assert auditory_environment.audition.renderer.sample_cursor == 0


def test_body_attached_source_produces_positioned_binaural_pressure(
    auditory_environment: BlightWorldEnv,
) -> None:
    """A real bundled clip follows the ball and reaches both virtual ears."""

    auditory_environment.reset(seed=17)
    assert auditory_environment.audition is not None
    if "ball_chime" not in auditory_environment.audition.renderer.source_ids:
        auditory_environment.audition.register_source(
            AudioSourceSpec(
                source_id="ball_chime",
                clip_id="chime",
                attachment=BodyAttachment("ball"),
            )
        )
    auditory_environment.audition.play("ball_chime")

    observation, reward, terminated, truncated, info = auditory_environment.step(
        np.zeros(auditory_environment.action_space.shape, dtype=np.float32)
    )
    pressure = observation["audition"]

    assert auditory_environment.observation_space.contains(observation)
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert set(info) == {
        "is_success",
        "is_failure",
        "achieved_goal",
        "desired_goal",
    }
    assert info["is_success"] is False
    assert info["is_failure"] is False
    for goal_key in ("achieved_goal", "desired_goal"):
        assert info[goal_key].shape == (0,)
        assert info[goal_key].dtype == np.float64
    assert pressure.shape == (2, 480)
    assert pressure.dtype == np.float32
    assert np.all(np.isfinite(pressure))
    assert np.max(np.abs(pressure)) > 0
    assert not np.array_equal(pressure[0], pressure[1])
    assert auditory_environment.audition.renderer.sample_cursor == 480


def test_ball_table_impact_triggers_positioned_audio(
    auditory_environment: BlightWorldEnv,
) -> None:
    """A newly dropped ball sounds without a registered semantic source."""

    auditory_environment.reset(seed=23)
    assert auditory_environment.audition is not None
    qpos = auditory_environment.data.qpos.copy()
    qvel = np.zeros_like(auditory_environment.data.qvel)
    ball_joint = auditory_environment.model.joint("ball_joint")
    qpos[ball_joint.qposadr + 2] = 0.76
    qvel[ball_joint.dofadr + 2] = -2.0
    auditory_environment.set_state(qpos, qvel)
    mujoco.mj_forward(auditory_environment.model, auditory_environment.data)
    auditory_environment.audition.reset(seed=23)

    blocks = []
    for _ in range(16):
        observation, *_ = auditory_environment.step(
            np.zeros(auditory_environment.action_space.shape, dtype=np.float32)
        )
        blocks.append(observation["audition"])
    pressure = np.concatenate(blocks, axis=1)

    assert pressure.shape == (2, 16 * 480)
    assert np.all(np.isfinite(pressure))
    assert np.max(np.abs(pressure)) > 0
    assert not np.array_equal(pressure[0], pressure[1])


def test_reset_replays_registered_source_deterministically(
    auditory_environment: BlightWorldEnv,
) -> None:
    """Episode reset clears DSP state while preserving source definitions."""

    assert auditory_environment.audition is not None
    source_id = "reset_ball_chime"
    if source_id not in auditory_environment.audition.renderer.source_ids:
        auditory_environment.audition.register_source(
            AudioSourceSpec(
                source_id=source_id,
                clip_id="chime",
                attachment=BodyAttachment("ball"),
            )
        )
    action = np.zeros(auditory_environment.action_space.shape, dtype=np.float32)
    outputs = []
    for _ in range(2):
        auditory_environment.reset(seed=41)
        assert source_id in auditory_environment.audition.renderer.source_ids
        auditory_environment.audition.play(source_id)
        observation, *_ = auditory_environment.step(action)
        outputs.append(observation["audition"])

    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_training_profile_exposes_24khz_live_audio() -> None:
    """The lower-cost profile keeps the 10 ms step with 240 samples."""

    environment = BlightWorldEnv(
        width=16,
        height=12,
        audition=AuditionConfig.training_24k(),
    )
    try:
        observation, _ = environment.reset(seed=5)
        assert observation["audition"].shape == (2, 240)
        assert observation["audition"].dtype == np.float32
        assert environment.observation_space.contains(observation)
        assert environment.audition is not None
        environment.audition.register_source(
            AudioSourceSpec("training_click", "click", BodyAttachment("block"))
        )
        environment.audition.play("training_click")
        observation, *_ = environment.step(
            np.zeros(environment.action_space.shape, dtype=np.float32)
        )
        assert observation["audition"].shape == (2, 240)
        assert np.max(np.abs(observation["audition"])) > 0
    finally:
        environment.close()


def test_enabling_audition_does_not_change_rgb_or_proprioception(
    auditory_environment: BlightWorldEnv,
) -> None:
    """The optional callbacks are observational and preserve MIMo dynamics."""

    baseline = BlightWorldEnv(width=32, height=24)
    try:
        enabled_reset, _ = auditory_environment.reset(seed=53)
        baseline_reset, _ = baseline.reset(seed=53)
        for key in ("observation", "eye_left", "eye_right"):
            np.testing.assert_array_equal(enabled_reset[key], baseline_reset[key])

        enabled_step, *_ = auditory_environment.step(
            np.zeros(auditory_environment.action_space.shape, dtype=np.float32)
        )
        baseline_step, *_ = baseline.step(
            np.zeros(baseline.action_space.shape, dtype=np.float32)
        )
        for key in ("observation", "eye_left", "eye_right"):
            np.testing.assert_array_equal(enabled_step[key], baseline_step[key])
    finally:
        baseline.close()


def test_enabled_environment_pickle_reconstructs_sensor(
    auditory_environment: BlightWorldEnv,
) -> None:
    """Gymnasium EzPickle retains the immutable audition configuration."""

    reconstructed = pickle.loads(pickle.dumps(auditory_environment))
    try:
        observation, _ = reconstructed.reset(seed=61)
        assert reconstructed.audition is not None
        assert reconstructed.audition.config == AuditionConfig.quality_48k()
        assert observation["audition"].shape == (2, 480)
        assert reconstructed.observation_space.contains(observation)
    finally:
        reconstructed.close()


def test_virtual_ears_attachments_and_mujoco_occlusion(
    auditory_environment: BlightWorldEnv,
) -> None:
    """Head, body, site, and ray geometry resolve in current MuJoCo state."""

    auditory_environment.reset(seed=67)
    assert auditory_environment.audition is not None
    scene = auditory_environment.audition.scene
    head = auditory_environment.data.body("head")
    head_rotation = head.xmat.reshape(3, 3)
    left, right = scene.ear_positions()
    np.testing.assert_allclose(left, head.xpos + head_rotation @ LEFT_EAR_LOCAL_M)
    np.testing.assert_allclose(right, head.xpos + head_rotation @ RIGHT_EAR_LOCAL_M)

    static_position, static_body_id = scene.attachment_pose(
        StaticAttachment((1.25, -0.5, 1.0))
    )
    np.testing.assert_array_equal(static_position, (1.25, -0.5, 1.0))
    assert static_body_id == -1

    local_offset = np.asarray((0.01, -0.02, 0.03))
    body_position, body_id = scene.attachment_pose(
        BodyAttachment("ball", tuple(local_offset))
    )
    ball = auditory_environment.data.body("ball")
    np.testing.assert_allclose(
        body_position,
        ball.xpos + ball.xmat.reshape(3, 3) @ local_offset,
    )
    assert body_id == ball.id

    site_position, site_body_id = scene.attachment_pose(SiteAttachment("torque_neck"))
    neck_site = auditory_environment.data.site("torque_neck")
    np.testing.assert_array_equal(site_position, neck_site.xpos)
    assert site_body_id == auditory_environment.model.site_bodyid[neck_site.id]

    listener_position, _ = scene.listener_pose()
    block_position = auditory_environment.data.body("block").xpos.copy()
    source_behind_block = listener_position + 1.5 * (
        block_position - listener_position
    )
    assert scene.direct_path_occluded(
        listener_position,
        source_behind_block,
        source_body_id=-1,
    )

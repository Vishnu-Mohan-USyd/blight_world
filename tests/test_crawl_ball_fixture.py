"""Physical and sensory regressions for the visible crawl/ball fixture."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Iterator

import mimoEnv
import mujoco
import numpy as np
import pytest
from numpy.typing import NDArray

from blight_world.crawl_ball_fixture import (
    CRAWL_BALL_CONTACT_GEOM_NAMES,
    CRAWL_BALL_FRICTION,
    CRAWL_BALL_INITIAL_POSITION_M,
    CRAWL_BALL_MASS_KG,
    CRAWL_BALL_RADIUS_M,
    CRAWL_BALL_TABLE_POSITION_M,
    CrawlBallContact,
    CrawlBallFixtureEnv,
)
from blight_world.environment import BlightWorldEnv
from blight_world.locomotion_skills import (
    BELLY_CRAWL_STOP_SECONDS,
    BELLY_SETTLE_STOP_SECONDS,
    BELLY_TRIAL_STOP_SECONDS,
    UnsupportedBellyCrawlController,
)
from mimo_audition import AuditionConfig, ContinuousSoundConfig


RESET_SEED = 1401
PUBLIC_STEP_SECONDS = 0.010
TRIAL_STEPS = int(round(BELLY_TRIAL_STOP_SECONDS / PUBLIC_STEP_SECONDS))
ACTIVE_START_STEP = int(round(BELLY_SETTLE_STOP_SECONDS / PUBLIC_STEP_SECONDS))
ACTIVE_STOP_STEP = int(round(BELLY_CRAWL_STOP_SECONDS / PUBLIC_STEP_SECONDS))
CONTACT_STEP = 2851
EXPECTED_SCENE_SHA256 = (
    "819783d00b8983ae26053e814a3af50439f1ac0f9ad223f081f309265455c50f"
)
EXPECTED_ACTION_SHA256 = (
    "46d8fde52bf2bd78c0a4d2a9791a2e056194761816422be58118ee02464fff67"
)
EXPECTED_BALL_TRAJECTORY_SHA256 = (
    "a2123ee5d9c7ad2f100ff6e59569f641770e618a346ad4930e00631f3d645b54"
)
EXPECTED_QPOS_SHA256 = (
    "e5379268f771e4f05578297a6df5257e1380e381056bc852e916f9f0a3d7e4ec"
)
EXPECTED_AUDIO_SHA256 = (
    "79fb1a31c8d4672b17b7b4f9fd407c316aa42e936eca3affa00d08bb0cd5d7a5"
)
EXPECTED_CONTACT_LEFT_EYE_SHA256 = (
    "23ad078ea1ffc360a0fcdf77beb74f3df86639cf71b3e1bdbd3866b776891153"
)
EXPECTED_CONTACT_RIGHT_EYE_SHA256 = (
    "5bc700fae754596b28cb9a55f1f129d13a340507bad72fb41e6b9d4fd3191c13"
)
EXPECTED_REST_ACTION_SHA256 = (
    "92904870ee2d0e806258653469929aff7660f4c3fa87e787f8693a7461f7066a"
)
EXPECTED_REST_BALL_TRAJECTORY_SHA256 = (
    "84d36aea9be98889c4712b2e49130cee147294eb5b6ac4aa4e607ccd3ab4641f"
)


def _file_sha256(path: Path) -> str:
    # Keep the exact scene-byte check independent of the MIMo checkout path.
    content = path.read_bytes()
    assets = Path(mimoEnv.__file__).resolve().parent / "assets"
    for resolved, token in (
        (assets / "mimo" / "MIMo_metav2.xml", "{{MIMO_META_XML}}"),
        (assets / "mimo" / "MIMo_modelv2.xml", "{{MIMO_MODEL_XML}}"),
        (assets, "{{MIMO_TEXTURE_DIR}}"),
    ):
        content = content.replace(
            escape(resolved.as_posix(), quote=True).encode("utf-8"),
            token.encode("ascii"),
        )
    return sha256(content).hexdigest()


@pytest.fixture
def environment() -> Iterator[CrawlBallFixtureEnv]:
    instance = CrawlBallFixtureEnv(width=32, height=24)
    try:
        yield instance
    finally:
        instance.close()


def test_scene_is_exact_free_ball_fixture_with_production_binocular_render(
    environment: CrawlBallFixtureEnv,
) -> None:
    observation, info = environment.reset(seed=RESET_SEED)
    assert info == {}
    assert environment.observation_space.contains(observation)
    assert CrawlBallFixtureEnv.render is BlightWorldEnv.render
    assert _file_sha256(environment.scene_path) == EXPECTED_SCENE_SHA256

    model = environment.model
    ball_body_id = int(model.body("ball").id)
    ball_geom_id = int(model.geom("ball_geom").id)
    ball_joint = model.joint("ball_joint")
    assert int(np.asarray(ball_joint.type).item()) == int(mujoco.mjtJoint.mjJNT_FREE)
    np.testing.assert_array_equal(
        model.body_pos[ball_body_id], np.asarray(CRAWL_BALL_INITIAL_POSITION_M)
    )
    assert int(model.geom_type[ball_geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    assert float(model.geom_size[ball_geom_id, 0]) == CRAWL_BALL_RADIUS_M
    assert float(model.body_mass[ball_body_id]) == CRAWL_BALL_MASS_KG
    np.testing.assert_array_equal(
        model.geom_friction[ball_geom_id], np.asarray(CRAWL_BALL_FRICTION)
    )
    np.testing.assert_array_equal(
        model.body_pos[int(model.body("table").id)],
        np.asarray(CRAWL_BALL_TABLE_POSITION_M),
    )
    for object_name in ("block", "cylinder"):
        geom_id = int(model.geom(f"{object_name}_geom").id)
        np.testing.assert_array_equal(
            model.body_pos[int(model.body(object_name).id)],
            np.asarray((2.2, 2.2, 2.2)),
        )
        assert int(model.geom_contype[geom_id]) == 0
        assert int(model.geom_conaffinity[geom_id]) == 0

    body_equalities = [
        equality_id
        for equality_id in range(model.neq)
        if int(model.eq_type[equality_id])
        in {int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)}
    ]
    assert body_equalities == []
    ball_contacts = []
    mimo_root_id = int(model.body("mimo_location").id)
    for contact in environment.data.contact:
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if ball_geom_id not in (geom1, geom2):
            continue
        other_geom = geom2 if geom1 == ball_geom_id else geom1
        other_body = int(model.geom_bodyid[other_geom])
        if int(model.body_rootid[other_body]) == mimo_root_id:
            ball_contacts.append(other_geom)
    assert ball_contacts == []

    recorder = environment.start_contact_recording()
    assert recorder.records == ()
    assert environment.stop_contact_recording() is recorder


@dataclass(frozen=True, slots=True)
class _ReplayEvidence:
    """Immutable hashes and observable evidence from one public replay."""

    cycles_enabled: bool
    audition_enabled: bool
    action_sha256: str
    qpos_sha256: str
    ball_trajectory_sha256: str
    audio_sha256: str | None
    contacts: tuple[CrawlBallContact, ...]
    initial_ball_position_m: NDArray[np.float64]
    final_ball_position_m: NDArray[np.float64]
    contact_left_red_pixels: int | None
    contact_right_red_pixels: int | None
    contact_left_sha256: str | None
    contact_right_sha256: str | None
    audio_nonzero_blocks: int
    audio_peak_abs_pa: float
    audio_left_rms_pa: float
    audio_right_rms_pa: float


def _red_pixel_count(image: NDArray[np.uint8]) -> int:
    mask = (
        (image[..., 0] > 150)
        & (image[..., 1] < 140)
        & (image[..., 2] < 140)
    )
    return int(np.count_nonzero(mask))


def _run_replay(
    *,
    cycles_enabled: bool,
    audition_enabled: bool,
) -> _ReplayEvidence:
    """Run the exact public controller path with read-only instrumentation."""

    audition = (
        AuditionConfig.quality_48k(continuous=ContinuousSoundConfig())
        if audition_enabled
        else None
    )
    width, height = ((320, 240) if audition_enabled else (32, 24))
    environment = CrawlBallFixtureEnv(
        width=width,
        height=height,
        audition=audition,
    )
    try:
        observation, info = environment.reset(seed=RESET_SEED)
        assert info == {}
        assert environment.observation_space.contains(observation)
        recorder = environment.start_contact_recording()
        controller = UnsupportedBellyCrawlController(
            environment, cycles_enabled=cycles_enabled
        )
        state = controller.reset()

        ball_body_id = int(environment.model.body("ball").id)
        initial_ball_position = np.asarray(
            environment.data.xpos[ball_body_id], dtype=np.float64
        ).copy()
        action_digest = sha256()
        qpos_digest = sha256()
        trajectory_digest = sha256()
        audio_digest = sha256() if audition_enabled else None
        audio_nonzero_blocks = 0
        audio_peak = 0.0
        audio_sum_squares = np.zeros(2, dtype=np.float64)
        audio_sample_count = 0
        contact_left: NDArray[np.uint8] | None = None
        contact_right: NDArray[np.uint8] | None = None

        model_snapshots = {
            "body_pos": environment.model.body_pos.copy(),
            "body_quat": environment.model.body_quat.copy(),
            "geom_pos": environment.model.geom_pos.copy(),
            "geom_quat": environment.model.geom_quat.copy(),
            "geom_size": environment.model.geom_size.copy(),
            "geom_friction": environment.model.geom_friction.copy(),
            "eq_active0": environment.model.eq_active0.copy(),
            "gravity": environment.model.opt.gravity.copy(),
        }
        mocap_pos = environment.data.mocap_pos.copy()
        mocap_quat = environment.data.mocap_quat.copy()

        for step_index in range(TRIAL_STEPS):
            qpos_before = environment.data.qpos.copy()
            qvel_before = environment.data.qvel.copy()
            xfrc_before = environment.data.xfrc_applied.copy()
            action, state, _ = controller.compose(environment, state)
            np.testing.assert_array_equal(environment.data.qpos, qpos_before)
            np.testing.assert_array_equal(environment.data.qvel, qvel_before)
            np.testing.assert_array_equal(environment.data.xfrc_applied, xfrc_before)
            assert environment.action_space.contains(action)
            action_bytes = np.ascontiguousarray(action, dtype=np.float32).tobytes(
                order="C"
            )
            action_digest.update(action_bytes)

            observation, _, terminated, truncated, _ = environment.step(action)
            assert not terminated
            assert not truncated
            assert environment.observation_space.contains(observation)
            np.testing.assert_array_equal(
                environment.data.xfrc_applied,
                np.zeros_like(environment.data.xfrc_applied),
            )
            for name, expected in model_snapshots.items():
                actual = (
                    environment.model.opt.gravity
                    if name == "gravity"
                    else getattr(environment.model, name)
                )
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(environment.data.mocap_pos, mocap_pos)
            np.testing.assert_array_equal(environment.data.mocap_quat, mocap_quat)

            qpos_digest.update(
                np.ascontiguousarray(
                    environment.data.qpos, dtype=np.float64
                ).tobytes(order="C")
            )
            trajectory_digest.update(
                np.ascontiguousarray(
                    environment.data.xpos[ball_body_id], dtype=np.float64
                ).tobytes(order="C")
            )
            if step_index == CONTACT_STEP:
                contact_left = np.ascontiguousarray(
                    observation["eye_left"], dtype=np.uint8
                ).copy()
                contact_right = np.ascontiguousarray(
                    observation["eye_right"], dtype=np.uint8
                ).copy()
            if audition_enabled:
                audio_block = observation["audition"]
                assert audio_block.shape == (2, 480)
                assert audio_block.dtype == np.float32
                assert audio_block.flags.c_contiguous
                assert np.all(np.isfinite(audio_block))
                audio_values = np.ascontiguousarray(audio_block, dtype=np.float32)
                assert audio_digest is not None
                audio_digest.update(audio_values.tobytes(order="C"))
                audio_nonzero_blocks += int(np.any(audio_values != 0.0))
                audio_peak = max(audio_peak, float(np.max(np.abs(audio_values))))
                audio_sum_squares += np.sum(
                    audio_values.astype(np.float64) ** 2,
                    axis=1,
                )
                audio_sample_count += audio_values.shape[1]

        final_ball_position = np.asarray(
            environment.data.xpos[ball_body_id], dtype=np.float64
        ).copy()
        return _ReplayEvidence(
            cycles_enabled=cycles_enabled,
            audition_enabled=audition_enabled,
            action_sha256=action_digest.hexdigest(),
            qpos_sha256=qpos_digest.hexdigest(),
            ball_trajectory_sha256=trajectory_digest.hexdigest(),
            audio_sha256=(
                None if audio_digest is None else audio_digest.hexdigest()
            ),
            contacts=recorder.records,
            initial_ball_position_m=initial_ball_position,
            final_ball_position_m=final_ball_position,
            contact_left_red_pixels=(
                None if contact_left is None else _red_pixel_count(contact_left)
            ),
            contact_right_red_pixels=(
                None if contact_right is None else _red_pixel_count(contact_right)
            ),
            contact_left_sha256=(
                None
                if contact_left is None
                else sha256(contact_left.tobytes(order="C")).hexdigest()
            ),
            contact_right_sha256=(
                None
                if contact_right is None
                else sha256(contact_right.tobytes(order="C")).hexdigest()
            ),
            audio_nonzero_blocks=audio_nonzero_blocks,
            audio_peak_abs_pa=audio_peak,
            audio_left_rms_pa=(
                0.0
                if audio_sample_count == 0
                else float(np.sqrt(audio_sum_squares[0] / audio_sample_count))
            ),
            audio_right_rms_pa=(
                0.0
                if audio_sample_count == 0
                else float(np.sqrt(audio_sum_squares[1] / audio_sample_count))
            ),
        )
    finally:
        environment.close()


@pytest.fixture(scope="module")
def physical_replays() -> tuple[_ReplayEvidence, ...]:
    """Run no-audition, audition, and belly-rest paths once each."""

    return (
        _run_replay(cycles_enabled=True, audition_enabled=False),
        _run_replay(cycles_enabled=True, audition_enabled=True),
        _run_replay(cycles_enabled=False, audition_enabled=False),
    )


def test_crawl_has_exact_finger_contact_and_free_ball_displacement(
    physical_replays: tuple[_ReplayEvidence, ...],
) -> None:
    active, with_audition, _ = physical_replays
    assert active.action_sha256 == EXPECTED_ACTION_SHA256
    assert active.qpos_sha256 == EXPECTED_QPOS_SHA256
    assert active.ball_trajectory_sha256 == EXPECTED_BALL_TRAJECTORY_SHA256
    assert with_audition.action_sha256 == active.action_sha256
    assert with_audition.qpos_sha256 == active.qpos_sha256
    assert with_audition.ball_trajectory_sha256 == active.ball_trajectory_sha256
    assert with_audition.contacts == active.contacts

    assert len(active.contacts) == 5
    first = active.contacts[0]
    assert first.public_step_index == CONTACT_STEP
    assert first.physics_substep_index == 5702
    expected_time_seconds = (first.physics_substep_index + 1) * (
        PUBLIC_STEP_SECONDS / 2.0
    )
    assert first.time_seconds == pytest.approx(expected_time_seconds, abs=5e-11)
    assert {contact.mimo_geom_name for contact in active.contacts} == (
        CRAWL_BALL_CONTACT_GEOM_NAMES
    )
    assert all(contact.ball_geom_name == "ball_geom" for contact in active.contacts)
    assert not any(
        contact.public_step_index < ACTIVE_START_STEP for contact in active.contacts
    )
    assert not any(
        contact.public_step_index >= ACTIVE_STOP_STEP for contact in active.contacts
    )
    assert max(contact.normal_force_n for contact in active.contacts) == pytest.approx(
        0.27134175246687636, abs=1e-12
    )

    planar_displacement = float(
        np.linalg.norm(
            active.final_ball_position_m[:2]
            - active.initial_ball_position_m[:2]
        )
    )
    assert planar_displacement == pytest.approx(0.3279796105546315, abs=1e-12)
    np.testing.assert_allclose(
        active.final_ball_position_m,
        np.asarray((0.8282788213116993, 0.35641593947774347, 0.024552576685847535)),
        rtol=0,
        atol=1e-12,
    )


def test_ball_is_publicly_visible_and_binocular_at_contact(
    physical_replays: tuple[_ReplayEvidence, ...],
) -> None:
    _, with_audition, _ = physical_replays
    assert with_audition.contact_left_red_pixels == 2270
    assert with_audition.contact_right_red_pixels == 2163
    assert with_audition.contact_left_sha256 == EXPECTED_CONTACT_LEFT_EYE_SHA256
    assert with_audition.contact_right_sha256 == EXPECTED_CONTACT_RIGHT_EYE_SHA256
    assert with_audition.contact_left_sha256 != with_audition.contact_right_sha256


def test_quality_48k_continuous_audition_is_nonzero_and_physics_neutral(
    physical_replays: tuple[_ReplayEvidence, ...],
) -> None:
    active, with_audition, _ = physical_replays
    assert with_audition.qpos_sha256 == active.qpos_sha256
    assert with_audition.ball_trajectory_sha256 == active.ball_trajectory_sha256
    assert with_audition.action_sha256 == active.action_sha256
    assert with_audition.audio_sha256 == EXPECTED_AUDIO_SHA256
    assert with_audition.audio_nonzero_blocks == 3665
    assert with_audition.audio_peak_abs_pa == pytest.approx(
        12.987203598022461, abs=1e-12
    )
    assert with_audition.audio_left_rms_pa == pytest.approx(
        0.3599577686644569, abs=1e-12
    )
    assert with_audition.audio_right_rms_pa == pytest.approx(
        0.35238572050749417, abs=1e-12
    )


def test_belly_rest_control_never_contacts_or_translates_ball(
    physical_replays: tuple[_ReplayEvidence, ...],
) -> None:
    _, _, rest = physical_replays
    assert rest.action_sha256 == EXPECTED_REST_ACTION_SHA256
    assert rest.ball_trajectory_sha256 == EXPECTED_REST_BALL_TRAJECTORY_SHA256
    assert rest.contacts == ()
    np.testing.assert_array_equal(
        rest.final_ball_position_m[:2], rest.initial_ball_position_m[:2]
    )

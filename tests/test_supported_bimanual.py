"""Validation for the supported bimanual foam-block clamp."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from hashlib import sha256
from typing import Iterator, Literal

import mujoco
import numpy as np
import pytest
from numpy.typing import NDArray

from blight_world import (
    BimanualClampPhase,
    FoamBlockContactClassifier,
    SupportedBimanualFoamBlockClampController,
    SupportedBimanualFoamBlockEnv,
    resolve_motor_channel,
)
from blight_world.supported_bimanual import (
    APPROACH_ARM_REFERENCE_RAD,
    CONTACT_ARM_REFERENCE_RAD,
    LEFT_BIMANUAL_ARM_ACTUATOR_NAMES,
    RIGHT_BIMANUAL_ARM_ACTUATOR_NAMES,
    SQUEEZE_ARM_REFERENCE_RAD,
)


RESET_SEED = 1401
TRIAL_STEPS = 700
EXPECTED_ACTION_SHA256 = (
    "35249db1947b783793245e43214855953f0a9565030f6d22cc2c28a0ef4af925"
)
EXPECTED_QPOS_SHA256 = (
    "4ca47a9085ad6eb3e39f4db5a7af521fa42065bfa901f57fe692421c5ae684fb"
)


@dataclass(frozen=True, slots=True)
class _ClampReplay:
    """Deterministic public-step evidence for one clamp-control mode."""

    action_trace: NDArray[np.float32]
    qpos_trace: NDArray[np.float64]
    action_sha256: str
    qpos_sha256: str
    bilateral_hold_seconds: float
    force_window_hold_seconds: float
    maximum_distinct_palm_geoms: int
    other_contact_geom_names: tuple[str, ...]
    supported_bilateral_ratio: float
    maximum_normal_force_n: float
    saturated_actuators: tuple[str, ...]
    initial_block_position_m: NDArray[np.float64]
    final_block_position_m: NDArray[np.float64]
    maximum_hand_relative_drift_m: float
    maximum_oriented_bottom_gap_m: float


def _longest_true_run_seconds(values: list[bool], dt_seconds: float) -> float:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest * dt_seconds)


def _block_in_hand_frame(
    environment: SupportedBimanualFoamBlockEnv,
    hand_body_name: str,
    block_position_m: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Express the block center in one hand body's instantaneous frame."""

    hand = environment.data.body(hand_body_name)
    rotation_world_from_hand = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(
        rotation_world_from_hand,
        np.asarray(hand.xquat, dtype=np.float64),
    )
    rotation_world_from_hand = rotation_world_from_hand.reshape(3, 3)
    return rotation_world_from_hand.T @ (
        block_position_m - np.asarray(hand.xpos, dtype=np.float64)
    )


def _oriented_block_bottom_gap_m(
    environment: SupportedBimanualFoamBlockEnv,
    block_geom_id: int,
    plinth_top_height_m: float,
) -> float:
    """Return positive clearance between the rotated block and plinth top."""

    half_size = np.asarray(environment.model.geom_size[block_geom_id], dtype=np.float64)
    center = np.asarray(environment.data.geom_xpos[block_geom_id], dtype=np.float64)
    rotation = np.asarray(
        environment.data.geom_xmat[block_geom_id], dtype=np.float64
    ).reshape(3, 3)
    local_corners = np.asarray(
        [
            (x, y, z)
            for x in (-half_size[0], half_size[0])
            for y in (-half_size[1], half_size[1])
            for z in (-half_size[2], half_size[2])
        ],
        dtype=np.float64,
    )
    world_corners = center + local_corners @ rotation.T
    return max(0.0, float(np.min(world_corners[:, 2]) - plinth_top_height_m))


def _run_clamp(mode: Literal["active", "open", "one_arm"]) -> _ClampReplay:
    """Run one seeded actuator-only clamp or negative control."""

    environment = SupportedBimanualFoamBlockEnv(width=16, height=12)
    try:
        observation, info = environment.reset(seed=RESET_SEED)
        assert info == {}
        assert environment.observation_space.contains(observation)
        controller = SupportedBimanualFoamBlockClampController(environment)
        state = controller.reset(environment)
        classifier = FoamBlockContactClassifier(environment)

        block_body_id = int(environment.model.body("foam_block").id)
        block_geom_id = int(environment.model.geom("foam_block_geom").id)
        plinth_top_height = 0.206 + 0.206
        initial_block_position = np.asarray(
            environment.data.xpos[block_body_id], dtype=np.float64
        ).copy()
        controlled_indices = {
            resolve_motor_channel(environment, name).action_index
            for name in controller.controlled_actuator_names
        }
        action_digest = sha256()
        qpos_digest = sha256()
        action_rows: list[NDArray[np.float32]] = []
        qpos_rows: list[NDArray[np.float64]] = []
        bilateral_rows: list[bool] = []
        force_window_rows: list[bool] = []
        contact_rows = []
        other_contact_names: set[str] = set()
        maximum_distinct_palm_geoms = 0
        supported_bilateral_rows = 0
        maximum_oriented_bottom_gap = 0.0
        saturated: set[str] = set()
        hand_reference: dict[str, NDArray[np.float64]] | None = None
        maximum_hand_relative_drift = 0.0

        for _ in range(TRIAL_STEPS):
            qpos_before = environment.data.qpos.copy()
            qvel_before = environment.data.qvel.copy()
            xfrc_before = environment.data.xfrc_applied.copy()
            gravity_before = environment.model.opt.gravity.copy()
            equality_before = environment.model.eq_active0.copy()
            action, state, diagnostics = controller.compose(
                environment,
                state,
                mode=mode,
            )
            np.testing.assert_array_equal(environment.data.qpos, qpos_before)
            np.testing.assert_array_equal(environment.data.qvel, qvel_before)
            np.testing.assert_array_equal(environment.data.xfrc_applied, xfrc_before)
            np.testing.assert_array_equal(environment.model.opt.gravity, gravity_before)
            np.testing.assert_array_equal(environment.model.eq_active0, equality_before)
            assert environment.action_space.contains(action)
            assert set(np.flatnonzero(action)).issubset(controlled_indices)
            saturated.update(diagnostics.saturated_actuators)
            action_copy = np.ascontiguousarray(action, dtype=np.float32)
            action_rows.append(action_copy.copy())
            action_digest.update(action_copy.tobytes(order="C"))

            observation, _, terminated, truncated, _ = environment.step(action)
            assert not terminated
            assert not truncated
            assert environment.observation_space.contains(observation)
            np.testing.assert_array_equal(
                environment.data.xfrc_applied,
                np.zeros_like(environment.data.xfrc_applied),
            )
            qpos_copy = np.ascontiguousarray(environment.data.qpos, dtype=np.float64)
            qpos_rows.append(qpos_copy.copy())
            qpos_digest.update(qpos_copy.tobytes(order="C"))

            contacts = classifier.classify()
            contact_rows.append(contacts)
            other_contact_names.update(contact.geom_name for contact in contacts.other)
            maximum_distinct_palm_geoms = max(
                maximum_distinct_palm_geoms,
                contacts.distinct_palm_geom_count,
            )
            bilateral_rows.append(contacts.bilateral_palm)
            if contacts.bilateral_palm and contacts.support_normal_force_n > 0.0:
                supported_bilateral_rows += 1
            force_window_rows.append(
                contacts.bilateral_palm
                and 0.3 <= contacts.left_normal_force_n <= 3.0
                and 0.3 <= contacts.right_normal_force_n <= 3.0
            )
            block_position = np.asarray(
                environment.data.xpos[block_body_id], dtype=np.float64
            ).copy()
            if contacts.bilateral_palm and hand_reference is None:
                hand_reference = {
                    hand_name: _block_in_hand_frame(
                        environment, hand_name, block_position
                    )
                    for hand_name in ("left_hand", "right_hand")
                }
            if hand_reference is not None:
                for hand_name, reference in hand_reference.items():
                    maximum_hand_relative_drift = max(
                        maximum_hand_relative_drift,
                        float(
                            np.linalg.norm(
                                _block_in_hand_frame(
                                    environment, hand_name, block_position
                                )
                                - reference
                            )
                        ),
                    )

            maximum_oriented_bottom_gap = max(
                maximum_oriented_bottom_gap,
                _oriented_block_bottom_gap_m(
                    environment,
                    block_geom_id,
                    plinth_top_height,
                ),
            )

        bilateral_count = sum(bilateral_rows)
        supported_bilateral_ratio = (
            supported_bilateral_rows / bilateral_count if bilateral_count else 0.0
        )
        final_block_position = np.asarray(
            environment.data.xpos[block_body_id], dtype=np.float64
        ).copy()
        return _ClampReplay(
            action_trace=np.asarray(action_rows, dtype=np.float32),
            qpos_trace=np.asarray(qpos_rows, dtype=np.float64),
            action_sha256=action_digest.hexdigest(),
            qpos_sha256=qpos_digest.hexdigest(),
            bilateral_hold_seconds=_longest_true_run_seconds(
                bilateral_rows, environment.dt
            ),
            force_window_hold_seconds=_longest_true_run_seconds(
                force_window_rows, environment.dt
            ),
            maximum_distinct_palm_geoms=maximum_distinct_palm_geoms,
            other_contact_geom_names=tuple(sorted(other_contact_names)),
            supported_bilateral_ratio=float(supported_bilateral_ratio),
            maximum_normal_force_n=max(
                contacts.maximum_normal_force_n for contacts in contact_rows
            ),
            saturated_actuators=tuple(sorted(saturated)),
            initial_block_position_m=initial_block_position,
            final_block_position_m=final_block_position,
            maximum_hand_relative_drift_m=maximum_hand_relative_drift,
            maximum_oriented_bottom_gap_m=maximum_oriented_bottom_gap,
        )
    finally:
        environment.close()


@pytest.fixture(scope="module")
def physical_replays() -> tuple[_ClampReplay, ...]:
    """Run two active replays and the two frozen negative controls."""

    return (
        _run_clamp("active"),
        _run_clamp("active"),
        _run_clamp("open"),
        _run_clamp("one_arm"),
    )


@pytest.fixture(scope="module")
def environment() -> Iterator[SupportedBimanualFoamBlockEnv]:
    instance = SupportedBimanualFoamBlockEnv(width=16, height=12)
    try:
        yield instance
    finally:
        instance.close()


def test_scene_has_free_foam_block_and_only_root_support_body_equality(
    environment: SupportedBimanualFoamBlockEnv,
) -> None:
    observation, info = environment.reset(seed=RESET_SEED)
    assert info == {}
    assert environment.observation_space.contains(observation)
    model = environment.model
    block_body_id = int(model.body("foam_block").id)
    block_geom_id = int(model.geom("foam_block_geom").id)
    block_joint = model.joint("foam_block_joint")
    assert int(np.asarray(block_joint.type).item()) == int(mujoco.mjtJoint.mjJNT_FREE)
    assert float(model.body_mass[block_body_id]) == pytest.approx(0.060)
    assert int(model.geom_type[block_geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
    np.testing.assert_array_equal(
        model.geom_size[block_geom_id], np.asarray((0.04, 0.085, 0.04))
    )
    np.testing.assert_array_equal(
        model.body_pos[block_body_id], np.asarray((0.106, 0.0, 0.452))
    )
    np.testing.assert_array_equal(
        model.body_pos[int(model.body("foam_plinth").id)],
        np.asarray((0.106, 0.0, 0.206)),
    )
    assert model.nmocap == 0

    body_equalities: list[int] = []
    for equality_id in range(model.neq):
        if int(model.eq_type[equality_id]) in {
            int(mujoco.mjtEq.mjEQ_CONNECT),
            int(mujoco.mjtEq.mjEQ_WELD),
        }:
            body_equalities.append(equality_id)
    assert len(body_equalities) == 1
    equality_id = body_equalities[0]
    assert int(model.eq_type[equality_id]) == int(mujoco.mjtEq.mjEQ_WELD)
    assert mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_id
    ) == "root_support"
    assert int(model.eq_obj1id[equality_id]) == int(model.body("mimo_location").id)
    assert int(model.eq_obj2id[equality_id]) == 0
    assert block_body_id not in {
        int(model.eq_obj1id[equality_id]),
        int(model.eq_obj2id[equality_id]),
    }


def test_controller_is_name_resolved_sparse_immutable_and_state_read_only(
    environment: SupportedBimanualFoamBlockEnv,
) -> None:
    environment.reset(seed=RESET_SEED)
    controller = SupportedBimanualFoamBlockClampController(environment)
    state = controller.reset(environment)
    assert state == controller.reset(environment)
    with pytest.raises(FrozenInstanceError):
        state.head_reference_rad = 1.0  # type: ignore[misc]

    expected_names = {
        *RIGHT_BIMANUAL_ARM_ACTUATOR_NAMES,
        *LEFT_BIMANUAL_ARM_ACTUATOR_NAMES,
        "act:hip_bend",
        "act:head_tilt",
    }
    assert set(controller.controlled_actuator_names) == expected_names
    assert len(controller.controlled_actuator_names) == 16
    assert not any("ff_" in name or "thumb" in name for name in expected_names)
    controlled_indices = {
        resolve_motor_channel(environment, name).action_index
        for name in expected_names
    }

    qpos = environment.data.qpos.copy()
    qvel = environment.data.qvel.copy()
    xfrc = environment.data.xfrc_applied.copy()
    gravity = environment.model.opt.gravity.copy()
    equality_active = environment.model.eq_active0.copy()
    action, _, diagnostics = controller.compose(
        environment, state, time_seconds=3.5
    )
    assert diagnostics.phase is BimanualClampPhase.SQUEEZE
    assert diagnostics.mode == "active"
    assert diagnostics.right_reference_positions_rad == (
        diagnostics.left_reference_positions_rad
    )
    assert diagnostics.right_reference_velocities_rad_s == (
        diagnostics.left_reference_velocities_rad_s
    )
    assert environment.action_space.contains(action)
    assert set(np.flatnonzero(action)).issubset(controlled_indices)
    np.testing.assert_array_equal(environment.data.qpos, qpos)
    np.testing.assert_array_equal(environment.data.qvel, qvel)
    np.testing.assert_array_equal(environment.data.xfrc_applied, xfrc)
    np.testing.assert_array_equal(environment.model.opt.gravity, gravity)
    np.testing.assert_array_equal(environment.model.eq_active0, equality_active)

    open_action, _, open_diagnostics = controller.compose(
        environment, state, mode="open", time_seconds=3.5
    )
    assert open_diagnostics.phase is BimanualClampPhase.CONTROL_HOLD
    assert open_diagnostics.right_reference_positions_rad == (
        state.right_initial_positions_rad
    )
    assert open_diagnostics.left_reference_positions_rad == (
        state.left_initial_positions_rad
    )
    assert environment.action_space.contains(open_action)

    _, _, one_arm_diagnostics = controller.compose(
        environment, state, mode="one_arm", time_seconds=3.5
    )
    assert one_arm_diagnostics.left_reference_positions_rad == (
        state.left_initial_positions_rad
    )
    assert one_arm_diagnostics.right_reference_positions_rad != (
        state.right_initial_positions_rad
    )
    with pytest.raises(ValueError, match="unsupported clamp mode"):
        controller.compose(environment, state, mode="invalid")  # type: ignore[arg-type]


def test_trajectory_rows_and_velocity_are_exact_at_phase_boundaries() -> None:
    initial = (0.0,) * 7
    phase, references, velocities = (
        SupportedBimanualFoamBlockClampController._trajectory(initial, 0.3)
    )
    assert phase is BimanualClampPhase.APPROACH
    np.testing.assert_array_equal(references, initial)
    np.testing.assert_array_equal(velocities, np.zeros(7))

    phase, references, velocities = (
        SupportedBimanualFoamBlockClampController._trajectory(initial, 1.7)
    )
    assert phase is BimanualClampPhase.CONTACT
    np.testing.assert_array_equal(references, APPROACH_ARM_REFERENCE_RAD)
    np.testing.assert_array_equal(velocities, np.zeros(7))

    phase, references, velocities = (
        SupportedBimanualFoamBlockClampController._trajectory(initial, 2.8)
    )
    assert phase is BimanualClampPhase.SQUEEZE
    np.testing.assert_array_equal(references, CONTACT_ARM_REFERENCE_RAD)
    np.testing.assert_array_equal(velocities, np.zeros(7))

    phase, references, velocities = (
        SupportedBimanualFoamBlockClampController._trajectory(initial, 4.2)
    )
    assert phase is BimanualClampPhase.SQUEEZE_HOLD
    np.testing.assert_array_equal(references, SQUEEZE_ARM_REFERENCE_RAD)
    np.testing.assert_array_equal(velocities, np.zeros(7))


def test_contact_classifier_is_read_only_and_reports_initial_support(
    environment: SupportedBimanualFoamBlockEnv,
) -> None:
    environment.reset(seed=RESET_SEED)
    classifier = FoamBlockContactClassifier(environment)
    qpos = environment.data.qpos.copy()
    qvel = environment.data.qvel.copy()
    contacts = classifier.classify()
    assert contacts.left_palm == ()
    assert contacts.right_palm == ()
    assert not contacts.bilateral_palm
    assert contacts.support_normal_force_n >= 0.0
    assert contacts.other == ()
    separation_segment = np.zeros(6, dtype=np.float64)
    block_cb_distance = mujoco.mj_geomDistance(
        environment.model,
        environment.data,
        int(environment.model.geom("foam_block_geom").id),
        int(environment.model.geom("cb").id),
        1.0,
        separation_segment,
    )
    assert block_cb_distance > 0.0
    assert block_cb_distance == pytest.approx(0.006, abs=1e-9)
    np.testing.assert_array_equal(environment.data.qpos, qpos)
    np.testing.assert_array_equal(environment.data.qvel, qvel)


def test_active_clamp_is_bit_exact_and_holds_bilateral_force_window(
    physical_replays: tuple[_ClampReplay, ...],
) -> None:
    """Two seeded actuator-only runs reproduce the calibrated clamp."""

    active, repeated, _, _ = physical_replays
    np.testing.assert_array_equal(active.action_trace, repeated.action_trace)
    np.testing.assert_array_equal(active.qpos_trace, repeated.qpos_trace)
    np.testing.assert_array_equal(
        active.final_block_position_m,
        repeated.final_block_position_m,
    )
    assert active.action_sha256 == repeated.action_sha256
    assert active.qpos_sha256 == repeated.qpos_sha256
    assert active.action_sha256 == EXPECTED_ACTION_SHA256
    assert active.qpos_sha256 == EXPECTED_QPOS_SHA256

    assert active.bilateral_hold_seconds >= 2.5
    assert active.force_window_hold_seconds >= 2.5
    assert active.maximum_distinct_palm_geoms >= 2
    assert active.other_contact_geom_names == ()
    assert active.supported_bilateral_ratio >= 0.95
    assert active.maximum_normal_force_n <= 5.0
    assert active.saturated_actuators == ()

    displacement = float(
        np.linalg.norm(
            active.final_block_position_m - active.initial_block_position_m
        )
    )
    assert displacement <= 0.05
    assert active.maximum_hand_relative_drift_m <= 0.06
    assert active.maximum_oriented_bottom_gap_m <= 0.0005


def test_open_and_one_arm_controls_do_not_form_bilateral_clamp(
    physical_replays: tuple[_ClampReplay, ...],
) -> None:
    """Open and unilateral trajectories cannot satisfy the clamp gate."""

    _, _, open_control, one_arm_control = physical_replays
    assert open_control.bilateral_hold_seconds <= 0.01
    assert one_arm_control.bilateral_hold_seconds <= 0.01
    assert open_control.force_window_hold_seconds <= 0.01
    assert one_arm_control.force_window_hold_seconds <= 0.01
    assert open_control.saturated_actuators == ()
    assert one_arm_control.saturated_actuators == ()

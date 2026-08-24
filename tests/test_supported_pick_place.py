"""Unit contracts for the supported scripted sphere pick-and-place."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Iterator

import mujoco
import numpy as np
import pytest

from blight_world import resolve_motor_channel
from blight_world.supported_pick_place import (
    DIGIT_GROUPS,
    PREGRASP_ARM_SEED_RAD,
    RIGHT_DIGIT_ACTUATOR_NAMES,
    RIGHT_DIGIT_JOINT_RANGES_RAD,
    RIGHT_DIGIT_NOMINAL_CURL_RAD,
    RIGHT_DIGIT_SEARCH_CAP_RAD,
    HandTargetContact,
    PickPlaceContacts,
    PickPlacePhase,
    RightHandSpherePickPlaceController,
    SupportedPickPlaceConfig,
    SupportedPickPlaceEnv,
)


RESET_SEED = 1401


def _unit_config(**overrides: float | None) -> SupportedPickPlaceConfig:
    """Return explicit synthetic gains used only for state-machine unit tests."""

    values: dict[str, float | None] = {
        "digit_closure_fraction_per_second": 0.5,
        "digit_position_action_per_rad": 0.2,
        "digit_velocity_action_seconds_per_rad": 0.01,
    }
    values.update(overrides)
    return SupportedPickPlaceConfig(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def environment() -> Iterator[SupportedPickPlaceEnv]:
    instance = SupportedPickPlaceEnv(width=16, height=12)
    try:
        instance.reset(seed=RESET_SEED)
        yield instance
    finally:
        instance.close()


def _sample(
    geom_name: str,
    group: str,
    radial: tuple[float, float, float],
    *,
    force_n: float = 0.01,
) -> HandTargetContact:
    return HandTargetContact(
        geom_name=geom_name,
        group=group,
        radial_direction=radial,
        normal_force_n=force_n,
    )


def _secure_contacts(*, table_contact: bool = False) -> PickPlaceContacts:
    return PickPlaceContacts(
        samples=(
            _sample("geom:right_thdistal1", "thumb", (1.0, 0.0, 0.0)),
            _sample("geom:right_ffmiddle1", "ff", (-1.0, 0.1, 0.0)),
            _sample("geom:right_mfmiddle1", "mf", (-1.0, -0.1, 0.0)),
        ),
        table_contact=table_contact,
    )


def test_scene_reuses_free_target_and_only_body_support_equality(
    environment: SupportedPickPlaceEnv,
) -> None:
    """No target weld, grasp equality, mocap body, or alternate object exists."""

    target_body_id = int(environment.model.body("push_target").id)
    target_joint = environment.model.joint("push_target_joint")
    assert int(target_joint.type.item()) == int(mujoco.mjtJoint.mjJNT_FREE)
    target_geom = environment.model.geom("push_target_geom")
    assert float(environment.model.body_mass[target_body_id]) == pytest.approx(0.020)
    assert float(target_geom.size[0]) == pytest.approx(0.020)
    assert environment.model.nmocap == 0

    body_equalities: list[int] = []
    for equality_id in range(environment.model.neq):
        if int(environment.model.eq_type[equality_id]) in {
            int(mujoco.mjtEq.mjEQ_CONNECT),
            int(mujoco.mjtEq.mjEQ_WELD),
        }:
            body_equalities.append(equality_id)
            assert target_body_id not in {
                int(environment.model.eq_obj1id[equality_id]),
                int(environment.model.eq_obj2id[equality_id]),
            }
    support_id = int(environment.model.equality("root_support").id)
    assert body_equalities == [support_id]
    assert {
        int(environment.model.eq_obj1id[support_id]),
        int(environment.model.eq_obj2id[support_id]),
    } == {0, int(environment.model.body("mimo_location").id)}


def test_exact_twenty_two_digit_channels_resolve_by_name_and_range(
    environment: SupportedPickPlaceEnv,
) -> None:
    channels = tuple(
        resolve_motor_channel(environment, name)
        for name in RIGHT_DIGIT_ACTUATOR_NAMES
    )
    assert len(channels) == 22
    assert len({channel.actuator_id for channel in channels}) == 22
    assert [channel.action_index for channel in channels] == list(range(21, 43))
    compiled_ranges = np.asarray(
        [environment.model.jnt_range[channel.joint_id] for channel in channels]
    )
    np.testing.assert_allclose(
        compiled_ranges,
        RIGHT_DIGIT_JOINT_RANGES_RAD,
        rtol=0,
        atol=5e-6,
    )
    for reference_set in (
        RIGHT_DIGIT_NOMINAL_CURL_RAD,
        RIGHT_DIGIT_SEARCH_CAP_RAD,
    ):
        references = np.asarray(reference_set)
        assert np.all(references >= compiled_ranges[:, 0] - 5e-6)
        assert np.all(references <= compiled_ranges[:, 1] + 5e-6)

    controller = RightHandSpherePickPlaceController(environment, _unit_config())
    assert len(controller.controlled_actuator_names) == 31
    assert len(set(controller.controlled_actuator_names)) == 31
    expected = {
        *RIGHT_DIGIT_ACTUATOR_NAMES,
        "act:hip_bend",
        "act:head_tilt",
        "act:right_shoulder_horizontal",
        "act:right_shoulder_abduction",
        "act:right_shoulder_internal",
        "act:right_elbow",
        "act:right_wrist_rotation",
        "act:right_wrist_flexion",
        "act:right_wrist_ulnar",
    }
    assert set(controller.controlled_actuator_names) == expected


def test_uncalibrated_default_fails_closed_before_motor_command(
    environment: SupportedPickPlaceEnv,
) -> None:
    environment.reset(seed=RESET_SEED)
    controller = RightHandSpherePickPlaceController(environment)
    state = controller.reset(environment)
    qpos = environment.data.qpos.copy()
    qvel = environment.data.qvel.copy()
    with pytest.raises(RuntimeError, match="require physical calibration"):
        controller.compose(environment, state, contacts=PickPlaceContacts())
    np.testing.assert_array_equal(environment.data.qpos, qpos)
    np.testing.assert_array_equal(environment.data.qvel, qvel)


def test_opposition_requires_thumb_and_two_distinct_opposed_geometries() -> None:
    secure = _secure_contacts()
    assert secure.has_hand_contact
    assert secure.contacted_digit_groups == frozenset({"thumb", "ff", "mf"})
    assert secure.opposed_non_thumb_geom_names == (
        "geom:right_ffmiddle1",
        "geom:right_mfmiddle1",
    )
    assert secure.secure_geometry

    parallel = PickPlaceContacts(
        samples=(
            _sample("geom:right_thdistal1", "thumb", (1.0, 0.0, 0.0)),
            _sample("geom:right_ffmiddle1", "ff", (0.0, 1.0, 0.0)),
            _sample("geom:right_mfmiddle1", "mf", (1.0, 0.0, 0.0)),
        )
    )
    assert not parallel.secure_geometry
    two_geometries = PickPlaceContacts(samples=secure.samples[:2])
    assert not two_geometries.secure_geometry


def test_contact_classifier_covers_entire_right_hand_subtree(
    environment: SupportedPickPlaceEnv,
) -> None:
    classifier = RightHandSpherePickPlaceController(
        environment, _unit_config()
    ).contact_classifier
    names = {
        mujoco.mj_id2name(environment.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in classifier.hand_geom_ids
    }
    assert None not in names
    assert {
        "geom:right_hand1",
        "geom:right_ffknuckle1",
        "geom:right_mfdistal1",
        "geom:right_lfmetacarpal1",
        "geom:right_thdistal1",
    }.issubset(names)
    assert all(name is not None and name.startswith("geom:right_") for name in names)


def test_state_machine_freezes_arm_and_gates_each_digit_group(
    environment: SupportedPickPlaceEnv,
) -> None:
    environment.reset(seed=RESET_SEED)
    controller = RightHandSpherePickPlaceController(environment, _unit_config())
    state = controller.reset(environment)
    with pytest.raises(FrozenInstanceError):
        state.phase = PickPlacePhase.FAIL  # type: ignore[misc]
    qpos0 = environment.data.qpos.copy()
    qvel0 = environment.data.qvel.copy()
    xfrc0 = environment.data.xfrc_applied.copy()

    action, state, diagnostics = controller.compose(
        environment,
        state,
        contacts=PickPlaceContacts(table_contact=True),
        time_seconds=1.0,
    )
    assert state.phase is PickPlacePhase.REACH_PREGRASP
    assert diagnostics.phase is PickPlacePhase.REACH_PREGRASP
    assert environment.action_space.contains(action)
    controlled_indices = {
        resolve_motor_channel(environment, name).action_index
        for name in controller.controlled_actuator_names
    }
    assert set(np.flatnonzero(action)).issubset(controlled_indices)

    _, state, _ = controller.compose(
        environment,
        state,
        contacts=PickPlaceContacts(table_contact=True),
        time_seconds=4.5,
    )
    assert state.phase is PickPlacePhase.FINAL_APPROACH
    contact = PickPlaceContacts(
        samples=(
            _sample("geom:right_ffmiddle1", "ff", (-1.0, 0.0, 0.0)),
        ),
        table_contact=True,
    )
    final_approach_state = state
    _, stale_state, _ = controller.compose(
        environment,
        final_approach_state,
        contacts=contact,
        time_seconds=4.6,
    )
    assert stale_state.phase is PickPlacePhase.FINAL_APPROACH
    assert stale_state.first_hand_contact_seconds is None
    _, state, _ = controller.compose(
        environment,
        final_approach_state,
        contacts=PickPlaceContacts(table_contact=True),
        time_seconds=4.59,
    )
    frozen_expected = tuple(
        float(environment.data.qpos[resolve_motor_channel(environment, name).qpos_address])
        for name in controller.controlled_actuator_names[:7]
    )
    _, state, _ = controller.compose(
        environment,
        state,
        contacts=contact,
        time_seconds=4.60,
    )
    assert state.phase is PickPlacePhase.ENCLOSE
    assert state.frozen_arm_positions_rad == frozen_expected

    _, state, diagnostics = controller.compose(
        environment,
        state,
        contacts=contact,
        time_seconds=4.8,
    )
    fractions = dict(zip(DIGIT_GROUPS, state.digit_closure_fractions, strict=True))
    assert fractions["ff"] == 0.0
    for group in ("mf", "rf", "lf", "thumb"):
        assert fractions[group] == pytest.approx(0.1)
    ff_indices = [
        index
        for index, name in enumerate(RIGHT_DIGIT_ACTUATOR_NAMES)
        if name.startswith("act:right_ff_")
    ]
    assert all(diagnostics.digit_reference_positions_rad[index] == 0.0 for index in ff_indices)

    _, state, _ = controller.compose(
        environment,
        state,
        contacts=_secure_contacts(table_contact=True),
        time_seconds=4.9,
    )
    assert state.phase is PickPlacePhase.SECURE
    _, state, _ = controller.compose(
        environment,
        state,
        contacts=_secure_contacts(table_contact=True),
        time_seconds=5.151,
    )
    assert state.phase is PickPlacePhase.LIFT
    np.testing.assert_array_equal(environment.data.qpos, qpos0)
    np.testing.assert_array_equal(environment.data.qvel, qvel0)
    np.testing.assert_array_equal(environment.data.xfrc_applied, xfrc0)


def test_pregrasp_reference_and_no_contact_deadline_are_deterministic(
    environment: SupportedPickPlaceEnv,
) -> None:
    environment.reset(seed=RESET_SEED)
    controller = RightHandSpherePickPlaceController(environment, _unit_config())
    state = controller.reset(environment)
    _, state, _ = controller.compose(
        environment, state, contacts=PickPlaceContacts(), time_seconds=1.0
    )
    _, state, diagnostics = controller.compose(
        environment, state, contacts=PickPlaceContacts(), time_seconds=2.75
    )
    midpoint = 0.5 * (
        np.asarray(state.initial_arm_positions_rad) + np.asarray(PREGRASP_ARM_SEED_RAD)
    )
    np.testing.assert_allclose(
        diagnostics.arm_reference_positions_rad, midpoint, rtol=0, atol=1e-15
    )
    _, state, _ = controller.compose(
        environment, state, contacts=PickPlaceContacts(), time_seconds=4.5
    )
    assert state.phase is PickPlacePhase.FINAL_APPROACH
    _, state, diagnostics = controller.compose(
        environment, state, contacts=PickPlaceContacts(), time_seconds=6.501
    )
    assert state.phase is PickPlacePhase.FAIL
    assert diagnostics.failure_reason == "no hand-target contact before approach deadline"


def test_active_pregrasp_contact_is_latched_without_time_jump_or_state_write(
    environment: SupportedPickPlaceEnv,
) -> None:
    """A real-time pregrasp contact freezes the arm and bypasses timeout."""

    environment.reset(seed=RESET_SEED)
    controller = RightHandSpherePickPlaceController(environment, _unit_config())
    state = controller.reset(environment)
    _, state, _ = controller.compose(
        environment,
        state,
        contacts=PickPlaceContacts(),
        time_seconds=1.0,
    )
    assert state.phase is PickPlacePhase.REACH_PREGRASP

    active_contact = PickPlaceContacts(
        samples=(
            _sample("geom:right_ffmiddle1", "ff", (-1.0, 0.0, 0.0)),
        ),
        table_contact=True,
    )
    _, jumped_state, _ = controller.compose(
        environment,
        state,
        contacts=active_contact,
        time_seconds=3.98,
    )
    assert jumped_state.phase is PickPlacePhase.REACH_PREGRASP
    assert jumped_state.first_hand_contact_seconds is None
    assert jumped_state.frozen_arm_positions_rad is None

    _, state, _ = controller.compose(
        environment,
        state,
        contacts=PickPlaceContacts(),
        time_seconds=3.97,
    )
    qpos = environment.data.qpos.copy()
    qvel = environment.data.qvel.copy()
    xfrc = environment.data.xfrc_applied.copy()
    frozen_expected = tuple(
        float(environment.data.qpos[resolve_motor_channel(environment, name).qpos_address])
        for name in controller.controlled_actuator_names[:7]
    )
    _, state, diagnostics = controller.compose(
        environment,
        state,
        contacts=active_contact,
        time_seconds=3.98,
    )
    assert state.phase is PickPlacePhase.FINAL_APPROACH
    assert state.first_hand_contact_seconds == 3.98
    assert state.frozen_arm_positions_rad == frozen_expected
    assert diagnostics.arm_reference_positions_rad == frozen_expected
    np.testing.assert_array_equal(environment.data.qpos, qpos)
    np.testing.assert_array_equal(environment.data.qvel, qvel)
    np.testing.assert_array_equal(environment.data.xfrc_applied, xfrc)

    _, state, diagnostics = controller.compose(
        environment,
        state,
        contacts=PickPlaceContacts(table_contact=True),
        time_seconds=3.99,
    )
    assert state.phase is PickPlacePhase.ENCLOSE
    assert diagnostics.failure_reason is None

    environment.reset(seed=RESET_SEED)
    inactive_controller = RightHandSpherePickPlaceController(
        environment, _unit_config()
    )
    inactive_state = inactive_controller.reset(environment)
    _, inactive_state, _ = inactive_controller.compose(
        environment,
        inactive_state,
        contacts=PickPlaceContacts(),
        time_seconds=1.0,
    )
    _, inactive_state, _ = inactive_controller.compose(
        environment,
        inactive_state,
        contacts=PickPlaceContacts(),
        time_seconds=3.97,
    )
    inactive = PickPlaceContacts(
        samples=(
            _sample(
                "geom:right_ffmiddle1",
                "ff",
                (-1.0, 0.0, 0.0),
                force_n=0.0,
            ),
        )
    )
    _, inactive_state, _ = inactive_controller.compose(
        environment,
        inactive_state,
        contacts=inactive,
        time_seconds=3.98,
    )
    assert inactive_state.phase is PickPlacePhase.REACH_PREGRASP
    assert inactive_state.first_hand_contact_seconds is None

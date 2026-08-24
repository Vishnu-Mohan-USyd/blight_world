"""Physical regressions for the supported right-hand sphere push."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import mujoco
import numpy as np
import pytest
from numpy.typing import NDArray

from blight_world import (
    RIGHT_ARM_ACTUATOR_NAMES,
    RIGHT_ARM_TARGET_RAD,
    RightHandSpherePushController,
    SupportedPushEnv,
    resolve_motor_channel,
)


TRIAL_SECONDS = 8.0
RESET_SEED = 1401
ALLOWED_HAND_TARGET_GEOMS = {
    "geom:right_hand3",
    "geom:right_ffknuckle1",
    "geom:right_ffmiddle1",
}
ALLOWED_HAND_PUSH_TABLE_GEOMS = {
    "geom:right_ffknuckle1",
    "geom:right_mfdistal1",
    "geom:right_mfknuckle1",
    "geom:right_mfmiddle1",
    "geom:right_rfknuckle1",
}


@dataclass(frozen=True, slots=True)
class _TrialResult:
    trace: NDArray[np.float64]
    initial_target_position_m: NDArray[np.float64]
    final_target_position_m: NDArray[np.float64]
    final_target_speed_m_s: float
    target_speed_at_6_5_s_m_s: float
    first_hand_contact_time_s: float | None
    displacement_20mm_time_s: float | None
    target_hand_geoms: tuple[str, ...]
    push_table_hand_geoms: tuple[str, ...]
    unexpected_external_contacts: tuple[tuple[str, str], ...]
    max_target_contact_normal_force_n: float
    max_root_position_error_m: float
    max_root_rotation_error_deg: float
    saturated_actuators: tuple[str, ...]


@pytest.fixture(scope="module")
def environment() -> Iterator[SupportedPushEnv]:
    instance = SupportedPushEnv(width=16, height=12)
    try:
        yield instance
    finally:
        instance.close()


def _name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    value = mujoco.mj_id2name(model, object_type, object_id)
    assert value is not None
    return value


def _rotation_error_degrees(
    reference: NDArray[np.float64],
    current: NDArray[np.float64],
) -> float:
    relative = reference.reshape(3, 3).T @ current.reshape(3, 3)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _run_trial(*, execute_push: bool) -> _TrialResult:
    environment = SupportedPushEnv(width=16, height=12)
    try:
        observation, info = environment.reset(seed=RESET_SEED)
        assert info == {}
        assert environment.observation_space.contains(observation)
        controller = RightHandSpherePushController(environment)
        state = controller.reset(environment)

        target_body_id = int(environment.model.body("push_target").id)
        target_geom_id = int(environment.model.geom("push_target_geom").id)
        target_joint = environment.model.joint("push_target_joint")
        target_dof = int(np.asarray(target_joint.dofadr).item())
        mimo_root_id = int(environment.model.body("mimo_location").id)
        initial_target = np.asarray(
            environment.data.xpos[target_body_id], dtype=np.float64
        ).copy()
        root_position = np.asarray(
            environment.data.xpos[mimo_root_id], dtype=np.float64
        ).copy()
        root_rotation = np.asarray(
            environment.data.xmat[mimo_root_id], dtype=np.float64
        ).copy()

        trace: list[NDArray[np.float64]] = []
        hand_geoms: set[str] = set()
        push_table_hand_geoms: set[str] = set()
        unexpected_contacts: set[tuple[str, str]] = set()
        saturated: set[str] = set()
        first_hand_contact_time: float | None = None
        displacement_20mm_time: float | None = None
        speed_at_6_5: float | None = None
        max_normal_force = 0.0
        max_root_position_error = 0.0
        max_root_rotation_error = 0.0
        steps = int(round(TRIAL_SECONDS / environment.dt))

        controlled_indices = {
            resolve_motor_channel(environment, name).action_index
            for name in controller.controlled_actuator_names
        }
        for _ in range(steps):
            qpos_before = environment.data.qpos.copy()
            qvel_before = environment.data.qvel.copy()
            xfrc_before = environment.data.xfrc_applied.copy()
            action, state, diagnostics = controller.compose(
                environment,
                state,
                execute_push=execute_push,
            )
            np.testing.assert_array_equal(environment.data.qpos, qpos_before)
            np.testing.assert_array_equal(environment.data.qvel, qvel_before)
            np.testing.assert_array_equal(environment.data.xfrc_applied, xfrc_before)
            assert environment.action_space.contains(action)
            assert set(np.flatnonzero(action)).issubset(controlled_indices)
            saturated.update(diagnostics.saturated_actuators)

            observation, _, terminated, truncated, _ = environment.step(action)
            assert not terminated
            assert not truncated
            assert environment.observation_space.contains(observation)
            np.testing.assert_array_equal(
                environment.data.xfrc_applied,
                np.zeros_like(environment.data.xfrc_applied),
            )

            target_position = np.asarray(
                environment.data.xpos[target_body_id], dtype=np.float64
            ).copy()
            target_velocity = np.asarray(
                environment.data.qvel[target_dof : target_dof + 6],
                dtype=np.float64,
            ).copy()
            planar_displacement = float(
                np.linalg.norm(target_position[:2] - initial_target[:2])
            )
            if planar_displacement >= 0.020 and displacement_20mm_time is None:
                displacement_20mm_time = float(environment.data.time)
            target_speed = float(np.linalg.norm(target_velocity[:3]))
            if environment.data.time >= 6.5 and speed_at_6_5 is None:
                speed_at_6_5 = target_speed

            max_root_position_error = max(
                max_root_position_error,
                float(
                    np.linalg.norm(
                        environment.data.xpos[mimo_root_id] - root_position
                    )
                ),
            )
            max_root_rotation_error = max(
                max_root_rotation_error,
                _rotation_error_degrees(
                    root_rotation,
                    np.asarray(
                        environment.data.xmat[mimo_root_id], dtype=np.float64
                    ),
                ),
            )

            for contact_index, contact in enumerate(environment.data.contact):
                if int(contact.efc_address) < 0:
                    continue
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                body1 = int(environment.model.geom_bodyid[geom1])
                body2 = int(environment.model.geom_bodyid[geom2])
                first_is_mimo = int(environment.model.body_rootid[body1]) == mimo_root_id
                second_is_mimo = int(environment.model.body_rootid[body2]) == mimo_root_id
                if first_is_mimo != second_is_mimo:
                    mimo_geom = geom1 if first_is_mimo else geom2
                    external_geom = geom2 if first_is_mimo else geom1
                    names = (
                        _name(
                            environment.model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            mimo_geom,
                        ),
                        _name(
                            environment.model,
                            mujoco.mjtObj.mjOBJ_GEOM,
                            external_geom,
                        ),
                    )
                    if (
                        external_geom == target_geom_id
                        and names[0] in ALLOWED_HAND_TARGET_GEOMS
                    ):
                        pass
                    elif (
                        names[1] == "push_table_top"
                        and names[0] in ALLOWED_HAND_PUSH_TABLE_GEOMS
                    ):
                        push_table_hand_geoms.add(names[0])
                    else:
                        unexpected_contacts.add(names)
                if geom1 != target_geom_id and geom2 != target_geom_id:
                    continue
                other_geom = geom2 if geom1 == target_geom_id else geom1
                other_name = _name(
                    environment.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    other_geom,
                )
                if other_name not in ALLOWED_HAND_TARGET_GEOMS:
                    continue
                hand_geoms.add(other_name)
                if first_hand_contact_time is None:
                    first_hand_contact_time = float(environment.data.time)
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(
                    environment.model,
                    environment.data,
                    contact_index,
                    force,
                )
                max_normal_force = max(max_normal_force, float(force[0]))

            trace.append(
                np.concatenate(
                    (
                        action.astype(np.float64),
                        target_position,
                        target_velocity,
                        np.asarray(
                            (
                                max_root_position_error,
                                max_root_rotation_error,
                            ),
                            dtype=np.float64,
                        ),
                    )
                )
            )

        assert speed_at_6_5 is not None
        final_position = np.asarray(
            environment.data.xpos[target_body_id], dtype=np.float64
        ).copy()
        final_speed = float(
            np.linalg.norm(environment.data.qvel[target_dof : target_dof + 3])
        )
        return _TrialResult(
            trace=np.asarray(trace, dtype=np.float64),
            initial_target_position_m=initial_target,
            final_target_position_m=final_position,
            final_target_speed_m_s=final_speed,
            target_speed_at_6_5_s_m_s=speed_at_6_5,
            first_hand_contact_time_s=first_hand_contact_time,
            displacement_20mm_time_s=displacement_20mm_time,
            target_hand_geoms=tuple(sorted(hand_geoms)),
            push_table_hand_geoms=tuple(sorted(push_table_hand_geoms)),
            unexpected_external_contacts=tuple(sorted(unexpected_contacts)),
            max_target_contact_normal_force_n=max_normal_force,
            max_root_position_error_m=max_root_position_error,
            max_root_rotation_error_deg=max_root_rotation_error,
            saturated_actuators=tuple(sorted(saturated)),
        )
    finally:
        environment.close()


@pytest.fixture(scope="module")
def physical_trials() -> tuple[_TrialResult, _TrialResult, _TrialResult]:
    """Run two active seeded replays and one neutral-arm control."""

    return (
        _run_trial(execute_push=True),
        _run_trial(execute_push=True),
        _run_trial(execute_push=False),
    )


def test_supported_scene_is_free_target_with_only_root_support(
    environment: SupportedPushEnv,
) -> None:
    """The dedicated scene exposes support without welding the target."""

    target_body_id = int(environment.model.body("push_target").id)
    target_joint = environment.model.joint("push_target_joint")
    assert int(target_joint.type.item()) == int(mujoco.mjtJoint.mjJNT_FREE)
    assert int(environment.model.geom("push_target_geom").bodyid.item()) == target_body_id
    assert tuple(environment.model.geom("push_target_geom").size[:1]) == (0.02,)
    assert environment.touch is None
    assert environment.vestibular is None
    assert environment.audition is None

    support_id = int(environment.model.equality("root_support").id)
    mimo_root_id = int(environment.model.body("mimo_location").id)
    assert int(environment.model.eq_type[support_id]) == int(mujoco.mjtEq.mjEQ_WELD)
    assert {
        int(environment.model.eq_obj1id[support_id]),
        int(environment.model.eq_obj2id[support_id]),
    } == {0, mimo_root_id}
    np.testing.assert_array_equal(
        environment.model.eq_solref[support_id],
        np.asarray((0.01, 1.0)),
    )
    np.testing.assert_array_equal(
        environment.model.eq_solimp[support_id],
        np.asarray((0.99, 0.999, 0.001, 0.5, 2.0)),
    )
    for equality_id in range(environment.model.neq):
        if int(environment.model.eq_type[equality_id]) not in {
            int(mujoco.mjtEq.mjEQ_CONNECT),
            int(mujoco.mjtEq.mjEQ_WELD),
        }:
            continue
        assert target_body_id not in {
            int(environment.model.eq_obj1id[equality_id]),
            int(environment.model.eq_obj2id[equality_id]),
        }


def test_push_controller_is_sparse_smooth_and_state_read_only(
    environment: SupportedPushEnv,
) -> None:
    environment.reset(seed=RESET_SEED)
    controller = RightHandSpherePushController(environment)
    state = controller.reset(environment)
    qpos = environment.data.qpos.copy()
    qvel = environment.data.qvel.copy()

    action, next_state, diagnostics = controller.compose(
        environment,
        state,
        time_seconds=3.0,
    )
    np.testing.assert_array_equal(environment.data.qpos, qpos)
    np.testing.assert_array_equal(environment.data.qvel, qvel)
    assert diagnostics.phase == "reach"
    expected_midpoint = 0.5 * (
        np.asarray(state.initial_arm_positions_rad) + np.asarray(RIGHT_ARM_TARGET_RAD)
    )
    np.testing.assert_allclose(
        diagnostics.arm_reference_positions_rad,
        expected_midpoint,
        rtol=0,
        atol=1e-15,
    )
    controlled_indices = {
        resolve_motor_channel(environment, name).action_index
        for name in controller.controlled_actuator_names
    }
    assert set(np.flatnonzero(action)).issubset(controlled_indices)
    assert next_state.motor_state.step_index == 1
    assert next_state.initial_arm_positions_rad == state.initial_arm_positions_rad


def test_active_push_is_bit_exact_and_physically_contact_driven(
    physical_trials: tuple[_TrialResult, _TrialResult, _TrialResult],
) -> None:
    first, repeated, _ = physical_trials
    np.testing.assert_array_equal(first.trace, repeated.trace)
    np.testing.assert_array_equal(
        first.final_target_position_m,
        repeated.final_target_position_m,
    )
    assert first.first_hand_contact_time_s == repeated.first_hand_contact_time_s
    assert first.target_hand_geoms == repeated.target_hand_geoms

    displacement = float(
        np.linalg.norm(
            first.final_target_position_m[:2]
            - first.initial_target_position_m[:2]
        )
    )
    assert 0.020 <= displacement <= 0.040
    assert first.first_hand_contact_time_s is not None
    assert first.displacement_20mm_time_s is not None
    assert first.first_hand_contact_time_s < first.displacement_20mm_time_s
    assert "geom:right_hand3" in first.target_hand_geoms
    assert set(first.target_hand_geoms).issubset(ALLOWED_HAND_TARGET_GEOMS)
    assert set(first.push_table_hand_geoms) == ALLOWED_HAND_PUSH_TABLE_GEOMS
    assert first.max_target_contact_normal_force_n > 0.0
    assert first.unexpected_external_contacts == ()
    assert first.saturated_actuators == ()

    final = first.final_target_position_m
    assert 0.125 <= final[0] <= 0.595
    assert -0.35 <= final[1] <= -0.05
    assert 0.357 <= final[2] <= 0.361
    assert first.target_speed_at_6_5_s_m_s < 1e-5
    assert first.final_target_speed_m_s < 1e-5
    assert first.max_root_position_error_m <= 0.001
    assert first.max_root_rotation_error_deg <= 0.1


def test_neutral_arm_has_no_target_contact_or_planar_motion(
    physical_trials: tuple[_TrialResult, _TrialResult, _TrialResult],
) -> None:
    _, _, neutral = physical_trials
    planar_displacement = float(
        np.linalg.norm(
            neutral.final_target_position_m[:2]
            - neutral.initial_target_position_m[:2]
        )
    )
    assert neutral.first_hand_contact_time_s is None
    assert neutral.target_hand_geoms == ()
    assert neutral.push_table_hand_geoms == ()
    assert planar_displacement <= 1e-12
    assert neutral.unexpected_external_contacts == ()
    assert neutral.saturated_actuators == ()

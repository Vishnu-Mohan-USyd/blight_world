"""Validation for the unsupported static-support milestone."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from hashlib import sha256
import json
from typing import Iterator

import mujoco
import numpy as np
import pytest

from blight_world import resolve_motor_channel
from blight_world.locomotion_skills import (
    ACTION_RAMP_SECONDS,
    BILATERAL_STATIC_ACTIONS,
    CENTRAL_STATIC_ACTIONS,
    UNSUPPORTED_ROOT_QPOS,
    StaticSupportContact,
    StaticSupportContactClassifier,
    StaticSupportContacts,
    StaticSupportControllerState,
    SupportRegion,
    UnsupportedStaticSupportController,
    UnsupportedStaticSupportEnv,
)


RESET_SEED = 1401
TRIAL_STEPS = 2000
EVALUATION_STEPS = 500


@dataclass(frozen=True, slots=True)
class _StaticReplay:
    trace: np.ndarray
    canonical_payload: bytes
    canonical_sha256: str
    support_fraction: dict[str, float]
    region_mean_force_n: dict[str, float]
    central_force_mean_n: float
    central_force_peak_n: float
    head_peak_all_n: float
    external_contacts: tuple[tuple[str, str], ...]
    com_polygon_margin_m: float
    root_drift_eval_m: np.ndarray
    max_root_linear_speed_eval_m_s: float
    max_root_angular_speed_eval_rad_s: float
    final_root_linear_speed_m_s: float
    final_root_angular_speed_rad_s: float
    energy_abs_mechanical_j: float
    action_min_bound_margin: float
    knee_distances_m: dict[str, float]
    substep_evaluation_support_fraction: dict[str, float]


@pytest.fixture(scope="module")
def environment() -> Iterator[UnsupportedStaticSupportEnv]:
    instance = UnsupportedStaticSupportEnv(width=16, height=12)
    try:
        yield instance
    finally:
        instance.close()


def test_reset_stages_only_root_freejoint_without_support_equality(
    environment: UnsupportedStaticSupportEnv,
) -> None:
    observation, info = environment.reset(seed=RESET_SEED)
    assert info == {}
    assert environment.observation_space.contains(observation)
    root_joint = environment.model.joint("mimo_orientation")
    root_qpos = int(np.asarray(root_joint.qposadr).item())
    root_dof = int(np.asarray(root_joint.dofadr).item())
    np.testing.assert_array_equal(
        environment.data.qpos[root_qpos : root_qpos + 7],
        np.asarray(UNSUPPORTED_ROOT_QPOS),
    )
    np.testing.assert_array_equal(
        environment.data.qpos[:root_qpos],
        environment.init_qpos[:root_qpos],
    )
    np.testing.assert_array_equal(
        environment.data.qpos[root_qpos + 7 :],
        environment.init_qpos[root_qpos + 7 :],
    )
    np.testing.assert_array_equal(environment.data.qvel, np.zeros(environment.model.nv))
    np.testing.assert_array_equal(
        environment.data.qvel[root_dof : root_dof + 6], np.zeros(6)
    )
    assert environment.model.nmocap == 0
    for equality_id in range(environment.model.neq):
        assert int(environment.model.eq_type[equality_id]) not in {
            int(mujoco.mjtEq.mjEQ_CONNECT),
            int(mujoco.mjtEq.mjEQ_WELD),
        }


def test_static_controller_is_sparse_bounded_and_read_only(
    environment: UnsupportedStaticSupportEnv,
) -> None:
    environment.reset(seed=RESET_SEED)
    controller = UnsupportedStaticSupportController(environment)
    first_reset = controller.reset()
    second_reset = controller.reset()
    assert first_reset == second_reset
    with pytest.raises(FrozenInstanceError):
        first_reset.motor_state = first_reset.motor_state  # type: ignore[misc]

    expected_targets: dict[str, float] = {}
    for side in ("right", "left"):
        for suffix, value in BILATERAL_STATIC_ACTIONS.items():
            expected_targets[f"act:{side}_{suffix}"] = value
    for suffix, value in CENTRAL_STATIC_ACTIONS.items():
        expected_targets[f"act:{suffix}"] = value
    assert set(controller.controlled_actuator_names) == set(expected_targets)
    assert len(controller.controlled_actuator_names) == 18

    qpos = environment.data.qpos.copy()
    qvel = environment.data.qvel.copy()
    xfrc = environment.data.xfrc_applied.copy()
    gravity = environment.model.opt.gravity.copy()
    equality_active = environment.model.eq_active0.copy()
    half, state, half_diagnostics = controller.compose(
        environment,
        first_reset,
        time_seconds=ACTION_RAMP_SECONDS / 2.0,
    )
    assert half_diagnostics.ramp_fraction == pytest.approx(0.5)
    assert environment.action_space.contains(half)
    for name, target in expected_targets.items():
        channel = resolve_motor_channel(environment, name)
        assert float(half[channel.action_index]) == pytest.approx(
            0.5 * target, abs=2e-8
        )

    full, _, diagnostics = controller.compose(
        environment,
        state,
        time_seconds=ACTION_RAMP_SECONDS,
    )
    assert diagnostics.ramp_fraction == 1.0
    assert environment.action_space.contains(full)
    nonzero_indices = {
        resolve_motor_channel(environment, name).action_index
        for name in expected_targets
    }
    assert set(np.flatnonzero(full)) == nonzero_indices
    for name, target in expected_targets.items():
        channel = resolve_motor_channel(environment, name)
        assert float(full[channel.action_index]) == pytest.approx(target, abs=2e-8)
    assert diagnostics.minimum_action_bound_margin == pytest.approx(
        0.30000001192092896
    )
    np.testing.assert_array_equal(environment.data.qpos, qpos)
    np.testing.assert_array_equal(environment.data.qvel, qvel)
    np.testing.assert_array_equal(environment.data.xfrc_applied, xfrc)
    np.testing.assert_array_equal(environment.model.opt.gravity, gravity)
    np.testing.assert_array_equal(environment.model.eq_active0, equality_active)


def test_support_polygon_margin_uses_convex_contact_hull() -> None:
    contacts = StaticSupportContacts(
        time_seconds=0.0,
        contacts=tuple(
            StaticSupportContact(
                region=region,
                geom_name=region.value,
                position_m=position,
                normal_force_n=1.0,
            )
            for region, position in zip(
                SupportRegion,
                (
                    (-0.1, 0.1, 0.0),
                    (0.1, 0.1, 0.0),
                    (-0.1, -0.1, 0.0),
                    (0.1, -0.1, 0.0),
                ),
                strict=True,
            )
        ),
        central_force_n=0.0,
        head_force_n=0.0,
        unexpected_external_contacts=(),
    )
    assert contacts.active_regions == frozenset(SupportRegion)
    assert StaticSupportContactClassifier.support_polygon_margin_m(
        contacts, np.asarray((0.0, 0.0))
    ) == pytest.approx(0.1)
    assert StaticSupportContactClassifier.support_polygon_margin_m(
        contacts, np.asarray((0.2, 0.0))
    ) < 0.0


def test_distal_knee_cap_boundary_is_inclusive_and_narrow() -> None:
    radius = 0.037
    knee = np.asarray((0.2, -0.1, 0.03))
    boundary = knee + np.asarray((radius + 0.010, 0.0, 0.0))
    outside = knee + np.asarray((radius + 0.010001, 0.0, 0.0))
    assert StaticSupportContactClassifier._is_distal_knee_contact(
        boundary, knee, radius
    )
    assert not StaticSupportContactClassifier._is_distal_knee_contact(
        outside, knee, radius
    )


def _run_static_replay() -> _StaticReplay:
    environment = UnsupportedStaticSupportEnv(width=16, height=12)
    try:
        observation, info = environment.reset(seed=RESET_SEED)
        assert info == {}
        assert environment.observation_space.contains(observation)
        controller = UnsupportedStaticSupportController(environment)
        state = controller.reset()
        classifier = StaticSupportContactClassifier(environment)
        root_joint = environment.model.joint("mimo_orientation")
        root_qpos = int(np.asarray(root_joint.qposadr).item())
        root_dof = int(np.asarray(root_joint.dofadr).item())
        root_body_id = int(environment.model.body("mimo_location").id)
        actuator_ids = np.asarray(environment.mimo_actuators, dtype=np.int64)
        actuator_dofs = environment.model.jnt_dofadr[
            environment.model.actuator_trnid[actuator_ids, 0]
        ]
        actuator_gears = environment.model.actuator_gear[actuator_ids, 0]
        gravity = environment.model.opt.gravity.copy()
        equality_active = environment.model.eq_active0.copy()

        substep_contacts: list[StaticSupportContacts] = []
        original_substep_callback = environment._substep_callback

        def record_substep() -> None:
            original_substep_callback()
            substep_contacts.append(classifier.classify())

        environment._substep_callback = record_substep  # type: ignore[method-assign]
        trace: list[np.ndarray] = []
        evaluation_contacts: list[StaticSupportContacts] = []
        evaluation_root_positions: list[np.ndarray] = []
        evaluation_linear_speeds: list[float] = []
        evaluation_angular_speeds: list[float] = []
        normal_force_integral_by_geom: dict[str, float] = {}
        contact_position_sum_by_geom: dict[str, np.ndarray] = {}
        contact_count_by_geom: dict[str, int] = {}
        region_steps = {region: 0 for region in SupportRegion}
        region_force_integral = {region: 0.0 for region in SupportRegion}
        central_force_integral = 0.0
        central_force_peak = 0.0
        head_peak_all = 0.0
        external_contacts: set[tuple[str, str]] = set()
        final_support_points: list[tuple[float, float]] = []
        energy_abs_mechanical = 0.0
        minimum_bound_margin = float("inf")

        for step_index in range(TRIAL_STEPS):
            qpos_before = environment.data.qpos.copy()
            qvel_before = environment.data.qvel.copy()
            xfrc_before = environment.data.xfrc_applied.copy()
            action, state, diagnostics = controller.compose(environment, state)
            np.testing.assert_array_equal(environment.data.qpos, qpos_before)
            np.testing.assert_array_equal(environment.data.qvel, qvel_before)
            np.testing.assert_array_equal(environment.data.xfrc_applied, xfrc_before)
            assert environment.action_space.contains(action)
            minimum_bound_margin = min(
                minimum_bound_margin,
                diagnostics.minimum_action_bound_margin,
            )
            torque = actuator_gears * action.astype(np.float64)
            energy_abs_mechanical += float(
                np.sum(np.abs(torque * environment.data.qvel[actuator_dofs]))
                * environment.dt
            )
            observation, _, terminated, truncated, _ = environment.step(action)
            assert not terminated
            assert not truncated
            assert environment.observation_space.contains(observation)
            np.testing.assert_array_equal(
                environment.data.xfrc_applied,
                np.zeros_like(environment.data.xfrc_applied),
            )
            contact_sample = classifier.classify()
            head_peak_all = max(head_peak_all, contact_sample.head_force_n)
            external_contacts.update(contact_sample.unexpected_external_contacts)
            root_state = np.concatenate(
                (
                    environment.data.qpos[root_qpos : root_qpos + 7],
                    environment.data.qvel[root_dof : root_dof + 6],
                )
            )
            trace.append(np.concatenate((action.astype(np.float64), root_state)))

            if step_index < TRIAL_STEPS - EVALUATION_STEPS:
                continue
            evaluation_contacts.append(contact_sample)
            evaluation_root_positions.append(
                environment.data.qpos[root_qpos : root_qpos + 3].copy()
            )
            evaluation_linear_speeds.append(
                float(np.linalg.norm(environment.data.qvel[root_dof : root_dof + 3]))
            )
            evaluation_angular_speeds.append(
                float(
                    np.linalg.norm(
                        environment.data.qvel[root_dof + 3 : root_dof + 6]
                    )
                )
            )
            active = contact_sample.active_regions
            force_by_region = dict(contact_sample.force_by_region_n)
            for region in SupportRegion:
                region_steps[region] += region in active
                region_force_integral[region] += (
                    force_by_region[region] * environment.dt
                )
            central_force_integral += contact_sample.central_force_n * environment.dt
            central_force_peak = max(
                central_force_peak, contact_sample.central_force_n
            )
            if contact_sample.contacts:
                final_support_points = [
                    (contact.position_m[0], contact.position_m[1])
                    for contact in contact_sample.contacts
                ]
            for contact in contact_sample.contacts:
                normal_force_integral_by_geom[contact.geom_name] = (
                    normal_force_integral_by_geom.get(contact.geom_name, 0.0)
                    + contact.normal_force_n * environment.dt
                )
                contact_position_sum_by_geom.setdefault(
                    contact.geom_name, np.zeros(3, dtype=np.float64)
                )
                contact_position_sum_by_geom[contact.geom_name] += np.asarray(
                    contact.position_m
                )
                contact_count_by_geom[contact.geom_name] = (
                    contact_count_by_geom.get(contact.geom_name, 0) + 1
                )

        assert len(substep_contacts) == 2 * TRIAL_STEPS
        assert len(evaluation_contacts) == EVALUATION_STEPS
        evaluation_seconds = EVALUATION_STEPS * environment.dt
        support_fraction = {
            region.value: region_steps[region] / EVALUATION_STEPS
            for region in SupportRegion
        }
        region_mean_force = {
            region.value: region_force_integral[region] / evaluation_seconds
            for region in SupportRegion
        }
        substep_window = substep_contacts[-2 * EVALUATION_STEPS :]
        substep_fraction = {
            region.value: sum(
                region in sample.active_regions for sample in substep_window
            )
            / len(substep_window)
            for region in SupportRegion
        }
        final_com = environment.data.subtree_com[root_body_id].copy()
        final_contacts_snapshot = evaluation_contacts[-1]
        margin = StaticSupportContactClassifier.support_polygon_margin_m(
            final_contacts_snapshot,
            final_com[:2],
        )
        normal_force_mean = {
            name: integral / evaluation_seconds
            for name, integral in sorted(normal_force_integral_by_geom.items())
        }
        contact_position_mean = {
            name: (
                contact_position_sum_by_geom[name] / contact_count_by_geom[name]
            ).tolist()
            for name in sorted(contact_position_sum_by_geom)
        }
        root_drift = (
            evaluation_root_positions[-1] - evaluation_root_positions[0]
        )
        payload_result = {
            "duration_s": 20.0,
            "evaluation_duration_s": 5.0,
            "support_fraction": support_fraction,
            "central_force_mean_n": central_force_integral / evaluation_seconds,
            "central_force_peak_n": central_force_peak,
            "head_peak_all_n": head_peak_all,
            "normal_force_mean_by_geom_n": normal_force_mean,
            "contact_position_mean_by_geom_m": contact_position_mean,
            "root_drift_eval_m": root_drift.tolist(),
            "max_root_linear_speed_eval_m_s": max(evaluation_linear_speeds),
            "max_root_angular_speed_eval_rad_s": max(evaluation_angular_speeds),
            "final_root_linear_speed_m_s": evaluation_linear_speeds[-1],
            "final_root_angular_speed_rad_s": evaluation_angular_speeds[-1],
            "energy_abs_mechanical_j": energy_abs_mechanical,
            "final_support_points_xy_m": [list(point) for point in final_support_points],
            "final_com_xyz_m": final_com.tolist(),
            "external_contacts": [list(pair) for pair in sorted(external_contacts)],
            "final_root_qpos": environment.data.qpos[
                root_qpos : root_qpos + 7
            ].tolist(),
            "com_polygon_margin_m": margin,
            "action_min_bound_margin": minimum_bound_margin,
            "action_space_contains": True,
            "transition_s": ACTION_RAMP_SECONDS,
            "root_stage": [
                0,
                0,
                0.35,
                UNSUPPORTED_ROOT_QPOS[3],
                0,
                UNSUPPORTED_ROOT_QPOS[5],
                0,
            ],
        }
        payload = json.dumps(
            payload_result, sort_keys=True, separators=(",", ":")
        ).encode()
        knee_distances = {
            contact.geom_name: float(contact.distal_knee_distance_m)
            for contact in final_contacts_snapshot.contacts
            if contact.distal_knee_distance_m is not None
        }
        np.testing.assert_array_equal(environment.model.opt.gravity, gravity)
        np.testing.assert_array_equal(environment.model.eq_active0, equality_active)
        return _StaticReplay(
            trace=np.asarray(trace),
            canonical_payload=payload,
            canonical_sha256=sha256(payload).hexdigest(),
            support_fraction=support_fraction,
            region_mean_force_n=region_mean_force,
            central_force_mean_n=central_force_integral / evaluation_seconds,
            central_force_peak_n=central_force_peak,
            head_peak_all_n=head_peak_all,
            external_contacts=tuple(sorted(external_contacts)),
            com_polygon_margin_m=margin,
            root_drift_eval_m=root_drift,
            max_root_linear_speed_eval_m_s=max(evaluation_linear_speeds),
            max_root_angular_speed_eval_rad_s=max(evaluation_angular_speeds),
            final_root_linear_speed_m_s=evaluation_linear_speeds[-1],
            final_root_angular_speed_rad_s=evaluation_angular_speeds[-1],
            energy_abs_mechanical_j=energy_abs_mechanical,
            action_min_bound_margin=minimum_bound_margin,
            knee_distances_m=knee_distances,
            substep_evaluation_support_fraction=substep_fraction,
        )
    finally:
        environment.close()


@pytest.fixture(scope="module")
def static_replays() -> tuple[_StaticReplay, _StaticReplay]:
    return _run_static_replay(), _run_static_replay()


def test_two_twenty_second_static_support_replays_are_stable_and_bit_exact(
    static_replays: tuple[_StaticReplay, _StaticReplay],
) -> None:
    first, repeated = static_replays
    np.testing.assert_array_equal(first.trace, repeated.trace)
    assert first.canonical_payload == repeated.canonical_payload
    assert first.canonical_sha256 == repeated.canonical_sha256
    assert first.support_fraction == {region.value: 1.0 for region in SupportRegion}
    assert first.substep_evaluation_support_fraction == {
        region.value: 1.0 for region in SupportRegion
    }
    assert first.central_force_mean_n == 0.0
    assert first.central_force_peak_n == 0.0
    assert first.head_peak_all_n == 0.0
    assert first.external_contacts == ()
    assert first.com_polygon_margin_m >= 0.08
    assert float(np.linalg.norm(first.root_drift_eval_m)) <= 0.001
    assert first.max_root_linear_speed_eval_m_s <= 0.00025
    assert first.max_root_angular_speed_eval_rad_s <= 0.00075
    assert first.final_root_linear_speed_m_s <= 0.00025
    assert first.final_root_angular_speed_rad_s <= 0.00075
    assert first.action_min_bound_margin >= 0.30
    assert np.isfinite(first.energy_abs_mechanical_j)
    assert first.energy_abs_mechanical_j > 0.0

    left_front = first.region_mean_force_n[SupportRegion.FRONT_LEFT.value]
    right_front = first.region_mean_force_n[SupportRegion.FRONT_RIGHT.value]
    left_rear = first.region_mean_force_n[SupportRegion.REAR_LEFT.value]
    right_rear = first.region_mean_force_n[SupportRegion.REAR_RIGHT.value]
    assert left_front >= 20.0
    assert right_front >= 20.0
    assert left_rear >= 15.0
    assert right_rear >= 15.0
    assert abs(left_front - right_front) / (0.5 * (left_front + right_front)) <= 0.01
    assert abs(left_rear - right_rear) / (0.5 * (left_rear + right_rear)) <= 0.01
    assert set(first.knee_distances_m) == {
        "geom:left_upper_leg1",
        "geom:right_upper_leg1",
    }
    assert all(distance <= 0.047 for distance in first.knee_distances_m.values())

    assert first.com_polygon_margin_m == pytest.approx(
        0.08160546774420886, abs=1e-5
    )
    np.testing.assert_allclose(
        first.root_drift_eval_m,
        (-0.00041313121849195084, 2.8300139323551905e-08, 0.0006509036170482263),
        rtol=0,
        atol=1e-6,
    )

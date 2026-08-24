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
    BELLY_ACTION_DELTAS,
    BELLY_BUMP_PLATEAU_STOP_SECONDS,
    BELLY_BUMP_RISE_SECONDS,
    BELLY_CRAWL_STOP_SECONDS,
    BELLY_FIRST_SIDE,
    BELLY_LIMB_PHASE_SECONDS,
    BELLY_MOVING_SIDE_DELTAS,
    BELLY_SETTLE_STOP_SECONDS,
    BELLY_STANCE_FRONT,
    BELLY_TRANSITION_START_SECONDS,
    BELLY_TRANSITION_STOP_SECONDS,
    BILATERAL_STATIC_ACTIONS,
    BellyCrawlPhase,
    CENTRAL_STATIC_ACTIONS,
    UNSUPPORTED_ROOT_QPOS,
    StaticSupportContact,
    StaticSupportContactClassifier,
    StaticSupportContacts,
    StaticSupportControllerState,
    SupportRegion,
    UnsupportedBellyCrawlController,
    UnsupportedStaticSupportController,
    UnsupportedStaticSupportEnv,
)


RESET_SEED = 1401
TRIAL_STEPS = 2000
EVALUATION_STEPS = 500
BELLY_TRIAL_STEPS = 4180
EXPECTED_BELLY_ACTION_SHA256 = (
    "46d8fde52bf2bd78c0a4d2a9791a2e056194761816422be58118ee02464fff67"
)
EXPECTED_BELLY_QPOS_SHA256 = (
    "7c11ea7cf0f7fc3af78fc575ffbd01b51f4a918267fe9d952e486cae56bc5eb3"
)
EXPECTED_BELLY_REST_ACTION_SHA256 = (
    "92904870ee2d0e806258653469929aff7660f4c3fa87e787f8693a7461f7066a"
)
EXPECTED_BELLY_REST_QPOS_SHA256 = (
    "017c2431d2e77c62167dda674c7186262b3dc846df3c0159a329f76ed93c7694"
)
BELLY_CRAWL_SUBSTEP_START = 3600
BELLY_CRAWL_SUBSTEP_STOP = 7560
BELLY_CYCLE_ENDPOINT_STEPS = (2129, 2459, 2789, 3119, 3449, 3779)


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


@dataclass(frozen=True, slots=True)
class _CrawlSubstep:
    region_force_n: tuple[float, float, float, float]
    central_force_n: float
    head_force_n: float
    root_vertical_velocity_m_s: float
    unexpected_external_contacts: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _BellyReplay:
    action_trace: np.ndarray
    qpos_trace: np.ndarray
    action_sha256: str
    qpos_sha256: str
    forward_m: float
    lateral_m: float
    yaw_rad: float
    cycle_forward_m: np.ndarray
    support_duty: dict[str, float]
    quiet_tail_support_fraction: dict[str, float]
    central_mean_n: float
    central_peak_n: float
    central_impulse_n_s: float
    central_max_delta_n: float
    central_true_onset_max_abs_vz_m_s: float
    central_short_hard_episode_count: int
    head_peak_n: float
    unexpected_external_contacts: tuple[tuple[str, str], ...]
    bodyweight_n: float
    minimum_action_bound_margin: float
    tail_max_linear_speed_m_s: float
    tail_max_angular_speed_rad_s: float


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


def test_belly_crawl_controller_has_exact_c1_open_loop_waveform(
    environment: UnsupportedStaticSupportEnv,
) -> None:
    """The scripted phase schedule is sparse, continuous, and state-read-only."""

    environment.reset(seed=RESET_SEED)
    controller = UnsupportedBellyCrawlController(environment)
    state = controller.reset()
    assert state == controller.reset()
    with pytest.raises(FrozenInstanceError):
        state.motor_state = state.motor_state  # type: ignore[misc]
    assert len(controller.controlled_actuator_names) == 18

    qpos = environment.data.qpos.copy()
    qvel = environment.data.qvel.copy()
    xfrc = environment.data.xfrc_applied.copy()

    def sample(time_seconds: float) -> tuple[np.ndarray, object]:
        action, _, diagnostics = controller.compose(
            environment,
            state,
            time_seconds=time_seconds,
        )
        assert action.dtype == np.float32
        assert environment.action_space.contains(action)
        return action, diagnostics

    zero, diagnostics = sample(0.0)
    np.testing.assert_array_equal(zero, np.zeros(90, dtype=np.float32))
    assert diagnostics.phase is BellyCrawlPhase.STATIC_SUPPORT

    static, _ = sample(BELLY_TRANSITION_START_SECONDS)
    transition_midpoint, diagnostics = sample(
        0.5 * (BELLY_TRANSITION_START_SECONDS + BELLY_TRANSITION_STOP_SECONDS)
    )
    assert diagnostics.phase is BellyCrawlPhase.TRANSITION_TO_BELLY
    for side in ("right", "left"):
        for suffix, delta in BELLY_ACTION_DELTAS.items():
            channel = resolve_motor_channel(environment, f"act:{side}_{suffix}")
            assert float(transition_midpoint[channel.action_index]) == pytest.approx(
                float(static[channel.action_index]) + 0.5 * delta,
                abs=3e-8,
            )

    belly, diagnostics = sample(BELLY_SETTLE_STOP_SECONDS - 1.0)
    assert diagnostics.phase is BellyCrawlPhase.BELLY_SETTLE
    phase_start = BELLY_SETTLE_STOP_SECONDS
    at_rise_end, diagnostics = sample(phase_start + BELLY_BUMP_RISE_SECONDS)
    assert diagnostics.phase is BellyCrawlPhase.LIMB_PHASE
    assert diagnostics.cycle_index == 0
    assert diagnostics.moving_side == BELLY_FIRST_SIDE[0] == "right"
    assert diagnostics.bump_fraction == 1.0

    fall_start, fall_start_diagnostics = sample(
        phase_start + BELLY_BUMP_PLATEAU_STOP_SECONDS
    )
    assert fall_start_diagnostics.bump_fraction == 1.0
    np.testing.assert_array_equal(fall_start, at_rise_end)
    incorrect_inverted_fall_value = 1.0 - UnsupportedBellyCrawlController._smoothstep(
        (BELLY_LIMB_PHASE_SECONDS - BELLY_BUMP_PLATEAU_STOP_SECONDS)
        / BELLY_BUMP_RISE_SECONDS
    )
    assert incorrect_inverted_fall_value == 0.0
    assert fall_start_diagnostics.bump_fraction != incorrect_inverted_fall_value

    fall_midpoint_time = (
        phase_start
        + BELLY_BUMP_PLATEAU_STOP_SECONDS
        + 0.5 * BELLY_BUMP_RISE_SECONDS
    )
    fall_midpoint, fall_diagnostics = sample(fall_midpoint_time)
    assert fall_diagnostics.bump_fraction == pytest.approx(0.5)
    np.testing.assert_allclose(
        fall_midpoint,
        belly + 0.5 * (fall_start - belly),
        rtol=0,
        atol=np.finfo(np.float32).eps,
    )

    reload_action, reload_diagnostics = sample(
        phase_start + BELLY_LIMB_PHASE_SECONDS
    )
    assert reload_diagnostics.phase is BellyCrawlPhase.RELOAD
    np.testing.assert_array_equal(reload_action, belly)

    second_phase_time = (
        phase_start + BELLY_LIMB_PHASE_SECONDS + 0.30 + 0.5
    )
    _, second_diagnostics = sample(second_phase_time)
    assert second_diagnostics.phase is BellyCrawlPhase.LIMB_PHASE
    assert second_diagnostics.moving_side == "left"
    assert second_diagnostics.bump_fraction == 1.0

    peak_right_shoulder = resolve_motor_channel(
        environment, "act:right_shoulder_horizontal"
    )
    stance_left_shoulder = resolve_motor_channel(
        environment, "act:left_shoulder_horizontal"
    )

    def endpoint(base: float, delta: float) -> np.float32:
        return np.float32(float(np.float32(base)) + delta)

    def lerp(start: float, stop: float, fraction: float) -> np.float32:
        start32, stop32, fraction32 = map(np.float32, (start, stop, fraction))
        return np.float32(
            start32 + fraction32 * np.float32(stop32 - start32)
        )

    static_shoulder = np.float32(BILATERAL_STATIC_ACTIONS["shoulder_horizontal"])
    belly_shoulder = endpoint(
        static_shoulder,
        BELLY_ACTION_DELTAS["shoulder_horizontal"],
    )
    expected_right = lerp(
        belly_shoulder,
        endpoint(
            static_shoulder,
            BELLY_MOVING_SIDE_DELTAS["shoulder_horizontal"],
        ),
        1.0,
    )
    expected_left = lerp(
        belly_shoulder,
        endpoint(static_shoulder, 0.4 * BELLY_STANCE_FRONT[0]),
        1.0,
    )
    np.testing.assert_array_equal(
        fall_start[peak_right_shoulder.action_index], expected_right
    )
    np.testing.assert_array_equal(
        fall_start[stance_left_shoulder.action_index], expected_left
    )

    tail, tail_diagnostics = sample(BELLY_CRAWL_STOP_SECONDS)
    assert tail_diagnostics.phase is BellyCrawlPhase.BELLY_TAIL
    np.testing.assert_array_equal(tail, belly)
    assert set(np.flatnonzero(tail)).issubset(
        {
            resolve_motor_channel(environment, name).action_index
            for name in controller.controlled_actuator_names
        }
    )
    np.testing.assert_array_equal(environment.data.qpos, qpos)
    np.testing.assert_array_equal(environment.data.qvel, qvel)
    np.testing.assert_array_equal(environment.data.xfrc_applied, xfrc)


def test_belly_crawl_smoothstep_is_bounded_and_monotonic_at_float_edges() -> None:
    inputs = np.concatenate(
        (
            np.asarray((-1.0, 0.0, np.nextafter(0.0, 1.0))),
            np.linspace(0.0, 1.0, 1001),
            np.asarray((np.nextafter(1.0, 0.0), 1.0, 2.0)),
        )
    )
    inputs.sort()
    values = np.asarray(
        [UnsupportedBellyCrawlController._smoothstep(float(value)) for value in inputs]
    )
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert np.all((0.0 <= values) & (values <= 1.0))
    assert np.all(np.diff(values) >= 0.0)


def test_belly_crawl_float32_action_schedule_matches_calibrated_sha(
    environment: UnsupportedStaticSupportEnv,
) -> None:
    environment.reset(seed=RESET_SEED)
    controller = UnsupportedBellyCrawlController(environment)
    state = controller.reset()
    digest = sha256()
    for step_index in range(BELLY_TRIAL_STEPS):
        action, state, diagnostics = controller.compose(environment, state)
        assert action.dtype == np.float32
        assert action.flags.c_contiguous
        assert environment.action_space.contains(action)
        assert diagnostics.time_seconds == step_index * environment.dt
        if step_index == 0:
            assert not np.any(np.signbit(action))
        if step_index == 600:
            assert diagnostics.phase is BellyCrawlPhase.BELLY_SETTLE
        digest.update(action.tobytes(order="C"))
    assert environment.data.time == 0.0
    assert digest.hexdigest() == EXPECTED_BELLY_ACTION_SHA256


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


def _crawl_contact_sample(
    environment: UnsupportedStaticSupportEnv,
    classifier: StaticSupportContactClassifier,
    *,
    root_dof_address: int,
    head_body_ids: frozenset[int],
) -> _CrawlSubstep:
    classified = classifier.classify()
    model = environment.model
    data = environment.data
    mimo_root_id = int(model.body("mimo_location").id)
    floor_geom_id = int(model.geom("floor").id)
    total_admitted_floor_force = 0.0
    head_force = 0.0
    unexpected = set(classified.unexpected_external_contacts)
    for contact_index, contact in enumerate(data.contact):
        if int(contact.efc_address) < 0:
            continue
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        first_is_mimo = int(model.body_rootid[body1]) == mimo_root_id
        second_is_mimo = int(model.body_rootid[body2]) == mimo_root_id
        if first_is_mimo == second_is_mimo:
            continue
        mimo_geom = geom1 if first_is_mimo else geom2
        mimo_body = body1 if first_is_mimo else body2
        external_geom = geom2 if first_is_mimo else geom1
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force)
        normal_force = max(float(force[0]), 0.0)
        if external_geom != floor_geom_id:
            mimo_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, mimo_geom
            )
            external_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, external_geom
            )
            if mimo_name is None or external_name is None:
                raise AssertionError("every external crawl contact geom must be named")
            unexpected.add((mimo_name, external_name))
            continue
        if mimo_body in head_body_ids:
            head_force += normal_force
        if normal_force > 0.1:
            total_admitted_floor_force += normal_force

    support_force = sum(
        contact.normal_force_n for contact in classified.contacts
    )
    central_force = max(total_admitted_floor_force - support_force, 0.0)
    force_by_region = dict(classified.force_by_region_n)
    return _CrawlSubstep(
        region_force_n=tuple(force_by_region[region] for region in SupportRegion),
        central_force_n=central_force,
        head_force_n=head_force,
        root_vertical_velocity_m_s=float(data.qvel[root_dof_address + 2]),
        unexpected_external_contacts=tuple(sorted(unexpected)),
    )


def _normalized_heading_xy(environment: UnsupportedStaticSupportEnv) -> np.ndarray:
    heading = (
        np.asarray(environment.data.body("head").xpos[:2], dtype=np.float64)
        - np.asarray(environment.data.body("hip").xpos[:2], dtype=np.float64)
    )
    norm = float(np.linalg.norm(heading))
    if norm <= 0.0:
        raise AssertionError("head-to-hip heading must have nonzero planar length")
    return heading / norm


def _run_belly_replay(*, cycles_enabled: bool) -> _BellyReplay:
    environment = UnsupportedStaticSupportEnv(width=16, height=12)
    original_substep_callback = environment._substep_callback
    try:
        observation, info = environment.reset(seed=RESET_SEED)
        assert info == {}
        assert environment.observation_space.contains(observation)
        controller = UnsupportedBellyCrawlController(
            environment, cycles_enabled=cycles_enabled
        )
        state = controller.reset()
        classifier = StaticSupportContactClassifier(environment)
        root_joint = environment.model.joint("mimo_orientation")
        root_dof = int(np.asarray(root_joint.dofadr).item())
        root_body_id = int(environment.model.body("mimo_location").id)
        head_body_ids = StaticSupportContactClassifier._descendant_body_ids(
            environment.model,
            int(environment.model.body("head").id),
        )
        gravity = environment.model.opt.gravity.copy()
        equality_active = environment.model.eq_active0.copy()
        substeps: list[_CrawlSubstep] = []

        def record_substep() -> None:
            original_substep_callback()
            substeps.append(
                _crawl_contact_sample(
                    environment,
                    classifier,
                    root_dof_address=root_dof,
                    head_body_ids=head_body_ids,
                )
            )

        environment._substep_callback = record_substep  # type: ignore[method-assign]
        action_digest = sha256()
        qpos_digest = sha256()
        action_trace: list[np.ndarray] = []
        qpos_trace: list[np.ndarray] = []
        root_linear_speed: list[float] = []
        root_angular_speed: list[float] = []
        cycle_forward: list[float] = []
        minimum_bound_margin = float("inf")
        reference_com: np.ndarray | None = None
        reference_heading: np.ndarray | None = None

        for step_index in range(BELLY_TRIAL_STEPS):
            if step_index == 1800:
                reference_com = environment.data.subtree_com[root_body_id].copy()
                reference_heading = _normalized_heading_xy(environment)
            qpos_before = environment.data.qpos.copy()
            qvel_before = environment.data.qvel.copy()
            xfrc_before = environment.data.xfrc_applied.copy()
            action, state, diagnostics = controller.compose(environment, state)
            assert diagnostics.time_seconds == step_index * environment.dt
            np.testing.assert_array_equal(environment.data.qpos, qpos_before)
            np.testing.assert_array_equal(environment.data.qvel, qvel_before)
            np.testing.assert_array_equal(environment.data.xfrc_applied, xfrc_before)
            assert environment.action_space.contains(action)
            minimum_bound_margin = min(
                minimum_bound_margin,
                diagnostics.minimum_action_bound_margin,
            )
            action_bytes = np.ascontiguousarray(action, dtype=np.float32).tobytes()
            action_digest.update(action_bytes)
            action_trace.append(action.copy())
            observation, _, terminated, truncated, _ = environment.step(action)
            assert not terminated
            assert not truncated
            assert environment.observation_space.contains(observation)
            np.testing.assert_array_equal(
                environment.data.xfrc_applied,
                np.zeros_like(environment.data.xfrc_applied),
            )
            qpos = np.ascontiguousarray(environment.data.qpos, dtype=np.float64)
            qpos_digest.update(qpos.tobytes())
            qpos_trace.append(qpos.copy())
            root_linear_speed.append(
                float(np.linalg.norm(environment.data.qvel[root_dof : root_dof + 3]))
            )
            root_angular_speed.append(
                float(
                    np.linalg.norm(
                        environment.data.qvel[root_dof + 3 : root_dof + 6]
                    )
                )
            )
            if step_index in BELLY_CYCLE_ENDPOINT_STEPS:
                assert reference_com is not None
                assert reference_heading is not None
                displacement = (
                    environment.data.subtree_com[root_body_id][:2]
                    - reference_com[:2]
                )
                cycle_forward.append(float(np.dot(displacement, reference_heading)))

        assert reference_com is not None
        assert reference_heading is not None
        assert len(substeps) == 2 * BELLY_TRIAL_STEPS
        final_displacement = (
            environment.data.subtree_com[root_body_id][:2] - reference_com[:2]
        )
        forward = float(np.dot(final_displacement, reference_heading))
        lateral = float(
            reference_heading[0] * final_displacement[1]
            - reference_heading[1] * final_displacement[0]
        )
        final_heading = _normalized_heading_xy(environment)
        initial_angle = float(np.arctan2(reference_heading[1], reference_heading[0]))
        final_angle = float(np.arctan2(final_heading[1], final_heading[0]))
        yaw = float(
            np.arctan2(
                np.sin(final_angle - initial_angle),
                np.cos(final_angle - initial_angle),
            )
        )

        crawl_samples = substeps[BELLY_CRAWL_SUBSTEP_START:BELLY_CRAWL_SUBSTEP_STOP]
        crawl_region_force = np.asarray(
            [sample.region_force_n for sample in crawl_samples], dtype=np.float64
        )
        support_duty = {
            region.value: float(np.mean(crawl_region_force[:, index] > 0.1))
            for index, region in enumerate(SupportRegion)
        }
        central_all = np.asarray(
            [sample.central_force_n for sample in substeps], dtype=np.float64
        )
        central_crawl = central_all[
            BELLY_CRAWL_SUBSTEP_START:BELLY_CRAWL_SUBSTEP_STOP
        ]
        active = central_all > 0.1
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        stops = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
        true_onset_vertical_speeds: list[float] = []
        short_hard_count = 0
        bodyweight = float(
            environment.model.body_subtreemass[root_body_id] * 9.81
        )
        for start, stop in zip(starts, stops, strict=True):
            if stop <= BELLY_CRAWL_SUBSTEP_START or start >= BELLY_CRAWL_SUBSTEP_STOP:
                continue
            if start < BELLY_CRAWL_SUBSTEP_START:
                continue
            true_onset_vertical_speeds.append(
                substeps[start].root_vertical_velocity_m_s
            )
            duration_seconds = (stop - start) * 0.005
            if (
                float(np.max(central_all[start:stop])) > bodyweight
                and duration_seconds <= 0.020
            ):
                short_hard_count += 1

        quiet_tail = np.asarray(
            [sample.region_force_n for sample in substeps[-400:]],
            dtype=np.float64,
        )
        quiet_tail_fraction = {
            region.value: float(np.mean(quiet_tail[:, index] > 0.1))
            for index, region in enumerate(SupportRegion)
        }
        unexpected = tuple(
            sorted(
                {
                    pair
                    for sample in substeps
                    for pair in sample.unexpected_external_contacts
                }
            )
        )
        np.testing.assert_array_equal(environment.model.opt.gravity, gravity)
        np.testing.assert_array_equal(environment.model.eq_active0, equality_active)
        return _BellyReplay(
            action_trace=np.asarray(action_trace, dtype=np.float32),
            qpos_trace=np.asarray(qpos_trace, dtype=np.float64),
            action_sha256=action_digest.hexdigest(),
            qpos_sha256=qpos_digest.hexdigest(),
            forward_m=forward,
            lateral_m=lateral,
            yaw_rad=yaw,
            cycle_forward_m=np.asarray(cycle_forward, dtype=np.float64),
            support_duty=support_duty,
            quiet_tail_support_fraction=quiet_tail_fraction,
            central_mean_n=float(np.mean(central_crawl)),
            central_peak_n=float(np.max(central_crawl)),
            central_impulse_n_s=float(np.sum(central_crawl) * 0.005),
            central_max_delta_n=float(np.max(np.abs(np.diff(central_crawl)))),
            central_true_onset_max_abs_vz_m_s=(
                max(abs(value) for value in true_onset_vertical_speeds)
                if true_onset_vertical_speeds
                else 0.0
            ),
            central_short_hard_episode_count=short_hard_count,
            head_peak_n=max(sample.head_force_n for sample in substeps),
            unexpected_external_contacts=unexpected,
            bodyweight_n=bodyweight,
            minimum_action_bound_margin=minimum_bound_margin,
            tail_max_linear_speed_m_s=max(root_linear_speed[-200:]),
            tail_max_angular_speed_rad_s=max(root_angular_speed[-200:]),
        )
    finally:
        environment._substep_callback = original_substep_callback  # type: ignore[method-assign]
        environment.close()


@pytest.fixture(scope="module")
def belly_replays() -> tuple[_BellyReplay, _BellyReplay, _BellyReplay]:
    return (
        _run_belly_replay(cycles_enabled=True),
        _run_belly_replay(cycles_enabled=True),
        _run_belly_replay(cycles_enabled=False),
    )


def test_public_step_belly_crawl_is_deterministic_and_physically_bounded(
    belly_replays: tuple[_BellyReplay, _BellyReplay, _BellyReplay],
) -> None:
    active, repeated, rest = belly_replays
    np.testing.assert_array_equal(active.action_trace, repeated.action_trace)
    np.testing.assert_array_equal(active.qpos_trace, repeated.qpos_trace)
    assert active.action_sha256 == repeated.action_sha256
    assert active.qpos_sha256 == repeated.qpos_sha256
    assert active.action_sha256 == EXPECTED_BELLY_ACTION_SHA256
    assert active.qpos_sha256 == EXPECTED_BELLY_QPOS_SHA256
    assert rest.action_sha256 == EXPECTED_BELLY_REST_ACTION_SHA256
    assert rest.qpos_sha256 == EXPECTED_BELLY_REST_QPOS_SHA256

    corrected_forward = active.forward_m - rest.forward_m
    assert corrected_forward >= 0.05
    assert abs(rest.forward_m) <= 0.001
    cycle_increments = np.diff(np.r_[0.0, active.cycle_forward_m])
    assert len(cycle_increments) == 6
    assert np.all(cycle_increments >= 0.004)
    assert abs(active.lateral_m) <= 0.015
    assert abs(active.yaw_rad) <= 0.30
    assert active.forward_m == pytest.approx(0.0583732423, abs=2e-6)
    assert corrected_forward == pytest.approx(0.0583707354, abs=2e-6)
    assert active.lateral_m == pytest.approx(0.0089821222, abs=2e-6)
    assert active.yaw_rad == pytest.approx(0.2392100544, abs=2e-6)

    assert active.support_duty[SupportRegion.FRONT_LEFT.value] >= 0.50
    assert active.support_duty[SupportRegion.FRONT_RIGHT.value] >= 0.50
    assert active.support_duty[SupportRegion.REAR_LEFT.value] >= 0.90
    assert active.support_duty[SupportRegion.REAR_RIGHT.value] >= 0.90
    assert active.head_peak_n == 0.0
    assert active.unexpected_external_contacts == ()
    assert active.minimum_action_bound_margin > 0.0
    assert active.central_peak_n <= 2.0 * active.bodyweight_n
    assert active.central_max_delta_n <= 1.5 * active.bodyweight_n
    assert active.central_true_onset_max_abs_vz_m_s <= 0.03
    assert active.central_short_hard_episode_count == 0
    assert active.quiet_tail_support_fraction == {
        region.value: 1.0 for region in SupportRegion
    }
    assert active.tail_max_linear_speed_m_s <= 0.002
    assert active.tail_max_angular_speed_rad_s <= 0.03

"""Focused contracts for reusable name-based MIMo motor control."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import pytest

from blight_world import BlightWorldEnv
from blight_world.motor_control import (
    MotorControllerState,
    SparseActionComposer,
    coupled_hip_bias_compensated_pd,
    resolve_motor_channel,
    resolve_motor_channels,
    single_joint_bias_compensated_pd,
)


@pytest.fixture(scope="module")
def environment() -> Iterator[BlightWorldEnv]:
    """Create one small vision environment for model/action resolution."""

    instance = BlightWorldEnv(width=16, height=12)
    try:
        yield instance
    finally:
        instance.close()


def test_all_ninety_channels_resolve_uniquely_by_name(
    environment: BlightWorldEnv,
) -> None:
    channels = resolve_motor_channels(environment)

    assert len(channels) == 90
    assert [channel.action_index for channel in channels] == list(range(90))
    assert [channel.actuator_id for channel in channels] == list(
        np.asarray(environment.mimo_actuators, dtype=int)
    )
    assert len({channel.actuator_name for channel in channels}) == 90
    for channel in channels:
        assert resolve_motor_channel(environment, channel.actuator_name) == channel
        assert environment.model.joint(channel.joint_name).id == channel.joint_id
        assert environment.action_space.low[channel.action_index] == channel.lower_bound
        assert environment.action_space.high[channel.action_index] == channel.upper_bound

    hip = resolve_motor_channel(environment, "act:hip_bend")
    assert hip.joint_name == "robot:hip_bend1"
    assert hip.gear == 10.58
    assert (hip.lower_bound, hip.upper_bound) == (-1.0, 0.7649999856948853)
    head = resolve_motor_channel(environment, "act:head_tilt")
    assert head.joint_name == "robot:head_tilt"
    assert head.gear == 3.3
    assert (head.lower_bound, head.upper_bound) == (-1.0, 0.550000011920929)
    with pytest.raises(KeyError, match="unknown MIMo actuator"):
        resolve_motor_channel(environment, "act:not_present")


def test_sparse_composer_maps_only_owned_channels_to_ctrl(
    environment: BlightWorldEnv,
) -> None:
    channels = resolve_motor_channels(environment)
    composer = SparseActionComposer(channels)
    reset = composer.reset()
    action, state = composer.compose(
        {
            "posture": {"act:hip_bend": 0.25},
            "gaze": {"act:head_tilt": -0.125},
        },
        reset,
    )

    hip = resolve_motor_channel(environment, "act:hip_bend")
    head = resolve_motor_channel(environment, "act:head_tilt")
    assert action.shape == (90,)
    assert action.dtype == np.float32
    assert action.flags.c_contiguous
    assert np.flatnonzero(action).tolist() == [hip.action_index, head.action_index]
    environment.actuation_model.action(action)
    expected_ctrl = np.zeros(environment.model.nu)
    expected_ctrl[hip.actuator_id] = 0.25
    expected_ctrl[head.actuator_id] = -0.125
    np.testing.assert_array_equal(environment.data.ctrl, expected_ctrl)
    assert state.step_index == 1
    assert state.previous_action == tuple(float(value) for value in action)


def test_composer_enforces_conflicts_finiteness_clipping_and_slew(
    environment: BlightWorldEnv,
) -> None:
    channels = resolve_motor_channels(environment)
    hip = resolve_motor_channel(environment, "act:hip_bend")
    composer = SparseActionComposer(
        channels,
        slew_limits={"act:hip_bend": 0.1},
    )
    reset = composer.reset()
    first, first_state = composer.compose(
        {"posture": {hip: 100.0}},
        reset,
    )
    second, second_state = composer.compose(
        {"posture": {"act:hip_bend": 100.0}},
        first_state,
    )
    released, released_state = composer.compose({}, second_state)
    assert first[hip.action_index] == pytest.approx(0.1)
    assert second[hip.action_index] == pytest.approx(0.2)
    assert released[hip.action_index] == pytest.approx(0.1)
    assert released_state.step_index == 3

    clipped = SparseActionComposer(channels).compose(
        {"posture": {"act:hip_bend": 100.0}},
        reset,
    )[0]
    assert clipped[hip.action_index] == np.float32(hip.upper_bound)
    with pytest.raises(ValueError, match="claimed by both"):
        composer.compose(
            {
                "first": {"act:hip_bend": 0.1},
                "second": {hip: 0.2},
            },
            reset,
        )
    with pytest.raises(ValueError, match="must be finite"):
        composer.compose({"owner": {hip: np.nan}}, reset)


@pytest.mark.parametrize("bound", ["lower", "upper"])
def test_composer_rejects_out_of_bounds_prior_state_before_slew(
    environment: BlightWorldEnv,
    bound: str,
) -> None:
    """Corrupted history fails closed instead of bypassing action bounds."""

    channels = resolve_motor_channels(environment)
    hip = resolve_motor_channel(environment, "act:hip_bend")
    composer = SparseActionComposer(
        channels,
        slew_limits={hip.actuator_name: 0.1},
    )
    previous = list(composer.reset().previous_action)
    if bound == "lower":
        previous[hip.action_index] = hip.lower_bound - 1e-6
    else:
        previous[hip.action_index] = hip.upper_bound + 1e-6
    corrupted = MotorControllerState(
        step_index=1,
        previous_action=tuple(previous),
    )

    with pytest.raises(
        ValueError,
        match=rf"{hip.actuator_name}.*outside bounds",
    ):
        composer.compose({}, corrupted)


def test_all_channel_composition_preserves_bounds_and_slew(
    environment: BlightWorldEnv,
) -> None:
    """Seeded multi-step commands remain bounded with per-channel slew."""

    channels = resolve_motor_channels(environment)
    slew_limits = {
        channel.actuator_name: 0.005 + 0.0001 * channel.action_index
        for channel in channels
    }
    composer = SparseActionComposer(channels, slew_limits=slew_limits)
    random = np.random.default_rng(20260825)
    state = composer.reset()
    previous = np.asarray(state.previous_action, dtype=np.float32)
    limits = np.asarray(
        [slew_limits[channel.actuator_name] for channel in channels],
        dtype=np.float64,
    )

    for _ in range(500):
        requested = random.uniform(-2.0, 2.0, size=len(channels))
        action, state = composer.compose(
            {
                "seeded_controller": {
                    channel: float(requested[channel.action_index])
                    for channel in channels
                }
            },
            state,
        )
        assert environment.action_space.contains(action)
        assert np.all(
            np.abs(action.astype(np.float64) - previous.astype(np.float64))
            <= limits + np.finfo(np.float32).eps
        )
        previous = action


def test_controller_state_is_immutable_resettable_and_deterministic(
    environment: BlightWorldEnv,
) -> None:
    composer = SparseActionComposer(resolve_motor_channels(environment))
    first_reset = composer.reset()
    second_reset = composer.reset()
    assert first_reset == second_reset == MotorControllerState(
        step_index=0,
        previous_action=(0.0,) * 90,
    )
    with pytest.raises(FrozenInstanceError):
        first_reset.step_index = 1  # type: ignore[misc]
    commands = {"controller": {"act:head_tilt": 0.2}}
    first_action, first_next = composer.compose(commands, first_reset)
    second_action, second_next = composer.compose(commands, second_reset)
    np.testing.assert_array_equal(first_action, second_action)
    assert first_next == second_next


def test_single_joint_pd_has_expected_bias_sign_and_clipping(
    environment: BlightWorldEnv,
) -> None:
    channel = resolve_motor_channel(environment, "act:head_tilt")
    data = SimpleNamespace(
        qpos=np.zeros(environment.model.nq),
        qvel=np.zeros(environment.model.nv),
        qfrc_bias=np.zeros(environment.model.nv),
    )
    data.qpos[channel.qpos_address] = 0.2
    data.qvel[channel.dof_address] = -0.1
    data.qfrc_bias[channel.dof_address] = 0.3
    action = single_joint_bias_compensated_pd(
        channel,
        data,  # type: ignore[arg-type]
        reference_position_rad=0.4,
        stiffness_nm_per_rad=2.0,
        damping_nm_s_per_rad=0.5,
    )
    assert action == pytest.approx(0.75 / 3.3)

    negative = single_joint_bias_compensated_pd(
        channel,
        data,  # type: ignore[arg-type]
        reference_position_rad=-1.0,
        stiffness_nm_per_rad=2.0,
        damping_nm_s_per_rad=0.0,
    )
    assert negative < 0.0
    clipped = single_joint_bias_compensated_pd(
        channel,
        data,  # type: ignore[arg-type]
        reference_position_rad=100.0,
        stiffness_nm_per_rad=100.0,
        damping_nm_s_per_rad=0.0,
    )
    assert clipped == channel.upper_bound


def test_coupled_hip_pd_combines_two_biases_and_mean_state(
    environment: BlightWorldEnv,
) -> None:
    channel = resolve_motor_channel(environment, "act:hip_bend")
    second_joint = environment.model.joint("robot:hip_bend2")
    second_qpos = int(np.asarray(second_joint.qposadr).item())
    second_dof = int(np.asarray(second_joint.dofadr).item())
    data = SimpleNamespace(
        qpos=np.zeros(environment.model.nq),
        qvel=np.zeros(environment.model.nv),
        qfrc_bias=np.zeros(environment.model.nv),
    )
    data.qpos[channel.qpos_address] = 0.1
    data.qpos[second_qpos] = 0.3
    data.qvel[channel.dof_address] = 0.2
    data.qvel[second_dof] = 0.4
    data.qfrc_bias[channel.dof_address] = 0.5
    data.qfrc_bias[second_dof] = 0.7

    action = coupled_hip_bias_compensated_pd(
        channel,
        environment.model,
        data,  # type: ignore[arg-type]
        reference_mean_position_rad=0.4,
        stiffness_nm_per_rad=10.0,
        damping_nm_s_per_rad=4.5,
    )
    assert action == pytest.approx(1.85 / 10.58)
    wrong_channel = resolve_motor_channel(environment, "act:head_tilt")
    with pytest.raises(ValueError, match="requires the act:hip_bend channel"):
        coupled_hip_bias_compensated_pd(
            wrong_channel,
            environment.model,
            data,  # type: ignore[arg-type]
            reference_mean_position_rad=0.0,
            stiffness_nm_per_rad=1.0,
            damping_nm_s_per_rad=1.0,
        )


def test_environment_defaults_remain_exact_and_options_pass_through() -> None:
    """New constructor seams preserve defaults and enable upstream options."""

    baseline = BlightWorldEnv(width=16, height=12)
    default_actuation_type = type(baseline.actuation_model)
    configured = BlightWorldEnv(
        width=16,
        height=12,
        proprio_params={"components": ["velocity"]},
        vestibular_params={
            "sensors": ["vestibular_acc", "vestibular_gyro"],
        },
        actuation_model=default_actuation_type,
        motor_delay=2,
    )
    try:
        baseline_observation, baseline_info = baseline.reset(seed=41)
        assert set(baseline_observation) == {
            "observation",
            "eye_left",
            "eye_right",
        }
        assert baseline.observation_space.contains(baseline_observation)
        assert baseline_observation["observation"].shape == (93,)
        assert baseline.proprio_params == {"components": []}
        assert baseline.touch_params is None
        assert baseline.vestibular_params is None
        assert baseline.motor_delay == 0
        assert type(baseline.actuation_model) is default_actuation_type
        assert baseline_info == {}

        configured_observation, configured_info = configured.reset(seed=41)
        assert configured.proprio_params == {"components": ["velocity"]}
        assert configured.touch_params is None
        assert configured.vestibular_params == {
            "sensors": ["vestibular_acc", "vestibular_gyro"],
        }
        assert configured.motor_delay == 2
        assert type(configured.actuation_model) is default_actuation_type
        assert configured_observation["observation"].shape == (186,)
        assert configured_observation["vestibular"].shape == (6,)
        assert set(configured_observation) == {
            "observation",
            "eye_left",
            "eye_right",
            "vestibular",
        }
        assert configured.observation_space.contains(configured_observation)
        assert configured_info == {}
    finally:
        baseline.close()
        configured.close()

    with pytest.raises(ValueError, match="motor_delay"):
        BlightWorldEnv(width=16, height=12, motor_delay=-1)

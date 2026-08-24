"""Supported, actuator-driven manipulation primitives for MIMo.

The environment and controller in this module form one deliberately narrow
embodied milestone: MIMo reaches with its right hand and physically pushes a
small sphere across a tabletop.  The torso is held by an explicit world weld
so the experiment isolates arm control; this is supported manipulation, not
balance, locomotion, grasping, or learned behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import mujoco
import numpy as np
from numpy.typing import NDArray

from blight_world.environment import BlightWorldEnv
from blight_world.motor_control import (
    MotorChannel,
    MotorControllerState,
    SparseActionComposer,
    coupled_hip_bias_compensated_pd,
    resolve_motor_channel,
    resolve_motor_channels,
    single_joint_bias_compensated_pd,
)


SUPPORTED_PUSH_SCENE_XML = """<worldbody>
    <body name="push_table" pos=".36 -.20 .169">
      <geom name="push_table_top" type="box" size=".235 .15 .169"
            friction=".6 .03 .02" condim="6"/>
    </body>
    <body name="push_target" pos=".142 -.17 .360">
      <freejoint name="push_target_joint"/>
      <geom name="push_target_geom" type="sphere" size=".020" mass=".020"
            friction=".4 .02 .02" condim="6"/>
    </body>
  </worldbody>
  <equality>
    <weld name="root_support" body1="mimo_location"
          solref=".01 1" solimp=".99 .999 .001 .5 2"/>
  </equality>"""

SUPPORTED_PUSH_SUPPORT_LABEL = (
    "demo-only physical world weld supporting MIMo's root; no grasp weld"
)

RIGHT_ARM_ACTUATOR_NAMES = (
    "act:right_shoulder_horizontal",
    "act:right_shoulder_abduction",
    "act:right_shoulder_internal",
    "act:right_elbow",
    "act:right_wrist_rotation",
    "act:right_wrist_flexion",
    "act:right_wrist_ulnar",
)
RIGHT_ARM_TARGET_RAD = (
    1.292909606,
    1.024294074,
    1.088039922,
    -0.213853951,
    -0.144800168,
    -0.156875793,
    -0.150551313,
)
RIGHT_ARM_STIFFNESS_NM_RAD = (4.0, 5.0, 4.0, 4.0, 1.2, 1.6, 1.2)
RIGHT_ARM_DAMPING_NM_S_RAD = (0.45, 0.5, 0.45, 0.4, 0.14, 0.18, 0.14)

REACH_START_SECONDS = 1.0
REACH_STOP_SECONDS = 5.0
HIP_STIFFNESS_NM_RAD = 10.0
HIP_DAMPING_NM_S_RAD = 4.5
HEAD_STIFFNESS_NM_RAD = 1.0
HEAD_DAMPING_NM_S_RAD = 0.2

PushPhase = Literal["neutral_hold", "initial_hold", "reach", "target_hold"]


class SupportedPushEnv(BlightWorldEnv):
    """Reachable tabletop scene with an explicitly supported MIMo root.

    The only scene-specific equality is ``root_support`` between the world and
    ``mimo_location``.  The target remains a normal free body: it is moved only
    by MuJoCo gravity, contact, and actuator-driven hand motion.
    """

    _scene_extension_xml = SUPPORTED_PUSH_SCENE_XML


@dataclass(frozen=True, slots=True)
class SupportedPushControllerState:
    """Immutable state for one deterministic supported-push controller.

    ``initial_arm_positions_rad`` and the posture references are captured once
    after reset.  All evolving composer history is contained in
    ``motor_state``; the controller never writes MuJoCo state.
    """

    initial_arm_positions_rad: tuple[float, ...]
    hip_reference_rad: float
    head_reference_rad: float
    motor_state: MotorControllerState

    def __post_init__(self) -> None:
        if len(self.initial_arm_positions_rad) != len(RIGHT_ARM_ACTUATOR_NAMES):
            raise ValueError("initial_arm_positions_rad must contain seven values")
        values = (
            *self.initial_arm_positions_rad,
            self.hip_reference_rad,
            self.head_reference_rad,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("supported-push references must be finite")


@dataclass(frozen=True, slots=True)
class SupportedPushControlDiagnostics:
    """Immutable controller evidence for one public environment step."""

    time_seconds: float
    phase: PushPhase
    arm_reference_positions_rad: tuple[float, ...]
    arm_reference_velocities_rad_s: tuple[float, ...]
    unclipped_actions: tuple[tuple[str, float], ...]
    saturated_actuators: tuple[str, ...]


class RightHandSpherePushController:
    """Bias-compensated right-arm reach for :class:`SupportedPushEnv`.

    The seven arm references follow a four-second cubic smoothstep beginning at
    one second.  Paired hips hold their reset mean and head tilt holds its reset
    angle.  Every other one of MIMo's 90 action coordinates remains exactly
    zero through :class:`~blight_world.motor_control.SparseActionComposer`.
    """

    __slots__ = (
        "_arm_channels",
        "_composer",
        "_head_channel",
        "_hip_channel",
        "_hip_second_dof_address",
        "_hip_second_qpos_address",
        "_model",
    )

    def __init__(self, environment: SupportedPushEnv) -> None:
        self._model = environment.model
        self._arm_channels = tuple(
            resolve_motor_channel(environment, name)
            for name in RIGHT_ARM_ACTUATOR_NAMES
        )
        self._hip_channel = resolve_motor_channel(environment, "act:hip_bend")
        self._head_channel = resolve_motor_channel(environment, "act:head_tilt")
        hip_second = environment.model.joint("robot:hip_bend2")
        self._hip_second_qpos_address = int(np.asarray(hip_second.qposadr).item())
        self._hip_second_dof_address = int(np.asarray(hip_second.dofadr).item())
        self._composer = SparseActionComposer(resolve_motor_channels(environment))

    @property
    def controlled_actuator_names(self) -> tuple[str, ...]:
        """Return the nine channels this skill may command."""

        return (
            *RIGHT_ARM_ACTUATOR_NAMES,
            self._hip_channel.actuator_name,
            self._head_channel.actuator_name,
        )

    def reset(
        self,
        environment: SupportedPushEnv,
    ) -> SupportedPushControllerState:
        """Capture reset posture and return deterministic zero-history state."""

        self._require_environment(environment)
        initial_arm = tuple(
            float(environment.data.qpos[channel.qpos_address])
            for channel in self._arm_channels
        )
        hip_reference = float(
            np.mean(
                (
                    environment.data.qpos[self._hip_channel.qpos_address],
                    environment.data.qpos[self._hip_second_qpos_address],
                )
            )
        )
        return SupportedPushControllerState(
            initial_arm_positions_rad=initial_arm,
            hip_reference_rad=hip_reference,
            head_reference_rad=float(
                environment.data.qpos[self._head_channel.qpos_address]
            ),
            motor_state=self._composer.reset(),
        )

    def compose(
        self,
        environment: SupportedPushEnv,
        state: SupportedPushControllerState,
        *,
        execute_push: bool = True,
        time_seconds: float | None = None,
    ) -> tuple[
        NDArray[np.float32],
        SupportedPushControllerState,
        SupportedPushControlDiagnostics,
    ]:
        """Compose one action without mutating model, data, or controller state.

        Parameters
        ----------
        environment:
            The same live environment used to construct this controller.
        state:
            Immutable state returned by :meth:`reset` or a prior call.
        execute_push:
            When false, hold all seven arm joints at their reset positions.  It
            provides the neutral physical control used by validation.
        time_seconds:
            Optional finite non-negative public-step time.  By default the
            current MuJoCo simulation clock is read.
        """

        self._require_environment(environment)
        time_value = (
            float(environment.data.time)
            if time_seconds is None
            else float(time_seconds)
        )
        if not isfinite(time_value) or time_value < 0.0:
            raise ValueError("time_seconds must be finite and non-negative")
        phase, references, velocities = self._trajectory(
            state.initial_arm_positions_rad,
            time_value,
            execute_push=execute_push,
        )

        raw_actions: dict[str, float] = {}
        arm_commands: dict[MotorChannel, float] = {}
        for channel, reference, velocity_reference, stiffness, damping in zip(
            self._arm_channels,
            references,
            velocities,
            RIGHT_ARM_STIFFNESS_NM_RAD,
            RIGHT_ARM_DAMPING_NM_S_RAD,
            strict=True,
        ):
            raw_action = self._unclipped_joint_action(
                channel,
                environment.data,
                reference_position_rad=reference,
                reference_velocity_rad_s=velocity_reference,
                stiffness_nm_per_rad=stiffness,
                damping_nm_s_per_rad=damping,
            )
            raw_actions[channel.actuator_name] = raw_action
            arm_commands[channel] = raw_action

        hip_raw = self._unclipped_hip_action(
            environment.data,
            reference_rad=state.hip_reference_rad,
        )
        head_raw = self._unclipped_joint_action(
            self._head_channel,
            environment.data,
            reference_position_rad=state.head_reference_rad,
            reference_velocity_rad_s=0.0,
            stiffness_nm_per_rad=HEAD_STIFFNESS_NM_RAD,
            damping_nm_s_per_rad=HEAD_DAMPING_NM_S_RAD,
        )
        raw_actions[self._hip_channel.actuator_name] = hip_raw
        raw_actions[self._head_channel.actuator_name] = head_raw
        posture_commands = {
            self._hip_channel: coupled_hip_bias_compensated_pd(
                self._hip_channel,
                environment.model,
                environment.data,
                reference_mean_position_rad=state.hip_reference_rad,
                stiffness_nm_per_rad=HIP_STIFFNESS_NM_RAD,
                damping_nm_s_per_rad=HIP_DAMPING_NM_S_RAD,
            ),
            self._head_channel: single_joint_bias_compensated_pd(
                self._head_channel,
                environment.data,
                reference_position_rad=state.head_reference_rad,
                stiffness_nm_per_rad=HEAD_STIFFNESS_NM_RAD,
                damping_nm_s_per_rad=HEAD_DAMPING_NM_S_RAD,
            ),
        }
        action, motor_state = self._composer.compose(
            {"right_arm": arm_commands, "posture": posture_commands},
            state.motor_state,
        )
        channel_by_name = {
            channel.actuator_name: channel
            for channel in (*self._arm_channels, self._hip_channel, self._head_channel)
        }
        saturated = tuple(
            name
            for name, value in raw_actions.items()
            if value < channel_by_name[name].lower_bound
            or value > channel_by_name[name].upper_bound
        )
        next_state = SupportedPushControllerState(
            initial_arm_positions_rad=state.initial_arm_positions_rad,
            hip_reference_rad=state.hip_reference_rad,
            head_reference_rad=state.head_reference_rad,
            motor_state=motor_state,
        )
        diagnostics = SupportedPushControlDiagnostics(
            time_seconds=time_value,
            phase=phase,
            arm_reference_positions_rad=references,
            arm_reference_velocities_rad_s=velocities,
            unclipped_actions=tuple(raw_actions.items()),
            saturated_actuators=saturated,
        )
        return action, next_state, diagnostics

    def _require_environment(self, environment: SupportedPushEnv) -> None:
        if environment.model is not self._model:
            raise ValueError("controller belongs to a different MuJoCo model")

    @staticmethod
    def _trajectory(
        initial: tuple[float, ...],
        time_seconds: float,
        *,
        execute_push: bool,
    ) -> tuple[PushPhase, tuple[float, ...], tuple[float, ...]]:
        if not execute_push:
            return "neutral_hold", initial, (0.0,) * len(initial)
        if time_seconds <= REACH_START_SECONDS:
            return "initial_hold", initial, (0.0,) * len(initial)
        if time_seconds >= REACH_STOP_SECONDS:
            return "target_hold", RIGHT_ARM_TARGET_RAD, (0.0,) * len(initial)
        duration = REACH_STOP_SECONDS - REACH_START_SECONDS
        progress = (time_seconds - REACH_START_SECONDS) / duration
        blend = 3.0 * progress**2 - 2.0 * progress**3
        blend_rate = (6.0 * progress - 6.0 * progress**2) / duration
        delta = tuple(
            target - start
            for target, start in zip(RIGHT_ARM_TARGET_RAD, initial, strict=True)
        )
        references = tuple(
            start + blend * offset
            for start, offset in zip(initial, delta, strict=True)
        )
        velocities = tuple(blend_rate * offset for offset in delta)
        return "reach", references, velocities

    @staticmethod
    def _unclipped_joint_action(
        channel: MotorChannel,
        data: mujoco.MjData,
        *,
        reference_position_rad: float,
        reference_velocity_rad_s: float,
        stiffness_nm_per_rad: float,
        damping_nm_s_per_rad: float,
    ) -> float:
        torque_nm = (
            float(data.qfrc_bias[channel.dof_address])
            + stiffness_nm_per_rad
            * (reference_position_rad - float(data.qpos[channel.qpos_address]))
            + damping_nm_s_per_rad
            * (reference_velocity_rad_s - float(data.qvel[channel.dof_address]))
        )
        return torque_nm / channel.gear

    def _unclipped_hip_action(
        self,
        data: mujoco.MjData,
        *,
        reference_rad: float,
    ) -> float:
        mean_position = 0.5 * (
            float(data.qpos[self._hip_channel.qpos_address])
            + float(data.qpos[self._hip_second_qpos_address])
        )
        mean_velocity = 0.5 * (
            float(data.qvel[self._hip_channel.dof_address])
            + float(data.qvel[self._hip_second_dof_address])
        )
        torque_nm = (
            float(data.qfrc_bias[self._hip_channel.dof_address])
            + float(data.qfrc_bias[self._hip_second_dof_address])
            + HIP_STIFFNESS_NM_RAD * (reference_rad - mean_position)
            - HIP_DAMPING_NM_S_RAD * mean_velocity
        )
        return torque_nm / self._hip_channel.gear

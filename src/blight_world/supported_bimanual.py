"""Supported actuator-only bimanual foam-block clamp for MIMo.

This module implements one narrow, calibrated manipulation primitive: a
world-supported MIMo brings both open palms against a free foam block and
holds a bilateral clamp.  It is explicitly not pickup, lift, grasp-lift,
locomotion, or learned behavior.  The block remains free and is moved only by
MuJoCo gravity and contact from actuator-driven arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
from blight_world.motor_skills import (
    HEAD_DAMPING_NM_S_RAD,
    HEAD_STIFFNESS_NM_RAD,
    HIP_DAMPING_NM_S_RAD,
    HIP_STIFFNESS_NM_RAD,
)


SUPPORTED_BIMANUAL_SCENE_XML = """<worldbody>
    <body name="foam_plinth" pos="0.106 0 0.206">
      <geom name="foam_plinth_top" type="box" size="0.070 0.075 0.206"
            friction="1.0 .08 .02" condim="6"/>
    </body>
    <body name="foam_block" pos="0.106 0 0.452">
      <freejoint name="foam_block_joint"/>
      <geom name="foam_block_geom" type="box" size="0.04 0.085 0.04"
            mass="0.06" friction="1.2 .12 .04" condim="6"
            solref=".004 1" solimp=".95 .995 .001 .5 2"/>
    </body>
  </worldbody>
  <equality>
    <weld name="root_support" body1="mimo_location"
          solref=".01 1" solimp=".99 .999 .001 .5 2"/>
  </equality>"""

SUPPORTED_BIMANUAL_SUPPORT_LABEL = (
    "physical world weld supporting MIMo's root; the foam block remains free"
)

ARM_ACTUATOR_SUFFIXES = (
    "shoulder_horizontal",
    "shoulder_abduction",
    "shoulder_internal",
    "elbow",
    "wrist_rotation",
    "wrist_flexion",
    "wrist_ulnar",
)
RIGHT_BIMANUAL_ARM_ACTUATOR_NAMES = tuple(
    f"act:right_{suffix}" for suffix in ARM_ACTUATOR_SUFFIXES
)
LEFT_BIMANUAL_ARM_ACTUATOR_NAMES = tuple(
    f"act:left_{suffix}" for suffix in ARM_ACTUATOR_SUFFIXES
)

APPROACH_ARM_REFERENCE_RAD = (
    1.4317933503112434,
    0.7090671874996014,
    0.4244853076325576,
    -0.5476365301180084,
    1.2082119612214457,
    -0.47294133856767023,
    0.28251239202611494,
)
CONTACT_ARM_REFERENCE_RAD = (
    1.5690281233266343,
    0.6265706125172718,
    0.09446994139328434,
    -0.4843003077166465,
    1.4673264944459885,
    -0.5349494123373645,
    0.2367896738152049,
)
SQUEEZE_ARM_REFERENCE_RAD = (
    1.15,
    0.4849874275149149,
    0.10972051069246462,
    -0.5306940701003179,
    1.25,
    -0.5495846280529855,
    0.18381892405451292,
)
ARM_STIFFNESS_NM_RAD = (2.0, 2.5, 2.0, 2.0, 0.6, 0.8, 0.6)
ARM_DAMPING_NM_S_RAD = (0.225, 0.25, 0.225, 0.20, 0.07, 0.09, 0.07)

RESET_HOLD_STOP_SECONDS = 0.3
APPROACH_STOP_SECONDS = 1.7
CONTACT_STOP_SECONDS = 2.8
SQUEEZE_START_SECONDS = 4.2
TRIAL_STOP_SECONDS = 7.0

LEFT_PALM_GEOM_NAMES = tuple(f"geom:left_hand{index}" for index in range(1, 5))
RIGHT_PALM_GEOM_NAMES = tuple(
    f"geom:right_hand{index}" for index in range(1, 5)
)

ClampControlMode = Literal["active", "open", "one_arm"]


class BimanualClampPhase(str, Enum):
    """Named segment of the supported clamp trajectory."""

    RESET_HOLD = "reset_hold"
    APPROACH = "approach"
    CONTACT = "contact"
    SQUEEZE = "squeeze"
    SQUEEZE_HOLD = "squeeze_hold"
    CONTROL_HOLD = "control_hold"


class SupportedBimanualFoamBlockEnv(BlightWorldEnv):
    """Dedicated plinth and free foam-block scene with root support only."""

    _scene_extension_xml = SUPPORTED_BIMANUAL_SCENE_XML


@dataclass(frozen=True, slots=True)
class BimanualClampControllerState:
    """Immutable reset references and sparse-composer history."""

    right_initial_positions_rad: tuple[float, ...]
    left_initial_positions_rad: tuple[float, ...]
    hip_reference_rad: float
    head_reference_rad: float
    motor_state: MotorControllerState

    def __post_init__(self) -> None:
        if len(self.right_initial_positions_rad) != 7:
            raise ValueError("right_initial_positions_rad must contain seven values")
        if len(self.left_initial_positions_rad) != 7:
            raise ValueError("left_initial_positions_rad must contain seven values")
        values = (
            *self.right_initial_positions_rad,
            *self.left_initial_positions_rad,
            self.hip_reference_rad,
            self.head_reference_rad,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("bimanual clamp references must be finite")


@dataclass(frozen=True, slots=True)
class BimanualClampControlDiagnostics:
    """Immutable evidence for one controller composition."""

    time_seconds: float
    phase: BimanualClampPhase
    mode: ClampControlMode
    right_reference_positions_rad: tuple[float, ...]
    left_reference_positions_rad: tuple[float, ...]
    right_reference_velocities_rad_s: tuple[float, ...]
    left_reference_velocities_rad_s: tuple[float, ...]
    saturated_actuators: tuple[str, ...]


class SupportedBimanualFoamBlockClampController:
    """Symmetric bias-compensated arm controller for the supported clamp.

    Only the fourteen arm channels and paired-hip/head posture channels are
    owned.  Digits and every other action coordinate remain exactly zero.
    Runtime calls are pure with respect to MuJoCo model and data.
    """

    __slots__ = (
        "_composer",
        "_head_channel",
        "_hip_channel",
        "_hip_second_dof_address",
        "_hip_second_qpos_address",
        "_left_arm_channels",
        "_model",
        "_right_arm_channels",
    )

    def __init__(self, environment: SupportedBimanualFoamBlockEnv) -> None:
        self._model = environment.model
        self._right_arm_channels = tuple(
            resolve_motor_channel(environment, name)
            for name in RIGHT_BIMANUAL_ARM_ACTUATOR_NAMES
        )
        self._left_arm_channels = tuple(
            resolve_motor_channel(environment, name)
            for name in LEFT_BIMANUAL_ARM_ACTUATOR_NAMES
        )
        self._hip_channel = resolve_motor_channel(environment, "act:hip_bend")
        self._head_channel = resolve_motor_channel(environment, "act:head_tilt")
        hip_second = environment.model.joint("robot:hip_bend2")
        self._hip_second_qpos_address = int(np.asarray(hip_second.qposadr).item())
        self._hip_second_dof_address = int(np.asarray(hip_second.dofadr).item())
        self._composer = SparseActionComposer(resolve_motor_channels(environment))

    @property
    def controlled_actuator_names(self) -> tuple[str, ...]:
        """Return the exact sixteen name-resolved channels owned."""

        return (
            *RIGHT_BIMANUAL_ARM_ACTUATOR_NAMES,
            *LEFT_BIMANUAL_ARM_ACTUATOR_NAMES,
            self._hip_channel.actuator_name,
            self._head_channel.actuator_name,
        )

    def reset(
        self, environment: SupportedBimanualFoamBlockEnv
    ) -> BimanualClampControllerState:
        """Capture reset references and return deterministic zero history."""

        self._require_environment(environment)
        right_initial = tuple(
            float(environment.data.qpos[channel.qpos_address])
            for channel in self._right_arm_channels
        )
        left_initial = tuple(
            float(environment.data.qpos[channel.qpos_address])
            for channel in self._left_arm_channels
        )
        hip_reference = 0.5 * (
            float(environment.data.qpos[self._hip_channel.qpos_address])
            + float(environment.data.qpos[self._hip_second_qpos_address])
        )
        return BimanualClampControllerState(
            right_initial_positions_rad=right_initial,
            left_initial_positions_rad=left_initial,
            hip_reference_rad=hip_reference,
            head_reference_rad=float(
                environment.data.qpos[self._head_channel.qpos_address]
            ),
            motor_state=self._composer.reset(),
        )

    def compose(
        self,
        environment: SupportedBimanualFoamBlockEnv,
        state: BimanualClampControllerState,
        *,
        mode: ClampControlMode = "active",
        time_seconds: float | None = None,
    ) -> tuple[
        NDArray[np.float32],
        BimanualClampControllerState,
        BimanualClampControlDiagnostics,
    ]:
        """Return one bounded sparse action without writing simulation state."""

        self._require_environment(environment)
        if mode not in ("active", "open", "one_arm"):
            raise ValueError(f"unsupported clamp mode: {mode!r}")
        time_value = (
            float(state.motor_state.step_index * environment.dt)
            if time_seconds is None
            else float(time_seconds)
        )
        if not isfinite(time_value) or time_value < 0.0:
            raise ValueError("time_seconds must be finite and non-negative")

        if mode == "open":
            phase = BimanualClampPhase.CONTROL_HOLD
            right_references = state.right_initial_positions_rad
            left_references = state.left_initial_positions_rad
            right_velocities = (0.0,) * 7
            left_velocities = (0.0,) * 7
        else:
            phase, active_references, active_velocities = self._trajectory(
                state.right_initial_positions_rad,
                time_value,
            )
            right_references = active_references
            right_velocities = active_velocities
            if mode == "one_arm":
                left_references = state.left_initial_positions_rad
                left_velocities = (0.0,) * 7
            else:
                _, left_references, left_velocities = self._trajectory(
                    state.left_initial_positions_rad,
                    time_value,
                )

        raw_by_name: dict[str, float] = {}
        right_commands = self._arm_commands(
            environment,
            self._right_arm_channels,
            right_references,
            right_velocities,
            raw_by_name,
        )
        left_commands = self._arm_commands(
            environment,
            self._left_arm_channels,
            left_references,
            left_velocities,
            raw_by_name,
        )
        hip_raw = self._unclipped_hip_action(
            environment,
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
        raw_by_name[self._hip_channel.actuator_name] = hip_raw
        raw_by_name[self._head_channel.actuator_name] = head_raw
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
            {
                "right_arm": right_commands,
                "left_arm": left_commands,
                "posture": posture_commands,
            },
            state.motor_state,
        )
        channels = (
            *self._right_arm_channels,
            *self._left_arm_channels,
            self._hip_channel,
            self._head_channel,
        )
        channel_by_name = {channel.actuator_name: channel for channel in channels}
        saturated = tuple(
            name
            for name, value in raw_by_name.items()
            if value < channel_by_name[name].lower_bound
            or value > channel_by_name[name].upper_bound
        )
        return (
            action,
            BimanualClampControllerState(
                right_initial_positions_rad=state.right_initial_positions_rad,
                left_initial_positions_rad=state.left_initial_positions_rad,
                hip_reference_rad=state.hip_reference_rad,
                head_reference_rad=state.head_reference_rad,
                motor_state=motor_state,
            ),
            BimanualClampControlDiagnostics(
                time_seconds=time_value,
                phase=phase,
                mode=mode,
                right_reference_positions_rad=right_references,
                left_reference_positions_rad=left_references,
                right_reference_velocities_rad_s=right_velocities,
                left_reference_velocities_rad_s=left_velocities,
                saturated_actuators=saturated,
            ),
        )

    def _arm_commands(
        self,
        environment: SupportedBimanualFoamBlockEnv,
        channels: tuple[MotorChannel, ...],
        references: tuple[float, ...],
        velocities: tuple[float, ...],
        raw_by_name: dict[str, float],
    ) -> dict[MotorChannel, float]:
        commands: dict[MotorChannel, float] = {}
        for channel, reference, velocity, stiffness, damping in zip(
            channels,
            references,
            velocities,
            ARM_STIFFNESS_NM_RAD,
            ARM_DAMPING_NM_S_RAD,
            strict=True,
        ):
            raw = self._unclipped_joint_action(
                channel,
                environment.data,
                reference_position_rad=reference,
                reference_velocity_rad_s=velocity,
                stiffness_nm_per_rad=stiffness,
                damping_nm_s_per_rad=damping,
            )
            raw_by_name[channel.actuator_name] = raw
            commands[channel] = raw
        return commands

    @staticmethod
    def _trajectory(
        initial: tuple[float, ...], time_seconds: float
    ) -> tuple[BimanualClampPhase, tuple[float, ...], tuple[float, ...]]:
        if time_seconds < RESET_HOLD_STOP_SECONDS:
            return BimanualClampPhase.RESET_HOLD, initial, (0.0,) * 7
        if time_seconds < APPROACH_STOP_SECONDS:
            references, velocities = SupportedBimanualFoamBlockClampController._blend(
                initial,
                APPROACH_ARM_REFERENCE_RAD,
                time_seconds,
                RESET_HOLD_STOP_SECONDS,
                APPROACH_STOP_SECONDS,
            )
            return BimanualClampPhase.APPROACH, references, velocities
        if time_seconds < CONTACT_STOP_SECONDS:
            references, velocities = SupportedBimanualFoamBlockClampController._blend(
                APPROACH_ARM_REFERENCE_RAD,
                CONTACT_ARM_REFERENCE_RAD,
                time_seconds,
                APPROACH_STOP_SECONDS,
                CONTACT_STOP_SECONDS,
            )
            return BimanualClampPhase.CONTACT, references, velocities
        if time_seconds < SQUEEZE_START_SECONDS:
            references, velocities = SupportedBimanualFoamBlockClampController._blend(
                CONTACT_ARM_REFERENCE_RAD,
                SQUEEZE_ARM_REFERENCE_RAD,
                time_seconds,
                CONTACT_STOP_SECONDS,
                SQUEEZE_START_SECONDS,
            )
            return BimanualClampPhase.SQUEEZE, references, velocities
        return BimanualClampPhase.SQUEEZE_HOLD, SQUEEZE_ARM_REFERENCE_RAD, (0.0,) * 7

    @staticmethod
    def _blend(
        start: tuple[float, ...],
        stop: tuple[float, ...],
        time_seconds: float,
        start_seconds: float,
        stop_seconds: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        duration = stop_seconds - start_seconds
        progress = float(np.clip((time_seconds - start_seconds) / duration, 0.0, 1.0))
        blend = 3.0 * progress**2 - 2.0 * progress**3
        blend_rate = (6.0 * progress - 6.0 * progress**2) / duration
        delta = tuple(
            end - begin for begin, end in zip(start, stop, strict=True)
        )
        references = tuple(
            begin + blend * offset
            for begin, offset in zip(start, delta, strict=True)
        )
        velocities = tuple(blend_rate * offset for offset in delta)
        return references, velocities

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
        environment: SupportedBimanualFoamBlockEnv,
        *,
        reference_rad: float,
    ) -> float:
        mean_position = 0.5 * (
            float(environment.data.qpos[self._hip_channel.qpos_address])
            + float(environment.data.qpos[self._hip_second_qpos_address])
        )
        mean_velocity = 0.5 * (
            float(environment.data.qvel[self._hip_channel.dof_address])
            + float(environment.data.qvel[self._hip_second_dof_address])
        )
        torque_nm = (
            float(environment.data.qfrc_bias[self._hip_channel.dof_address])
            + float(environment.data.qfrc_bias[self._hip_second_dof_address])
            + HIP_STIFFNESS_NM_RAD * (reference_rad - mean_position)
            - HIP_DAMPING_NM_S_RAD * mean_velocity
        )
        return torque_nm / self._hip_channel.gear

    def _require_environment(
        self, environment: SupportedBimanualFoamBlockEnv
    ) -> None:
        if environment.model is not self._model:
            raise ValueError("controller belongs to a different MuJoCo model")


@dataclass(frozen=True, slots=True)
class FoamBlockContact:
    """One solved block contact in MuJoCo contact-array order."""

    geom_name: str
    normal_force_n: float
    position_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class FoamBlockContacts:
    """Immutable classification of one solved block-contact snapshot."""

    time_seconds: float
    left_palm: tuple[FoamBlockContact, ...]
    right_palm: tuple[FoamBlockContact, ...]
    support_normal_force_n: float
    other: tuple[FoamBlockContact, ...]
    maximum_normal_force_n: float

    @property
    def left_normal_force_n(self) -> float:
        return float(sum(contact.normal_force_n for contact in self.left_palm))

    @property
    def right_normal_force_n(self) -> float:
        return float(sum(contact.normal_force_n for contact in self.right_palm))

    @property
    def bilateral_palm(self) -> bool:
        return bool(self.left_palm and self.right_palm)

    @property
    def distinct_palm_geom_count(self) -> int:
        return len(
            {
                contact.geom_name
                for contact in (*self.left_palm, *self.right_palm)
            }
        )


class FoamBlockContactClassifier:
    """Classify solved plinth and palm contacts involving the free block."""

    __slots__ = (
        "_block_geom_id",
        "_environment",
        "_left_palm_geom_ids",
        "_right_palm_geom_ids",
        "_support_geom_id",
    )

    def __init__(self, environment: SupportedBimanualFoamBlockEnv) -> None:
        self._environment = environment
        self._block_geom_id = int(environment.model.geom("foam_block_geom").id)
        self._support_geom_id = int(environment.model.geom("foam_plinth_top").id)
        self._left_palm_geom_ids = frozenset(
            int(environment.model.geom(name).id) for name in LEFT_PALM_GEOM_NAMES
        )
        self._right_palm_geom_ids = frozenset(
            int(environment.model.geom(name).id) for name in RIGHT_PALM_GEOM_NAMES
        )

    def classify(self) -> FoamBlockContacts:
        """Read solved contacts without mutating MuJoCo state."""

        model = self._environment.model
        data = self._environment.data
        left: list[FoamBlockContact] = []
        right: list[FoamBlockContact] = []
        other: list[FoamBlockContact] = []
        support_force = 0.0
        maximum_force = 0.0
        for contact_index, contact in enumerate(data.contact):
            if int(contact.efc_address) < 0:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self._block_geom_id not in (geom1, geom2):
                continue
            opposing_geom = geom2 if geom1 == self._block_geom_id else geom1
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, force)
            normal_force = abs(float(force[0]))
            maximum_force = max(maximum_force, normal_force)
            if opposing_geom == self._support_geom_id:
                support_force += normal_force
                continue
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, opposing_geom
            )
            if geom_name is None:
                raise ValueError(f"geom {opposing_geom} has no name")
            sample = FoamBlockContact(
                geom_name=geom_name,
                normal_force_n=normal_force,
                position_m=tuple(float(value) for value in contact.pos),
            )
            if opposing_geom in self._left_palm_geom_ids:
                left.append(sample)
            elif opposing_geom in self._right_palm_geom_ids:
                right.append(sample)
            else:
                other.append(sample)
        return FoamBlockContacts(
            time_seconds=float(data.time),
            left_palm=tuple(left),
            right_palm=tuple(right),
            support_normal_force_n=support_force,
            other=tuple(other),
            maximum_normal_force_n=maximum_force,
        )

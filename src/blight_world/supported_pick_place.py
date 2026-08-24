"""Supported, scripted right-hand sphere pick-and-place control.

This module extends the validated supported-push scene with a deterministic
actuator-only manipulation state machine.  The target remains a free MuJoCo
body and the only body equality is the visibly named ``root_support`` weld.
The controller is an engineering fixture for supported manipulation: it is not
learned behavior, balance, locomotion, or an equality-assisted grasp.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Final, Literal

import mujoco
import numpy as np
from numpy.typing import NDArray

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
    RIGHT_ARM_ACTUATOR_NAMES,
    RIGHT_ARM_DAMPING_NM_S_RAD,
    RIGHT_ARM_STIFFNESS_NM_RAD,
    RIGHT_ARM_TARGET_RAD,
    SupportedPushEnv,
)


class PickPlacePhase(str, Enum):
    """Ordered phases of the supported scripted manipulation fixture."""

    OPEN_HOLD = "OPEN_HOLD"
    REACH_PREGRASP = "REACH_PREGRASP"
    FINAL_APPROACH = "FINAL_APPROACH"
    ENCLOSE = "ENCLOSE"
    SECURE = "SECURE"
    LIFT = "LIFT"
    CARRY = "CARRY"
    RETURN_LOWER = "RETURN_LOWER"
    RELEASE = "RELEASE"
    RETRACT = "RETRACT"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"


DigitGroup = Literal["ff", "mf", "rf", "lf", "thumb"]
DIGIT_GROUPS: Final[tuple[DigitGroup, ...]] = (
    "ff",
    "mf",
    "rf",
    "lf",
    "thumb",
)

RIGHT_DIGIT_ACTUATOR_NAMES: Final[tuple[str, ...]] = (
    "act:right_ff_side",
    "act:right_ff_knuckle",
    "act:right_ff_middle",
    "act:right_ff_distal",
    "act:right_mf_side",
    "act:right_mf_knuckle",
    "act:right_mf_middle",
    "act:right_mf_distal",
    "act:right_rf_side",
    "act:right_rf_knuckle",
    "act:right_rf_middle",
    "act:right_rf_distal",
    "act:right_lf_side",
    "act:right_lf_meta",
    "act:right_lf_knuckle",
    "act:right_lf_middle",
    "act:right_lf_distal",
    "act:right_thumb_side",
    "act:right_thumb_add",
    "act:right_thumb_pivot",
    "act:right_thumb_middle",
    "act:right_thumb_distal",
)

RIGHT_DIGIT_JOINT_RANGES_RAD: Final[tuple[tuple[float, float], ...]] = (
    (-0.43633, 0.43633),
    (-0.34907, 1.57080),
    (-0.08727, 1.74533),
    (-0.08727, 1.39626),
    (-0.43633, 0.43633),
    (-0.34907, 1.57080),
    (-0.08727, 1.74533),
    (-0.08727, 1.39626),
    (-0.43633, 0.43633),
    (-0.34907, 1.57080),
    (-0.08727, 1.74533),
    (-0.08727, 1.39626),
    (-0.43633, 0.43633),
    (0.0, 0.69813),
    (-0.34907, 1.57080),
    (-0.08727, 1.74533),
    (-0.08727, 1.39626),
    (-0.17453, 1.91986),
    (-1.04720, 0.0),
    (-0.17453, 0.17453),
    (0.0, 1.57080),
    (0.0, 1.57080),
)

RIGHT_DIGIT_NOMINAL_CURL_RAD: Final[tuple[float, ...]] = (
    0.0,
    np.pi / 4.0,
    np.pi / 4.0,
    np.pi / 6.0,
    0.0,
    np.pi / 4.0,
    np.pi / 4.0,
    np.pi / 6.0,
    0.0,
    np.pi / 4.0,
    np.pi / 4.0,
    np.pi / 6.0,
    0.0,
    0.349066,
    np.pi / 4.0,
    np.pi / 4.0,
    np.pi / 6.0,
    0.0,
    -0.9,
    0.0,
    0.4,
    0.3,
)

RIGHT_DIGIT_SEARCH_CAP_RAD: Final[tuple[float, ...]] = (
    0.0,
    1.10,
    1.22,
    0.98,
    0.0,
    1.10,
    1.22,
    0.98,
    0.0,
    1.10,
    1.22,
    0.98,
    0.0,
    0.49,
    1.10,
    1.22,
    0.98,
    0.0,
    -0.9,
    0.0,
    1.10,
    1.10,
)

RIGHT_DIGIT_GROUP_BY_ACTUATOR: Final[dict[str, DigitGroup]] = {
    name: (
        "thumb"
        if "thumb" in name
        else name.removeprefix("act:right_").split("_", maxsplit=1)[0]
    )
    for name in RIGHT_DIGIT_ACTUATOR_NAMES
}

PREGRASP_ARM_SEED_RAD: Final[tuple[float, ...]] = (
    1.0989731651,
    0.8706499629,
    0.9248339337,
    -0.1817758584,
    -0.1230801428,
    -0.1333444241,
    -0.1279686161,
)
LIFT_ARM_SEED_RAD: Final[tuple[float, ...]] = (
    1.1834514287077815,
    1.5365945906049567,
    1.1263333063585548,
    -0.7106572589996581,
    -0.2545961859903202,
    0.13084197814383444,
    0.1329125999065053,
)
CARRY_ARM_SEED_RAD: Final[tuple[float, ...]] = (
    1.334757918417238,
    1.2856910021585621,
    1.1252748456736834,
    -0.16051100136757557,
    -0.23680073085058856,
    -0.0335557496209037,
    -0.12699252848344258,
)
LIFT_HAND_POSITION_M: Final[tuple[float, float, float]] = (
    0.207480873,
    -0.132959261,
    0.624690035,
)
CARRY_HAND_POSITION_M: Final[tuple[float, float, float]] = (
    0.227480873,
    -0.132959261,
    0.624690035,
)
CARRY_DLS_STEP_CAP_RAD: Final[float] = 0.05
CARRY_DLS_DAMPING: Final[float] = 1e-5


class SupportedPickPlaceEnv(SupportedPushEnv):
    """Dedicated alias of the validated free-sphere/root-supported scene."""


@dataclass(frozen=True, slots=True)
class SupportedPickPlaceConfig:
    """Engineering timings and gains for the scripted supported controller.

    These defaults are deterministic control parameters, not measured human
    motor constants.  Durations and contact thresholds are exposed so physical
    validation can revise them explicitly rather than hiding empirical tuning.
    """

    open_hold_seconds: float = 1.00
    reach_pregrasp_seconds: float = 3.50
    final_approach_seconds: float = 1.50
    final_approach_deadline_seconds: float = 2.00
    digit_closure_fraction_per_second: float | None = None
    enclosure_deadline_seconds: float = 3.00
    secure_hold_seconds: float = 0.250
    lift_seconds: float = 2.00
    lift_gate_hold_seconds: float = 0.250
    lift_deadline_seconds: float = 4.00
    lift_minimum_target_bottom_m: float = 0.368
    carry_seconds: float = 1.25
    carry_deadline_seconds: float = 3.00
    carry_minimum_x_excursion_m: float = 0.015
    return_lower_seconds: float = 3.25
    return_carry_fraction: float = 0.30
    return_table_deadline_seconds: float = 4.00
    table_contact_hold_seconds: float = 0.250
    release_seconds: float = 1.00
    release_clear_hold_seconds: float = 0.250
    retract_seconds: float = 2.50
    total_deadline_seconds: float = 20.0
    precontact_maximum_planar_drift_m: float = 0.002
    digit_position_action_per_rad: float | None = None
    digit_velocity_action_seconds_per_rad: float | None = None
    digit_contact_preload_fraction: float | None = None
    maximum_contact_force_n: float | None = None
    maximum_orientation_drift_rad: float | None = None

    def __post_init__(self) -> None:
        positive = (
            "open_hold_seconds",
            "reach_pregrasp_seconds",
            "final_approach_seconds",
            "final_approach_deadline_seconds",
            "enclosure_deadline_seconds",
            "secure_hold_seconds",
            "lift_seconds",
            "lift_gate_hold_seconds",
            "lift_deadline_seconds",
            "lift_minimum_target_bottom_m",
            "carry_seconds",
            "carry_deadline_seconds",
            "carry_minimum_x_excursion_m",
            "return_lower_seconds",
            "return_table_deadline_seconds",
            "table_contact_hold_seconds",
            "release_seconds",
            "release_clear_hold_seconds",
            "retract_seconds",
            "total_deadline_seconds",
            "precontact_maximum_planar_drift_m",
        )
        for name in positive:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not isfinite(self.return_carry_fraction)
            or not 0.0 < self.return_carry_fraction < 1.0
        ):
            raise ValueError("return_carry_fraction must lie strictly within (0, 1)")
        if self.final_approach_deadline_seconds < self.final_approach_seconds:
            raise ValueError(
                "final_approach_deadline_seconds cannot precede approach duration"
            )
        if self.lift_deadline_seconds < self.lift_seconds:
            raise ValueError("lift_deadline_seconds cannot precede lift duration")
        if self.carry_deadline_seconds < self.carry_seconds:
            raise ValueError("carry_deadline_seconds cannot precede carry duration")
        for name in (
            "digit_closure_fraction_per_second",
            "digit_position_action_per_rad",
            "digit_velocity_action_seconds_per_rad",
            "maximum_contact_force_n",
            "maximum_orientation_drift_rad",
        ):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be None or finite and positive")
        if self.digit_contact_preload_fraction is not None and (
            not isfinite(self.digit_contact_preload_fraction)
            or not 0.0 <= self.digit_contact_preload_fraction <= 1.0
        ):
            raise ValueError(
                "digit_contact_preload_fraction must be None or lie within [0, 1]"
            )


@dataclass(frozen=True, slots=True)
class HandTargetContact:
    """One active right-hand/target contact with target-radial direction."""

    geom_name: str
    group: str
    radial_direction: tuple[float, float, float]
    normal_force_n: float = 0.0

    def __post_init__(self) -> None:
        if not self.geom_name:
            raise ValueError("geom_name cannot be empty")
        if self.group not in {*DIGIT_GROUPS, "palm"}:
            raise ValueError(f"unknown hand contact group: {self.group}")
        direction = np.asarray(self.radial_direction, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("radial_direction must contain three finite values")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("radial_direction must be nonzero")
        normalized = tuple(float(value) for value in direction / norm)
        object.__setattr__(self, "radial_direction", normalized)
        if not isfinite(self.normal_force_n) or self.normal_force_n < 0.0:
            raise ValueError("normal_force_n must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PickPlaceContacts:
    """Immutable privileged contact summary used only by the controller."""

    samples: tuple[HandTargetContact, ...] = ()
    table_contact: bool = False

    @property
    def has_hand_contact(self) -> bool:
        return any(sample.normal_force_n > 1e-9 for sample in self.samples)

    @property
    def contacted_digit_groups(self) -> frozenset[DigitGroup]:
        return frozenset(
            sample.group for sample in self.samples if sample.group in DIGIT_GROUPS
        )  # type: ignore[return-value]

    @property
    def opposed_non_thumb_geom_names(self) -> tuple[str, ...]:
        thumb_directions = [
            np.asarray(sample.radial_direction)
            for sample in self.samples
            if sample.group == "thumb"
        ]
        if not thumb_directions:
            return ()
        opposed = {
            sample.geom_name
            for sample in self.samples
            if sample.group != "thumb"
            and any(
                float(np.dot(thumb, sample.radial_direction)) < 0.0
                for thumb in thumb_directions
            )
        }
        return tuple(sorted(opposed))

    @property
    def secure_geometry(self) -> bool:
        return (
            any(sample.group == "thumb" for sample in self.samples)
            and len(self.opposed_non_thumb_geom_names) >= 2
        )


class RightHandTargetContactClassifier:
    """Classify active contacts for the free sphere and right-hand subtree."""

    __slots__ = ("_environment", "_hand_geom_ids", "_table_geom_id", "_target_geom_id")

    def __init__(self, environment: SupportedPickPlaceEnv) -> None:
        self._environment = environment
        self._target_geom_id = int(environment.model.geom("push_target_geom").id)
        self._table_geom_id = int(environment.model.geom("push_table_top").id)
        hand_body_id = int(environment.model.body("right_hand").id)
        hand_body_ids = {
            body_id
            for body_id in range(environment.model.nbody)
            if self._is_descendant(environment.model, body_id, hand_body_id)
        }
        self._hand_geom_ids = frozenset(
            geom_id
            for geom_id in range(environment.model.ngeom)
            if int(environment.model.geom_bodyid[geom_id]) in hand_body_ids
        )

    @property
    def hand_geom_ids(self) -> frozenset[int]:
        return self._hand_geom_ids

    def classify(self) -> PickPlaceContacts:
        """Read the just-solved MuJoCo contacts without mutating simulation."""

        environment = self._environment
        target_position = np.asarray(
            environment.data.geom_xpos[self._target_geom_id], dtype=np.float64
        )
        samples: dict[str, HandTargetContact] = {}
        table_contact = False
        for contact_index, contact in enumerate(environment.data.contact):
            if int(contact.efc_address) < 0:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self._target_geom_id not in (geom1, geom2):
                continue
            other = geom2 if geom1 == self._target_geom_id else geom1
            if other == self._table_geom_id:
                table_contact = True
                continue
            if other not in self._hand_geom_ids:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                environment.model,
                environment.data,
                contact_index,
                force,
            )
            normal_force = abs(float(force[0]))
            if normal_force <= 1e-9:
                continue
            geom_name = mujoco.mj_id2name(
                environment.model, mujoco.mjtObj.mjOBJ_GEOM, other
            )
            if geom_name is None:
                raise ValueError(f"right-hand geom {other} has no name")
            radial = np.asarray(contact.pos, dtype=np.float64) - target_position
            samples[geom_name] = HandTargetContact(
                geom_name=geom_name,
                group=self._geom_group(geom_name),
                radial_direction=tuple(float(value) for value in radial),
                normal_force_n=normal_force,
            )
        return PickPlaceContacts(
            samples=tuple(samples[name] for name in sorted(samples)),
            table_contact=table_contact,
        )

    @staticmethod
    def _is_descendant(model: mujoco.MjModel, body_id: int, root_id: int) -> bool:
        current = body_id
        while current != 0:
            if current == root_id:
                return True
            current = int(model.body_parentid[current])
        return current == root_id

    @staticmethod
    def _geom_group(geom_name: str) -> str:
        if geom_name.startswith("geom:right_th"):
            return "thumb"
        for prefix, group in (
            ("geom:right_ff", "ff"),
            ("geom:right_mf", "mf"),
            ("geom:right_rf", "rf"),
            ("geom:right_lf", "lf"),
        ):
            if geom_name.startswith(prefix):
                return group
        if geom_name.startswith("geom:right_hand"):
            return "palm"
        raise ValueError(f"unclassified right-hand geometry: {geom_name}")


@dataclass(frozen=True, slots=True)
class SupportedPickPlaceState:
    """Complete immutable state of the scripted pick-and-place controller."""

    phase: PickPlacePhase
    phase_started_seconds: float
    last_time_seconds: float
    initial_arm_positions_rad: tuple[float, ...]
    initial_target_position_m: tuple[float, float, float]
    initial_target_quaternion_wxyz: tuple[float, float, float, float]
    hip_reference_rad: float
    head_reference_rad: float
    digit_closure_fractions: tuple[float, ...]
    preloaded_digit_groups: tuple[DigitGroup, ...]
    frozen_arm_positions_rad: tuple[float, ...] | None
    first_hand_contact_seconds: float | None
    enclosure_started_seconds: float | None
    secure_started_seconds: float | None
    lift_gate_started_seconds: float | None
    carry_origin_x_m: float | None
    table_contact_started_seconds: float | None
    release_clear_started_seconds: float | None
    failure_reason: str | None
    motor_state: MotorControllerState

    def __post_init__(self) -> None:
        if len(self.initial_arm_positions_rad) != 7:
            raise ValueError("initial_arm_positions_rad must contain seven values")
        if len(self.digit_closure_fractions) != len(DIGIT_GROUPS):
            raise ValueError("digit_closure_fractions must contain five values")
        if any(group not in DIGIT_GROUPS for group in self.preloaded_digit_groups):
            raise ValueError("preloaded_digit_groups contains an unknown group")
        if len(set(self.preloaded_digit_groups)) != len(self.preloaded_digit_groups):
            raise ValueError("preloaded_digit_groups cannot contain duplicates")
        if self.frozen_arm_positions_rad is not None and len(
            self.frozen_arm_positions_rad
        ) != 7:
            raise ValueError("frozen_arm_positions_rad must contain seven values")
        if self.first_hand_contact_seconds is not None and (
            not isfinite(self.first_hand_contact_seconds)
            or self.first_hand_contact_seconds < 0.0
        ):
            raise ValueError("first_hand_contact_seconds must be finite and non-negative")
        values = (
            self.phase_started_seconds,
            self.last_time_seconds,
            *self.initial_arm_positions_rad,
            *self.initial_target_position_m,
            *self.initial_target_quaternion_wxyz,
            self.hip_reference_rad,
            self.head_reference_rad,
            *self.digit_closure_fractions,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("pick-place controller state must be finite")
        if any(not 0.0 <= value <= 1.0 for value in self.digit_closure_fractions):
            raise ValueError("digit closure fractions must lie within [0, 1]")
        quaternion_norm = float(np.linalg.norm(self.initial_target_quaternion_wxyz))
        if not np.isclose(quaternion_norm, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("initial target quaternion must be normalized")


@dataclass(frozen=True, slots=True)
class SupportedPickPlaceDiagnostics:
    """Immutable per-step state, contact, and command evidence."""

    time_seconds: float
    phase: PickPlacePhase
    phase_elapsed_seconds: float
    contacts: PickPlaceContacts
    arm_reference_positions_rad: tuple[float, ...]
    digit_reference_positions_rad: tuple[float, ...]
    controlled_actuator_names: tuple[str, ...]
    saturated_actuators: tuple[str, ...]
    failure_reason: str | None


class RightHandSpherePickPlaceController:
    """Deterministic supported right-hand sphere pick-and-place controller."""

    __slots__ = (
        "_arm_channels",
        "_classifier",
        "_composer",
        "_config",
        "_digit_channels",
        "_head_channel",
        "_hip_channel",
        "_hip_second_dof_address",
        "_hip_second_qpos_address",
        "_model",
    )

    def __init__(
        self,
        environment: SupportedPickPlaceEnv,
        config: SupportedPickPlaceConfig | None = None,
    ) -> None:
        self._model = environment.model
        self._config = config or SupportedPickPlaceConfig()
        self._arm_channels = tuple(
            resolve_motor_channel(environment, name)
            for name in RIGHT_ARM_ACTUATOR_NAMES
        )
        self._digit_channels = tuple(
            resolve_motor_channel(environment, name)
            for name in RIGHT_DIGIT_ACTUATOR_NAMES
        )
        self._hip_channel = resolve_motor_channel(environment, "act:hip_bend")
        self._head_channel = resolve_motor_channel(environment, "act:head_tilt")
        hip_second = environment.model.joint("robot:hip_bend2")
        self._hip_second_qpos_address = int(np.asarray(hip_second.qposadr).item())
        self._hip_second_dof_address = int(np.asarray(hip_second.dofadr).item())
        self._composer = SparseActionComposer(resolve_motor_channels(environment))
        self._classifier = RightHandTargetContactClassifier(environment)

    @property
    def config(self) -> SupportedPickPlaceConfig:
        return self._config

    @property
    def controlled_actuator_names(self) -> tuple[str, ...]:
        return (
            *RIGHT_ARM_ACTUATOR_NAMES,
            *RIGHT_DIGIT_ACTUATOR_NAMES,
            self._hip_channel.actuator_name,
            self._head_channel.actuator_name,
        )

    @property
    def contact_classifier(self) -> RightHandTargetContactClassifier:
        return self._classifier

    def reset(
        self,
        environment: SupportedPickPlaceEnv,
    ) -> SupportedPickPlaceState:
        """Capture the reset posture without changing MuJoCo state."""

        self._require_environment(environment)
        time_seconds = float(environment.data.time)
        target_body = environment.data.body("push_target")
        initial_arm = tuple(
            float(environment.data.qpos[channel.qpos_address])
            for channel in self._arm_channels
        )
        hip_reference = 0.5 * (
            float(environment.data.qpos[self._hip_channel.qpos_address])
            + float(environment.data.qpos[self._hip_second_qpos_address])
        )
        return SupportedPickPlaceState(
            phase=PickPlacePhase.OPEN_HOLD,
            phase_started_seconds=time_seconds,
            last_time_seconds=time_seconds,
            initial_arm_positions_rad=initial_arm,
            initial_target_position_m=tuple(
                float(value) for value in target_body.xpos
            ),
            initial_target_quaternion_wxyz=tuple(
                float(value) for value in target_body.xquat
            ),
            hip_reference_rad=hip_reference,
            head_reference_rad=float(
                environment.data.qpos[self._head_channel.qpos_address]
            ),
            digit_closure_fractions=(0.0,) * len(DIGIT_GROUPS),
            preloaded_digit_groups=(),
            frozen_arm_positions_rad=None,
            first_hand_contact_seconds=None,
            enclosure_started_seconds=None,
            secure_started_seconds=None,
            lift_gate_started_seconds=None,
            carry_origin_x_m=None,
            table_contact_started_seconds=None,
            release_clear_started_seconds=None,
            failure_reason=None,
            motor_state=self._composer.reset(),
        )

    def compose(
        self,
        environment: SupportedPickPlaceEnv,
        state: SupportedPickPlaceState,
        *,
        contacts: PickPlaceContacts | None = None,
        time_seconds: float | None = None,
    ) -> tuple[
        NDArray[np.float32],
        SupportedPickPlaceState,
        SupportedPickPlaceDiagnostics,
    ]:
        """Advance the phase machine and compose one sparse bounded action."""

        self._require_environment(environment)
        time_value = (
            float(environment.data.time)
            if time_seconds is None
            else float(time_seconds)
        )
        if not isfinite(time_value) or time_value < state.last_time_seconds:
            raise ValueError("time_seconds must be finite and monotonic")
        contact_state = self._classifier.classify() if contacts is None else contacts
        digit_position_gain, digit_velocity_gain, _ = (
            self._calibrated_digit_parameters()
        )
        advanced = self._advance_state(environment, state, contact_state, time_value)
        arm_references, arm_velocities = self._arm_references(advanced, time_value)
        digit_references = self._digit_references(advanced, time_value)

        raw_actions: dict[str, float] = {}
        arm_commands: dict[MotorChannel, float] = {}
        for channel, reference, velocity, stiffness, damping in zip(
            self._arm_channels,
            arm_references,
            arm_velocities,
            RIGHT_ARM_STIFFNESS_NM_RAD,
            RIGHT_ARM_DAMPING_NM_S_RAD,
            strict=True,
        ):
            raw = self._unclipped_joint_action(
                channel,
                environment.data,
                reference_position_rad=reference,
                reference_velocity_rad_s=velocity,
                position_action_per_rad=stiffness / abs(channel.gear),
                velocity_action_seconds_per_rad=damping / abs(channel.gear),
            )
            raw_actions[channel.actuator_name] = raw
            arm_commands[channel] = raw

        digit_commands: dict[MotorChannel, float] = {}
        for channel, reference in zip(
            self._digit_channels, digit_references, strict=True
        ):
            raw = self._unclipped_joint_action(
                channel,
                environment.data,
                reference_position_rad=reference,
                reference_velocity_rad_s=0.0,
                position_action_per_rad=digit_position_gain,
                velocity_action_seconds_per_rad=digit_velocity_gain,
            )
            raw_actions[channel.actuator_name] = raw
            digit_commands[channel] = raw

        hip_raw = self._unclipped_hip_action(
            environment.data, advanced.hip_reference_rad
        )
        head_raw = self._unclipped_joint_action(
            self._head_channel,
            environment.data,
            reference_position_rad=advanced.head_reference_rad,
            reference_velocity_rad_s=0.0,
            position_action_per_rad=HEAD_STIFFNESS_NM_RAD
            / abs(self._head_channel.gear),
            velocity_action_seconds_per_rad=HEAD_DAMPING_NM_S_RAD
            / abs(self._head_channel.gear),
        )
        raw_actions[self._hip_channel.actuator_name] = hip_raw
        raw_actions[self._head_channel.actuator_name] = head_raw
        posture_commands = {
            self._hip_channel: coupled_hip_bias_compensated_pd(
                self._hip_channel,
                environment.model,
                environment.data,
                reference_mean_position_rad=advanced.hip_reference_rad,
                stiffness_nm_per_rad=HIP_STIFFNESS_NM_RAD,
                damping_nm_s_per_rad=HIP_DAMPING_NM_S_RAD,
            ),
            self._head_channel: single_joint_bias_compensated_pd(
                self._head_channel,
                environment.data,
                reference_position_rad=advanced.head_reference_rad,
                stiffness_nm_per_rad=HEAD_STIFFNESS_NM_RAD,
                damping_nm_s_per_rad=HEAD_DAMPING_NM_S_RAD,
            ),
        }
        action, motor_state = self._composer.compose(
            {
                "right_arm": arm_commands,
                "right_digits": digit_commands,
                "posture": posture_commands,
            },
            advanced.motor_state,
        )
        channels = (
            *self._arm_channels,
            *self._digit_channels,
            self._hip_channel,
            self._head_channel,
        )
        channel_by_name = {channel.actuator_name: channel for channel in channels}
        saturated = tuple(
            name
            for name, value in raw_actions.items()
            if value < channel_by_name[name].lower_bound
            or value > channel_by_name[name].upper_bound
        )
        next_state = self._replace_state(advanced, motor_state=motor_state)
        diagnostics = SupportedPickPlaceDiagnostics(
            time_seconds=time_value,
            phase=next_state.phase,
            phase_elapsed_seconds=time_value - next_state.phase_started_seconds,
            contacts=contact_state,
            arm_reference_positions_rad=arm_references,
            digit_reference_positions_rad=digit_references,
            controlled_actuator_names=self.controlled_actuator_names,
            saturated_actuators=saturated,
            failure_reason=next_state.failure_reason,
        )
        return action, next_state, diagnostics

    def _advance_state(
        self,
        environment: SupportedPickPlaceEnv,
        state: SupportedPickPlaceState,
        contacts: PickPlaceContacts,
        time_seconds: float,
    ) -> SupportedPickPlaceState:
        config = self._config
        phase = state.phase
        phase_started = state.phase_started_seconds
        closures = list(state.digit_closure_fractions)
        preloaded_groups = set(state.preloaded_digit_groups)
        frozen = state.frozen_arm_positions_rad
        first_hand_contact = state.first_hand_contact_seconds
        enclosure_started = state.enclosure_started_seconds
        secure_started = state.secure_started_seconds
        lift_gate_started = state.lift_gate_started_seconds
        carry_origin_x = state.carry_origin_x_m
        table_contact_started = state.table_contact_started_seconds
        release_clear_started = state.release_clear_started_seconds
        failure_reason = state.failure_reason
        elapsed = time_seconds - phase_started
        delta_time = time_seconds - state.last_time_seconds
        current_hand_contact = (
            contacts.has_hand_contact
            and delta_time <= float(environment.dt) + 1e-12
        )
        target_body = environment.data.body("push_target")
        target_position = np.asarray(target_body.xpos, dtype=np.float64)
        target_quaternion = np.asarray(target_body.xquat, dtype=np.float64)
        precontact_drift = float(
            np.linalg.norm(
                target_position[:2]
                - np.asarray(state.initial_target_position_m[:2], dtype=np.float64)
            )
        )
        maximum_force = max(
            (sample.normal_force_n for sample in contacts.samples),
            default=0.0,
        )
        orientation_drift = self._quaternion_distance_rad(
            np.asarray(state.initial_target_quaternion_wxyz, dtype=np.float64),
            target_quaternion,
        )

        if phase not in {PickPlacePhase.COMPLETE, PickPlacePhase.FAIL} and (
            time_seconds > config.total_deadline_seconds
        ):
            phase = PickPlacePhase.FAIL
            phase_started = time_seconds
            failure_reason = "total deadline exceeded"
        elif (
            phase
            in {
                PickPlacePhase.OPEN_HOLD,
                PickPlacePhase.REACH_PREGRASP,
                PickPlacePhase.FINAL_APPROACH,
            }
            and not contacts.has_hand_contact
            and precontact_drift > config.precontact_maximum_planar_drift_m
        ):
            phase = PickPlacePhase.FAIL
            phase_started = time_seconds
            failure_reason = "target moved before hand contact"
        elif (
            config.maximum_contact_force_n is not None
            and maximum_force > config.maximum_contact_force_n
        ):
            phase = PickPlacePhase.FAIL
            phase_started = time_seconds
            failure_reason = "hand-target contact force ceiling exceeded"
        elif (
            config.maximum_orientation_drift_rad is not None
            and orientation_drift > config.maximum_orientation_drift_rad
        ):
            phase = PickPlacePhase.FAIL
            phase_started = time_seconds
            failure_reason = "target orientation drift ceiling exceeded"
        elif phase is PickPlacePhase.OPEN_HOLD and elapsed >= config.open_hold_seconds:
            phase = PickPlacePhase.REACH_PREGRASP
            phase_started = time_seconds
        elif (
            phase is PickPlacePhase.REACH_PREGRASP
        ):
            if current_hand_contact:
                frozen = tuple(
                    float(environment.data.qpos[channel.qpos_address])
                    for channel in self._arm_channels
                )
                first_hand_contact = time_seconds
                phase = PickPlacePhase.FINAL_APPROACH
                phase_started = time_seconds
            elif elapsed >= config.reach_pregrasp_seconds:
                phase = PickPlacePhase.FINAL_APPROACH
                phase_started = time_seconds
        elif phase is PickPlacePhase.FINAL_APPROACH:
            if frozen is not None and first_hand_contact is not None:
                phase = PickPlacePhase.ENCLOSE
                phase_started = time_seconds
                enclosure_started = time_seconds
            elif current_hand_contact:
                frozen = tuple(
                    float(environment.data.qpos[channel.qpos_address])
                    for channel in self._arm_channels
                )
                first_hand_contact = time_seconds
                phase = PickPlacePhase.ENCLOSE
                phase_started = time_seconds
                enclosure_started = time_seconds
            elif elapsed >= config.final_approach_deadline_seconds:
                phase = PickPlacePhase.FAIL
                phase_started = time_seconds
                failure_reason = "no hand-target contact before approach deadline"
        elif phase is PickPlacePhase.ENCLOSE:
            contacted = contacts.contacted_digit_groups
            _, _, closure_rate = self._calibrated_digit_parameters()
            for index, group in enumerate(DIGIT_GROUPS):
                if group in contacted:
                    preload = config.digit_contact_preload_fraction
                    if preload is not None and group not in preloaded_groups:
                        closures[index] = min(1.0, closures[index] + preload)
                        preloaded_groups.add(group)
                else:
                    closures[index] = min(
                        1.0,
                        closures[index]
                        + closure_rate * delta_time,
                    )
            if contacts.secure_geometry:
                phase = PickPlacePhase.SECURE
                phase_started = time_seconds
                secure_started = time_seconds
            elif (
                enclosure_started is not None
                and time_seconds - enclosure_started
                >= config.enclosure_deadline_seconds
            ):
                phase = PickPlacePhase.FAIL
                phase_started = time_seconds
                failure_reason = "secure opposition not achieved before deadline"
        elif phase is PickPlacePhase.SECURE:
            if not contacts.secure_geometry:
                phase = PickPlacePhase.ENCLOSE
                phase_started = time_seconds
                secure_started = None
            elif (
                secure_started is not None
                and time_seconds - secure_started >= config.secure_hold_seconds
            ):
                phase = PickPlacePhase.LIFT
                phase_started = time_seconds
                lift_gate_started = None
        elif phase is PickPlacePhase.LIFT:
            lift_gate = (
                float(target_position[2]) - 0.020
                >= config.lift_minimum_target_bottom_m
                and not contacts.table_contact
            )
            if lift_gate:
                if lift_gate_started is None:
                    lift_gate_started = time_seconds
            else:
                lift_gate_started = None
            if (
                elapsed >= config.lift_seconds
                and lift_gate_started is not None
                and time_seconds - lift_gate_started >= config.lift_gate_hold_seconds
            ):
                phase = PickPlacePhase.CARRY
                phase_started = time_seconds
                carry_origin_x = float(target_position[0])
            elif elapsed >= config.lift_deadline_seconds:
                phase = PickPlacePhase.FAIL
                phase_started = time_seconds
                failure_reason = "target failed sustained 60 mm lift gate"
        elif phase is PickPlacePhase.CARRY:
            lifted = (
                float(target_position[2]) - 0.020
                >= config.lift_minimum_target_bottom_m
                and not contacts.table_contact
            )
            carried = (
                carry_origin_x is not None
                and float(target_position[0]) - carry_origin_x
                >= config.carry_minimum_x_excursion_m
            )
            if elapsed >= config.carry_seconds and lifted and carried:
                phase = PickPlacePhase.RETURN_LOWER
                phase_started = time_seconds
                table_contact_started = None
            elif elapsed >= config.carry_deadline_seconds:
                phase = PickPlacePhase.FAIL
                phase_started = time_seconds
                failure_reason = "target failed sustained carry gate"
        elif phase is PickPlacePhase.RETURN_LOWER:
            if contacts.table_contact:
                if table_contact_started is None:
                    table_contact_started = time_seconds
            else:
                table_contact_started = None
            if (
                table_contact_started is not None
                and time_seconds - table_contact_started
                >= config.table_contact_hold_seconds
            ):
                phase = PickPlacePhase.RELEASE
                phase_started = time_seconds
                release_clear_started = None
            elif elapsed >= config.return_table_deadline_seconds:
                phase = PickPlacePhase.FAIL
                phase_started = time_seconds
                failure_reason = "target did not return to table before deadline"
        elif phase is PickPlacePhase.RELEASE:
            if contacts.has_hand_contact:
                release_clear_started = None
            elif release_clear_started is None:
                release_clear_started = time_seconds
            if (
                elapsed >= config.release_seconds
                and release_clear_started is not None
                and time_seconds - release_clear_started
                >= config.release_clear_hold_seconds
            ):
                phase = PickPlacePhase.RETRACT
                phase_started = time_seconds
        elif phase is PickPlacePhase.RETRACT and elapsed >= config.retract_seconds:
            phase = PickPlacePhase.COMPLETE
            phase_started = time_seconds

        return SupportedPickPlaceState(
            phase=phase,
            phase_started_seconds=phase_started,
            last_time_seconds=time_seconds,
            initial_arm_positions_rad=state.initial_arm_positions_rad,
            initial_target_position_m=state.initial_target_position_m,
            initial_target_quaternion_wxyz=(
                state.initial_target_quaternion_wxyz
            ),
            hip_reference_rad=state.hip_reference_rad,
            head_reference_rad=state.head_reference_rad,
            digit_closure_fractions=tuple(closures),
            preloaded_digit_groups=tuple(
                group for group in DIGIT_GROUPS if group in preloaded_groups
            ),
            frozen_arm_positions_rad=frozen,
            first_hand_contact_seconds=first_hand_contact,
            enclosure_started_seconds=enclosure_started,
            secure_started_seconds=secure_started,
            lift_gate_started_seconds=lift_gate_started,
            carry_origin_x_m=carry_origin_x,
            table_contact_started_seconds=table_contact_started,
            release_clear_started_seconds=release_clear_started,
            failure_reason=failure_reason,
            motor_state=state.motor_state,
        )

    def _arm_references(
        self,
        state: SupportedPickPlaceState,
        time_seconds: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        phase = state.phase
        elapsed = time_seconds - state.phase_started_seconds
        frozen = state.frozen_arm_positions_rad or RIGHT_ARM_TARGET_RAD
        if phase is PickPlacePhase.OPEN_HOLD:
            return state.initial_arm_positions_rad, (0.0,) * 7
        if phase is PickPlacePhase.REACH_PREGRASP:
            return self._smooth_reference(
                state.initial_arm_positions_rad,
                PREGRASP_ARM_SEED_RAD,
                elapsed,
                self._config.reach_pregrasp_seconds,
            )
        if phase is PickPlacePhase.FINAL_APPROACH:
            if state.frozen_arm_positions_rad is not None:
                return state.frozen_arm_positions_rad, (0.0,) * 7
            return self._smooth_reference(
                PREGRASP_ARM_SEED_RAD,
                RIGHT_ARM_TARGET_RAD,
                elapsed,
                self._config.final_approach_seconds,
            )
        if phase in {PickPlacePhase.ENCLOSE, PickPlacePhase.SECURE}:
            return frozen, (0.0,) * 7
        if phase is PickPlacePhase.LIFT:
            return self._smooth_reference(
                frozen,
                LIFT_ARM_SEED_RAD,
                elapsed,
                self._config.lift_seconds,
            )
        if phase is PickPlacePhase.CARRY:
            return self._smooth_reference(
                LIFT_ARM_SEED_RAD,
                CARRY_ARM_SEED_RAD,
                elapsed,
                self._config.carry_seconds,
            )
        if phase is PickPlacePhase.RETURN_LOWER:
            split = (
                self._config.return_lower_seconds
                * self._config.return_carry_fraction
            )
            if elapsed <= split:
                return self._smooth_reference(
                    CARRY_ARM_SEED_RAD, LIFT_ARM_SEED_RAD, elapsed, split
                )
            return self._smooth_reference(
                LIFT_ARM_SEED_RAD,
                frozen,
                elapsed - split,
                self._config.return_lower_seconds - split,
            )
        if phase is PickPlacePhase.RELEASE:
            return frozen, (0.0,) * 7
        if phase in {PickPlacePhase.RETRACT, PickPlacePhase.COMPLETE}:
            if phase is PickPlacePhase.COMPLETE:
                return state.initial_arm_positions_rad, (0.0,) * 7
            return self._smooth_reference(
                frozen,
                state.initial_arm_positions_rad,
                elapsed,
                self._config.retract_seconds,
            )
        return tuple(
            float(self._current_qpos(channel)) for channel in self._arm_channels
        ), (0.0,) * 7

    def _digit_references(
        self,
        state: SupportedPickPlaceState,
        time_seconds: float,
    ) -> tuple[float, ...]:
        fractions = state.digit_closure_fractions
        if state.phase in {
            PickPlacePhase.OPEN_HOLD,
            PickPlacePhase.REACH_PREGRASP,
            PickPlacePhase.FINAL_APPROACH,
            PickPlacePhase.RETRACT,
            PickPlacePhase.COMPLETE,
            PickPlacePhase.FAIL,
        }:
            fractions = (0.0,) * len(DIGIT_GROUPS)
        elif state.phase is PickPlacePhase.RELEASE:
            progress = np.clip(
                (time_seconds - state.phase_started_seconds)
                / self._config.release_seconds,
                0.0,
                1.0,
            )
            fractions = tuple((1.0 - progress) * value for value in fractions)
        fraction_by_group = dict(zip(DIGIT_GROUPS, fractions, strict=True))
        return tuple(
            fraction_by_group[RIGHT_DIGIT_GROUP_BY_ACTUATOR[name]] * target
            for name, target in zip(
                RIGHT_DIGIT_ACTUATOR_NAMES,
                RIGHT_DIGIT_SEARCH_CAP_RAD,
                strict=True,
            )
        )

    def _current_qpos(self, channel: MotorChannel) -> float:
        # This method exists only for a fail-state hold and does not mutate data.
        return float(self._classifier._environment.data.qpos[channel.qpos_address])

    @staticmethod
    def _smooth_reference(
        start: tuple[float, ...],
        stop: tuple[float, ...],
        elapsed: float,
        duration: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        progress = float(np.clip(elapsed / duration, 0.0, 1.0))
        blend = 3.0 * progress**2 - 2.0 * progress**3
        blend_rate = (
            (6.0 * progress - 6.0 * progress**2) / duration
            if 0.0 < progress < 1.0
            else 0.0
        )
        delta = tuple(end - begin for begin, end in zip(start, stop, strict=True))
        return (
            tuple(begin + blend * offset for begin, offset in zip(start, delta, strict=True)),
            tuple(blend_rate * offset for offset in delta),
        )

    @staticmethod
    def _unclipped_joint_action(
        channel: MotorChannel,
        data: mujoco.MjData,
        *,
        reference_position_rad: float,
        reference_velocity_rad_s: float,
        position_action_per_rad: float,
        velocity_action_seconds_per_rad: float,
    ) -> float:
        return (
            float(data.qfrc_bias[channel.dof_address]) / channel.gear
            + position_action_per_rad
            * (reference_position_rad - float(data.qpos[channel.qpos_address]))
            + velocity_action_seconds_per_rad
            * (reference_velocity_rad_s - float(data.qvel[channel.dof_address]))
        )

    def _unclipped_hip_action(
        self, data: mujoco.MjData, reference_rad: float
    ) -> float:
        mean_position = 0.5 * (
            float(data.qpos[self._hip_channel.qpos_address])
            + float(data.qpos[self._hip_second_qpos_address])
        )
        mean_velocity = 0.5 * (
            float(data.qvel[self._hip_channel.dof_address])
            + float(data.qvel[self._hip_second_dof_address])
        )
        torque = (
            float(data.qfrc_bias[self._hip_channel.dof_address])
            + float(data.qfrc_bias[self._hip_second_dof_address])
            + HIP_STIFFNESS_NM_RAD * (reference_rad - mean_position)
            - HIP_DAMPING_NM_S_RAD * mean_velocity
        )
        return torque / self._hip_channel.gear

    def _calibrated_digit_parameters(self) -> tuple[float, float, float]:
        values = (
            self._config.digit_position_action_per_rad,
            self._config.digit_velocity_action_seconds_per_rad,
            self._config.digit_closure_fraction_per_second,
        )
        if any(value is None for value in values):
            raise RuntimeError(
                "digit PD gains and closure rate require physical calibration"
            )
        return tuple(float(value) for value in values)  # type: ignore[return-value]

    @staticmethod
    def _quaternion_distance_rad(
        reference_wxyz: NDArray[np.float64],
        current_wxyz: NDArray[np.float64],
    ) -> float:
        reference = reference_wxyz / np.linalg.norm(reference_wxyz)
        current = current_wxyz / np.linalg.norm(current_wxyz)
        cosine = float(np.clip(abs(np.dot(reference, current)), 0.0, 1.0))
        return float(2.0 * np.arccos(cosine))

    @staticmethod
    def _replace_state(
        state: SupportedPickPlaceState,
        *,
        motor_state: MotorControllerState,
    ) -> SupportedPickPlaceState:
        return SupportedPickPlaceState(
            phase=state.phase,
            phase_started_seconds=state.phase_started_seconds,
            last_time_seconds=state.last_time_seconds,
            initial_arm_positions_rad=state.initial_arm_positions_rad,
            initial_target_position_m=state.initial_target_position_m,
            initial_target_quaternion_wxyz=(
                state.initial_target_quaternion_wxyz
            ),
            hip_reference_rad=state.hip_reference_rad,
            head_reference_rad=state.head_reference_rad,
            digit_closure_fractions=state.digit_closure_fractions,
            preloaded_digit_groups=state.preloaded_digit_groups,
            frozen_arm_positions_rad=state.frozen_arm_positions_rad,
            first_hand_contact_seconds=state.first_hand_contact_seconds,
            enclosure_started_seconds=state.enclosure_started_seconds,
            secure_started_seconds=state.secure_started_seconds,
            lift_gate_started_seconds=state.lift_gate_started_seconds,
            carry_origin_x_m=state.carry_origin_x_m,
            table_contact_started_seconds=state.table_contact_started_seconds,
            release_clear_started_seconds=state.release_clear_started_seconds,
            failure_reason=state.failure_reason,
            motor_state=motor_state,
        )

    def _require_environment(self, environment: SupportedPickPlaceEnv) -> None:
        if environment.model is not self._model:
            raise ValueError("controller belongs to a different MuJoCo model")

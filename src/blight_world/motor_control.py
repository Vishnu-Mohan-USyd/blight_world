"""Typed, name-based motor-control primitives for MIMo environments.

This module deliberately provides mechanism rather than behavior.  It maps the
model's named actuators onto Gymnasium action indices, composes sparse commands
without hidden nonzero outputs, and supplies small bias-compensated PD helpers.
It does not select goals, trajectories, rewards, or behavior policies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, TypeAlias

from gymnasium.spaces import Box
import mujoco
import numpy as np
from numpy.typing import NDArray


class MotorEnvironment(Protocol):
    """Structural interface needed to resolve MIMo motor channels."""

    model: mujoco.MjModel
    data: mujoco.MjData
    mimo_actuators: NDArray[np.integer]
    action_space: Box


@dataclass(frozen=True, slots=True)
class MotorChannel:
    """Immutable mapping from one named actuator to one action coordinate.

    Attributes
    ----------
    actuator_name:
        MuJoCo actuator name, for example ``"act:head_tilt"``.
    joint_name:
        Name of the scalar joint targeted by the actuator.
    actuator_id, joint_id:
        IDs in the loaded MuJoCo model.
    action_index:
        Unique coordinate in the environment's Gymnasium action vector.
    qpos_address, dof_address:
        Scalar joint addresses in ``data.qpos`` and ``data.qvel``/
        ``data.qfrc_bias`` respectively.
    gear:
        Scalar actuator gear converting control input to joint torque.
    lower_bound, upper_bound:
        Inclusive action-space limits for this coordinate.
    """

    actuator_name: str
    joint_name: str
    actuator_id: int
    joint_id: int
    action_index: int
    qpos_address: int
    dof_address: int
    gear: float
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        for name, value in (
            ("actuator_name", self.actuator_name),
            ("joint_name", self.joint_name),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name, value in (
            ("actuator_id", self.actuator_id),
            ("joint_id", self.joint_id),
            ("action_index", self.action_index),
            ("qpos_address", self.qpos_address),
            ("dof_address", self.dof_address),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name, value in (
            ("gear", self.gear),
            ("lower_bound", self.lower_bound),
            ("upper_bound", self.upper_bound),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.gear == 0.0:
            raise ValueError("gear must be nonzero")
        if self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound cannot exceed upper_bound")


@dataclass(frozen=True, slots=True)
class MotorControllerState:
    """Immutable state passed between deterministic composer calls.

    ``previous_action`` is a tuple rather than an ndarray so callers cannot
    mutate a saved controller state through an array alias.
    """

    step_index: int
    previous_action: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError("step_index must be a non-negative integer")
        if not all(isfinite(value) for value in self.previous_action):
            raise ValueError("previous_action must contain only finite values")


ChannelReference: TypeAlias = str | MotorChannel
OwnedCommands: TypeAlias = Mapping[str, Mapping[ChannelReference, float]]


def resolve_motor_channels(environment: MotorEnvironment) -> tuple[MotorChannel, ...]:
    """Resolve every MIMo actuator into action-vector order.

    Resolution follows ``environment.mimo_actuators`` rather than assuming
    actuator IDs and action indices are interchangeable.  Duplicate actuator
    IDs, names, or action indices fail closed because they would make sparse
    ownership ambiguous.
    """

    actuator_ids = np.asarray(environment.mimo_actuators, dtype=np.int64)
    if actuator_ids.ndim != 1 or actuator_ids.size == 0:
        raise ValueError("mimo_actuators must be a non-empty one-dimensional array")
    if np.unique(actuator_ids).size != actuator_ids.size:
        raise ValueError("mimo_actuators contains duplicate actuator IDs")
    if not isinstance(environment.action_space, Box):
        raise TypeError("motor resolution requires a Gymnasium Box action space")
    if environment.action_space.shape != (actuator_ids.size,):
        raise ValueError("action-space shape does not match MIMo actuator count")

    channels: list[MotorChannel] = []
    names: set[str] = set()
    for action_index, raw_actuator_id in enumerate(actuator_ids):
        actuator_id = int(raw_actuator_id)
        if not 0 <= actuator_id < environment.model.nu:
            raise ValueError(f"invalid actuator ID: {actuator_id}")
        actuator_name = mujoco.mj_id2name(
            environment.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        if actuator_name is None:
            raise ValueError(f"actuator {actuator_id} has no name")
        if actuator_name in names:
            raise ValueError(f"duplicate actuator name: {actuator_name}")
        names.add(actuator_name)

        joint_id = int(environment.model.actuator_trnid[actuator_id, 0])
        if not 0 <= joint_id < environment.model.njnt:
            raise ValueError(
                f"actuator {actuator_name} does not target a model joint"
            )
        joint_name = mujoco.mj_id2name(
            environment.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )
        if joint_name is None:
            raise ValueError(f"joint {joint_id} has no name")
        channels.append(
            MotorChannel(
                actuator_name=actuator_name,
                joint_name=joint_name,
                actuator_id=actuator_id,
                joint_id=joint_id,
                action_index=action_index,
                qpos_address=int(environment.model.jnt_qposadr[joint_id]),
                dof_address=int(environment.model.jnt_dofadr[joint_id]),
                gear=float(environment.model.actuator_gear[actuator_id, 0]),
                lower_bound=float(environment.action_space.low[action_index]),
                upper_bound=float(environment.action_space.high[action_index]),
            )
        )
    return tuple(channels)


def resolve_motor_channel(
    environment: MotorEnvironment,
    actuator_name: str,
) -> MotorChannel:
    """Resolve one actuator by exact name or raise ``KeyError``."""

    matches = [
        channel
        for channel in resolve_motor_channels(environment)
        if channel.actuator_name == actuator_name
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown MIMo actuator: {actuator_name}")
    return matches[0]


class SparseActionComposer:
    """Compose deterministic zero-default actions from disjoint owners.

    Parameters
    ----------
    channels:
        Complete channel map in action-vector order.
    slew_limits:
        Optional maximum absolute action change per call, keyed by actuator
        name.  Slew limits also govern return-to-zero when an owner omits a
        channel on a later call.

    Notes
    -----
    ``compose`` is pure with respect to the composer: all history is supplied
    through and returned as an immutable :class:`MotorControllerState`.
    """

    __slots__ = ("_channels", "_channels_by_name", "_slew_by_index")

    def __init__(
        self,
        channels: Sequence[MotorChannel],
        *,
        slew_limits: Mapping[str, float] | None = None,
    ) -> None:
        ordered = tuple(sorted(channels, key=lambda channel: channel.action_index))
        if not ordered:
            raise ValueError("channels cannot be empty")
        if tuple(channel.action_index for channel in ordered) != tuple(
            range(len(ordered))
        ):
            raise ValueError("channels must cover contiguous unique action indices")
        names = [channel.actuator_name for channel in ordered]
        if len(set(names)) != len(names):
            raise ValueError("channels contain duplicate actuator names")
        self._channels = ordered
        self._channels_by_name = dict(zip(names, ordered, strict=True))
        self._slew_by_index: dict[int, float] = {}
        for actuator_name, raw_limit in (slew_limits or {}).items():
            try:
                channel = self._channels_by_name[actuator_name]
            except KeyError as error:
                raise KeyError(
                    f"slew limit names unknown actuator: {actuator_name}"
                ) from error
            limit = float(raw_limit)
            if not isfinite(limit) or limit < 0.0:
                raise ValueError("slew limits must be finite and non-negative")
            self._slew_by_index[channel.action_index] = limit

    @property
    def channels(self) -> tuple[MotorChannel, ...]:
        """Return the immutable channel order used by this composer."""

        return self._channels

    def reset(self) -> MotorControllerState:
        """Return a deterministic all-zero immutable controller state."""

        return MotorControllerState(
            step_index=0,
            previous_action=(0.0,) * len(self._channels),
        )

    def compose(
        self,
        commands_by_owner: OwnedCommands,
        state: MotorControllerState | None = None,
    ) -> tuple[NDArray[np.float32], MotorControllerState]:
        """Compose one bounded action and its next immutable state.

        Every unspecified channel targets zero.  If two owners name the same
        channel (including once by name and once by ``MotorChannel``), a
        ``ValueError`` is raised instead of silently resolving precedence.
        """

        previous_state = self.reset() if state is None else state
        if len(previous_state.previous_action) != len(self._channels):
            raise ValueError("controller state length does not match channels")
        for channel, value in zip(
            self._channels,
            previous_state.previous_action,
            strict=True,
        ):
            if value < channel.lower_bound or value > channel.upper_bound:
                raise ValueError(
                    f"controller state for {channel.actuator_name} has value "
                    f"{value!r} outside bounds "
                    f"[{channel.lower_bound!r}, {channel.upper_bound!r}]"
                )
        previous = np.asarray(previous_state.previous_action, dtype=np.float64)
        target = np.zeros(len(self._channels), dtype=np.float64)
        claimed_by: dict[int, str] = {}
        for owner, assignments in commands_by_owner.items():
            if not isinstance(owner, str) or not owner:
                raise ValueError("command owner must be a non-empty string")
            for reference, raw_value in assignments.items():
                channel = self._resolve_reference(reference)
                prior_owner = claimed_by.get(channel.action_index)
                if prior_owner is not None:
                    raise ValueError(
                        f"motor {channel.actuator_name} claimed by both "
                        f"{prior_owner!r} and {owner!r}"
                    )
                value = float(raw_value)
                if not isfinite(value):
                    raise ValueError(
                        f"command for {channel.actuator_name} must be finite"
                    )
                claimed_by[channel.action_index] = owner
                target[channel.action_index] = np.clip(
                    value,
                    channel.lower_bound,
                    channel.upper_bound,
                )

        for action_index, limit in self._slew_by_index.items():
            target[action_index] = previous[action_index] + np.clip(
                target[action_index] - previous[action_index],
                -limit,
                limit,
            )
        action = np.ascontiguousarray(target, dtype=np.float32)
        next_state = MotorControllerState(
            step_index=previous_state.step_index + 1,
            previous_action=tuple(float(value) for value in action),
        )
        return action, next_state

    def _resolve_reference(self, reference: ChannelReference) -> MotorChannel:
        if isinstance(reference, str):
            try:
                return self._channels_by_name[reference]
            except KeyError as error:
                raise KeyError(f"unknown actuator: {reference}") from error
        if not isinstance(reference, MotorChannel):
            raise TypeError("motor references must be actuator names or MotorChannel")
        try:
            canonical = self._channels_by_name[reference.actuator_name]
        except KeyError as error:
            raise KeyError(f"unknown actuator: {reference.actuator_name}") from error
        if canonical != reference:
            raise ValueError(
                f"channel metadata disagrees for {reference.actuator_name}"
            )
        return canonical


def single_joint_bias_compensated_pd(
    channel: MotorChannel,
    data: mujoco.MjData,
    *,
    reference_position_rad: float,
    stiffness_nm_per_rad: float,
    damping_nm_s_per_rad: float,
) -> float:
    """Return one clipped single-joint PD action with bias compensation.

    The commanded torque is
    ``bias + stiffness * (reference - q) - damping * qvel`` and is divided by
    the channel gear before clipping to the action bounds.
    """

    reference, stiffness, damping = _validated_pd_parameters(
        reference_position_rad,
        stiffness_nm_per_rad,
        damping_nm_s_per_rad,
    )
    torque_nm = (
        float(data.qfrc_bias[channel.dof_address])
        + stiffness * (reference - float(data.qpos[channel.qpos_address]))
        - damping * float(data.qvel[channel.dof_address])
    )
    return float(
        np.clip(
            torque_nm / channel.gear,
            channel.lower_bound,
            channel.upper_bound,
        )
    )


def coupled_hip_bias_compensated_pd(
    channel: MotorChannel,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    reference_mean_position_rad: float,
    stiffness_nm_per_rad: float,
    damping_nm_s_per_rad: float,
) -> float:
    """Return a clipped PD action for MIMo's coupled sagittal hip pair.

    The helper is explicit to MIMo v2's ``act:hip_bend`` actuator and combines
    the bias torque of ``robot:hip_bend1`` and ``robot:hip_bend2``.  Position
    and velocity feedback use the arithmetic mean of the two joint states.
    """

    if (
        channel.actuator_name != "act:hip_bend"
        or channel.joint_name != "robot:hip_bend1"
    ):
        raise ValueError("coupled hip helper requires the act:hip_bend channel")
    reference, stiffness, damping = _validated_pd_parameters(
        reference_mean_position_rad,
        stiffness_nm_per_rad,
        damping_nm_s_per_rad,
    )
    second_joint = model.joint("robot:hip_bend2")
    second_qpos = int(np.asarray(second_joint.qposadr).item())
    second_dof = int(np.asarray(second_joint.dofadr).item())
    mean_position = 0.5 * (
        float(data.qpos[channel.qpos_address]) + float(data.qpos[second_qpos])
    )
    mean_velocity = 0.5 * (
        float(data.qvel[channel.dof_address]) + float(data.qvel[second_dof])
    )
    bias_torque = float(data.qfrc_bias[channel.dof_address]) + float(
        data.qfrc_bias[second_dof]
    )
    torque_nm = (
        bias_torque
        + stiffness * (reference - mean_position)
        - damping * mean_velocity
    )
    return float(
        np.clip(
            torque_nm / channel.gear,
            channel.lower_bound,
            channel.upper_bound,
        )
    )


def _validated_pd_parameters(
    reference_position_rad: float,
    stiffness_nm_per_rad: float,
    damping_nm_s_per_rad: float,
) -> tuple[float, float, float]:
    values = tuple(
        float(value)
        for value in (
            reference_position_rad,
            stiffness_nm_per_rad,
            damping_nm_s_per_rad,
        )
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("PD reference and gains must be finite")
    if values[1] < 0.0 or values[2] < 0.0:
        raise ValueError("PD gains must be non-negative")
    return values

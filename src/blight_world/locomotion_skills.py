"""Unsupported open-loop support and belly-crawl primitives for MIMo.

The static milestone is a staged torque posture that keeps MIMo supported by
the floor without a root weld.  The crawl milestone layers a scripted
actuator-only belly/army crawl on that posture.  Neither is learned, adaptive,
an all-fours gait, or a perturbation-recovery controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

import mujoco
import numpy as np
from numpy.typing import NDArray

from blight_world.environment import BlightWorldEnv
from blight_world.motor_control import (
    MotorChannel,
    MotorControllerState,
    SparseActionComposer,
    resolve_motor_channel,
    resolve_motor_channels,
)


UNSUPPORTED_ROOT_QPOS = (
    0.0,
    0.0,
    0.35,
    0.7253743710122876,
    0.0,
    0.6883545756937539,
    0.0,
)
ACTION_RAMP_SECONDS = 0.01

BELLY_TRANSITION_START_SECONDS = 4.0
BELLY_TRANSITION_STOP_SECONDS = 6.0
BELLY_SETTLE_STOP_SECONDS = 18.0
BELLY_CRAWL_STOP_SECONDS = 37.8
BELLY_TRIAL_STOP_SECONDS = 41.8
BELLY_LIMB_PHASE_SECONDS = 1.35
BELLY_RELOAD_SECONDS = 0.30
BELLY_BUMP_RISE_SECONDS = 0.3375
BELLY_BUMP_PLATEAU_STOP_SECONDS = 1.0125
BELLY_CYCLE_SECONDS = 2.0 * (BELLY_LIMB_PHASE_SECONDS + BELLY_RELOAD_SECONDS)
BELLY_FIRST_SIDE = ("right", "left", "right", "left", "left", "left")
BELLY_STANCE_FRONT = (0.29, 0.34, 0.34, 0.34, 0.34, 0.34)

BILATERAL_STATIC_ACTIONS = {
    "shoulder_horizontal": 0.5495666434717998,
    "shoulder_abduction": 0.4060872441575624,
    "shoulder_internal": 0.30308058542063065,
    "elbow": -0.3933576072208938,
    "wrist_flexion": -0.05195329701711351,
    "hip_flex": -0.35678670760597825,
    "knee": -0.11676130364288884,
    "foot_flexion": 0.07328405467757337,
}
CENTRAL_STATIC_ACTIONS = {
    "hip_bend": -0.11562250893837525,
    "head_tilt": -0.5,
}
BELLY_ACTION_DELTAS = {
    "shoulder_horizontal": -0.9,
    "elbow": 0.8,
    "knee": -0.4,
    "hip_flex": 0.2,
}
BELLY_MOVING_SIDE_DELTAS = {
    "shoulder_horizontal": -1.2,
    "elbow": 0.985,
    "hip_flex": 0.146,
}
BELLY_OPPOSITE_SIDE_DELTAS = {
    "knee": -0.22,
    "hip_flex": 0.042,
    "foot_flexion": -0.25,
}


class UnsupportedStaticSupportEnv(BlightWorldEnv):
    """BlightWorld staged once into an unsupported four-region floor posture.

    Every reset starts from the normal MIMo joint configuration, changes only
    the ``mimo_orientation`` freejoint to :data:`UNSUPPORTED_ROOT_QPOS`, and
    zeros velocities.  Runtime control never writes state, forces, gravity,
    model parameters, equality activation, or mocap data.
    """

    def reset_model(self) -> dict[str, NDArray[np.generic]]:
        qpos = np.asarray(self.init_qpos, dtype=np.float64).copy()
        qvel = np.zeros_like(self.init_qvel, dtype=np.float64)
        root_joint = self.model.joint("mimo_orientation")
        root_qpos_address = int(np.asarray(root_joint.qposadr).item())
        qpos[root_qpos_address : root_qpos_address + 7] = UNSUPPORTED_ROOT_QPOS
        self.set_state(qpos, qvel)
        if self.audition is not None:
            self.audition.reset(self.np_random_seed)
        return self._get_obs()


@dataclass(frozen=True, slots=True)
class StaticSupportControllerState:
    """Immutable sparse-composer state for open-loop static support."""

    motor_state: MotorControllerState


@dataclass(frozen=True, slots=True)
class StaticSupportControlDiagnostics:
    """Immutable command evidence for one public environment step."""

    time_seconds: float
    ramp_fraction: float
    commanded_actuator_names: tuple[str, ...]
    minimum_action_bound_margin: float


class UnsupportedStaticSupportController:
    """Name-resolved open-loop SpringDamper torque posture.

    Commands ramp with a cubic smoothstep over the first 10 ms and then remain
    constant.  These values are normalized torque actions, not joint-angle
    references or PD targets.
    """

    __slots__ = ("_channels", "_composer", "_model", "_targets")

    def __init__(self, environment: UnsupportedStaticSupportEnv) -> None:
        self._model = environment.model
        target_by_name: dict[str, float] = {}
        for side in ("right", "left"):
            for suffix, value in BILATERAL_STATIC_ACTIONS.items():
                target_by_name[f"act:{side}_{suffix}"] = value
        for suffix, value in CENTRAL_STATIC_ACTIONS.items():
            target_by_name[f"act:{suffix}"] = value
        self._channels = tuple(
            resolve_motor_channel(environment, name) for name in target_by_name
        )
        self._targets = tuple(
            (channel, float(target_by_name[channel.actuator_name]))
            for channel in self._channels
        )
        self._composer = SparseActionComposer(resolve_motor_channels(environment))

    @property
    def controlled_actuator_names(self) -> tuple[str, ...]:
        return tuple(channel.actuator_name for channel in self._channels)

    def reset(self) -> StaticSupportControllerState:
        return StaticSupportControllerState(motor_state=self._composer.reset())

    def compose(
        self,
        environment: UnsupportedStaticSupportEnv,
        state: StaticSupportControllerState,
        *,
        time_seconds: float | None = None,
    ) -> tuple[
        NDArray[np.float32],
        StaticSupportControllerState,
        StaticSupportControlDiagnostics,
    ]:
        """Return one sparse bounded action without modifying MuJoCo state."""

        if environment.model is not self._model:
            raise ValueError("controller belongs to a different MuJoCo model")
        time_value = (
            float(environment.data.time)
            if time_seconds is None
            else float(time_seconds)
        )
        if not isfinite(time_value) or time_value < 0.0:
            raise ValueError("time_seconds must be finite and non-negative")
        progress = float(np.clip(time_value / ACTION_RAMP_SECONDS, 0.0, 1.0))
        ramp = 3.0 * progress**2 - 2.0 * progress**3
        commands = {
            channel: ramp * target for channel, target in self._targets
        }
        action, motor_state = self._composer.compose(
            {"unsupported_static_support": commands}, state.motor_state
        )
        margins = (
            min(
                float(action[channel.action_index]) - channel.lower_bound,
                channel.upper_bound - float(action[channel.action_index]),
            )
            for channel in self._composer.channels
        )
        diagnostics = StaticSupportControlDiagnostics(
            time_seconds=time_value,
            ramp_fraction=ramp,
            commanded_actuator_names=self.controlled_actuator_names,
            minimum_action_bound_margin=min(margins),
        )
        return (
            action,
            StaticSupportControllerState(motor_state=motor_state),
            diagnostics,
        )


class BellyCrawlPhase(str, Enum):
    """Named segment of the scripted unsupported belly-crawl waveform."""

    STATIC_SUPPORT = "static_support"
    TRANSITION_TO_BELLY = "transition_to_belly"
    BELLY_SETTLE = "belly_settle"
    LIMB_PHASE = "limb_phase"
    RELOAD = "reload"
    BELLY_TAIL = "belly_tail"


@dataclass(frozen=True, slots=True)
class BellyCrawlControllerState:
    """Immutable sparse-composer state for the scripted crawl primitive."""

    motor_state: MotorControllerState


@dataclass(frozen=True, slots=True)
class BellyCrawlControlDiagnostics:
    """Immutable command evidence for one public environment step."""

    time_seconds: float
    phase: BellyCrawlPhase
    cycle_index: int | None
    moving_side: str | None
    bump_fraction: float
    minimum_action_bound_margin: float


class UnsupportedBellyCrawlController:
    """Scripted actuator-only unsupported belly/army crawl.

    The controller is deterministic open-loop playback of normalized
    SpringDamper actions.  It is not a learned policy, feedback locomotion
    controller, or an all-fours gait.  The optional rest-control mode follows
    the same static-to-belly transition and duration but omits crawl phases.
    """

    __slots__ = (
        "_channels",
        "_composer",
        "_cycles_enabled",
        "_model",
        "_static_targets",
    )

    def __init__(
        self,
        environment: UnsupportedStaticSupportEnv,
        *,
        cycles_enabled: bool = True,
    ) -> None:
        self._model = environment.model
        self._cycles_enabled = bool(cycles_enabled)
        static_targets: dict[str, float] = {}
        for side in ("right", "left"):
            for suffix, value in BILATERAL_STATIC_ACTIONS.items():
                static_targets[f"act:{side}_{suffix}"] = float(np.float32(value))
        for suffix, value in CENTRAL_STATIC_ACTIONS.items():
            static_targets[f"act:{suffix}"] = float(np.float32(value))
        self._static_targets = static_targets
        self._channels = {
            name: resolve_motor_channel(environment, name)
            for name in static_targets
        }
        self._composer = SparseActionComposer(resolve_motor_channels(environment))

    @property
    def controlled_actuator_names(self) -> tuple[str, ...]:
        """Return the exact name-resolved actuators owned by this primitive."""

        return tuple(self._channels)

    def reset(self) -> BellyCrawlControllerState:
        """Return a deterministic zero-action composer state."""

        return BellyCrawlControllerState(motor_state=self._composer.reset())

    def compose(
        self,
        environment: UnsupportedStaticSupportEnv,
        state: BellyCrawlControllerState,
        *,
        time_seconds: float | None = None,
    ) -> tuple[
        NDArray[np.float32],
        BellyCrawlControllerState,
        BellyCrawlControlDiagnostics,
    ]:
        """Compose one bounded action without writing simulation state."""

        if environment.model is not self._model:
            raise ValueError("controller belongs to a different MuJoCo model")
        time_value = (
            float(state.motor_state.step_index * environment.dt)
            if time_seconds is None
            else float(time_seconds)
        )
        if not isfinite(time_value) or time_value < 0.0:
            raise ValueError("time_seconds must be finite and non-negative")
        targets, phase, cycle_index, moving_side, bump = self._targets_at_time(
            time_value
        )
        commands = {
            self._channels[name]: value for name, value in targets.items()
        }
        action, motor_state = self._composer.compose(
            {"unsupported_belly_crawl": commands}, state.motor_state
        )
        margins = (
            min(
                float(action[channel.action_index]) - channel.lower_bound,
                channel.upper_bound - float(action[channel.action_index]),
            )
            for channel in self._composer.channels
        )
        return (
            action,
            BellyCrawlControllerState(motor_state=motor_state),
            BellyCrawlControlDiagnostics(
                time_seconds=time_value,
                phase=phase,
                cycle_index=cycle_index,
                moving_side=moving_side,
                bump_fraction=bump,
                minimum_action_bound_margin=min(margins),
            ),
        )

    def _targets_at_time(
        self, time_seconds: float
    ) -> tuple[dict[str, float], BellyCrawlPhase, int | None, str | None, float]:
        static = self._static_targets
        if time_seconds < BELLY_TRANSITION_START_SECONDS:
            if time_seconds == 0.0:
                return (
                    {name: 0.0 for name in static},
                    BellyCrawlPhase.STATIC_SUPPORT,
                    None,
                    None,
                    0.0,
                )
            progress = float(
                np.clip(time_seconds / ACTION_RAMP_SECONDS, 0.0, 1.0)
            )
            ramp = self._smoothstep(progress)
            return (
                {
                    name: self._lerp_float32(0.0, value, ramp)
                    for name, value in static.items()
                },
                BellyCrawlPhase.STATIC_SUPPORT,
                None,
                None,
                0.0,
            )

        belly = self._belly_targets()
        if time_seconds < BELLY_TRANSITION_STOP_SECONDS:
            progress = (
                time_seconds - BELLY_TRANSITION_START_SECONDS
            ) / (
                BELLY_TRANSITION_STOP_SECONDS - BELLY_TRANSITION_START_SECONDS
            )
            blend = self._smoothstep(progress)
            return (
                {
                    name: self._lerp_float32(static[name], belly[name], blend)
                    for name in static
                },
                BellyCrawlPhase.TRANSITION_TO_BELLY,
                None,
                None,
                0.0,
            )

        if (
            time_seconds < BELLY_SETTLE_STOP_SECONDS
            or not self._cycles_enabled
        ):
            phase = (
                BellyCrawlPhase.BELLY_SETTLE
                if time_seconds < BELLY_SETTLE_STOP_SECONDS
                else BellyCrawlPhase.BELLY_TAIL
            )
            return belly, phase, None, None, 0.0
        if time_seconds >= BELLY_CRAWL_STOP_SECONDS:
            return belly, BellyCrawlPhase.BELLY_TAIL, None, None, 0.0

        cycle_position = time_seconds - BELLY_SETTLE_STOP_SECONDS
        cycle_index = min(
            int(cycle_position / BELLY_CYCLE_SECONDS),
            len(BELLY_FIRST_SIDE) - 1,
        )
        local = cycle_position - cycle_index * BELLY_CYCLE_SECONDS
        first_side = BELLY_FIRST_SIDE[cycle_index]
        if local < BELLY_LIMB_PHASE_SECONDS:
            moving_side = first_side
            phase_time = local
        elif local < BELLY_LIMB_PHASE_SECONDS + BELLY_RELOAD_SECONDS:
            return belly, BellyCrawlPhase.RELOAD, cycle_index, None, 0.0
        elif local < 2.0 * BELLY_LIMB_PHASE_SECONDS + BELLY_RELOAD_SECONDS:
            moving_side = "left" if first_side == "right" else "right"
            phase_time = local - BELLY_LIMB_PHASE_SECONDS - BELLY_RELOAD_SECONDS
        else:
            return belly, BellyCrawlPhase.RELOAD, cycle_index, None, 0.0

        bump = self._limb_phase_bump(phase_time)
        peak = self._phase_peak_targets(
            moving_side=moving_side,
            stance_front=BELLY_STANCE_FRONT[cycle_index],
        )
        return (
            {
                name: self._lerp_float32(belly[name], peak[name], bump)
                for name in belly
            },
            BellyCrawlPhase.LIMB_PHASE,
            cycle_index,
            moving_side,
            bump,
        )

    def _belly_targets(self) -> dict[str, float]:
        targets = dict(self._static_targets)
        for side in ("right", "left"):
            for suffix, delta in BELLY_ACTION_DELTAS.items():
                name = f"act:{side}_{suffix}"
                targets[name] = self._endpoint_with_delta(targets[name], delta)
        return targets

    def _phase_peak_targets(
        self, *, moving_side: str, stance_front: float
    ) -> dict[str, float]:
        targets = dict(self._static_targets)
        opposite_side = "left" if moving_side == "right" else "right"
        for suffix, delta in BELLY_MOVING_SIDE_DELTAS.items():
            name = f"act:{moving_side}_{suffix}"
            targets[name] = self._endpoint_with_delta(targets[name], delta)
        for suffix, delta in BELLY_OPPOSITE_SIDE_DELTAS.items():
            name = f"act:{opposite_side}_{suffix}"
            targets[name] = self._endpoint_with_delta(targets[name], delta)
        shoulder_name = f"act:{opposite_side}_shoulder_horizontal"
        targets[shoulder_name] = self._endpoint_with_delta(
            targets[shoulder_name], 0.4 * stance_front
        )
        elbow_name = f"act:{opposite_side}_elbow"
        targets[elbow_name] = self._endpoint_with_delta(
            targets[elbow_name], -0.5 * stance_front
        )
        return targets

    @staticmethod
    def _endpoint_with_delta(base: float, delta: float) -> float:
        """Build one calibrated endpoint with exactly one float32 cast."""

        return float(np.float32(float(np.float32(base)) + delta))

    @staticmethod
    def _lerp_float32(start: float, stop: float, fraction: float) -> float:
        """Interpolate exactly as the calibrated float32 command generator."""

        start32 = np.float32(start)
        stop32 = np.float32(stop)
        fraction32 = np.float32(fraction)
        return float(
            np.float32(
                start32 + fraction32 * np.float32(stop32 - start32)
            )
        )

    @staticmethod
    def _smoothstep(value: float) -> float:
        bounded = float(np.clip(value, 0.0, 1.0))
        polynomial = 3.0 * bounded**2 - 2.0 * bounded**3
        return float(np.clip(polynomial, 0.0, 1.0))

    @classmethod
    def _limb_phase_bump(cls, phase_time_seconds: float) -> float:
        """Return the C1 rise/plateau/fall envelope for a 1.35 s phase."""

        if phase_time_seconds <= 0.0 or phase_time_seconds >= BELLY_LIMB_PHASE_SECONDS:
            return 0.0
        if phase_time_seconds < BELLY_BUMP_RISE_SECONDS:
            return cls._smoothstep(
                phase_time_seconds / BELLY_BUMP_RISE_SECONDS
            )
        if phase_time_seconds <= BELLY_BUMP_PLATEAU_STOP_SECONDS:
            return 1.0
        fall_progress = (
            phase_time_seconds - BELLY_BUMP_PLATEAU_STOP_SECONDS
        ) / BELLY_BUMP_RISE_SECONDS
        return 1.0 - cls._smoothstep(fall_progress)


class SupportRegion(str, Enum):
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    REAR_LEFT = "rear_left"
    REAR_RIGHT = "rear_right"


@dataclass(frozen=True, slots=True)
class StaticSupportContact:
    """One solved MIMo-floor contact assigned to a support region."""

    region: SupportRegion
    geom_name: str
    position_m: tuple[float, float, float]
    normal_force_n: float
    distal_knee_distance_m: float | None = None


@dataclass(frozen=True, slots=True)
class StaticSupportContacts:
    """Strict 5 ms support/contact accounting snapshot."""

    time_seconds: float
    contacts: tuple[StaticSupportContact, ...]
    central_force_n: float
    head_force_n: float
    unexpected_external_contacts: tuple[tuple[str, str], ...]

    @property
    def force_by_region_n(self) -> tuple[tuple[SupportRegion, float], ...]:
        return tuple(
            (
                region,
                float(
                    sum(
                        contact.normal_force_n
                        for contact in self.contacts
                        if contact.region is region
                    )
                ),
            )
            for region in SupportRegion
        )

    @property
    def active_regions(self) -> frozenset[SupportRegion]:
        return frozenset(
            region for region, force in self.force_by_region_n if force > 0.0
        )


class StaticSupportContactClassifier:
    """Classify hand, toe, and distal knee-cap floor contacts at 5 ms."""

    __slots__ = (
        "_environment",
        "_floor_geom_id",
        "_head_body_ids",
        "_mimo_root_id",
        "_region_by_geom_id",
        "_upper_leg_knee",
    )

    def __init__(self, environment: UnsupportedStaticSupportEnv) -> None:
        self._environment = environment
        model = environment.model
        self._floor_geom_id = int(model.geom("floor").id)
        self._mimo_root_id = int(model.body("mimo_location").id)
        self._head_body_ids = self._descendant_body_ids(
            model, int(model.body("head").id)
        )
        self._region_by_geom_id: dict[int, SupportRegion] = {}
        digit_suffixes = (
            "ffknuckle1",
            "ffmiddle1",
            "ffdistal1",
            "mfknuckle1",
            "mfmiddle1",
            "mfdistal1",
            "rfknuckle1",
            "rfmiddle1",
            "rfdistal1",
            "lfmetacarpal1",
            "lfmetacarpal2",
            "lfknuckle1",
            "lfmiddle1",
            "lfdistal1",
            "thbase1",
            "thhub1",
            "thdistal1",
        )
        for side, front, rear in (
            ("left", SupportRegion.FRONT_LEFT, SupportRegion.REAR_LEFT),
            ("right", SupportRegion.FRONT_RIGHT, SupportRegion.REAR_RIGHT),
        ):
            front_names = {
                f"{side}_larm",
                *(f"geom:{side}_hand{index}" for index in range(1, 5)),
                *(f"geom:{side}_{suffix}" for suffix in digit_suffixes),
            }
            rear_names = {
                *(f"geom:{side}_lower_leg{index}" for index in range(1, 3)),
                *(f"geom:{side}_foot{index}" for index in range(1, 4)),
                *(f"geom:{side}_big_toe{index}" for index in range(1, 3)),
                *(f"geom:{side}_toes{index}" for index in range(1, 3)),
            }
            for name in front_names:
                self._region_by_geom_id[int(model.geom(name).id)] = front
            for name in rear_names:
                self._region_by_geom_id[int(model.geom(name).id)] = rear
        self._upper_leg_knee: dict[int, tuple[SupportRegion, int, float]] = {}
        for side, region in (
            ("left", SupportRegion.REAR_LEFT),
            ("right", SupportRegion.REAR_RIGHT),
        ):
            geom_id = int(model.geom(f"geom:{side}_upper_leg1").id)
            knee_body_id = int(model.body(f"{side}_lower_leg").id)
            radius = float(model.geom_size[geom_id, 0])
            self._upper_leg_knee[geom_id] = (region, knee_body_id, radius)

    def classify(self) -> StaticSupportContacts:
        """Read all solved external contacts without mutating simulation."""

        environment = self._environment
        model = environment.model
        data = environment.data
        contacts: list[StaticSupportContact] = []
        central_force = 0.0
        head_force = 0.0
        unexpected: set[tuple[str, str]] = set()
        for contact_index, contact in enumerate(data.contact):
            if int(contact.efc_address) < 0:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            first_is_mimo = int(model.body_rootid[body1]) == self._mimo_root_id
            second_is_mimo = int(model.body_rootid[body2]) == self._mimo_root_id
            if first_is_mimo == second_is_mimo:
                continue
            mimo_geom = geom1 if first_is_mimo else geom2
            external_geom = geom2 if first_is_mimo else geom1
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, force)
            normal_force = max(float(force[0]), 0.0)
            mimo_name = self._geom_name(model, mimo_geom)
            external_name = self._geom_name(model, external_geom)
            if external_geom != self._floor_geom_id:
                unexpected.add((mimo_name, external_name))
                continue
            body_id = int(model.geom_bodyid[mimo_geom])
            if body_id in self._head_body_ids:
                head_force = max(head_force, normal_force)
            if normal_force <= 0.1:
                continue

            region = self._region_by_geom_id.get(mimo_geom)
            knee_distance: float | None = None
            knee = self._upper_leg_knee.get(mimo_geom)
            if knee is not None:
                candidate_region, knee_body_id, radius = knee
                knee_distance = float(
                    np.linalg.norm(
                        np.asarray(contact.pos, dtype=np.float64)
                        - data.xpos[knee_body_id]
                    )
                )
                if self._is_distal_knee_contact(
                    np.asarray(contact.pos, dtype=np.float64),
                    data.xpos[knee_body_id],
                    radius,
                ):
                    region = candidate_region
            if region is not None:
                contacts.append(
                    StaticSupportContact(
                        region=region,
                        geom_name=mimo_name,
                        position_m=tuple(float(value) for value in contact.pos),
                        normal_force_n=normal_force,
                        distal_knee_distance_m=knee_distance,
                    )
                )
            else:
                central_force = max(central_force, normal_force)
        return StaticSupportContacts(
            time_seconds=float(data.time),
            contacts=tuple(contacts),
            central_force_n=central_force,
            head_force_n=head_force,
            unexpected_external_contacts=tuple(sorted(unexpected)),
        )

    @staticmethod
    def support_polygon_margin_m(
        contacts: StaticSupportContacts,
        com_xy_m: NDArray[np.floating],
    ) -> float:
        """Return signed distance from COM to the convex support-polygon edge."""

        points = sorted(
            {
                (contact.position_m[0], contact.position_m[1])
                for contact in contacts.contacts
                if contact.normal_force_n > 0.0
            }
        )
        hull = StaticSupportContactClassifier._convex_hull(points)
        if len(hull) < 3:
            return float("-inf")
        point = np.asarray(com_xy_m, dtype=np.float64)
        signed_distances: list[float] = []
        for start, stop in zip(hull, (*hull[1:], hull[0]), strict=True):
            edge = np.asarray(stop) - np.asarray(start)
            offset = point - np.asarray(start)
            signed_distances.append(
                float((edge[0] * offset[1] - edge[1] * offset[0]) / np.linalg.norm(edge))
            )
        return min(signed_distances)

    @staticmethod
    def _convex_hull(
        points: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], ...]:
        if len(points) <= 1:
            return tuple(points)

        def cross(
            origin: tuple[float, float],
            first: tuple[float, float],
            second: tuple[float, float],
        ) -> float:
            return (first[0] - origin[0]) * (second[1] - origin[1]) - (
                first[1] - origin[1]
            ) * (second[0] - origin[0])

        lower: list[tuple[float, float]] = []
        for point in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
                lower.pop()
            lower.append(point)
        upper: list[tuple[float, float]] = []
        for point in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
                upper.pop()
            upper.append(point)
        return tuple(lower[:-1] + upper[:-1])

    @staticmethod
    def _is_distal_knee_contact(
        contact_position_m: NDArray[np.floating],
        knee_position_m: NDArray[np.floating],
        capsule_radius_m: float,
    ) -> bool:
        distance = float(
            np.linalg.norm(
                np.asarray(contact_position_m, dtype=np.float64)
                - np.asarray(knee_position_m, dtype=np.float64)
            )
        )
        return distance <= capsule_radius_m + 0.010

    @staticmethod
    def _descendant_body_ids(model: mujoco.MjModel, root_id: int) -> frozenset[int]:
        descendants: set[int] = set()
        for body_id in range(model.nbody):
            current = body_id
            while current != 0:
                if current == root_id:
                    descendants.add(body_id)
                    break
                current = int(model.body_parentid[current])
        return frozenset(descendants)

    @staticmethod
    def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None:
            raise ValueError(f"geom {geom_id} has no name")
        return name

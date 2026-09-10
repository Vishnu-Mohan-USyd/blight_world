"""Short, constraint-assisted ball throw built from the validated held pose.

The recorded behavior begins after the existing three-object controller has
placed the ball in its disclosed collision-suppressed, mocap-weld-assisted
hold.  The throw itself is actuator driven: the grip anchor follows the
``right_hand`` body frame without a world-space offset, the seven arm joints
execute one smooth reference change, and the weld is disabled while the hand
is moving.  The free ball must then inherit the hand velocity and reach the
floor through MuJoCo dynamics.  This is a throw-only demonstration, not proof
of a physical pickup or grasp.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from numpy.typing import NDArray

from blight_world.constraint_assisted_three_object import (
    ASSISTED_HOLD_CENTER_M,
    ASSISTED_HOLD_DIGIT_CLOSURE_FRACTION,
    GRIP_DIGIT_REFERENCE_RAD,
    ROOT_YAW_ROTATION,
    ConstraintAssistedControllerState,
    ConstraintAssistedThreeObjectController,
)
from blight_world.motor_control import (
    MotorChannel,
    resolve_motor_channels,
    single_joint_bias_compensated_pd,
)
from blight_world.motor_skills import (
    RIGHT_ARM_ACTUATOR_NAMES,
    RIGHT_ARM_DAMPING_NM_S_RAD,
    RIGHT_ARM_STIFFNESS_NM_RAD,
)
from blight_world.supported_pick_place import (
    LIFT_ARM_SEED_RAD,
    RIGHT_DIGIT_ACTUATOR_NAMES,
)
from blight_world.three_object_scenario import (
    SCENARIO_BODY_NAMES,
    SCENARIO_SEED,
    ScenarioObject,
    ScenarioSchedule,
    ThreeObjectScenarioEnv,
    dispatch_controller,
)
from mimo_audition.clips import AudioClip


PRE_ROLL_PUBLIC_STEPS = 700
RECORDED_PUBLIC_STEPS = 500
WINDUP_STOP_RECORDED_STEP = 30
RETURN_STOP_RECORDED_STEP = 70
PRE_SWING_PUBLIC_STEPS = 92
SWING_PUBLIC_STEPS = 7
RELEASE_RECORDED_STEP = PRE_SWING_PUBLIC_STEPS + SWING_PUBLIC_STEPS
FINGER_OPEN_START_SWING_STEP = 3
FINGER_OPEN_STOP_SWING_STEP = SWING_PUBLIC_STEPS
COLLISION_CLEARANCE_M = 0.001
# Demo-only tolerance for the disclosed collision-suppressed mocap hold.
MAXIMUM_HAND_RELATIVE_ERROR_M = 0.008
MAXIMUM_RELEASE_VELOCITY_ERROR_M_S = 0.10
MINIMUM_RELEASE_SPEED_M_S = 0.50

WINDUP_ARM_REFERENCE_RAD = (
    -0.2,
    1.5366,
    1.1263333063585548,
    -2.5,
    -0.2545961859903202,
    0.13084197814383444,
    0.1329125999065053,
)
LAUNCH_ARM_REFERENCE_RAD = (
    0.7,
    1.5366,
    1.1263333063585548,
    -1.7,
    -0.2545961859903202,
    0.13084197814383444,
    0.1329125999065053,
)
THROW_ARM_TARGET_RAD = LAUNCH_ARM_REFERENCE_RAD
THROW_PULSE_RAMP_STEPS = 5
THROW_PULSE_ACTIONS = (
    ("act:right_shoulder_horizontal", 1.0),
    ("act:right_elbow", 0.30),
    ("act:right_shoulder_abduction", 0.152),
)

BALL_THROW_IMPACT_CLIP_ID = "demo_ball_floor_single_hit_rubber"
BALL_THROW_IMPACT_SOURCE_CLIP_ID = "impact_hard_rubber_2"


Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class HeldBallPreparation:
    """Audited cycle-zero held state produced before recording begins."""

    baseline_action: Float32Array
    digit_open_reference_rad: tuple[float, ...]
    digit_held_reference_rad: tuple[float, ...]
    hold_relative_quaternion_wxyz: tuple[float, float, float, float]
    original_ball_collision_mask: tuple[int, int]
    base_controller: ConstraintAssistedThreeObjectController

    def __post_init__(self) -> None:
        action = np.frombuffer(
            np.ascontiguousarray(self.baseline_action, dtype=np.float32).tobytes(),
            dtype=np.float32,
        )
        if action.ndim != 1 or not np.all(np.isfinite(action)):
            raise ValueError("baseline action must be a finite vector")
        if len(self.digit_open_reference_rad) != len(RIGHT_DIGIT_ACTUATOR_NAMES):
            raise ValueError("digit-open reference has the wrong length")
        if len(self.digit_held_reference_rad) != len(RIGHT_DIGIT_ACTUATOR_NAMES):
            raise ValueError("digit-held reference has the wrong length")
        object.__setattr__(self, "baseline_action", action)


@dataclass(frozen=True, slots=True)
class BallThrowState:
    """Immutable public-step state and release measurements."""

    step_index: int = 0
    released: bool = False
    collisions_restored: bool = False
    pre_release_ball_velocity_m_s: Vector3 | None = None
    pre_release_hand_velocity_m_s: Vector3 | None = None
    post_release_ball_velocity_m_s: Vector3 | None = None
    post_release_hand_velocity_m_s: Vector3 | None = None
    maximum_hand_relative_error_m: float = 0.0
    collision_restore_step: int | None = None
    saturation_rows: int = 0
    held_pose_sync_count: int = 0
    release_velocity_write_count: int = 0


@dataclass(frozen=True, slots=True)
class BallThrowStep:
    """One bounded action plus its next immutable controller state."""

    action: Float32Array
    state: BallThrowState

    def __post_init__(self) -> None:
        values = np.frombuffer(
            np.ascontiguousarray(self.action, dtype=np.float32).tobytes(),
            dtype=np.float32,
        )
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("throw action must be a finite vector")
        object.__setattr__(self, "action", values)


def derive_single_hit_rubber_clip(source: AudioClip) -> Float64Array:
    """Derive one deterministic, single-onset rubber hit from variant 2.

    The source transient is retained through 9 ms after its first 2%-peak
    onset and then half-cosine faded over 3 ms.  The remaining clip contains
    only a phase-continuous, smoothly introduced 160/260 Hz rubber resonance.
    The source clip is never modified.
    """

    samples = np.asarray(source.samples, dtype=np.float64)
    peak = float(np.max(np.abs(samples)))
    onset_candidates = np.flatnonzero(np.abs(samples) >= 0.02 * peak)
    if onset_candidates.size == 0:
        raise ValueError("source rubber clip has no measurable onset")
    onset = int(onset_candidates[0])
    retained_stop = min(
        samples.size,
        onset + round(0.009 * source.sample_rate_hz),
    )
    fade_samples = round(0.003 * source.sample_rate_hz)
    fade_stop = min(samples.size, retained_stop + fade_samples)

    output = np.zeros_like(samples)
    output[:retained_stop] = samples[:retained_stop]
    if fade_stop > retained_stop:
        phase = np.linspace(
            0.0,
            math.pi,
            fade_stop - retained_stop,
            endpoint=True,
            dtype=np.float64,
        )
        output[retained_stop:fade_stop] = (
            samples[retained_stop:fade_stop] * 0.5 * (1.0 + np.cos(phase))
        )

    indices = np.arange(samples.size - onset, dtype=np.float64)
    time_seconds = indices / source.sample_rate_hz
    resonance = (
        np.sin(2.0 * math.pi * 160.0 * time_seconds)
        * np.exp(-time_seconds / 0.045)
        + 0.45
        * np.sin(2.0 * math.pi * 260.0 * time_seconds)
        * np.exp(-time_seconds / 0.025)
    )
    resonance_peak = float(np.max(np.abs(resonance)))
    if resonance_peak <= 0.0:
        raise AssertionError("rubber resonance became silent")
    resonance *= 0.12 * peak / resonance_peak
    attack_samples = max(1, round(0.001 * source.sample_rate_hz))
    attack = np.ones_like(resonance)
    attack[:attack_samples] = 0.5 * (
        1.0
        - np.cos(
            np.linspace(
                0.0,
                math.pi,
                attack_samples,
                endpoint=True,
                dtype=np.float64,
            )
        )
    )
    output[onset:] += resonance * attack
    return np.ascontiguousarray(output, dtype=np.float64)


def onset_cluster_starts(
    samples: NDArray[np.floating],
    sample_rate_hz: int,
) -> tuple[int, ...]:
    """Return robust onset-cluster starts from a 1 ms moving-RMS envelope."""

    values = np.asarray(samples, dtype=np.float64)
    window = max(1, round(0.001 * sample_rate_hz))
    envelope = np.sqrt(
        np.convolve(values * values, np.ones(window) / window, mode="same")
    )
    threshold = 0.03 * float(np.max(envelope))
    active = envelope >= threshold
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return ()
    gap = max(1, round(0.012 * sample_rate_hz))
    starts = [int(indices[0])]
    for previous, current in zip(indices, indices[1:]):
        if int(current - previous) > gap:
            starts.append(int(current))
    return tuple(starts)


def prepare_held_ball(
    environment: ThreeObjectScenarioEnv,
) -> HeldBallPreparation:
    """Run the validated cycle-zero controller through local step 699."""

    schedule = ScenarioSchedule.default()
    base_controller = ConstraintAssistedThreeObjectController(environment, schedule)
    base_state = base_controller.reset(environment, schedule, SCENARIO_SEED)
    ball_geom_id = int(environment.model.geom("demo_ball_geom").id)
    original_mask = (
        int(environment.model.geom_contype[ball_geom_id]),
        int(environment.model.geom_conaffinity[ball_geom_id]),
    )
    baseline_action: Float32Array | None = None
    for step_index in range(PRE_ROLL_PUBLIC_STEPS):
        command = dispatch_controller(
            base_controller,
            environment,
            base_state,
            schedule.context_at(step_index),
        )
        baseline_action = np.ascontiguousarray(command.action, dtype=np.float32)
        base_state = command.state
        observation, _, terminated, truncated, _ = environment.step(baseline_action)
        if terminated or truncated or not environment.observation_space.contains(
            observation
        ):
            raise AssertionError(f"held-pose pre-roll failed at step {step_index}")
    if not isinstance(base_state, ConstraintAssistedControllerState):
        raise TypeError("base controller returned an unexpected state")
    if baseline_action is None or base_state.step_index != PRE_ROLL_PUBLIC_STEPS:
        raise AssertionError("held-pose pre-roll clock changed")
    open_reference = base_state.digit_start_references_rad[0]
    relative_quaternion = base_state.hold_relative_quaternions[0]
    if open_reference is None or relative_quaternion is None:
        raise AssertionError("cycle-zero held-pose state is incomplete")
    fraction = ASSISTED_HOLD_DIGIT_CLOSURE_FRACTION[ScenarioObject.BALL]
    held_reference = tuple(
        start + fraction * (stop - start)
        for start, stop in zip(
            open_reference,
            GRIP_DIGIT_REFERENCE_RAD[ScenarioObject.BALL],
            strict=True,
        )
    )
    ball_equality_id = base_controller.grip_equality_ids[ScenarioObject.BALL]
    if int(environment.data.eq_active[ball_equality_id]) != 1:
        raise AssertionError("ball grip weld is not active after pre-roll")
    if (
        int(environment.model.geom_contype[ball_geom_id]) != 0
        or int(environment.model.geom_conaffinity[ball_geom_id]) != 0
    ):
        raise AssertionError("held ball collisions are not suppressed")
    return HeldBallPreparation(
        baseline_action=baseline_action,
        digit_open_reference_rad=open_reference,
        digit_held_reference_rad=held_reference,
        hold_relative_quaternion_wxyz=relative_quaternion,
        original_ball_collision_mask=original_mask,
        base_controller=base_controller,
    )


class AssistedBallThrowController:
    """Actuator-driven swing followed by one velocity-inheriting weld release."""

    def __init__(
        self,
        environment: ThreeObjectScenarioEnv,
        preparation: HeldBallPreparation,
    ) -> None:
        self.environment = environment
        self.preparation = preparation
        self.channels = {
            channel.actuator_name: channel
            for channel in resolve_motor_channels(environment)
        }
        self.ball_body_id = int(environment.model.body("demo_ball").id)
        self.ball_geom_id = int(environment.model.geom("demo_ball_geom").id)
        self.ball_joint_name = "demo_ball_joint"
        self.hand_body_id = int(environment.model.body("right_hand").id)
        self.grip_mocap_id = preparation.base_controller.grip_mocap_id
        self.grip_equality_id = preparation.base_controller.grip_equality_ids[
            ScenarioObject.BALL
        ]
        self.hand_geom_ids = self._descendant_geom_ids(self.hand_body_id)

    def initial_state(self) -> BallThrowState:
        """Return the immutable state for recorded public step zero."""

        return BallThrowState()

    def compose(self, state: BallThrowState) -> BallThrowStep:
        """Compose one action without writing object qpos, qvel, or forces."""

        if state.step_index >= RECORDED_PUBLIC_STEPS:
            raise IndexError("recorded ball-throw episode is complete")
        step_index = state.step_index
        released = state.released
        pre_ball = state.pre_release_ball_velocity_m_s
        pre_hand = state.pre_release_hand_velocity_m_s

        if not released:
            self._follow_hand_exactly()
        if step_index == RELEASE_RECORDED_STEP:
            hand_spatial_velocity = self._hand_spatial_velocity()
            ball_qvel = self.environment.data.joint(self.ball_joint_name).qvel
            ball_qvel[:3] = hand_spatial_velocity[3:]
            ball_qvel[3:6] = hand_spatial_velocity[:3]
            pre_ball = self._vector3(ball_qvel[:3])
            pre_hand = self._vector3(hand_spatial_velocity[3:])
            self.environment.data.eq_active[self.grip_equality_id] = 0
            mujoco.mj_forward(self.environment.model, self.environment.data)
            released = True

        action = np.array(
            self.preparation.baseline_action,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        if step_index < WINDUP_STOP_RECORDED_STEP:
            arm_blend = self._smooth(
                (step_index + 1) / WINDUP_STOP_RECORDED_STEP
            )
            arm_reference = tuple(
                start + arm_blend * (stop - start)
                for start, stop in zip(
                    LIFT_ARM_SEED_RAD,
                    WINDUP_ARM_REFERENCE_RAD,
                    strict=True,
                )
            )
        elif step_index < RETURN_STOP_RECORDED_STEP:
            arm_blend = self._smooth(
                (step_index - WINDUP_STOP_RECORDED_STEP + 1)
                / (RETURN_STOP_RECORDED_STEP - WINDUP_STOP_RECORDED_STEP)
            )
            arm_reference = tuple(
                start + arm_blend * (stop - start)
                for start, stop in zip(
                    WINDUP_ARM_REFERENCE_RAD,
                    LAUNCH_ARM_REFERENCE_RAD,
                    strict=True,
                )
            )
        else:
            arm_reference = LAUNCH_ARM_REFERENCE_RAD
        saturation_rows = state.saturation_rows
        for name, reference, stiffness, damping in zip(
            RIGHT_ARM_ACTUATOR_NAMES,
            arm_reference,
            RIGHT_ARM_STIFFNESS_NM_RAD,
            RIGHT_ARM_DAMPING_NM_S_RAD,
            strict=True,
        ):
            saturation_rows += self._command_pd(
                action, name, reference, stiffness, damping
            )

        if PRE_SWING_PUBLIC_STEPS <= step_index <= RELEASE_RECORDED_STEP:
            pulse_blend = self._smooth(
                (step_index - PRE_SWING_PUBLIC_STEPS + 1)
                / THROW_PULSE_RAMP_STEPS
            )
            for name, pulse_action in THROW_PULSE_ACTIONS:
                action_index = self.channels[name].action_index
                action[action_index] = np.float32(
                    (1.0 - pulse_blend) * float(action[action_index])
                    + pulse_blend * pulse_action
                )

        swing_step = step_index - PRE_SWING_PUBLIC_STEPS
        open_blend = self._smooth(
            (swing_step - FINGER_OPEN_START_SWING_STEP)
            / (FINGER_OPEN_STOP_SWING_STEP - FINGER_OPEN_START_SWING_STEP)
        )
        for name, closed, opened in zip(
            RIGHT_DIGIT_ACTUATOR_NAMES,
            self.preparation.digit_held_reference_rad,
            self.preparation.digit_open_reference_rad,
            strict=True,
        ):
            channel = self.channels[name]
            saturation_rows += self._command_pd(
                action,
                name,
                closed + open_blend * (opened - closed),
                0.20 * channel.gear,
                0.01 * channel.gear,
            )
        saturation_rows += self._command_ball_gaze(action)

        next_state = BallThrowState(
            step_index=step_index + 1,
            released=released,
            collisions_restored=state.collisions_restored,
            pre_release_ball_velocity_m_s=pre_ball,
            pre_release_hand_velocity_m_s=pre_hand,
            post_release_ball_velocity_m_s=state.post_release_ball_velocity_m_s,
            post_release_hand_velocity_m_s=state.post_release_hand_velocity_m_s,
            maximum_hand_relative_error_m=state.maximum_hand_relative_error_m,
            collision_restore_step=state.collision_restore_step,
            saturation_rows=saturation_rows,
            held_pose_sync_count=state.held_pose_sync_count,
            release_velocity_write_count=(
                state.release_velocity_write_count
                + int(step_index == RELEASE_RECORDED_STEP)
            ),
        )
        return BallThrowStep(action=action, state=next_state)

    def after_step(self, state: BallThrowState) -> BallThrowState:
        """Measure release inheritance and restore collision after clearance."""

        held_pose_sync_count = state.held_pose_sync_count
        if not state.released:
            self._synchronize_ball_pose_to_hand()
            held_pose_sync_count += 1
        maximum_error = state.maximum_hand_relative_error_m
        if not state.released:
            maximum_error = max(maximum_error, self.hand_relative_error_m())
        post_ball = state.post_release_ball_velocity_m_s
        post_hand = state.post_release_hand_velocity_m_s
        if state.step_index == RELEASE_RECORDED_STEP + 1:
            post_ball = self._vector3(self._ball_linear_velocity())
            post_hand = self._vector3(self._hand_linear_velocity())

        collisions_restored = state.collisions_restored
        restore_step = state.collision_restore_step
        if state.released and not collisions_restored:
            if self.minimum_hand_clearance_m() > COLLISION_CLEARANCE_M:
                contype, conaffinity = self.preparation.original_ball_collision_mask
                self.environment.model.geom_contype[self.ball_geom_id] = contype
                self.environment.model.geom_conaffinity[
                    self.ball_geom_id
                ] = conaffinity
                mujoco.mj_forward(self.environment.model, self.environment.data)
                collisions_restored = True
                restore_step = state.step_index - 1

        return BallThrowState(
            step_index=state.step_index,
            released=state.released,
            collisions_restored=collisions_restored,
            pre_release_ball_velocity_m_s=state.pre_release_ball_velocity_m_s,
            pre_release_hand_velocity_m_s=state.pre_release_hand_velocity_m_s,
            post_release_ball_velocity_m_s=post_ball,
            post_release_hand_velocity_m_s=post_hand,
            maximum_hand_relative_error_m=maximum_error,
            collision_restore_step=restore_step,
            saturation_rows=state.saturation_rows,
            held_pose_sync_count=held_pose_sync_count,
            release_velocity_write_count=state.release_velocity_write_count,
        )

    def hand_relative_error_m(self) -> float:
        """Distance from the exact disclosed hand-local assisted-hold center."""

        hand = self.environment.data.body("right_hand")
        local = np.asarray(hand.xmat, dtype=np.float64).reshape(3, 3).T @ (
            np.asarray(self.environment.data.xpos[self.ball_body_id], dtype=np.float64)
            - np.asarray(hand.xpos, dtype=np.float64)
        )
        return float(
            np.linalg.norm(
                local
                - np.asarray(
                    ASSISTED_HOLD_CENTER_M[ScenarioObject.BALL], dtype=np.float64
                )
            )
        )

    def minimum_hand_clearance_m(self) -> float:
        """Minimum signed ball-to-right-hand geom distance in metres."""

        from_to = np.zeros(6, dtype=np.float64)
        return min(
            float(
                mujoco.mj_geomDistance(
                    self.environment.model,
                    self.environment.data,
                    self.ball_geom_id,
                    geom_id,
                    0.25,
                    from_to,
                )
            )
            for geom_id in self.hand_geom_ids
        )

    def release_velocity_metrics(self, state: BallThrowState) -> dict[str, float]:
        """Return the pre/post hand-to-ball velocity inheritance evidence."""

        vectors = (
            state.pre_release_ball_velocity_m_s,
            state.pre_release_hand_velocity_m_s,
            state.post_release_ball_velocity_m_s,
            state.post_release_hand_velocity_m_s,
        )
        if any(vector is None for vector in vectors):
            raise AssertionError("release velocity evidence is incomplete")
        pre_ball, pre_hand, post_ball, post_hand = (
            np.asarray(vector, dtype=np.float64) for vector in vectors
        )
        return {
            "pre_release_velocity_error_m_s": float(
                np.linalg.norm(pre_ball - pre_hand)
            ),
            "post_release_velocity_error_m_s": float(
                np.linalg.norm(post_ball - post_hand)
            ),
            "pre_release_ball_speed_m_s": float(np.linalg.norm(pre_ball)),
            "post_release_ball_speed_m_s": float(np.linalg.norm(post_ball)),
        }

    def close(self) -> None:
        """Restore the ball collision mask even after an interrupted run."""

        contype, conaffinity = self.preparation.original_ball_collision_mask
        self.environment.model.geom_contype[self.ball_geom_id] = contype
        self.environment.model.geom_conaffinity[self.ball_geom_id] = conaffinity
        self.preparation.base_controller.close()

    def _follow_hand_exactly(self) -> None:
        hand = self.environment.data.body("right_hand")
        hand_rotation = np.asarray(hand.xmat, dtype=np.float64).reshape(3, 3)
        self.environment.data.mocap_pos[self.grip_mocap_id] = (
            np.asarray(hand.xpos, dtype=np.float64)
            + hand_rotation
            @ np.asarray(
                ASSISTED_HOLD_CENTER_M[ScenarioObject.BALL], dtype=np.float64
            )
        )
        hand_quaternion = np.asarray(hand.xquat, dtype=np.float64)
        relative = np.asarray(
            self.preparation.hold_relative_quaternion_wxyz, dtype=np.float64
        )
        result = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(result, hand_quaternion, relative)
        self.environment.data.mocap_quat[self.grip_mocap_id] = result

    def _command_pd(
        self,
        action: Float32Array,
        name: str,
        reference: float,
        stiffness: float,
        damping: float,
    ) -> int:
        channel: MotorChannel = self.channels[name]
        value = single_joint_bias_compensated_pd(
            channel,
            self.environment.data,
            reference_position_rad=reference,
            stiffness_nm_per_rad=stiffness,
            damping_nm_s_per_rad=damping,
        )
        action[channel.action_index] = value
        return int(
            min(
                value - channel.lower_bound,
                channel.upper_bound - value,
            )
            < 1e-6
        )

    def _command_ball_gaze(self, action: Float32Array) -> int:
        """Aim the proven physical head-and-eye controller at the live ball."""

        environment = self.environment
        look_position = np.asarray(
            environment.data.xpos[self.ball_body_id], dtype=np.float64
        )
        delta_local = ROOT_YAW_ROTATION.T @ (
            look_position
            - np.asarray(environment.data.body("head").xpos, dtype=np.float64)
        )
        head_swivel = float(
            np.clip(math.atan2(delta_local[1], delta_local[0]), -1.6, 1.6)
        )
        head_tilt = float(
            np.clip(
                math.atan2(
                    -delta_local[2],
                    math.hypot(delta_local[0], delta_local[1]),
                ),
                -1.0,
                1.2,
            )
        )
        saturation_rows = self._command_pd(
            action, "act:head_swivel", head_swivel, 1.0, 0.2
        )
        saturation_rows += self._command_pd(
            action, "act:head_tilt", head_tilt, 1.0, 0.2
        )
        saturation_rows += self._command_pd(
            action, "act:head_tilt_side", 0.0, 2.0, 0.5
        )
        for side in ("left", "right"):
            saturation_rows += self._command_pd(
                action,
                f"act:{side}_eye_torsional",
                0.0,
                0.002,
                0.0004,
            )

        left_camera_id = int(environment.model.camera("eye_left").id)
        right_camera_id = int(environment.model.camera("eye_right").id)
        eye_midpoint = 0.5 * (
            np.asarray(
                environment.data.cam_xpos[left_camera_id], dtype=np.float64
            )
            + np.asarray(
                environment.data.cam_xpos[right_camera_id], dtype=np.float64
            )
        )
        camera_rotation = np.asarray(
            environment.data.cam_xmat[left_camera_id], dtype=np.float64
        ).reshape(3, 3)
        direction = look_position - eye_midpoint
        distance = float(np.linalg.norm(direction))
        if distance <= 1e-9:
            raise AssertionError("ball coincides with the binocular midpoint")
        direction /= distance
        forward_component = float(direction @ (-camera_rotation[:, 2]))
        horizontal_reference = math.atan2(
            float(direction @ camera_rotation[:, 0]), forward_component
        )
        vertical_reference = math.atan2(
            float(direction @ camera_rotation[:, 1]), forward_component
        )

        for side, horizontal_sign in (("left", -1.0), ("right", 1.0)):
            horizontal = self.channels[f"act:{side}_eye_horizontal"]
            vertical = self.channels[f"act:{side}_eye_vertical"]
            current_horizontal = float(
                environment.data.qpos[horizontal.qpos_address]
            )
            current_vertical = float(
                environment.data.qpos[vertical.qpos_address]
            )
            commanded_horizontal = current_horizontal + float(
                np.clip(
                    horizontal_sign * horizontal_reference - current_horizontal,
                    -0.15,
                    0.15,
                )
            )
            commanded_vertical = current_vertical + float(
                np.clip(
                    vertical_reference - current_vertical,
                    -0.15,
                    0.15,
                )
            )
            horizontal_range = environment.model.jnt_range[horizontal.joint_id]
            vertical_range = environment.model.jnt_range[vertical.joint_id]
            clipped_horizontal = float(
                np.clip(
                    commanded_horizontal,
                    horizontal_range[0] + 0.04,
                    horizontal_range[1] - 0.04,
                )
            )
            clipped_vertical = float(
                np.clip(
                    commanded_vertical,
                    vertical_range[0] + 0.04,
                    vertical_range[1] - 0.04,
                )
            )
            saturation_rows += self._command_pd(
                action,
                f"act:{side}_eye_horizontal",
                clipped_horizontal,
                0.02,
                0.002,
            )
            saturation_rows += self._command_pd(
                action,
                f"act:{side}_eye_vertical",
                clipped_vertical,
                0.02,
                0.002,
            )
        return saturation_rows

    def _hand_spatial_velocity(self) -> Float64Array:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.environment.model,
            self.environment.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.hand_body_id,
            velocity,
            0,
        )
        return velocity

    def _hand_linear_velocity(self) -> Float64Array:
        return self._hand_spatial_velocity()[3:].copy()

    def _synchronize_ball_pose_to_hand(self) -> None:
        """Place the already-held ball at the exact measured hand transform."""

        hand = self.environment.data.body("right_hand")
        hand_rotation = np.asarray(hand.xmat, dtype=np.float64).reshape(3, 3)
        target_position = np.asarray(hand.xpos, dtype=np.float64) + hand_rotation @ (
            np.asarray(ASSISTED_HOLD_CENTER_M[ScenarioObject.BALL], dtype=np.float64)
        )
        target_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(
            target_quaternion,
            np.asarray(hand.xquat, dtype=np.float64),
            np.asarray(
                self.preparation.hold_relative_quaternion_wxyz,
                dtype=np.float64,
            ),
        )
        ball_qpos = self.environment.data.joint(self.ball_joint_name).qpos
        ball_qpos[:3] = target_position
        ball_qpos[3:7] = target_quaternion
        mujoco.mj_forward(self.environment.model, self.environment.data)

    def _ball_linear_velocity(self) -> Float64Array:
        return np.asarray(
            self.environment.data.joint(self.ball_joint_name).qvel[:3],
            dtype=np.float64,
        ).copy()

    def _descendant_geom_ids(self, root_body_id: int) -> tuple[int, ...]:
        body_ids = {
            body_id
            for body_id in range(self.environment.model.nbody)
            if self._is_descendant(body_id, root_body_id)
        }
        geom_ids = tuple(
            geom_id
            for geom_id in range(self.environment.model.ngeom)
            if int(self.environment.model.geom_bodyid[geom_id]) in body_ids
        )
        if not geom_ids:
            raise AssertionError("right-hand subtree contains no geoms")
        return geom_ids

    def _is_descendant(self, body_id: int, root_body_id: int) -> bool:
        current = body_id
        while current != 0:
            if current == root_body_id:
                return True
            current = int(self.environment.model.body_parentid[current])
        return current == root_body_id

    @staticmethod
    def _smooth(value: float) -> float:
        bounded = float(np.clip(value, 0.0, 1.0))
        return 3.0 * bounded * bounded - 2.0 * bounded * bounded * bounded

    @staticmethod
    def _vector3(values: NDArray[np.floating]) -> Vector3:
        return float(values[0]), float(values[1]), float(values[2])

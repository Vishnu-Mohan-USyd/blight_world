"""Calibrated controller for the constraint-assisted three-object demo.

The controller reproduces the proven 3 x 15 s actuator/mocap trace. Root
motion is carried by ``demo_root_support``; pickup uses exactly one target
specific grip weld from local t=6 through t=12 s. It never writes an object's
qpos/qvel or any applied force. The mechanism is not free walking and not a
physical grasp.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
from numpy.typing import NDArray

from blight_world.motor_control import (
    MotorChannel,
    coupled_hip_bias_compensated_pd,
    resolve_motor_channels,
    single_joint_bias_compensated_pd,
)
from blight_world.motor_skills import (
    RIGHT_ARM_ACTUATOR_NAMES,
    RIGHT_ARM_DAMPING_NM_S_RAD,
    RIGHT_ARM_STIFFNESS_NM_RAD,
    RIGHT_ARM_TARGET_RAD,
)
from blight_world.supported_pick_place import (
    LIFT_ARM_SEED_RAD,
    RIGHT_DIGIT_ACTUATOR_NAMES,
)
from blight_world.three_object_scenario import (
    ControllerStep,
    SCENARIO_BODY_NAMES,
    SCENARIO_SEED,
    ScenarioContext,
    ScenarioEvent,
    ScenarioEventKind,
    ScenarioObject,
    ScenarioSchedule,
    ThreeObjectScenarioEnv,
    next_object,
)


OBJECT_Y_M = (-0.30, 0.0, 0.30)
ROOT_STATIONS_M = tuple(
    (-0.038, object_y - 0.17, 0.4) for object_y in OBJECT_Y_M
)
ROOT_YAW_QUATERNION_WXYZ = (0.0, 0.0, 0.0, 1.0)
ROOT_YAW_ROTATION = np.diag((-1.0, -1.0, 1.0))
CONTROLLER_ID = "constraint-assisted-three-object-v1"

GRIP_DIGIT_REFERENCE_RAD = {
    ScenarioObject.BALL: (
        0.0, 1.012, 1.1224, 0.9016,
        0.0, 1.012, 1.1224, 0.9016,
        0.0, 1.012, 1.1224, 0.9016,
        0.0, 0.4508, 1.012, 1.1224, 0.9016,
        0.0, -0.828, 0.0, 1.012, 1.012,
    ),
    ScenarioObject.BLOCK: (
        0.0, 1.012, 1.1224, 0.9016,
        0.0, 1.012, 1.1224, 0.9016,
        0.0, 1.012, 1.1224, 0.9016,
        0.0, 0.4508, 1.012, 1.1224, 0.9016,
        0.0, -0.828, 0.0, 1.012, 1.012,
    ),
    ScenarioObject.CYLINDER: (
        0.0, 0.77, 0.854, 0.686,
        0.0, 0.77, 0.854, 0.686,
        0.0, 0.77, 0.854, 0.686,
        0.0, 0.343, 0.77, 0.854, 0.686,
        0.0, -0.63, 0.0, 0.77, 0.77,
    ),
}
ASSISTED_HOLD_CENTER_M = {
    ScenarioObject.BALL: (
        0.016284136472370515,
        -0.02759,
        0.02624554751923478,
    ),
    ScenarioObject.BLOCK: (-0.010, -0.060, 0.020),
    ScenarioObject.CYLINDER: (-0.006, -0.016, 0.022),
}
ASSISTED_HOLD_DIGIT_CLOSURE_FRACTION = {
    ScenarioObject.BALL: 0.45,
    ScenarioObject.BLOCK: 1.0,
    ScenarioObject.CYLINDER: 0.85,
}
BLOCK_ASSISTED_HOLD_RELATIVE_QUATERNION_WXYZ = (
    -0.1662171599,
    0.4587052977,
    -0.7899618450,
    -0.3713779599,
)
CYLINDER_ASSISTED_HOLD_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
)
HEAD_TILT_SIDE_KNOTS_RAD = {
    0: (
        (0, 0.0),
        (299, -0.560000),
        (599, -0.9265),
        (699, -0.2888),
        (849, -0.5119),
        (999, -0.3905),
        (1_200, -0.2839),
    ),
    1: (
        (0, 0.0),
        (299, -0.9707),
        (599, -0.9397),
        (699, -0.1006),
        (849, -0.5254),
        (999, -0.2256),
        (1_200, -0.2744),
    ),
    2: (
        (0, 0.0),
        (299, -1.0000),
        (599, -0.9433),
        (699, -0.1345),
        (849, -0.1757),
        (999, -0.4143),
        (1_200, -0.2638),
    ),
}
LEFT_EYE_TORSION_KNOTS_RAD = {
    0: (
        (0, 0.0),
        (299, -0.00327),
        (599, 0.00022),
        (699, -0.00206),
        (849, -0.00044),
        (999, 0.00019),
        (1_200, 0.00058),
    ),
    1: (
        (0, 0.0),
        (299, -0.00242),
        (599, 0.00126),
        (699, 0.00405),
        (849, -0.00153),
        (999, -0.00017),
        (1_200, 0.00068),
    ),
    2: (
        (0, 0.0),
        (299, 0.00137),
        (599, 0.00111),
        (699, 0.00047),
        (849, 0.00071),
        (999, -0.00134),
        (1_200, -0.00087),
    ),
}


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
DigitReference = tuple[float, ...]
EventKey = tuple[int, ScenarioEventKind]

@dataclass(frozen=True, slots=True)
class ConstraintAssistedControllerState:
    """Immutable deterministic state passed between public control steps."""

    step_index: int
    initial_arm_reference_rad: tuple[float, ...]
    attach_positions_m: tuple[Vector3 | None, Vector3 | None, Vector3 | None]
    inspect_positions_m: tuple[Vector3 | None, Vector3 | None, Vector3 | None]
    attach_relative_quaternions: tuple[
        Quaternion | None,
        Quaternion | None,
        Quaternion | None,
    ]
    hold_relative_quaternions: tuple[
        Quaternion | None,
        Quaternion | None,
        Quaternion | None,
    ]
    digit_start_references_rad: tuple[
        DigitReference | None,
        DigitReference | None,
        DigitReference | None,
    ]
    emitted_events: frozenset[EventKey]
    saturation_rows: int

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if len(self.initial_arm_reference_rad) != len(RIGHT_ARM_ACTUATOR_NAMES):
            raise ValueError("initial arm reference has the wrong length")
        if any(
            reference is not None
            and len(reference) != len(RIGHT_DIGIT_ACTUATOR_NAMES)
            for reference in self.digit_start_references_rad
        ):
            raise ValueError("digit start reference has the wrong length")
        if any(
            quaternion is not None and len(quaternion) != 4
            for quaternion in (
                *self.attach_relative_quaternions,
                *self.hold_relative_quaternions,
            )
        ):
            raise ValueError("relative attachment quaternion has the wrong length")
        if self.saturation_rows < 0:
            raise ValueError("saturation_rows must be non-negative")


class ConstraintAssistedThreeObjectController:
    """Exact deterministic root-anchor/grip-anchor demonstration controller."""

    __slots__ = (
        "_channels",
        "_environment",
        "_equality_ids",
        "_floor_geom_id",
        "_grip_mocap_id",
        "_model",
        "_object_collision_masks",
        "_object_geom_ids",
        "_root_mocap_id",
        "_root_support_equality_id",
    )

    def __init__(
        self,
        environment: ThreeObjectScenarioEnv,
        schedule: ScenarioSchedule,
    ) -> None:
        if schedule != ScenarioSchedule.default():
            raise ValueError("controller requires the frozen default schedule")
        self._environment = environment
        self._model = environment.model
        self._channels = {
            channel.actuator_name: channel
            for channel in resolve_motor_channels(environment)
        }
        self._root_mocap_id = int(
            np.asarray(environment.model.body("demo_root_anchor").mocapid).item()
        )
        self._grip_mocap_id = int(
            np.asarray(environment.model.body("demo_grip_anchor").mocapid).item()
        )
        self._root_support_equality_id = int(
            environment.model.equality("demo_root_support").id
        )
        self._equality_ids = {
            target: int(
                environment.model.equality(f"demo_grip_{target.value}").id
            )
            for target in ScenarioObject
        }
        self._object_geom_ids = {
            target: int(
                environment.model.geom(f"demo_{target.value}_geom").id
            )
            for target in ScenarioObject
        }
        self._object_collision_masks = {
            target: (
                int(environment.model.geom_contype[geom_id]),
                int(environment.model.geom_conaffinity[geom_id]),
            )
            for target, geom_id in self._object_geom_ids.items()
        }
        self._floor_geom_id = int(environment.model.geom("floor").id)

    @property
    def controller_id(self) -> str:
        return CONTROLLER_ID

    @property
    def root_mocap_id(self) -> int:
        return self._root_mocap_id

    @property
    def grip_mocap_id(self) -> int:
        return self._grip_mocap_id

    @property
    def root_support_equality_id(self) -> int:
        return self._root_support_equality_id

    @property
    def grip_equality_ids(self) -> dict[ScenarioObject, int]:
        return dict(self._equality_ids)

    def reset(
        self,
        environment: ThreeObjectScenarioEnv,
        schedule: ScenarioSchedule,
        seed: int,
    ) -> ConstraintAssistedControllerState:
        """Apply the one-time proven root orientation stage and reset anchors."""

        self._validate_environment(environment)
        self._restore_object_collisions()
        if schedule != ScenarioSchedule.default() or seed != SCENARIO_SEED:
            raise ValueError("controller requires the frozen schedule and seed 1401")
        qpos = np.asarray(environment.data.qpos, dtype=np.float64).copy()
        qvel = np.zeros_like(environment.data.qvel, dtype=np.float64)
        root_qpos = int(
            np.asarray(
                environment.model.joint("mimo_orientation").qposadr
            ).item()
        )
        qpos[root_qpos + 3 : root_qpos + 7] = ROOT_YAW_QUATERNION_WXYZ
        environment.set_state(qpos, qvel)
        mujoco.mj_forward(environment.model, environment.data)

        environment.data.mocap_pos[self._root_mocap_id] = (0.0, 0.0, 0.4)
        environment.data.mocap_quat[
            self._root_mocap_id
        ] = ROOT_YAW_QUATERNION_WXYZ
        first_body = SCENARIO_BODY_NAMES[ScenarioObject.BALL]
        environment.data.mocap_pos[self._grip_mocap_id] = environment.data.body(
            first_body
        ).xpos
        environment.data.mocap_quat[self._grip_mocap_id] = environment.data.body(
            first_body
        ).xquat
        environment.data.eq_active[self._root_support_equality_id] = 1
        for equality_id in self._equality_ids.values():
            environment.data.eq_active[equality_id] = 0

        initial_arm = tuple(
            float(environment.data.qpos[self._channels[name].qpos_address])
            for name in RIGHT_ARM_ACTUATOR_NAMES
        )
        return ConstraintAssistedControllerState(
            step_index=0,
            initial_arm_reference_rad=initial_arm,
            attach_positions_m=(None, None, None),
            inspect_positions_m=(None, None, None),
            attach_relative_quaternions=(None, None, None),
            hold_relative_quaternions=(None, None, None),
            digit_start_references_rad=(None, None, None),
            emitted_events=frozenset(),
            saturation_rows=0,
        )

    def compose_locomotion(
        self,
        environment: ThreeObjectScenarioEnv,
        state: object,
        context: ScenarioContext,
    ) -> ControllerStep:
        return self._compose(environment, state, context)

    def compose_grasp(
        self,
        environment: ThreeObjectScenarioEnv,
        state: object,
        context: ScenarioContext,
    ) -> ControllerStep:
        return self._compose(environment, state, context)

    def compose_gaze(
        self,
        environment: ThreeObjectScenarioEnv,
        state: object,
        context: ScenarioContext,
    ) -> ControllerStep:
        return self._compose(environment, state, context)

    def compose_release(
        self,
        environment: ThreeObjectScenarioEnv,
        state: object,
        context: ScenarioContext,
    ) -> ControllerStep:
        return self._compose(environment, state, context)

    def compose_settle(
        self,
        environment: ThreeObjectScenarioEnv,
        state: object,
        context: ScenarioContext,
    ) -> ControllerStep:
        return self._compose(environment, state, context)

    def _compose(
        self,
        environment: ThreeObjectScenarioEnv,
        raw_state: object,
        context: ScenarioContext,
    ) -> ControllerStep:
        self._validate_environment(environment)
        if not isinstance(raw_state, ConstraintAssistedControllerState):
            raise TypeError("state must be ConstraintAssistedControllerState")
        state = raw_state
        if state.step_index != context.step_index:
            raise ValueError("controller state and scenario clocks disagree")

        cycle = context.cycle_index
        local_step = context.step_index % 1_500
        local_time = local_step * environment.dt
        target = context.target
        target_body_name = SCENARIO_BODY_NAMES[target]
        station = np.asarray(ROOT_STATIONS_M[cycle], dtype=np.float64)
        start_station = (
            np.asarray((0.0, 0.0, 0.4), dtype=np.float64)
            if cycle == 0
            else np.asarray(ROOT_STATIONS_M[cycle - 1], dtype=np.float64)
        )
        events: list[ScenarioEvent] = []
        emitted = set(state.emitted_events)
        attach_positions = list(state.attach_positions_m)
        inspect_positions = list(state.inspect_positions_m)
        attach_relative_quaternions = list(state.attach_relative_quaternions)
        hold_relative_quaternions = list(state.hold_relative_quaternions)
        digit_start_references = list(state.digit_start_references_rad)

        self._maybe_emit_floor_impact(
            environment,
            state,
            context,
            events,
            emitted,
        )

        if local_step < 300:
            progress = local_step / 300.0
            root_target = start_station + self._smooth(progress) * (
                station - start_station
            )
            root_target[2] = 0.4 - 0.06 * math.sin(math.pi * progress)
        else:
            root_target = station.copy()
        environment.data.mocap_pos[self._root_mocap_id] = root_target
        environment.data.mocap_quat[
            self._root_mocap_id
        ] = ROOT_YAW_QUATERNION_WXYZ

        if local_step == 300:
            root_error = float(
                np.linalg.norm(
                    np.asarray(environment.data.body("mimo_location").xpos)
                    - root_target
                )
            )
            self._emit_once(
                events,
                emitted,
                context,
                ScenarioEventKind.STATION_REACHED,
                metrics=(("root_position_error_m", root_error),),
            )

        if local_step == 600:
            object_position = np.asarray(
                environment.data.body(target_body_name).xpos
            ).copy()
            object_quaternion = np.asarray(
                environment.data.body(target_body_name).xquat,
                dtype=np.float64,
            ).copy()
            hand_quaternion = np.asarray(
                environment.data.body("right_hand").xquat,
                dtype=np.float64,
            ).copy()
            attach_positions[cycle] = self._vector3(object_position)
            attach_relative = self._multiply_quaternions(
                self._conjugate_quaternion(hand_quaternion),
                object_quaternion,
            )
            attach_relative_quaternions[cycle] = self._quaternion(attach_relative)
            if target is ScenarioObject.BALL:
                hold_relative = attach_relative
            elif target is ScenarioObject.BLOCK:
                hold_relative = self._normalize_quaternion(
                    np.asarray(
                        BLOCK_ASSISTED_HOLD_RELATIVE_QUATERNION_WXYZ,
                        dtype=np.float64,
                    )
                )
            else:
                hold_relative = self._rotation_matrix_to_quaternion(
                    np.asarray(
                        CYLINDER_ASSISTED_HOLD_ROTATION,
                        dtype=np.float64,
                    )
                )
            hold_relative_quaternions[cycle] = self._quaternion(hold_relative)
            environment.data.mocap_pos[self._grip_mocap_id] = object_position
            environment.data.mocap_quat[
                self._grip_mocap_id
            ] = object_quaternion
            self._suppress_object_collisions(target)
            environment.data.eq_active[self._equality_ids[target]] = 1
            mujoco.mj_forward(environment.model, environment.data)
            inspect = station + ROOT_YAW_ROTATION @ np.asarray(
                (0.067, -0.257, 0.031), dtype=np.float64
            )
            inspect_positions[cycle] = self._vector3(inspect)
            digit_start_references[cycle] = tuple(
                float(environment.data.qpos[self._channels[name].qpos_address])
                for name in RIGHT_DIGIT_ACTUATOR_NAMES
            )
            hand_distance = float(
                np.linalg.norm(
                    object_position
                    - np.asarray(environment.data.body("right_hand").xpos)
                )
            )
            self._emit_once(
                events,
                emitted,
                context,
                ScenarioEventKind.GRIP_CONSTRAINT_ACTIVATED,
                metrics=(("hand_center_distance_m", hand_distance),),
            )

        attach = attach_positions[cycle]
        inspect = inspect_positions[cycle]
        attach_relative_quaternion = attach_relative_quaternions[cycle]
        hold_relative_quaternion = hold_relative_quaternions[cycle]
        grip_anchor: NDArray[np.float64] | None = None
        if (
            attach is not None
            and attach_relative_quaternion is not None
            and hold_relative_quaternion is not None
            and 600 <= local_step < 1_200
        ):
            hand = environment.data.body("right_hand")
            hand_rotation = np.asarray(hand.xmat, dtype=np.float64).reshape(3, 3)
            hand_target = np.asarray(hand.xpos, dtype=np.float64) + hand_rotation @ (
                np.asarray(ASSISTED_HOLD_CENTER_M[target], dtype=np.float64)
            )
            blend = self._smooth((local_step - 600) / 99.0)
            if local_step < 700:
                grip_anchor = (
                    (1.0 - blend) * np.asarray(attach, dtype=np.float64)
                    + blend * hand_target
                )
            elif local_step < 1_140:
                grip_anchor = hand_target
            else:
                tau = (local_step - 1_140) * environment.dt
                grip_anchor = hand_target + np.asarray(
                    (0.0, 0.75 * tau * tau, 0.4 * tau * tau)
                )
        elif attach is not None and local_step == 1_200:
            joint_velocity = np.asarray(
                environment.data.joint(f"{target_body_name}_joint").qvel[:3]
            ).copy()
            environment.data.eq_active[self._equality_ids[target]] = 0
            self._restore_object_collision(target)
            mujoco.mj_forward(environment.model, environment.data)
            self._emit_once(
                events,
                emitted,
                context,
                ScenarioEventKind.GRIP_CONSTRAINT_RELEASED,
                metrics=(
                    ("linear_velocity_x_m_s", float(joint_velocity[0])),
                    ("linear_velocity_y_m_s", float(joint_velocity[1])),
                    ("linear_velocity_z_m_s", float(joint_velocity[2])),
                ),
            )
        if grip_anchor is not None:
            environment.data.mocap_pos[self._grip_mocap_id] = grip_anchor
            if (
                attach_relative_quaternion is not None
                and hold_relative_quaternion is not None
            ):
                hand_quaternion = np.asarray(
                    environment.data.body("right_hand").xquat,
                    dtype=np.float64,
                )
                relative_orientation = self._slerp_quaternions(
                    np.asarray(attach_relative_quaternion, dtype=np.float64),
                    np.asarray(hold_relative_quaternion, dtype=np.float64),
                    self._smooth((local_step - 600) / 99.0),
                )
                environment.data.mocap_quat[self._grip_mocap_id] = (
                    self._multiply_quaternions(
                        hand_quaternion,
                        relative_orientation,
                    )
                )

        action = np.zeros(environment.action_space.shape, dtype=np.float32)
        if local_step < 300:
            arm_reference = state.initial_arm_reference_rad
        elif local_step < 600:
            blend = self._smooth((local_step - 300) / 300.0)
            arm_reference = tuple(
                start + blend * (stop - start)
                for start, stop in zip(
                    state.initial_arm_reference_rad,
                    RIGHT_ARM_TARGET_RAD,
                    strict=True,
                )
            )
        elif local_step < 700:
            blend = self._smooth((local_step - 600) / 100.0)
            arm_reference = tuple(
                start + blend * (stop - start)
                for start, stop in zip(
                    RIGHT_ARM_TARGET_RAD,
                    LIFT_ARM_SEED_RAD,
                    strict=True,
                )
            )
        elif local_step < 1_200:
            arm_reference = LIFT_ARM_SEED_RAD
        elif local_step < 1_400:
            blend = self._smooth((local_step - 1_200) / 200.0)
            arm_reference = tuple(
                start + blend * (stop - start)
                for start, stop in zip(
                    LIFT_ARM_SEED_RAD,
                    state.initial_arm_reference_rad,
                    strict=True,
                )
            )
        else:
            arm_reference = state.initial_arm_reference_rad

        saturation_rows = state.saturation_rows
        for actuator_name, reference, stiffness, damping in zip(
            RIGHT_ARM_ACTUATOR_NAMES,
            arm_reference,
            RIGHT_ARM_STIFFNESS_NM_RAD,
            RIGHT_ARM_DAMPING_NM_S_RAD,
            strict=True,
        ):
            saturation_rows += self._command_pd(
                action,
                actuator_name,
                reference,
                stiffness,
                damping,
            )

        if 600 <= local_step < 1_200:
            closure = self._smooth((local_step - 599) / 100.0)
            closure *= ASSISTED_HOLD_DIGIT_CLOSURE_FRACTION[target]
            start_references = digit_start_references[cycle]
            if start_references is None:
                raise AssertionError("digit closure lacks its live start reference")
            for actuator_name, start_reference, terminal_reference in zip(
                RIGHT_DIGIT_ACTUATOR_NAMES,
                start_references,
                GRIP_DIGIT_REFERENCE_RAD[target],
                strict=True,
            ):
                channel = self._channels[actuator_name]
                saturation_rows += self._command_pd(
                    action,
                    actuator_name,
                    start_reference
                    + closure * (terminal_reference - start_reference),
                    0.20 * channel.gear,
                    0.01 * channel.gear,
                )

        if local_step < 300:
            sine = math.sin(2.0 * math.pi * local_time)
            for side, sign in (("right", 1.0), ("left", -1.0)):
                saturation_rows += self._command_pd(
                    action,
                    f"act:{side}_hip_flex",
                    -0.25 + 0.20 * sign * sine,
                    5.0,
                    0.5,
                )
                saturation_rows += self._command_pd(
                    action,
                    f"act:{side}_knee",
                    -0.28 - 0.30 * max(sign * sine, 0.0),
                    5.0,
                    0.5,
                )
                saturation_rows += self._command_pd(
                    action,
                    f"act:{side}_foot_flexion",
                    0.05 - 0.08 * sign * sine,
                    1.0,
                    0.12,
                )

        if local_step < 700:
            look_target = target
        elif local_step < 850:
            look_target = next_object(target)
        elif local_step < 1_000:
            look_target = next_object(next_object(target))
        else:
            look_target = target
        look_position = np.asarray(
            environment.data.body(SCENARIO_BODY_NAMES[look_target]).xpos
        )
        delta_local = ROOT_YAW_ROTATION.T @ (
            look_position
            - np.asarray(environment.data.body("head").xpos)
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
        saturation_rows += self._command_pd(
            action, "act:head_swivel", head_swivel, 1.0, 0.2
        )
        saturation_rows += self._command_pd(
            action, "act:head_tilt", head_tilt, 1.0, 0.2
        )
        head_tilt_side = self._smooth_knot_reference(
            local_step,
            HEAD_TILT_SIDE_KNOTS_RAD[cycle],
        )
        saturation_rows += self._command_pd(
            action,
            "act:head_tilt_side",
            head_tilt_side,
            2.0,
            0.5,
        )
        left_torsion = self._smooth_knot_reference(
            local_step,
            LEFT_EYE_TORSION_KNOTS_RAD[cycle],
        )
        for side, reference in (("left", left_torsion), ("right", -left_torsion)):
            saturation_rows += self._command_pd(
                action,
                f"act:{side}_eye_torsional",
                reference,
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
        direction /= np.linalg.norm(direction)
        forward_component = float(direction @ (-camera_rotation[:, 2]))
        horizontal_reference = math.atan2(
            float(direction @ camera_rotation[:, 0]), forward_component
        )
        vertical_reference = math.atan2(
            float(direction @ camera_rotation[:, 1]), forward_component
        )
        for side, horizontal_sign in (("left", -1.0), ("right", 1.0)):
            horizontal = self._channels[f"act:{side}_eye_horizontal"]
            vertical = self._channels[f"act:{side}_eye_vertical"]
            current_horizontal = float(
                environment.data.qpos[horizontal.qpos_address]
            )
            current_vertical = float(environment.data.qpos[vertical.qpos_address])
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
            clipped_horizontal_reference = float(
                np.clip(
                    commanded_horizontal,
                    horizontal_range[0] + 0.04,
                    horizontal_range[1] - 0.04,
                )
            )
            clipped_vertical_reference = float(
                np.clip(
                    commanded_vertical,
                    vertical_range[0] + 0.04,
                    vertical_range[1] - 0.04,
                )
            )
            saturation_rows += self._command_pd(
                action,
                f"act:{side}_eye_horizontal",
                clipped_horizontal_reference,
                0.02,
                0.002,
            )
            saturation_rows += self._command_pd(
                action,
                f"act:{side}_eye_vertical",
                clipped_vertical_reference,
                0.02,
                0.002,
            )
        hip_channel = self._channels["act:hip_bend"]
        action[hip_channel.action_index] = coupled_hip_bias_compensated_pd(
            hip_channel,
            environment.model,
            environment.data,
            reference_mean_position_rad=0.0,
            stiffness_nm_per_rad=10.0,
            damping_nm_s_per_rad=4.5,
        )

        if local_step == 699:
            angles = self._binocular_angles_deg(environment, target)
            self._emit_once(
                events,
                emitted,
                context,
                ScenarioEventKind.HELD_OBJECT_INSPECTED,
                subject=target,
                metrics=(
                    ("eye_left_angle_deg", angles[0]),
                    ("eye_right_angle_deg", angles[1]),
                ),
            )
        if local_step == 849:
            glance_target = next_object(target)
            angles = self._binocular_angles_deg(environment, glance_target)
            self._emit_once(
                events,
                emitted,
                context,
                ScenarioEventKind.NEXT_OBJECT_GLANCE_MEASURED,
                subject=glance_target,
                metrics=(
                    ("eye_left_angle_deg", angles[0]),
                    ("eye_right_angle_deg", angles[1]),
                ),
            )
        if local_step == 999:
            glance_target = next_object(next_object(target))
            angles = self._binocular_angles_deg(environment, glance_target)
            self._emit_once(
                events,
                emitted,
                context,
                ScenarioEventKind.SECOND_OTHER_OBJECT_GLANCE_MEASURED,
                subject=glance_target,
                metrics=(
                    ("eye_left_angle_deg", angles[0]),
                    ("eye_right_angle_deg", angles[1]),
                ),
            )

        next_state = ConstraintAssistedControllerState(
            step_index=state.step_index + 1,
            initial_arm_reference_rad=state.initial_arm_reference_rad,
            attach_positions_m=tuple(attach_positions),  # type: ignore[arg-type]
            inspect_positions_m=tuple(inspect_positions),  # type: ignore[arg-type]
            attach_relative_quaternions=tuple(  # type: ignore[arg-type]
                attach_relative_quaternions
            ),
            hold_relative_quaternions=tuple(  # type: ignore[arg-type]
                hold_relative_quaternions
            ),
            digit_start_references_rad=tuple(  # type: ignore[arg-type]
                digit_start_references
            ),
            emitted_events=frozenset(emitted),
            saturation_rows=saturation_rows,
        )
        return ControllerStep(
            action=action,
            state=next_state,
            events=tuple(events),
            diagnostics=(
                ("root_target_x_m", float(root_target[0])),
                ("root_target_y_m", float(root_target[1])),
                ("root_target_z_m", float(root_target[2])),
                (
                    "active_grip_welds",
                    float(
                        sum(
                            int(environment.data.eq_active[equality_id])
                            for equality_id in self._equality_ids.values()
                        )
                    ),
                ),
            ),
        )

    def _maybe_emit_floor_impact(
        self,
        environment: ThreeObjectScenarioEnv,
        state: ConstraintAssistedControllerState,
        context: ScenarioContext,
        events: list[ScenarioEvent],
        emitted: set[EventKey],
    ) -> None:
        local_step = context.step_index % 1_500
        key = (context.cycle_index, ScenarioEventKind.FLOOR_IMPACT)
        if local_step < 1_200 or key in state.emitted_events:
            return
        target_geom_id = self._object_geom_ids[context.target]
        for contact_index in range(environment.data.ncon):
            contact = environment.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if {geom1, geom2} != {target_geom_id, self._floor_geom_id}:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                environment.model,
                environment.data,
                contact_index,
                force,
            )
            position = np.asarray(
                environment.data.body(SCENARIO_BODY_NAMES[context.target]).xpos
            )
            self._emit_once(
                events,
                emitted,
                context,
                ScenarioEventKind.FLOOR_IMPACT,
                metrics=(
                    ("normal_force_n", max(float(force[0]), 0.0)),
                    ("position_x_m", float(position[0])),
                    ("position_y_m", float(position[1])),
                    ("position_z_m", float(position[2])),
                    (
                        "grip_active",
                        float(
                            environment.data.eq_active[
                                self._equality_ids[context.target]
                            ]
                        ),
                    ),
                ),
            )
            return

    def _command_pd(
        self,
        action: NDArray[np.float32],
        actuator_name: str,
        reference: float,
        stiffness: float,
        damping: float,
    ) -> int:
        channel: MotorChannel = self._channels[actuator_name]
        value = single_joint_bias_compensated_pd(
            channel,
            self._environment.data,
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

    def _binocular_angles_deg(
        self,
        environment: ThreeObjectScenarioEnv,
        subject: ScenarioObject,
    ) -> tuple[float, float]:
        subject_position = np.asarray(
            environment.data.body(SCENARIO_BODY_NAMES[subject]).xpos
        )
        angles: list[float] = []
        for eye in ("eye_left", "eye_right"):
            camera_id = int(environment.model.camera(eye).id)
            camera_matrix = np.asarray(
                environment.data.cam_xmat[camera_id]
            ).reshape(3, 3)
            forward = -camera_matrix[:, 2]
            direction = subject_position - np.asarray(
                environment.data.cam_xpos[camera_id]
            )
            angles.append(
                math.degrees(
                    math.acos(
                        np.clip(
                            float(
                                np.dot(forward, direction)
                                / np.linalg.norm(direction)
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        return angles[0], angles[1]

    @staticmethod
    def _emit_once(
        events: list[ScenarioEvent],
        emitted: set[EventKey],
        context: ScenarioContext,
        kind: ScenarioEventKind,
        *,
        subject: ScenarioObject | None = None,
        metrics: tuple[tuple[str, float], ...] = (),
    ) -> None:
        key = (context.cycle_index, kind)
        if key in emitted:
            return
        events.append(
            ScenarioEvent(
                kind=kind,
                step_index=context.step_index,
                time_seconds=context.nominal_time_seconds,
                cycle_index=context.cycle_index,
                target=context.target,
                subject=subject,
                metrics=metrics,
            )
        )
        emitted.add(key)

    @staticmethod
    def _smooth(value: float) -> float:
        bounded = float(np.clip(value, 0.0, 1.0))
        return 3.0 * bounded * bounded - 2.0 * bounded * bounded * bounded

    @staticmethod
    def _smooth_knot_reference(
        step_index: int,
        knots: tuple[tuple[int, float], ...],
    ) -> float:
        """Interpolate calibrated joint references with a C1 smoothstep."""

        if not knots:
            raise ValueError("reference knots must not be empty")
        if step_index <= knots[0][0]:
            return knots[0][1]
        for (start_step, start), (stop_step, stop) in zip(
            knots,
            knots[1:],
        ):
            if stop_step <= start_step:
                raise ValueError("reference knot steps must be strictly increasing")
            if step_index <= stop_step:
                fraction = (step_index - start_step) / (stop_step - start_step)
                blend = ConstraintAssistedThreeObjectController._smooth(fraction)
                return start + blend * (stop - start)
        return knots[-1][1]

    @staticmethod
    def _conjugate_quaternion(
        quaternion: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the inverse of one unit MuJoCo quaternion in WXYZ order."""

        return np.asarray(
            (
                quaternion[0],
                -quaternion[1],
                -quaternion[2],
                -quaternion[3],
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _multiply_quaternions(
        left: NDArray[np.float64],
        right: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Multiply WXYZ quaternions and return a normalized result."""

        result = np.empty(4, dtype=np.float64)
        mujoco.mju_mulQuat(result, left, right)
        return ConstraintAssistedThreeObjectController._normalize_quaternion(result)

    @staticmethod
    def _normalize_quaternion(
        quaternion: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return a finite unit WXYZ quaternion."""

        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("attachment quaternion must contain four finite values")
        result = np.asarray(quaternion, dtype=np.float64).copy()
        norm = float(np.linalg.norm(result))
        if norm <= 0.0:
            raise ValueError("attachment quaternion became invalid")
        return result / norm

    @staticmethod
    def _rotation_matrix_to_quaternion(
        rotation: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert one hand-local 3x3 rotation to a WXYZ quaternion."""

        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("assisted-hold rotation must be finite and 3x3")
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            quaternion,
            np.ascontiguousarray(rotation, dtype=np.float64).reshape(9),
        )
        return ConstraintAssistedThreeObjectController._normalize_quaternion(
            quaternion
        )

    @staticmethod
    def _slerp_quaternions(
        start: NDArray[np.float64],
        stop: NDArray[np.float64],
        fraction: float,
    ) -> NDArray[np.float64]:
        """Shortest-path spherical interpolation between unit WXYZ quaternions."""

        start_unit = ConstraintAssistedThreeObjectController._normalize_quaternion(
            start
        )
        stop_unit = ConstraintAssistedThreeObjectController._normalize_quaternion(
            stop
        )
        bounded = float(np.clip(fraction, 0.0, 1.0))
        dot = float(np.dot(start_unit, stop_unit))
        if dot < 0.0:
            stop_unit = -stop_unit
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return ConstraintAssistedThreeObjectController._normalize_quaternion(
                start_unit + bounded * (stop_unit - start_unit)
            )
        angle = math.acos(dot)
        denominator = math.sin(angle)
        result = (
            math.sin((1.0 - bounded) * angle) / denominator * start_unit
            + math.sin(bounded * angle) / denominator * stop_unit
        )
        return ConstraintAssistedThreeObjectController._normalize_quaternion(result)

    @staticmethod
    def _quaternion(values: NDArray[np.floating]) -> Quaternion:
        return (
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
        )

    @staticmethod
    def _vector3(values: NDArray[np.floating]) -> Vector3:
        return float(values[0]), float(values[1]), float(values[2])

    def _validate_environment(self, environment: ThreeObjectScenarioEnv) -> None:
        if environment is not self._environment or environment.model is not self._model:
            raise ValueError("controller belongs to a different environment")

    def _suppress_object_collisions(self, target: ScenarioObject) -> None:
        geom_id = self._object_geom_ids[target]
        self._model.geom_contype[geom_id] = 0
        self._model.geom_conaffinity[geom_id] = 0

    def _restore_object_collision(self, target: ScenarioObject) -> None:
        geom_id = self._object_geom_ids[target]
        contype, conaffinity = self._object_collision_masks[target]
        self._model.geom_contype[geom_id] = contype
        self._model.geom_conaffinity[geom_id] = conaffinity

    def _restore_object_collisions(self) -> None:
        for target in ScenarioObject:
            self._restore_object_collision(target)

    def close(self) -> None:
        """Restore collision masks changed by the assisted hold."""

        self._restore_object_collisions()


def constraint_assisted_three_object_controller_factory(
    environment: ThreeObjectScenarioEnv,
    schedule: ScenarioSchedule,
) -> ConstraintAssistedThreeObjectController:
    """Construct the exact name-resolved calibrated controller."""

    return ConstraintAssistedThreeObjectController(environment, schedule)

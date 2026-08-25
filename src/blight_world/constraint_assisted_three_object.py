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
from blight_world.supported_pick_place import LIFT_ARM_SEED_RAD
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


Vector3 = tuple[float, float, float]
EventKey = tuple[int, ScenarioEventKind]


@dataclass(frozen=True, slots=True)
class ConstraintAssistedControllerState:
    """Immutable deterministic state passed between public control steps."""

    step_index: int
    initial_arm_reference_rad: tuple[float, ...]
    attach_positions_m: tuple[Vector3 | None, Vector3 | None, Vector3 | None]
    inspect_positions_m: tuple[Vector3 | None, Vector3 | None, Vector3 | None]
    emitted_events: frozenset[EventKey]
    saturation_rows: int

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if len(self.initial_arm_reference_rad) != len(RIGHT_ARM_ACTUATOR_NAMES):
            raise ValueError("initial arm reference has the wrong length")
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
            attach_positions[cycle] = self._vector3(object_position)
            environment.data.mocap_pos[self._grip_mocap_id] = object_position
            environment.data.mocap_quat[
                self._grip_mocap_id
            ] = environment.data.body(target_body_name).xquat
            environment.data.eq_active[self._equality_ids[target]] = 1
            mujoco.mj_forward(environment.model, environment.data)
            inspect = station + ROOT_YAW_ROTATION @ np.asarray(
                (0.067, -0.257, 0.031), dtype=np.float64
            )
            inspect_positions[cycle] = self._vector3(inspect)
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
        if attach is not None and 600 <= local_step < 700:
            attach_array = np.asarray(attach, dtype=np.float64)
            inspect_array = np.asarray(inspect, dtype=np.float64)
            environment.data.mocap_pos[self._grip_mocap_id] = (
                attach_array
                + self._smooth((local_step - 600) / 100.0)
                * (inspect_array - attach_array)
            )
        elif attach is not None and 700 <= local_step < 1_140:
            environment.data.mocap_pos[self._grip_mocap_id] = inspect
        elif attach is not None and 1_140 <= local_step < 1_200:
            tau = (local_step - 1_140) * environment.dt
            environment.data.mocap_pos[self._grip_mocap_id] = np.asarray(
                inspect, dtype=np.float64
            ) + np.asarray((0.0, 0.75 * tau * tau, 0.4 * tau * tau))
        elif attach is not None and local_step == 1_200:
            joint_velocity = np.asarray(
                environment.data.joint(f"{target_body_name}_joint").qvel[:3]
            ).copy()
            environment.data.eq_active[self._equality_ids[target]] = 0
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

        for side, horizontal_sign in (("left", -1.0), ("right", 1.0)):
            camera_id = int(environment.model.camera(f"eye_{side}").id)
            camera_rotation = np.asarray(
                environment.data.cam_xmat[camera_id], dtype=np.float64
            ).reshape(3, 3)
            direction = look_position - np.asarray(
                environment.data.cam_xpos[camera_id], dtype=np.float64
            )
            direction /= np.linalg.norm(direction)
            forward_component = float(direction @ (-camera_rotation[:, 2]))
            horizontal_error = math.atan2(
                float(direction @ camera_rotation[:, 0]), forward_component
            )
            vertical_error = math.atan2(
                float(direction @ camera_rotation[:, 1]), forward_component
            )
            horizontal = self._channels[f"act:{side}_eye_horizontal"]
            vertical = self._channels[f"act:{side}_eye_vertical"]
            horizontal_range = environment.model.jnt_range[horizontal.joint_id]
            vertical_range = environment.model.jnt_range[vertical.joint_id]
            horizontal_reference = float(
                np.clip(
                    environment.data.qpos[horizontal.qpos_address]
                    + horizontal_sign
                    * np.clip(horizontal_error, -0.15, 0.15),
                    horizontal_range[0] + 0.04,
                    horizontal_range[1] - 0.04,
                )
            )
            vertical_reference = float(
                np.clip(
                    environment.data.qpos[vertical.qpos_address]
                    + np.clip(vertical_error, -0.15, 0.15),
                    vertical_range[0] + 0.04,
                    vertical_range[1] - 0.04,
                )
            )
            saturation_rows += self._command_pd(
                action,
                f"act:{side}_eye_horizontal",
                horizontal_reference,
                0.02,
                0.002,
            )
            saturation_rows += self._command_pd(
                action,
                f"act:{side}_eye_vertical",
                vertical_reference,
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
    def _vector3(values: NDArray[np.floating]) -> Vector3:
        return float(values[0]), float(values[1]), float(values[2])

    def _validate_environment(self, environment: ThreeObjectScenarioEnv) -> None:
        if environment is not self._environment or environment.model is not self._model:
            raise ValueError("controller belongs to a different environment")


def constraint_assisted_three_object_controller_factory(
    environment: ThreeObjectScenarioEnv,
    schedule: ScenarioSchedule,
) -> ConstraintAssistedThreeObjectController:
    """Construct the exact name-resolved calibrated controller."""

    return ConstraintAssistedThreeObjectController(environment, schedule)

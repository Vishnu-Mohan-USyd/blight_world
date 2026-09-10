"""Scripted running strides with explicit pelvis balance and pacing assistance.

The existing bounded SpringDamper motors move the limbs, and MuJoCo resolves
foot contacts. External root forces assist balance, height and forward pace;
this demonstration is not a learned or independently balancing running policy.
"""

from __future__ import annotations

from copy import copy
from math import acos, atan2, cos, pi, sin
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from blight_world.environment import BlightWorldEnv
from blight_world.motor_control import resolve_motor_channels


STRIDE_SECONDS = 0.60
STANCE_FRACTION = 0.396
SETTLE_SECONDS = 0.60
RAMP_SECONDS = 0.90
RECORDING_START_SECONDS = 2.0
SUPPORT_LABEL = (
    "adult-reference torque gait with external pelvis balance/height/pacing forces; "
    "demo sagittal joint damping: hip 0.40, knee 0.18, ankle 0.06 N m s/rad"
)

# Fukuchi et al. (2017), RBDS001 at 2.5 m/s, CC BY 4.0:
# https://figshare.com/articles/dataset/A_public_data_set_of_running_biomechanics_and_the_effects_of_running_speed_on_lower_extremity_kinematics_and_kinetics/4543435
# Global sagittal hip, knee and ankle angles reconstructed from measured markers,
# sampled every 5% of a same-foot stride. Mirror one cycle for a symmetric demo.
# This is an adult-reference engineering retarget, not measured infant running.
_LEG_CYCLE_DEG = np.array([
    [-15.1761, -8.2028, -9.4827], [-16.1675, -18.3255, -1.2338],
    [-16.6130, -30.3451, 11.2748], [-14.5411, -37.1270, 19.3193],
    [-8.4421, -36.4613, 23.1844], [0.3080, -30.0003, 20.6425],
    [9.3136, -21.8593, 9.7380], [16.4488, -15.0788, -6.8913],
    [20.0616, -13.3226, -21.9781], [19.4105, -20.1595, -27.6529],
    [15.0178, -34.5257, -20.6271], [8.6509, -50.5782, -14.3748],
    [0.8310, -65.6201, -8.5959], [-8.3483, -76.7669, -2.5630],
    [-17.4415, -81.1109, 1.2255], [-23.5244, -76.9939, 3.9135],
    [-25.5035, -65.0452, 4.9128], [-23.6428, -46.4852, 1.0311],
    [-18.8192, -24.3195, -5.5360], [-16.0238, -10.6317, -7.6796],
])
_LEG_CYCLE_DEG[:, 0] -= 0.5
# The rigid MIMo sole needs a small heel-first angle and a truly flat loading
# segment; retain the measured plantarflexion/toe-off and recovery afterwards.
_LEG_CYCLE_DEG[0, 2] = _LEG_CYCLE_DEG[0, 0] - _LEG_CYCLE_DEG[0, 1] + 5.0
for _index in (1, 2):
    _LEG_CYCLE_DEG[_index, 2] = _LEG_CYCLE_DEG[_index, 0] - _LEG_CYCLE_DEG[_index, 1]
_LEG_CYCLE = np.deg2rad(_LEG_CYCLE_DEG)


def _leg_reference(phase: float) -> NDArray[np.float64]:
    """Periodic cubic interpolation of the coordinated measured joint cycle."""

    coordinate = (phase % 1.0) * len(_LEG_CYCLE)
    index = int(coordinate)
    u = coordinate - index
    p0, p1, p2, p3 = (_LEG_CYCLE[i % len(_LEG_CYCLE)] for i in (
        index - 1, index, index + 1, index + 2,
    ))
    return 0.5 * ((2*p1) + (-p0+p2)*u + (2*p0-5*p1+4*p2-p3)*u*u
                  + (-p0+3*p1-3*p2+p3)*u*u*u)


def _contact_height(phase: float) -> float:
    """Pelvis height placing the retargeted rigid foot's lowest point on z=0."""

    hip, knee, ankle = _leg_reference(phase)
    pitch = hip - knee - ankle
    foot_z = -0.007 - 0.135*cos(hip) - 0.162*cos(hip-knee) - 0.015*cos(pitch)
    s, c = sin(pitch), cos(pitch)
    # Native heel cylinder, midfoot box, forefoot cylinder and neutral toe tips.
    lowest = min(
        foot_z + 0.007*s - 0.0249*abs(s) - 0.0101*abs(c),
        foot_z - 0.023*s - 0.030*abs(s) - 0.010*abs(c),
        foot_z - 0.053*s - 0.0101,
        foot_z - 0.0625*s - 0.0095*abs(s) - 0.0099*abs(c),
        foot_z - 0.072*s - 0.010,
    )
    return -lowest


def _running_height(phase: float) -> float:
    half_phase = phase % 0.5
    if half_phase <= STANCE_FRACTION:
        return _contact_height(half_phase)
    u = (half_phase - STANCE_FRACTION) / (0.5 - STANCE_FRACTION)
    return ((1-u)*_contact_height(STANCE_FRACTION) + u*_contact_height(0.0)
            + 0.006 * 4*u*(1-u))


def _pace(time_seconds: float, speed: float) -> tuple[float, float, float]:
    """Integrated smooth speed ramp: displacement, speed, and ramp fraction."""

    elapsed = max(0.0, time_seconds - SETTLE_SECONDS)
    u = min(elapsed / RAMP_SECONDS, 1.0)
    fraction = u * u * (3.0 - 2.0 * u)
    distance = speed * (
        RAMP_SECONDS * (u**3 - 0.5 * u**4)
        + max(0.0, elapsed - RAMP_SECONDS)
    )
    return distance, speed * fraction, fraction


def _height_reference(time_seconds: float) -> float:
    _, _, ramp = _pace(time_seconds, 1.0)
    phase = (time_seconds - SETTLE_SECONDS) / STRIDE_SECONDS
    return (1-ramp)*0.316 + ramp*_running_height(phase)


def _leg_angles(ankle_x: float, ankle_down: float) -> tuple[float, float, float]:
    """Planar MIMo v2 IK using its actual hip/knee/ankle pivot distances."""

    thigh, shank = 0.135, 0.162
    bend = acos(float(np.clip(
        (ankle_x**2 + ankle_down**2 - thigh**2 - shank**2)
        / (2.0 * thigh * shank), -0.995, 0.995,
    )))
    hip = atan2(-ankle_x, ankle_down) - atan2(
        shank * sin(bend), thigh + shank * cos(bend),
    )
    knee = -bend
    return hip, knee, hip - knee


class StreetRunEnv(BlightWorldEnv):
    """A +X road run assisted by forces, with ordinary MuJoCo limb contacts.

    Pass the street MJCF through ``scene_template`` and use the inherited vision
    and audition options. Only reset writes the initial pose. During stepping,
    this subclass owns ``xfrc_applied[hip]`` and applies the explicitly
    declared balance/pacing wrench once per physics substep.
    """

    def __init__(self, *, speed_m_s: float = 1.0, **kwargs: Any) -> None:
        if not np.isfinite(speed_m_s) or speed_m_s <= 0.0:
            raise ValueError("speed_m_s must be positive and finite")
        self.speed_m_s = float(speed_m_s)
        kwargs.setdefault("snapshot_sensors", True)
        super().__init__(**kwargs)

    def _initialize_simulation(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        model, data = super()._initialize_simulation()
        root = model.joint("mimo_orientation")
        self._root_qpos = int(root.qposadr[0])
        self._root_dof = int(root.dofadr[0])
        self._root_body = int(model.body("hip").id)
        self._body_mass = float(model.body_subtreemass[self._root_body])
        self._foot_geoms = {
            side: tuple(int(model.geom(f"geom:{side}_{name}").id) for name in (
                "foot1", "foot2", "foot3", "big_toe1", "big_toe2", "toes1", "toes2",
            ))
            for side in ("right", "left")
        }
        self._upright_quaternion = np.array((cos(pi/60), 0.0, sin(pi/60), 0.0))
        data.qpos[self._root_qpos + 2] = 0.316
        data.qpos[self._root_qpos + 3:self._root_qpos + 7] = self._upright_quaternion
        self._origin = data.qpos[self._root_qpos:self._root_qpos + 3].copy()
        neutral_leg = _leg_angles(0.0, 0.316 - 0.007 - 0.0251)
        for side in ("right", "left"):
            for suffix, value in zip(("hip1", "knee", "foot1"), neutral_leg):
                if suffix == "hip1":
                    value -= pi/30
                data.qpos[int(model.joint(f"robot:{side}_{suffix}").qposadr[0])] = value
            # Native passive resistance exceeds the available motors at running
            # rates (ankle: .60 Nms/rad against only +1.89 Nm). Demo-only values;
            # retain all original stiffness, armature, ranges and torque limits.
            for suffix, damping in (("hip1", 0.40), ("knee", 0.18), ("foot1", 0.06)):
                model.dof_damping[int(model.joint(f"robot:{side}_{suffix}").dofadr[0])] = damping
            data.qpos[int(model.joint(f"robot:{side}_elbow").qposadr[0])] = -1.30899694
            data.qpos[int(model.joint(f"robot:{side}_shoulder_ad_ab").qposadr[0])] = 0.34906585
        mujoco.mj_forward(model, data)
        model.vis.map.znear = 0.005 / model.stat.extent
        return model, data

    def reset_model(self) -> dict[str, NDArray[Any]]:
        self.data.xfrc_applied[self._root_body] = 0.0
        return super().reset_model()

    def _foot_clearance(self, side: str) -> float:
        """Lowest actual foot surface above the flat road."""

        if self.sensor_context is not None:
            data = self.sensor_context.data
        else:
            data = copy(self.data)
            mujoco.mj_kinematics(self.model, data)
        lowest_point = np.array((0.0, 0.0, np.inf))
        for geom_id in self._foot_geoms[side]:
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            up = rotation[2]
            size = self.model.geom_size[geom_id]
            if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                local_point = -np.sign(up) * size
            else:  # All remaining native foot/toe geoms are cylinders.
                radial = float(np.linalg.norm(up[:2]))
                local_point = np.array((0.0, 0.0, -np.sign(up[2]) * size[1]))
                if radial > 1e-12:
                    local_point[:2] = -size[0] * up[:2] / radial
            point = data.geom_xpos[geom_id] + rotation @ local_point
            if point[2] < lowest_point[2]:
                lowest_point = point
        return float(lowest_point[2])

    def _single_mujoco_step(self) -> None:
        time_seconds = float(self.data.time)
        displacement, velocity, ramp = _pace(time_seconds, self.speed_m_s)
        phase = (time_seconds - SETTLE_SECONDS) / STRIDE_SECONDS
        height = _height_reference(time_seconds)
        height_acceleration = (
            _height_reference(time_seconds + 0.01) - 2*height
            + _height_reference(time_seconds - 0.01)
        ) / 0.01**2
        height_velocity = ramp * (
            _running_height(phase + 0.001/STRIDE_SECONDS) - _running_height(phase)
        ) / 0.001
        position = self.data.qpos[self._root_qpos:self._root_qpos + 3]
        linear_velocity = self.data.qvel[self._root_dof:self._root_dof + 3]
        cycle_phase = phase % 1.0
        stance_side = (
            "right" if cycle_phase < STANCE_FRACTION else
            "left" if 0.5 <= cycle_phase < 0.5 + STANCE_FRACTION else None
        )
        if stance_side is not None and ramp >= 1.0:
            clearance = self._foot_clearance(stance_side)
            # Correct the measured stance gap without giving up pelvis support
            # when the loaded leg yields. Flight keeps the original reference.
            height += float(np.clip(-3.0 * (clearance + 0.005), -0.015, 0.005))
        vertical_force = (
            self._body_mass * (9.81 + height_acceleration)
            + 2500.0 * (height - position[2])
            + 150.0 * (height_velocity - linear_velocity[2])
        )
        force = np.array((
            300.0 * (self._origin[0] + displacement - position[0])
            + 80.0 * (velocity - linear_velocity[0]),
            600.0 * (self._origin[1] - position[1]) - 90.0 * linear_velocity[1],
            vertical_force,
        ))
        quaternion = self.data.qpos[self._root_qpos + 3:self._root_qpos + 7]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quaternion)
        error = np.empty(3)
        mujoco.mju_subQuat(error, self._upright_quaternion, quaternion)
        angular_velocity = self.data.qvel[self._root_dof + 3:self._root_dof + 6]
        torque = rotation.reshape(3, 3) @ (80.0 * error - 2.0 * angular_velocity)
        self.data.xfrc_applied[self._root_body, :3] = np.clip(
            force, (-80.0, -80.0, -60.0), (80.0, 80.0, 220.0),
        )
        self.data.xfrc_applied[self._root_body, 3:] = np.clip(torque, -40.0, 40.0)
        super()._single_mujoco_step()


class StreetRunController:
    """Alternating foot trajectories and arm counter-swing through 90 motors."""

    def __init__(self, environment: StreetRunEnv) -> None:
        self.env = environment
        self.channels = resolve_motor_channels(environment)
        self._by_name = {channel.actuator_name: channel for channel in self.channels}
        self._sagittal_actuators = tuple(
            f"act:{side}_{joint}"
            for side in ("right", "left")
            for joint in ("hip_flex", "knee", "foot_flexion")
        )
        self._neutral = {
            channel.actuator_name: float(environment.init_qpos[channel.qpos_address])
            for channel in self.channels
        }

    def _references(self, time_seconds: float) -> dict[str, float]:
        _, _, ramp = _pace(time_seconds, self.env.speed_m_s)
        targets = self._neutral.copy()
        phase = (time_seconds - SETTLE_SECONDS) / STRIDE_SECONDS
        root_matrix = np.empty(9)
        mujoco.mju_quat2Mat(
            root_matrix,
            self.env.data.qpos[self.env._root_qpos+3:self.env._root_qpos+7],
        )
        pelvis_pitch = atan2(root_matrix.reshape(3, 3)[0, 2], root_matrix.reshape(3, 3)[2, 2])
        for side, offset in (("right", 0.0), ("left", 0.5)):
            angles = _leg_reference(phase + offset)
            angles[0] -= pelvis_pitch
            for suffix, value in zip(("hip_flex", "knee", "foot_flexion"), angles):
                name = f"act:{side}_{suffix}"
                targets[name] = (1.0 - ramp) * self._neutral[name] + ramp * value
            targets[f"act:{side}_hip_abduction"] = -0.035
            swing = -ramp * cos(2.0 * pi * (phase + offset))
            # Matched shoulder/internal rotation keeps the bent arms at the sides.
            targets[f"act:{side}_shoulder_horizontal"] = 0.38218692 * swing
            targets[f"act:{side}_shoulder_internal"] = -0.44402543 * swing
            targets[f"act:{side}_shoulder_abduction"] = 0.34906585 + 0.05070927 * swing**2
        for name, value in targets.items():
            channel = self._by_name[name]
            limits = self.env.model.jnt_range[channel.joint_id]
            targets[name] = float(np.clip(value, limits[0] + 0.005, limits[1] - 0.005))
        return targets

    def action(self) -> NDArray[np.float32]:
        """Return the bounded public-step torque action without writing state."""

        data = self.env.sensor_context.data if self.env.sensor_context else self.env.data
        time_seconds = float(self.env.data.time)
        targets = self._references(time_seconds)
        future = self._references(time_seconds + 0.001)
        past = self._references(time_seconds - 0.001)
        reference_acceleration = np.zeros(self.env.model.nv)
        for name in self._sagittal_actuators:
            channel = self._by_name[name]
            reference_acceleration[channel.dof_address] = (
                future[name] - 2.0 * targets[name] + past[name]
            ) / 0.001**2
        # The assisted pelvis acceleration also loads the articulated legs.
        reference_acceleration[self.env._root_dof + 2] = (
            _height_reference(time_seconds + 0.01)
            - 2.0 * _height_reference(time_seconds)
            + _height_reference(time_seconds - 0.01)
        ) / 0.01**2
        inertial_torque = np.empty(self.env.model.nv)
        mujoco.mj_mulM(self.env.model, data, inertial_torque, reference_acceleration)
        action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        for channel in self.channels:
            name = channel.actuator_name
            q = float(data.qpos[channel.qpos_address])
            velocity = float(data.qvel[channel.dof_address])
            bias = float(data.qfrc_bias[channel.dof_address] - data.qfrc_passive[channel.dof_address])
            stiffness, damping = 2.0, 0.2
            if "_hip_" in name:
                stiffness, damping = 35.0, 0.5
            elif name.endswith("_knee"):
                stiffness, damping = 35.0, 0.3
            elif "_foot_" in name:
                stiffness, damping = 12.0, 0.15
            elif "shoulder" in name or name.endswith("_elbow"):
                stiffness, damping = 6.0, 0.4
            elif "eye_" in name:
                stiffness, damping = 0.008, 0.0004
            elif "head_" in name:
                stiffness, damping = 4.0, 0.4
            elif "_toes" in name or "_big_toe" in name:
                stiffness, damping = 0.2, 0.02
            if name in ("act:hip_bend", "act:hip_lean", "act:hip_twist"):
                second_name, ratio = {
                    "act:hip_bend": ("robot:hip_bend2", 1.0),
                    "act:hip_lean": ("robot:hip_lean2", 2.0),
                    "act:hip_twist": ("robot:hip_rot2", 1.0),
                }[name]
                second = self.env.model.joint(second_name)
                dof = int(second.dofadr[0])
                bias += ratio * float(data.qfrc_bias[dof] - data.qfrc_passive[dof])
                stiffness, damping = 40.0, 4.0
            desired_velocity = (future[name] - targets[name]) / 0.001
            torque = bias + stiffness * (targets[name] - q) + damping * (desired_velocity - velocity)
            if name in self._sagittal_actuators:
                torque += inertial_torque[channel.dof_address]
            action[channel.action_index] = np.clip(
                torque / channel.gear, channel.lower_bound, channel.upper_bound,
            )
        return action

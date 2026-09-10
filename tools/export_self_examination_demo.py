#!/usr/bin/env python3
"""Export a short root-supported first-person self-examination episode.

MIMo looks at and gently moves its right hand, left hand, right lower leg,
and left lower leg in one uninterrupted ten-second episode.  Motion is driven
only through named SpringDamper action channels.  The world weld supporting
the root is explicit; this is supported scripted motion, not balance or a
learned policy.  Video is the two unmodified public eye observations.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from blight_world.environment import BlightWorldEnv
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
from mimo_audition import AuditionConfig

from export_ball_throw import (
    EYE_INTERIOR_GEOM_NAMES,
    _public_binocular_frame,
    _segmentation_pixels,
)
from export_continuous_embodied_demo import (
    _git_state,
    _sha256_file,
    _validate_ffmpeg,
)
from export_three_object_walk_pick_throw import (
    _encode_playback,
    _finish_video_encoder,
    _start_video_encoder,
    _validate_media,
)


DEFAULT_OUTPUT_DIRECTORY = Path(
    "/home/vishnu/neuroips_outputs/self_examination"
)
EYE_WIDTH = 320
EYE_HEIGHT = 240
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 240
VIDEO_FRAME_RATE_HZ = 50
AUDIO_SAMPLE_RATE_HZ = 48_000
AUDIO_BLOCK_SAMPLES = 480
PUBLIC_STEP_SECONDS = 0.01
TOTAL_PUBLIC_STEPS = 1_000
DURATION_SECONDS = TOTAL_PUBLIC_STEPS * PUBLIC_STEP_SECONDS
VIDEO_FRAMES = TOTAL_PUBLIC_STEPS // 2
AUDIO_SAMPLES_PER_EAR = TOTAL_PUBLIC_STEPS * AUDIO_BLOCK_SAMPLES
SEED = 1_401
DEFAULT_MONITOR_GAIN = 0.04

SELF_EXAMINATION_SCENE_XML = """<equality>
    <weld name="self_examination_root_support" body1="mimo_location"
          solref=".01 1" solimp=".99 .999 .001 .5 2"/>
  </equality>"""
SUPPORT_LABEL = "demo-only physical world weld supporting MIMo's root"

ARM_SUFFIXES = tuple(
    name.removeprefix("act:right_") for name in RIGHT_ARM_ACTUATOR_NAMES
)
RIGHT_ARM_NAMES = RIGHT_ARM_ACTUATOR_NAMES
LEFT_ARM_NAMES = tuple(f"act:left_{suffix}" for suffix in ARM_SUFFIXES)
EXAMINATION_ARM_TARGET_RAD = (
    *RIGHT_ARM_TARGET_RAD[:3],
    math.radians(-70.0),
    *RIGHT_ARM_TARGET_RAD[4:],
)
LEG_SUFFIXES = ("hip_flex", "knee", "foot_flexion")
RIGHT_LEG_NAMES = tuple(f"act:right_{suffix}" for suffix in LEG_SUFFIXES)
LEFT_LEG_NAMES = tuple(f"act:left_{suffix}" for suffix in LEG_SUFFIXES)
LEG_REFERENCE_DELTAS_RAD = (0.12, -0.18, 0.08)
LEG_STIFFNESS_NM_RAD = (5.0, 5.0, 1.0)
LEG_DAMPING_NM_S_RAD = (0.5, 0.5, 0.12)
HIP_STIFFNESS_NM_RAD = 10.0
HIP_DAMPING_NM_S_RAD = 4.5

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Uint8Array = NDArray[np.uint8]
FrameSink = Callable[[bytes], None]


class SelfExaminationEnv(BlightWorldEnv):
    """BlightWorld with one explicit root-support weld and no task controller."""

    _scene_extension_xml = SELF_EXAMINATION_SCENE_XML

    def _initialize_simulation(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Use a 5 mm near plane for the close views of MIMo's body."""

        model, data = super()._initialize_simulation()
        model.vis.map.znear = 0.005 / model.stat.extent
        return model, data


@dataclass(frozen=True, slots=True)
class ExaminationSegment:
    """One half-open public-step interval in the uninterrupted episode."""

    name: str
    start_step: int
    stop_step: int
    target_body: str | None
    target_root_body: str | None
    moving_side: str | None
    moving_kind: str | None

    @property
    def midpoint_step(self) -> int:
        """Return the odd post-step evidence sample at the interval midpoint."""

        return (self.start_step + self.stop_step) // 2


SEGMENTS = (
    ExaminationSegment("neutral_settle", 0, 120, None, None, None, None),
    ExaminationSegment(
        "right_hand", 120, 310, "right_hand", "right_hand", "right", "arm"
    ),
    ExaminationSegment(
        "left_hand", 310, 500, "left_hand", "left_hand", "left", "arm"
    ),
    ExaminationSegment(
        "right_lower_leg_foot",
        500,
        690,
        "right_foot",
        "right_lower_leg",
        "right",
        "leg",
    ),
    ExaminationSegment(
        "left_lower_leg_foot",
        690,
        880,
        "left_foot",
        "left_lower_leg",
        "left",
        "leg",
    ),
    ExaminationSegment("neutral_finish", 880, 1_000, None, None, None, None),
)
EVIDENCE_SEGMENTS = tuple(
    segment for segment in SEGMENTS if segment.target_root_body is not None
)
EVIDENCE_STEPS = {
    segment.midpoint_step: segment for segment in EVIDENCE_SEGMENTS
}


@dataclass(frozen=True, slots=True)
class SelfExaminationCommand:
    """One bounded action and its segment-level audit metadata."""

    action: Float32Array
    segment_name: str
    saturated_actuators: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Public-eye visibility and body motion at one segment midpoint."""

    segment_name: str
    public_step: int
    target_body: str
    body_displacement_m: float
    left_target_pixels: int
    right_target_pixels: int
    left_eye_interior_pixels: int
    right_eye_interior_pixels: int


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """In-memory public sensor and physical evidence for one episode."""

    pressure_channels_first: Float32Array
    frame_count: int
    raw_video_sha256: str
    action_sha256: str
    qpos_sha256: str
    audio_transport_sha256: str
    first_frame_sha256: str
    final_frame_sha256: str
    binocular_equal_frame_count: int
    evidence: tuple[EvidenceRecord, ...]
    maximum_saturated_actuators_per_step: int


@dataclass(frozen=True, slots=True)
class OutputPaths:
    """Files exclusively owned by this exporter."""

    output_directory: Path
    playback_mp4: Path
    pressure_wav: Path
    manifest_json: Path


class SelfExaminationController:
    """Sparse name-resolved PD control for the supported scripted sequence."""

    def __init__(self, environment: SelfExaminationEnv) -> None:
        self.environment = environment
        self.channels = {
            channel.actuator_name: channel
            for channel in resolve_motor_channels(environment)
        }
        self.initial_references = {
            name: float(environment.data.qpos[self.channels[name].qpos_address])
            for name in (
                *RIGHT_ARM_NAMES,
                *LEFT_ARM_NAMES,
                *RIGHT_LEG_NAMES,
                *LEFT_LEG_NAMES,
                "act:hip_lean",
                "act:hip_twist",
            )
        }
        hip = self.channels["act:hip_bend"]
        second_hip = environment.model.joint("robot:hip_bend2")
        self.hip_reference = 0.5 * (
            float(environment.data.qpos[hip.qpos_address])
            + float(
                environment.data.qpos[
                    int(np.asarray(second_hip.qposadr).item())
                ]
            )
        )

    def compose(self, step_index: int) -> SelfExaminationCommand:
        """Compose one action without writing MuJoCo state or applied force."""

        segment = segment_at(step_index)
        envelope = movement_envelope(step_index, segment)
        action = np.zeros(self.environment.action_space.shape, dtype=np.float32)
        saturated: list[str] = []

        for side, names in (("right", RIGHT_ARM_NAMES), ("left", LEFT_ARM_NAMES)):
            side_envelope = envelope if (
                segment.moving_kind == "arm" and segment.moving_side == side
            ) else 0.0
            for name, target, stiffness, damping in zip(
                names,
                EXAMINATION_ARM_TARGET_RAD,
                RIGHT_ARM_STIFFNESS_NM_RAD,
                RIGHT_ARM_DAMPING_NM_S_RAD,
                strict=True,
            ):
                initial = self.initial_references[name]
                reference = initial + side_envelope * (target - initial)
                self._command_pd(
                    action, name, reference, stiffness, damping, saturated
                )

        for side, names in (("right", RIGHT_LEG_NAMES), ("left", LEFT_LEG_NAMES)):
            side_envelope = envelope if (
                segment.moving_kind == "leg" and segment.moving_side == side
            ) else 0.0
            for name, delta, stiffness, damping in zip(
                names,
                LEG_REFERENCE_DELTAS_RAD,
                LEG_STIFFNESS_NM_RAD,
                LEG_DAMPING_NM_S_RAD,
                strict=True,
            ):
                initial = self.initial_references[name]
                reference = self._safe_joint_reference(
                    self.channels[name], initial + side_envelope * delta
                )
                self._command_pd(
                    action, name, reference, stiffness, damping, saturated
                )

        hip = self.channels["act:hip_bend"]
        hip_reference = self.hip_reference + (
            0.35 * envelope if segment.moving_kind == "leg" else 0.0
        )
        action[hip.action_index] = np.float32(
            coupled_hip_bias_compensated_pd(
                hip,
                self.environment.model,
                self.environment.data,
                reference_mean_position_rad=hip_reference,
                stiffness_nm_per_rad=HIP_STIFFNESS_NM_RAD,
                damping_nm_s_per_rad=HIP_DAMPING_NM_S_RAD,
            )
        )
        for name, second_joint_name, ratio in (
            ("act:hip_lean", "robot:hip_lean2", 2.0),
            ("act:hip_twist", "robot:hip_rot2", 1.0),
        ):
            channel = self.channels[name]
            second_joint = self.environment.model.joint(second_joint_name)
            second_qpos = int(np.asarray(second_joint.qposadr).item())
            second_dof = int(np.asarray(second_joint.dofadr).item())
            data = self.environment.data
            mean_position = 0.5 * (
                float(data.qpos[channel.qpos_address])
                + float(data.qpos[second_qpos]) / ratio
            )
            mean_velocity = 0.5 * (
                float(data.qvel[channel.dof_address])
                + float(data.qvel[second_dof]) / ratio
            )
            # The tendon couples q2 = ratio * q1, including its bias torque.
            torque = (
                float(data.qfrc_bias[channel.dof_address])
                + ratio * float(data.qfrc_bias[second_dof])
                + HIP_STIFFNESS_NM_RAD
                * (self.initial_references[name] - mean_position)
                - HIP_DAMPING_NM_S_RAD * mean_velocity
            )
            value = float(np.clip(
                torque / channel.gear, channel.lower_bound, channel.upper_bound
            ))
            action[channel.action_index] = np.float32(value)
            if min(value - channel.lower_bound, channel.upper_bound - value) < 1e-6:
                saturated.append(name)
        self._command_gaze(action, self._look_position(segment), saturated)
        if not self.environment.action_space.contains(action):
            raise AssertionError(f"invalid action at step {step_index}")
        return SelfExaminationCommand(
            action=np.ascontiguousarray(action),
            segment_name=segment.name,
            saturated_actuators=tuple(sorted(set(saturated))),
        )

    def _look_position(self, segment: ExaminationSegment) -> Float64Array:
        if segment.target_body is not None:
            if segment.moving_kind == "leg":
                lower_leg = np.asarray(
                    self.environment.data.body(
                        f"{segment.moving_side}_lower_leg"
                    ).xpos,
                    dtype=np.float64,
                )
                foot = np.asarray(
                    self.environment.data.body(segment.target_body).xpos,
                    dtype=np.float64,
                )
                return 0.5 * (lower_leg + foot)
            return np.asarray(
                self.environment.data.body(segment.target_body).xpos,
                dtype=np.float64,
            ).copy()
        root = self.environment.data.body("mimo_location")
        root_rotation = np.asarray(root.xmat, dtype=np.float64).reshape(3, 3)
        return (
            np.asarray(self.environment.data.body("head").xpos, dtype=np.float64)
            + root_rotation @ np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        )

    def _command_gaze(
        self,
        action: Float32Array,
        look_position: Float64Array,
        saturated: list[str],
    ) -> None:
        environment = self.environment
        upper_body_rotation = np.asarray(
            environment.data.body("upper_body").xmat, dtype=np.float64
        ).reshape(3, 3)
        head_pivot = np.asarray(
            environment.data.xanchor[self.channels["act:head_swivel"].joint_id],
            dtype=np.float64,
        )
        delta_local = upper_body_rotation.T @ (look_position - head_pivot)
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
        self._command_pd(
            action, "act:head_swivel", head_swivel, 1.0, 0.2, saturated
        )
        self._command_pd(
            action, "act:head_tilt", head_tilt, 1.0, 0.2, saturated
        )
        self._command_pd(
            action, "act:head_tilt_side", 0.0, 2.0, 0.5, saturated
        )
        for side in ("left", "right"):
            self._command_pd(
                action,
                f"act:{side}_eye_torsional",
                0.0,
                0.002,
                0.0004,
                saturated,
            )

        for side, horizontal_sign in (("left", -1.0), ("right", 1.0)):
            camera_id = int(environment.model.camera(f"eye_{side}").id)
            camera_rotation = np.asarray(
                environment.data.cam_xmat[camera_id], dtype=np.float64
            ).reshape(3, 3)
            direction = np.asarray(look_position, dtype=np.float64) - np.asarray(
                environment.data.cam_xpos[camera_id], dtype=np.float64
            )
            distance = float(np.linalg.norm(direction))
            if distance <= 1e-9:
                raise AssertionError(f"gaze target coincides with {side} eye")
            direction /= distance
            forward = float(direction @ (-camera_rotation[:, 2]))
            horizontal_error = math.atan2(
                float(direction @ camera_rotation[:, 0]), forward
            )
            vertical_error = math.atan2(
                float(direction @ camera_rotation[:, 1]), forward
            )
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
                    horizontal_sign * horizontal_error,
                    -0.15,
                    0.15,
                )
            )
            commanded_vertical = current_vertical + float(
                np.clip(
                    vertical_error, -0.15, 0.15
                )
            )
            horizontal_reference_clipped = self._safe_joint_reference(
                horizontal, commanded_horizontal, margin=0.04
            )
            vertical_reference_clipped = self._safe_joint_reference(
                vertical, commanded_vertical, margin=0.04
            )
            self._command_pd(
                action,
                f"act:{side}_eye_horizontal",
                horizontal_reference_clipped,
                0.02,
                0.002,
                saturated,
            )
            self._command_pd(
                action,
                f"act:{side}_eye_vertical",
                vertical_reference_clipped,
                0.02,
                0.002,
                saturated,
            )

    def _command_pd(
        self,
        action: Float32Array,
        actuator_name: str,
        reference: float,
        stiffness: float,
        damping: float,
        saturated: list[str],
    ) -> None:
        channel = self.channels[actuator_name]
        value = single_joint_bias_compensated_pd(
            channel,
            self.environment.data,
            reference_position_rad=reference,
            stiffness_nm_per_rad=stiffness,
            damping_nm_s_per_rad=damping,
        )
        action[channel.action_index] = np.float32(value)
        if min(
            value - channel.lower_bound,
            channel.upper_bound - value,
        ) < 1e-6:
            saturated.append(actuator_name)

    def _safe_joint_reference(
        self,
        channel: MotorChannel,
        reference: float,
        *,
        margin: float = 1e-4,
    ) -> float:
        limits = self.environment.model.jnt_range[channel.joint_id]
        return float(np.clip(reference, limits[0] + margin, limits[1] - margin))


def segment_at(step_index: int) -> ExaminationSegment:
    """Resolve one public step to its single half-open segment."""

    if not 0 <= step_index < TOTAL_PUBLIC_STEPS:
        raise IndexError("self-examination step is outside the episode")
    return next(
        segment
        for segment in SEGMENTS
        if segment.start_step <= step_index < segment.stop_step
    )


def movement_envelope(
    step_index: int, segment: ExaminationSegment
) -> float:
    """Return a C1 raise/hold/lower envelope for one movement segment."""

    if segment.moving_kind is None:
        return 0.0
    ramp_steps = 60
    local_step = step_index - segment.start_step
    remaining = segment.stop_step - step_index
    if local_step < ramp_steps:
        return _smoothstep((local_step + 1) / ramp_steps)
    if remaining <= ramp_steps:
        return _smoothstep(remaining / ramp_steps)
    return 1.0


def _smoothstep(value: float) -> float:
    clipped = float(np.clip(value, 0.0, 1.0))
    return float(np.clip(clipped * clipped * (3.0 - 2.0 * clipped), 0.0, 1.0))


def _descendant_geom_ids(
    model: mujoco.MjModel, root_body_name: str
) -> tuple[int, ...]:
    root_body_id = int(model.body(root_body_name).id)
    body_ids: set[int] = set()
    for body_id in range(model.nbody):
        current = body_id
        while current != 0:
            if current == root_body_id:
                body_ids.add(body_id)
                break
            current = int(model.body_parentid[current])
    geom_ids = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    )
    if not geom_ids:
        raise AssertionError(f"{root_body_name} subtree has no geoms")
    return geom_ids


def output_paths(output_directory: Path) -> OutputPaths:
    output = output_directory.resolve()
    return OutputPaths(
        output,
        output / "self_examination.mp4",
        output / "self_examination_pressure_pa.wav",
        output / "self_examination_manifest.json",
    )


def _prepare_output(paths: OutputPaths, *, overwrite: bool) -> None:
    paths.output_directory.mkdir(parents=True, exist_ok=True)
    existing = tuple(
        path
        for path in (paths.playback_mp4, paths.pressure_wav, paths.manifest_json)
        if path.exists()
    )
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite exporter-owned files: "
            + ", ".join(map(str, existing))
        )


def run_episode(
    *,
    width: int,
    height: int,
    frame_sink: FrameSink | None = None,
) -> EpisodeResult:
    """Run the uninterrupted supported episode and validate public sensors."""

    environment = SelfExaminationEnv(
        width=width,
        height=height,
        audition=AuditionConfig.quality_48k(),
    )
    segmentation_renderer: mujoco.Renderer | None = None
    try:
        observation, info = environment.reset(seed=SEED)
        if info != {} or not environment.observation_space.contains(observation):
            raise AssertionError("initial public observation contract changed")
        controller = SelfExaminationController(environment)
        support_id = int(
            environment.model.equality("self_examination_root_support").id
        )
        if int(environment.data.eq_active[support_id]) != 1:
            raise AssertionError("root-support weld is inactive")
        body_equalities = [
            equality_id
            for equality_id in range(environment.model.neq)
            if int(environment.model.eq_type[equality_id])
            in {
                int(mujoco.mjtEq.mjEQ_CONNECT),
                int(mujoco.mjtEq.mjEQ_WELD),
            }
        ]
        if body_equalities != [support_id]:
            raise AssertionError(
                f"unexpected body equality set: {body_equalities}"
            )

        segmentation_renderer = mujoco.Renderer(
            environment.model, height=height, width=width
        )
        segmentation_renderer.enable_segmentation_rendering()
        interior_ids = tuple(
            int(environment.model.geom(name).id)
            for name in EYE_INTERIOR_GEOM_NAMES
        )
        target_geom_ids = {
            segment.name: _descendant_geom_ids(
                environment.model, segment.target_root_body  # type: ignore[arg-type]
            )
            for segment in EVIDENCE_SEGMENTS
        }
        baseline_positions = {
            segment.name: np.asarray(
                environment.data.body(segment.target_body).xpos,  # type: ignore[arg-type]
                dtype=np.float64,
            ).copy()
            for segment in EVIDENCE_SEGMENTS
        }
        baseline_mocap_pos = environment.data.mocap_pos.copy()
        baseline_mocap_quat = environment.data.mocap_quat.copy()
        baseline_eq_active = environment.data.eq_active.copy()

        audio_blocks: list[Float32Array] = []
        action_digest = sha256()
        qpos_digest = sha256()
        audio_digest = sha256()
        video_digest = sha256()
        first_frame_sha = ""
        final_frame_sha = ""
        frame_count = 0
        binocular_equal = 0
        evidence: list[EvidenceRecord] = []
        maximum_saturated = 0

        for step_index in range(TOTAL_PUBLIC_STEPS):
            qpos_before = environment.data.qpos.copy()
            qvel_before = environment.data.qvel.copy()
            xfrc_before = environment.data.xfrc_applied.copy()
            command = controller.compose(step_index)
            if (
                not np.array_equal(environment.data.qpos, qpos_before)
                or not np.array_equal(environment.data.qvel, qvel_before)
                or not np.array_equal(environment.data.xfrc_applied, xfrc_before)
                or not np.array_equal(environment.data.mocap_pos, baseline_mocap_pos)
                or not np.array_equal(environment.data.mocap_quat, baseline_mocap_quat)
                or not np.array_equal(environment.data.eq_active, baseline_eq_active)
            ):
                raise AssertionError("controller mutated MuJoCo state")
            action_digest.update(command.action.tobytes(order="C"))
            maximum_saturated = max(
                maximum_saturated, len(command.saturated_actuators)
            )
            observation, _, terminated, truncated, _ = environment.step(
                command.action
            )
            if terminated or truncated:
                raise AssertionError(f"episode ended at step {step_index}")
            if not environment.observation_space.contains(observation):
                raise AssertionError(f"invalid public observation at {step_index}")
            if (
                np.any(environment.data.xfrc_applied != 0.0)
                or not np.array_equal(environment.data.mocap_pos, baseline_mocap_pos)
                or not np.array_equal(environment.data.mocap_quat, baseline_mocap_quat)
                or not np.array_equal(environment.data.eq_active, baseline_eq_active)
            ):
                raise AssertionError("runtime force, mocap, or equality state changed")
            qpos_digest.update(
                np.ascontiguousarray(
                    environment.data.qpos, dtype=np.float64
                ).tobytes(order="C")
            )
            audio = np.ascontiguousarray(
                observation["audition"], dtype=np.float32
            )
            if (
                audio.shape != (2, AUDIO_BLOCK_SAMPLES)
                or not audio.flags.c_contiguous
                or not np.all(np.isfinite(audio))
            ):
                raise AssertionError(f"invalid public audio at step {step_index}")
            audio_blocks.append(audio.copy())
            audio_digest.update(audio.tobytes(order="C"))

            if step_index % 2 == 1:
                left, right, stereo = _public_binocular_frame(
                    observation, width=width, height=height
                )
                binocular_equal += int(np.array_equal(left, right))
                frame_bytes = stereo.tobytes(order="C")
                if frame_sink is not None:
                    frame_sink(frame_bytes)
                video_digest.update(frame_bytes)
                frame_sha = sha256(frame_bytes).hexdigest()
                if frame_count == 0:
                    first_frame_sha = frame_sha
                final_frame_sha = frame_sha
                frame_count += 1

                segment = EVIDENCE_STEPS.get(step_index)
                if segment is not None:
                    pixels: dict[str, tuple[int, int]] = {}
                    for eye_name in ("eye_left", "eye_right"):
                        segmentation_renderer.update_scene(
                            environment.data, camera=eye_name
                        )
                        segmentation = np.asarray(
                            segmentation_renderer.render()
                        ).copy()
                        pixels[eye_name] = (
                            _segmentation_pixels(
                                segmentation, target_geom_ids[segment.name]
                            ),
                            _segmentation_pixels(segmentation, interior_ids),
                        )
                    target_position = np.asarray(
                        environment.data.body(segment.target_body).xpos,  # type: ignore[arg-type]
                        dtype=np.float64,
                    )
                    displacement = float(
                        np.linalg.norm(
                            target_position - baseline_positions[segment.name]
                        )
                    )
                    record = EvidenceRecord(
                        segment_name=segment.name,
                        public_step=step_index,
                        target_body=segment.target_body or "",
                        body_displacement_m=displacement,
                        left_target_pixels=pixels["eye_left"][0],
                        right_target_pixels=pixels["eye_right"][0],
                        left_eye_interior_pixels=pixels["eye_left"][1],
                        right_eye_interior_pixels=pixels["eye_right"][1],
                    )
                    if min(record.left_target_pixels, record.right_target_pixels) <= 0:
                        raise AssertionError(
                            f"{segment.name} is absent from a public eye: {record}"
                        )
                    if (
                        record.left_eye_interior_pixels != 0
                        or record.right_eye_interior_pixels != 0
                    ):
                        raise AssertionError(
                            f"head/eye interior visible at {segment.name}: {record}"
                        )
                    minimum_displacement = (
                        0.03 if segment.moving_kind == "arm" else 0.005
                    )
                    if displacement < minimum_displacement:
                        raise AssertionError(
                            f"{segment.name} movement is not measurable: {record}"
                        )
                    evidence.append(record)

        pressure = np.ascontiguousarray(
            np.concatenate(audio_blocks, axis=1), dtype=np.float32
        )
        if pressure.shape != (2, AUDIO_SAMPLES_PER_EAR):
            raise AssertionError("public pressure shape changed")
        if not np.all(np.isfinite(pressure)):
            raise AssertionError("public pressure contains non-finite values")
        if frame_count != VIDEO_FRAMES or len(evidence) != len(EVIDENCE_SEGMENTS):
            raise AssertionError("episode evidence or frame count is incomplete")
        if binocular_equal:
            raise AssertionError("public binocular frames became identical")
        return EpisodeResult(
            pressure_channels_first=pressure,
            frame_count=frame_count,
            raw_video_sha256=video_digest.hexdigest(),
            action_sha256=action_digest.hexdigest(),
            qpos_sha256=qpos_digest.hexdigest(),
            audio_transport_sha256=audio_digest.hexdigest(),
            first_frame_sha256=first_frame_sha,
            final_frame_sha256=final_frame_sha,
            binocular_equal_frame_count=binocular_equal,
            evidence=tuple(evidence),
            maximum_saturated_actuators_per_step=maximum_saturated,
        )
    finally:
        if segmentation_renderer is not None:
            segmentation_renderer.close()
        environment.close()


def _result_record(result: EpisodeResult) -> dict[str, Any]:
    pressure = result.pressure_channels_first
    return {
        "clock": {
            "duration_seconds": DURATION_SECONDS,
            "public_steps": TOTAL_PUBLIC_STEPS,
            "video_frames": result.frame_count,
            "audio_samples_per_ear": pressure.shape[1],
        },
        "behavior": {
            "label": "root_supported_scripted_self_examination",
            "support": SUPPORT_LABEL,
            "learned_policy": False,
            "runtime_qpos_qvel_mocap_xfrc_writes": False,
            "segments": [asdict(segment) for segment in SEGMENTS],
            "maximum_saturated_actuators_per_step": (
                result.maximum_saturated_actuators_per_step
            ),
        },
        "public_sources": {
            "video": "public_eye_left_plus_public_eye_right",
            "camera_overrides": False,
            "audio": "public_audition_float32_pressure_pa",
            "binocular_equal_frame_count": result.binocular_equal_frame_count,
            "evidence": [asdict(record) for record in result.evidence],
            "raw_video_sha256": result.raw_video_sha256,
            "first_frame_sha256": result.first_frame_sha256,
            "final_frame_sha256": result.final_frame_sha256,
            "audio_transport_sha256": result.audio_transport_sha256,
            "pressure_channels_first_sha256": sha256(
                pressure.tobytes(order="C")
            ).hexdigest(),
            "raw_peak_pressure_pa": float(np.max(np.abs(pressure))),
            "raw_rms_pressure_pa": float(
                np.sqrt(np.mean(pressure.astype(np.float64) ** 2))
            ),
        },
        "physics_hashes": {
            "action_sha256": result.action_sha256,
            "qpos_sha256": result.qpos_sha256,
        },
    }


def run_smoke() -> dict[str, Any]:
    """Run the complete behavior at a smaller public-eye resolution."""

    return _result_record(run_episode(width=128, height=96))


def export_self_examination(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    overwrite: bool = False,
    monitor_gain: float = DEFAULT_MONITOR_GAIN,
) -> dict[str, Any]:
    """Run, encode, validate, and atomically publish the demonstration."""

    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("set MUJOCO_GL=egl before starting the exporter")
    paths = output_paths(output_directory)
    _prepare_output(paths, overwrite=overwrite)
    repository_root = Path(__file__).resolve().parents[1]
    ffmpeg_tools = _validate_ffmpeg(repository_root)
    video_encoder: subprocess.Popen[bytes] | None = None
    with TemporaryDirectory(
        prefix="self-examination-", dir=paths.output_directory
    ) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        video_only = temporary_directory / "video_only.mp4"
        temporary_wav = temporary_directory / paths.pressure_wav.name
        temporary_mp4 = temporary_directory / paths.playback_mp4.name
        temporary_manifest = temporary_directory / paths.manifest_json.name
        try:
            video_encoder = _start_video_encoder(
                video_only, ffmpeg_tools["ffmpeg_path"], repository_root
            )
            if video_encoder.stdin is None:
                raise RuntimeError("video encoder stdin was not created")
            stream: BinaryIO = video_encoder.stdin
            result = run_episode(
                width=EYE_WIDTH, height=EYE_HEIGHT, frame_sink=stream.write
            )
            _finish_video_encoder(video_encoder)
            video_encoder = None
            pressure = result.pressure_channels_first
            peak = float(np.max(np.abs(pressure)))
            if monitor_gain <= 0.0 or monitor_gain * peak >= 1.0:
                raise AssertionError(
                    f"fixed monitor gain would clip public pressure: {monitor_gain * peak}"
                )
            interleaved = np.ascontiguousarray(pressure.T, dtype=np.dtype("<f4"))
            wavfile.write(temporary_wav, AUDIO_SAMPLE_RATE_HZ, interleaved)
            wav_rate, wav_pressure = wavfile.read(temporary_wav)
            if (
                wav_rate != AUDIO_SAMPLE_RATE_HZ
                or wav_pressure.dtype != np.float32
                or not np.array_equal(wav_pressure, interleaved)
            ):
                raise AssertionError("lossless public pressure WAV changed")
            _encode_playback(
                video_only,
                temporary_wav,
                temporary_mp4,
                monitor_gain=monitor_gain,
                duration_seconds=DURATION_SECONDS,
                ffmpeg=ffmpeg_tools["ffmpeg_path"],
                repository_root=repository_root,
            )
            media = _validate_media(
                temporary_mp4,
                expected_duration_seconds=DURATION_SECONDS,
                expected_frames=VIDEO_FRAMES,
                ffmpeg_tools=ffmpeg_tools,
                repository_root=repository_root,
            )
            manifest = {
                "title": "BlightWorld supported self-examination",
                "seed": SEED,
                "monitor_gain": monitor_gain,
                "monitor_mapping": "1 Pa equals 1 full-scale unit before fixed gain",
                "headphones_recommended": True,
                "result": _result_record(result),
                "media_validation": media,
                "provenance": {
                    "script_sha256": _sha256_file(Path(__file__).resolve()),
                    "git": _git_state(repository_root),
                    "ffmpeg": ffmpeg_tools,
                    "mp4_sha256": _sha256_file(temporary_mp4),
                    "wav_sha256": _sha256_file(temporary_wav),
                },
            }
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_mp4.replace(paths.playback_mp4)
            temporary_wav.replace(paths.pressure_wav)
            temporary_manifest.replace(paths.manifest_json)
            return manifest
        finally:
            if video_encoder is not None:
                if video_encoder.stdin is not None:
                    video_encoder.stdin.close()
                video_encoder.terminate()
                video_encoder.wait(timeout=10)


def export_unreal_self_examination(
    output_directory: Path, *, overwrite: bool, spear_config: Path
) -> Path:
    """Render the existing supported controller through the two Unreal eyes."""
    import cv2
    from blight_world.unreal_vision import make_unreal_vision_provider

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "limb_sequence.mp4"
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    os.environ["BLIGHT_UNREAL_CONFIG"] = str(spear_config.resolve())
    repository_root = Path(__file__).resolve().parents[1]
    ffmpeg = _validate_ffmpeg(repository_root)["ffmpeg_path"]
    environment = SelfExaminationEnv(
        width=EYE_WIDTH,
        height=EYE_HEIGHT,
        vision_provider_factory=make_unreal_vision_provider,
    )
    encoder: subprocess.Popen[bytes] | None = None
    try:
        environment.reset(seed=SEED)
        controller = SelfExaminationController(environment)
        encoder = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", "640x272",
             "-framerate", str(VIDEO_FRAME_RATE_HZ), "-i", "-", "-an",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert encoder.stdin is not None
        labels = {
            "neutral_settle": "NEUTRAL",
            "right_hand": "RIGHT HAND",
            "left_hand": "LEFT HAND",
            "right_lower_leg_foot": "RIGHT LOWER LEG / FOOT",
            "left_lower_leg_foot": "LEFT LOWER LEG / FOOT",
            "neutral_finish": "NEUTRAL",
        }
        frame_count = 0
        for step_index in range(TOTAL_PUBLIC_STEPS):
            command = controller.compose(step_index)
            observation, _, terminated, truncated, _ = environment.step(command.action)
            if terminated or truncated:
                raise RuntimeError(f"Episode ended at public step {step_index}")
            if step_index % 2 == 0:
                continue
            _, _, stereo = _public_binocular_frame(
                observation, width=EYE_WIDTH, height=EYE_HEIGHT
            )
            canvas = np.zeros((272, 640, 3), dtype=np.uint8)
            canvas[32:] = stereo
            cv2.putText(canvas, labels[command.segment_name], (8, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(canvas, "LEFT EYE", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (190, 190, 190), 1,
                        cv2.LINE_AA)
            cv2.putText(canvas, "RIGHT EYE", (328, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (190, 190, 190), 1,
                        cv2.LINE_AA)
            encoder.stdin.write(canvas.tobytes())
            if step_index in EVIDENCE_STEPS:
                name = EVIDENCE_STEPS[step_index].name
                cv2.imwrite(str(output_directory / f"{name}.png"),
                            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Unreal video: {frame_count}/{VIDEO_FRAMES} frames", flush=True)
        _finish_video_encoder(encoder)
        encoder = None
        if frame_count != VIDEO_FRAMES:
            raise RuntimeError(f"Expected {VIDEO_FRAMES} frames; got {frame_count}")
        return output_path
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--renderer", choices=("mujoco", "unreal"), default="mujoco")
    parser.add_argument("--spear-config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config/spear_unreal.yaml")
    arguments = parser.parse_args()
    if arguments.renderer == "unreal":
        path = export_unreal_self_examination(
            arguments.output, overwrite=arguments.overwrite,
            spear_config=arguments.spear_config,
        )
        print(f"wrote {path}")
        return
    if arguments.smoke:
        print(json.dumps(run_smoke(), indent=2, sort_keys=True))
        return
    manifest = export_self_examination(
        arguments.output, overwrite=arguments.overwrite
    )
    paths = output_paths(arguments.output)
    print(f"wrote {paths.playback_mp4}")
    print(f"wrote {paths.pressure_wav}")
    print(f"wrote {paths.manifest_json}")
    print(json.dumps(manifest["result"]["public_sources"]["evidence"], sort_keys=True))


if __name__ == "__main__":
    main()

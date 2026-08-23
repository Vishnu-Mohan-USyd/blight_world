"""Export a deterministic binocular demonstration driven by MuJoCo physics.

The five-second storyboard stages three objects once, then advances only through
the public Gymnasium ``step`` API.  A finite force pulse moves the block and a
second pulse rolls the cylinder; gravity and contact dynamics govern every
subsequent object pose.  Video is made directly from MIMo's public left- and
right-eye observations, and audio is made directly from the public audition
observation.  The torso support and neutral hip/head posture stabilization are
scripted demo fixtures, not walking, autonomy, or learned behavior.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import json
from math import atan2, degrees, sqrt
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from blight_world import AuditionConfig, BlightWorldEnv, ContinuousSoundConfig
import mimo_audition.renderer as audition_renderer_module
from mimo_audition.sources import (
    Attachment,
    BodyAttachment,
    SiteAttachment,
    StaticAttachment,
)


Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Uint8Array = NDArray[np.uint8]

DEMO_SEED = 1_401
DURATION_SECONDS = 5.0
PHYSICS_TIMESTEP_SECONDS = 0.005
ENVIRONMENT_STEP_SECONDS = 0.010
PUBLIC_STEPS = 500
PHYSICS_SUBSTEPS = 1_000
EYE_WIDTH = 320
EYE_HEIGHT = 240
VIDEO_WIDTH = EYE_WIDTH * 2
VIDEO_HEIGHT = EYE_HEIGHT
VIDEO_FRAME_RATE_HZ = 50
VIDEO_FRAMES = 250
AUDIO_SAMPLE_RATE_HZ = 48_000
AUDIO_BLOCK_SAMPLES = 480
AUDIO_BLOCKS = PUBLIC_STEPS
AUDIO_SAMPLES_PER_EAR = 240_000
SAMPLES_PER_VIDEO_FRAME = 960

BALL_INITIAL_POSITION_M = (1.15, 0.0, 1.20)
BLOCK_INITIAL_POSITION_M = (0.65, 0.30, 0.699)
CYLINDER_INITIAL_POSITION_M = (0.65, -0.30, 0.699)
IDENTITY_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
CYLINDER_QUATERNION_WXYZ = (sqrt(0.5), sqrt(0.5), 0.0, 0.0)

BLOCK_FORCE_START_SECONDS = 1.25
BLOCK_FORCE_STOP_SECONDS = 1.45
BLOCK_FORCE_X_N = 1.0
CYLINDER_FORCE_START_SECONDS = 2.40
CYLINDER_FORCE_STOP_SECONDS = 2.55
CYLINDER_FORCE_X_N = 0.30
HIP_POSTURE_STIFFNESS_NM_RAD = 10.0
HIP_POSTURE_DAMPING_NM_S_RAD = 4.5
HIP_ACTUATOR_GEAR = 10.58
HEAD_POSTURE_STIFFNESS_NM_RAD = 1.0
HEAD_POSTURE_DAMPING_NM_S_RAD = 0.20
HEAD_ACTUATOR_GEAR = 3.3

TABLE_REST_CENTER_Z_M = 0.700
TABLE_X_BOUNDS_M = (0.55, 1.45)
TABLE_Y_BOUNDS_M = (-0.55, 0.55)
EXPECTED_REBOUND_CLEARANCES_M = (0.075555, 0.001735, 0.000062)
TARGET_VISIBILITY_WINDOWS = {
    "ball": (0, 31),
    "block": (62, 86),
    "cylinder": (120, 141),
}
TARGET_MINIMUM_MASK_PIXELS = {
    "ball": 192,
    "block": 768,
    "cylinder": 768,
}

DEMO_WORLD_SUPPORT_XML = """<equality>
    <weld name="demo_mimo_root_support" body1="mimo_location"
          solref=".01 1" solimp=".99 .999 .001 .5 2"/>
  </equality>"""


class _BinocularPhysicsDemoEnv(BlightWorldEnv):
    """BlightWorld variant with the explicitly demo-only torso support weld."""

    _scene_extension_xml = DEMO_WORLD_SUPPORT_XML


@dataclass(frozen=True, slots=True)
class DemoPaths:
    """Final artifact paths for one export."""

    output_directory: Path
    master_mkv: Path
    playback_mp4: Path
    pressure_wav: Path
    manifest_json: Path


@dataclass(frozen=True, slots=True)
class DemoAddresses:
    """Resolved MuJoCo identifiers needed after one-time state staging."""

    object_qpos_dof: dict[str, tuple[int, int]]
    object_body_ids: dict[str, int]
    root_body_id: int
    hip_qpos_addresses: tuple[int, int]
    hip_dof_addresses: tuple[int, int]
    hip_reference_rad: float
    hip_actuator_id: int
    hip_action_index: int
    head_qpos_address: int
    head_dof_address: int
    head_reference_rad: float
    head_actuator_id: int
    head_action_index: int


def _scalar_int(value: Any) -> int:
    """Extract one integer from MuJoCo's one-element address arrays."""

    return int(np.asarray(value).item())


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """Run a text-mode subprocess and preserve complete failure evidence."""

    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {rendered}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _paths(output_directory: Path) -> DemoPaths:
    output_directory = output_directory.resolve()
    return DemoPaths(
        output_directory=output_directory,
        master_mkv=output_directory / "blight_world_binocular_physics_master.mkv",
        playback_mp4=output_directory / "blight_world_binocular_physics.mp4",
        pressure_wav=(
            output_directory / "blight_world_binocular_physics_pressure_pa.wav"
        ),
        manifest_json=(
            output_directory / "blight_world_binocular_physics_validation.json"
        ),
    )


def _prepare_output(paths: DemoPaths, *, overwrite: bool) -> None:
    paths.output_directory.mkdir(parents=True, exist_ok=True)
    for path in (
        paths.master_mkv,
        paths.playback_mp4,
        paths.pressure_wav,
        paths.manifest_json,
    ):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")


def _validate_ffmpeg(repository_root: Path) -> dict[str, str]:
    """Fail closed unless ffmpeg, ffprobe, and all required codecs exist."""

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise FileNotFoundError("ffmpeg and ffprobe must both be available on PATH")
    version = _run_checked((ffmpeg, "-hide_banner", "-version"), cwd=repository_root)
    encoders = _run_checked((ffmpeg, "-hide_banner", "-encoders"), cwd=repository_root)
    for encoder in ("ffv1", "libx264", "pcm_f32le", "aac"):
        if encoder not in encoders.stdout:
            raise RuntimeError(f"required ffmpeg encoder is unavailable: {encoder}")
    return {
        "ffmpeg_path": str(Path(ffmpeg).resolve()),
        "ffprobe_path": str(Path(ffprobe).resolve()),
        "ffmpeg_version": version.stdout.splitlines()[0],
    }


def _git_state(repository_root: Path) -> dict[str, Any]:
    commit = _run_checked(
        ("git", "rev-parse", "HEAD"),
        cwd=repository_root,
    ).stdout.strip()
    branch = _run_checked(
        ("git", "branch", "--show-current"),
        cwd=repository_root,
    ).stdout.strip()
    status = _run_checked(
        ("git", "status", "--short"),
        cwd=repository_root,
    ).stdout.splitlines()
    return {"commit": commit, "branch": branch, "status_short": status}


def make_demo_environment() -> BlightWorldEnv:
    """Create the fixed-resolution demo environment with physical torso support."""

    return _BinocularPhysicsDemoEnv(
        width=EYE_WIDTH,
        height=EYE_HEIGHT,
        audition=AuditionConfig.quality_48k(
            continuous=ContinuousSoundConfig()
        ),
    )


def _free_joint_addresses(
    environment: BlightWorldEnv,
    body_name: str,
) -> tuple[int, int]:
    joint = environment.model.joint(f"{body_name}_joint")
    return _scalar_int(joint.qposadr), _scalar_int(joint.dofadr)


def _stage_demo_state(
    environment: BlightWorldEnv,
    *,
    seed: int = DEMO_SEED,
) -> DemoAddresses:
    """Stage object poses exactly once before the public-step storyboard."""

    environment.reset(seed=seed)
    qpos = np.asarray(environment.init_qpos, dtype=np.float64).copy()
    qvel = np.zeros_like(environment.init_qvel)
    states = {
        "ball": (BALL_INITIAL_POSITION_M, IDENTITY_QUATERNION_WXYZ),
        "block": (BLOCK_INITIAL_POSITION_M, IDENTITY_QUATERNION_WXYZ),
        "cylinder": (CYLINDER_INITIAL_POSITION_M, CYLINDER_QUATERNION_WXYZ),
    }
    object_qpos_dof: dict[str, tuple[int, int]] = {}
    object_body_ids: dict[str, int] = {}
    for name, (position, quaternion) in states.items():
        qpos_address, dof_address = _free_joint_addresses(environment, name)
        object_qpos_dof[name] = (qpos_address, dof_address)
        object_body_ids[name] = _scalar_int(environment.model.body(name).id)
        qpos[qpos_address : qpos_address + 3] = position
        qpos[qpos_address + 3 : qpos_address + 7] = quaternion
        qvel[dof_address : dof_address + 6] = 0.0

    environment.set_state(qpos, qvel)
    environment.data.time = 0.0
    environment.data.ctrl.fill(0.0)
    environment.data.qfrc_applied.fill(0.0)
    environment.data.xfrc_applied.fill(0.0)
    environment.data.qacc_warmstart.fill(0.0)
    mujoco.mj_forward(environment.model, environment.data)
    if environment.audition is None:
        raise RuntimeError("the binocular physics demo requires audition")
    environment.audition.reset(seed)

    hip_joints = tuple(
        environment.model.joint(name)
        for name in ("robot:hip_bend1", "robot:hip_bend2")
    )
    hip_qpos_addresses = tuple(_scalar_int(joint.qposadr) for joint in hip_joints)
    hip_dof_addresses = tuple(_scalar_int(joint.dofadr) for joint in hip_joints)
    hip_reference_rad = float(
        np.mean(environment.data.qpos[list(hip_qpos_addresses)])
    )
    head_joint = environment.model.joint("robot:head_tilt")
    head_qpos_address = _scalar_int(head_joint.qposadr)
    head_dof_address = _scalar_int(head_joint.dofadr)
    head_reference_rad = float(environment.data.qpos[head_qpos_address])

    hip_actuator_id = _scalar_int(environment.model.actuator("act:hip_bend").id)
    head_actuator_id = _scalar_int(environment.model.actuator("act:head_tilt").id)
    action_indices: dict[str, int] = {}
    for label, actuator_id in (
        ("hip", hip_actuator_id),
        ("head", head_actuator_id),
    ):
        matches = np.flatnonzero(
            np.asarray(environment.mimo_actuators) == actuator_id
        )
        if matches.shape != (1,):
            raise AssertionError(f"{label} actuator did not map to one action")
        action_indices[label] = _scalar_int(matches)
    return DemoAddresses(
        object_qpos_dof=object_qpos_dof,
        object_body_ids=object_body_ids,
        root_body_id=_scalar_int(environment.model.body("mimo_location").id),
        hip_qpos_addresses=hip_qpos_addresses,  # type: ignore[arg-type]
        hip_dof_addresses=hip_dof_addresses,  # type: ignore[arg-type]
        hip_reference_rad=hip_reference_rad,
        hip_actuator_id=hip_actuator_id,
        hip_action_index=action_indices["hip"],
        head_qpos_address=head_qpos_address,
        head_dof_address=head_dof_address,
        head_reference_rad=head_reference_rad,
        head_actuator_id=head_actuator_id,
        head_action_index=action_indices["head"],
    )


def force_commands(time_seconds: float) -> tuple[float, float]:
    """Return the exact block/cylinder +X forces for one public step."""

    block = (
        BLOCK_FORCE_X_N
        if BLOCK_FORCE_START_SECONDS <= time_seconds < BLOCK_FORCE_STOP_SECONDS
        else 0.0
    )
    cylinder = (
        CYLINDER_FORCE_X_N
        if CYLINDER_FORCE_START_SECONDS
        <= time_seconds
        < CYLINDER_FORCE_STOP_SECONDS
        else 0.0
    )
    return block, cylinder


def _apply_step_inputs(
    environment: BlightWorldEnv,
    addresses: DemoAddresses,
    time_seconds: float,
) -> tuple[Float32Array, tuple[float, float]]:
    """Apply finite object forces and neutral hip/head posture torques."""

    environment.data.xfrc_applied.fill(0.0)
    block_force, cylinder_force = force_commands(time_seconds)
    environment.data.xfrc_applied[
        addresses.object_body_ids["block"], 0
    ] = block_force
    environment.data.xfrc_applied[
        addresses.object_body_ids["cylinder"], 0
    ] = cylinder_force
    action = np.zeros(environment.action_space.shape, dtype=np.float32)
    hip_qpos = float(
        np.mean(environment.data.qpos[list(addresses.hip_qpos_addresses)])
    )
    hip_qvel = float(
        np.mean(environment.data.qvel[list(addresses.hip_dof_addresses)])
    )
    hip_torque_nm = (
        float(np.sum(environment.data.qfrc_bias[list(addresses.hip_dof_addresses)]))
        + HIP_POSTURE_STIFFNESS_NM_RAD
        * (addresses.hip_reference_rad - hip_qpos)
        - HIP_POSTURE_DAMPING_NM_S_RAD * hip_qvel
    )
    head_qpos = float(environment.data.qpos[addresses.head_qpos_address])
    head_qvel = float(environment.data.qvel[addresses.head_dof_address])
    head_torque_nm = (
        float(environment.data.qfrc_bias[addresses.head_dof_address])
        + HEAD_POSTURE_STIFFNESS_NM_RAD
        * (addresses.head_reference_rad - head_qpos)
        - HEAD_POSTURE_DAMPING_NM_S_RAD * head_qvel
    )
    action[addresses.hip_action_index] = np.float32(
        np.clip(
            hip_torque_nm / HIP_ACTUATOR_GEAR,
            environment.action_space.low[addresses.hip_action_index],
            environment.action_space.high[addresses.hip_action_index],
        )
    )
    action[addresses.head_action_index] = np.float32(
        np.clip(
            head_torque_nm / HEAD_ACTUATOR_GEAR,
            environment.action_space.low[addresses.head_action_index],
            environment.action_space.high[addresses.head_action_index],
        )
    )
    if not environment.action_space.contains(action):
        raise AssertionError("neutral posture action left the environment action space")
    return action, (block_force, cylinder_force)


def _pair_id(
    environment: BlightWorldEnv,
    geom_name1: str,
    geom_name2: str,
) -> int:
    expected = {
        _scalar_int(environment.model.geom(geom_name1).id),
        _scalar_int(environment.model.geom(geom_name2).id),
    }
    matches = [
        pair_id
        for pair_id in range(environment.model.npair)
        if {
            int(environment.model.pair_geom1[pair_id]),
            int(environment.model.pair_geom2[pair_id]),
        }
        == expected
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one explicit pair for {geom_name1}/{geom_name2}, got {matches}"
        )
    return matches[0]


def _validate_model(environment: BlightWorldEnv) -> dict[str, Any]:
    """Validate and serialize the exact pair-local and demo-weld parameters."""

    expected_pairs = {
        "ball_table": (
            "ball_geom",
            3,
            (0.015, 0.40),
            (0.55, 0.55, 0.01, 0.002, 0.002),
        ),
        "block_table": (
            "block_geom",
            3,
            (0.02, 1.0),
            (0.30, 0.30, 0.02, 0.001, 0.001),
        ),
        "cylinder_table": (
            "cylinder_geom",
            6,
            (0.02, 1.0),
            (0.50, 0.50, 0.02, 0.025, 0.025),
        ),
    }
    pairs: dict[str, Any] = {}
    for label, (object_geom, condim, solref, friction) in expected_pairs.items():
        pair_id = _pair_id(environment, object_geom, "table_top")
        np.testing.assert_array_equal(environment.model.pair_solref[pair_id], solref)
        np.testing.assert_array_equal(environment.model.pair_friction[pair_id], friction)
        if int(environment.model.pair_dim[pair_id]) != condim:
            raise AssertionError(f"compiled condim changed for {label}")
        pairs[label] = {
            "pair_id": pair_id,
            "geom_names": [object_geom, "table_top"],
            "condim": condim,
            "solref": list(solref),
            "friction": list(friction),
        }

    weld_id = mujoco.mj_name2id(
        environment.model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        "demo_mimo_root_support",
    )
    if weld_id < 0:
        raise AssertionError("demo-only MIMo root support weld is absent")
    expected_solref = np.asarray((0.01, 1.0))
    expected_solimp = np.asarray((0.99, 0.999, 0.001, 0.5, 2.0))
    np.testing.assert_array_equal(environment.model.eq_solref[weld_id], expected_solref)
    np.testing.assert_array_equal(environment.model.eq_solimp[weld_id], expected_solimp)
    if not bool(environment.model.eq_active0[weld_id]):
        raise AssertionError("demo-only MIMo root support weld is inactive")
    return {
        "contact_pairs": pairs,
        "demo_world_weld": {
            "name": "demo_mimo_root_support",
            "equality_id": int(weld_id),
            "active": True,
            "solref": expected_solref.tolist(),
            "solimp": expected_solimp.tolist(),
        },
    }


def _attachment_json(attachment: Attachment) -> dict[str, Any]:
    if isinstance(attachment, StaticAttachment):
        return {
            "type": "static",
            "world_position_m": list(attachment.world_position_m),
        }
    if isinstance(attachment, BodyAttachment):
        return {
            "type": "body",
            "body_name": attachment.body_name,
            "local_position_m": list(attachment.local_position_m),
        }
    if isinstance(attachment, SiteAttachment):
        return {"type": "site", "site_name": attachment.site_name}
    raise TypeError(f"unsupported attachment: {type(attachment)!r}")


def _body_kinematics(
    environment: BlightWorldEnv,
    body_id: int,
) -> dict[str, Any]:
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        environment.model,
        environment.data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity,
        0,
    )
    angular = velocity[:3]
    linear = velocity[3:]
    rotation = np.asarray(
        environment.data.ximat[body_id],
        dtype=np.float64,
    ).reshape(3, 3)
    inertia_world = (
        rotation
        @ np.diag(environment.model.body_inertia[body_id])
        @ rotation.T
    )
    mass = float(environment.model.body_mass[body_id])
    kinetic = 0.5 * mass * float(linear @ linear)
    kinetic += 0.5 * float(angular @ inertia_world @ angular)
    position = np.asarray(environment.data.xpos[body_id], dtype=np.float64)
    mechanical = kinetic + mass * 9.81 * (
        float(position[2]) - TABLE_REST_CENTER_Z_M
    )
    return {
        "position_m": position.copy(),
        "linear_velocity_m_s": linear.copy(),
        "angular_velocity_rad_s": angular.copy(),
        "kinetic_energy_j": kinetic,
        "mechanical_energy_above_table_rest_j": mechanical,
    }


def _table_contact(
    environment: BlightWorldEnv,
    object_name: str,
) -> bool:
    object_geom = _scalar_int(environment.model.geom(f"{object_name}_geom").id)
    table_geom = _scalar_int(environment.model.geom("table_top").id)
    return any(
        {int(contact.geom[0]), int(contact.geom[1])}
        == {object_geom, table_geom}
        for contact in environment.data.contact[: environment.data.ncon]
    )


def _target_mask(image: Uint8Array, target: str) -> NDArray[np.bool_]:
    """Return the frozen public-RGB material mask for one demo object."""

    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    if target == "ball":
        return (red > 180) & (green < 120) & (blue < 120)
    if target == "block":
        return (blue > 130) & (red < 100) & (green < 150)
    if target == "cylinder":
        return (green > 130) & (red < 100) & (blue < 130)
    raise KeyError(f"unknown visibility target: {target}")


def _camera_point_metrics(
    environment: BlightWorldEnv,
    eye_name: str,
    world_point_m: Float64Array,
) -> dict[str, float]:
    """Project a world point into MuJoCo's local -Z optical frame."""

    camera_id = _scalar_int(environment.model.camera(eye_name).id)
    rotation = np.asarray(
        environment.data.cam_xmat[camera_id],
        dtype=np.float64,
    ).reshape(3, 3)
    local = rotation.T @ (
        np.asarray(world_point_m, dtype=np.float64)
        - np.asarray(environment.data.cam_xpos[camera_id], dtype=np.float64)
    )
    depth = -float(local[2])
    return {
        "depth_m": depth,
        "horizontal_degrees": degrees(atan2(float(local[0]), depth)),
        "vertical_degrees": degrees(atan2(float(local[1]), depth)),
    }


def _visibility_measurements(
    environment: BlightWorldEnv,
    frame_index: int,
    left: Uint8Array,
    right: Uint8Array,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the table and scheduled target in both public eye arrays."""

    target_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    presentation_time = (frame_index + 1) / VIDEO_FRAME_RATE_HZ
    table_point = np.asarray(
        environment.data.geom("table_top").xpos,
        dtype=np.float64,
    )
    for eye_name, image in (("eye_left", left), ("eye_right", right)):
        table_metrics = _camera_point_metrics(
            environment,
            eye_name,
            table_point,
        )
        if table_metrics["depth_m"] <= 0.0 or abs(
            table_metrics["vertical_degrees"]
        ) > 14.0:
            raise AssertionError(
                f"table left the {eye_name} view at frame {frame_index}: "
                f"{table_metrics}"
            )
        table_rows.append(
            {
                "frame_index": frame_index,
                "presentation_time_s": presentation_time,
                "eye": eye_name,
                **table_metrics,
            }
        )

        for target, (first_frame, last_frame) in TARGET_VISIBILITY_WINDOWS.items():
            if not first_frame <= frame_index <= last_frame:
                continue
            metrics = _camera_point_metrics(
                environment,
                eye_name,
                np.asarray(environment.data.body(target).xpos, dtype=np.float64),
            )
            mask_pixels = int(np.count_nonzero(_target_mask(image, target)))
            if (
                metrics["depth_m"] <= 0.0
                or abs(metrics["horizontal_degrees"]) > 31.0
                or abs(metrics["vertical_degrees"]) > 23.0
                or mask_pixels < TARGET_MINIMUM_MASK_PIXELS[target]
            ):
                raise AssertionError(
                    f"{target} failed {eye_name} visibility at frame "
                    f"{frame_index}: {metrics}, mask_pixels={mask_pixels}"
                )
            target_rows.append(
                {
                    "frame_index": frame_index,
                    "presentation_time_s": presentation_time,
                    "eye": eye_name,
                    "target": target,
                    "mask_pixels": mask_pixels,
                    "mask_fraction": mask_pixels / (EYE_WIDTH * EYE_HEIGHT),
                    **metrics,
                }
            )
    return target_rows, table_rows


def _is_descendant(
    environment: BlightWorldEnv,
    body_id: int,
    ancestor_id: int,
) -> bool:
    while body_id:
        if body_id == ancestor_id:
            return True
        body_id = int(environment.model.body_parentid[body_id])
    return False


def _external_torso_contacts(
    environment: BlightWorldEnv,
    root_body_id: int,
    torso_body_ids: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contact in environment.data.contact[: environment.data.ncon]:
        body1 = int(environment.model.geom_bodyid[int(contact.geom[0])])
        body2 = int(environment.model.geom_bodyid[int(contact.geom[1])])
        if not ({body1, body2} & torso_body_ids):
            continue
        if _is_descendant(environment, body1, root_body_id) and _is_descendant(
            environment,
            body2,
            root_body_id,
        ):
            continue
        rows.append(
            {
                "time_s": float(environment.data.time),
                "body_names": [
                    mujoco.mj_id2name(
                        environment.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        body1,
                    ),
                    mujoco.mj_id2name(
                        environment.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        body2,
                    ),
                ],
            }
        )
    return rows


def _flight_apexes(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    first_contact = next(
        index for index, row in enumerate(rows) if row["table_contact"]
    )
    episodes: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows[first_contact + 1 :]:
        if not row["table_contact"]:
            current.append(row)
        elif current:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return [
        {
            "time_s": float(
                max(episode, key=lambda row: row["position_m"][2])["time_s"]
            ),
            "clearance_above_rest_m": float(
                max(float(row["position_m"][2]) for row in episode)
                - TABLE_REST_CENTER_Z_M
            ),
            "flight_substep_count": len(episode),
            "flight_duration_s": len(episode) * PHYSICS_TIMESTEP_SECONDS,
        }
        for episode in episodes
    ]


def _settle_time(rows: list[dict[str, Any]]) -> float | None:
    stable = np.asarray(
        [
            row["table_contact"]
            and abs(float(row["position_m"][2]) - TABLE_REST_CENTER_Z_M)
            <= 0.002
            and abs(float(row["linear_velocity_m_s"][2])) <= 0.01
            for row in rows
        ],
        dtype=bool,
    )
    suffix = np.logical_and.accumulate(stable[::-1])[::-1]
    indices = np.flatnonzero(suffix)
    return float(rows[int(indices[0])]["time_s"]) if indices.size else None


def _stop_time(
    rows: list[dict[str, Any]],
    force_stop_seconds: float,
) -> float | None:
    eligible = np.asarray(
        [
            row["time_s"] >= force_stop_seconds
            and abs(float(row["linear_velocity_m_s"][0])) <= 0.01
            for row in rows
        ],
        dtype=bool,
    )
    suffix = np.logical_and.accumulate(eligible[::-1])[::-1]
    indices = np.flatnonzero(suffix)
    return float(rows[int(indices[0])]["time_s"]) if indices.size else None


def _object_summary(
    name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    positions = np.asarray([row["position_m"] for row in rows])
    linear = np.asarray([row["linear_velocity_m_s"] for row in rows])
    angular = np.asarray([row["angular_velocity_rad_s"] for row in rows])
    return {
        "initial_staged_position_m": {
            "ball": list(BALL_INITIAL_POSITION_M),
            "block": list(BLOCK_INITIAL_POSITION_M),
            "cylinder": list(CYLINDER_INITIAL_POSITION_M),
        }[name],
        "first_recorded_position_m": positions[0].astype(float).tolist(),
        "final_position_m": positions[-1].astype(float).tolist(),
        "xyz_min_m": positions.min(axis=0).astype(float).tolist(),
        "xyz_max_m": positions.max(axis=0).astype(float).tolist(),
        "peak_abs_vx_m_s": float(np.max(np.abs(linear[:, 0]))),
        "final_linear_velocity_m_s": linear[-1].astype(float).tolist(),
        "peak_abs_omega_y_rad_s": float(np.max(np.abs(angular[:, 1]))),
        "table_contact_fraction": float(
            np.mean([row["table_contact"] for row in rows])
        ),
    }


def _physics_evidence(
    rows_by_object: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Derive and enforce the frozen physical-envelope acceptance criteria."""

    ball_rows = rows_by_object["ball"]
    ball_apexes = _flight_apexes(ball_rows)
    clearances = np.asarray(
        [row["clearance_above_rest_m"] for row in ball_apexes],
        dtype=np.float64,
    )
    if clearances.shape != (3,):
        raise AssertionError(f"ball rebound episode count changed: {clearances}")
    np.testing.assert_allclose(
        clearances,
        EXPECTED_REBOUND_CLEARANCES_M,
        rtol=0.15,
        atol=0.00010,
    )
    if not np.all(np.diff(clearances) < 0):
        raise AssertionError("ball rebound clearances are not strictly decreasing")
    ball_settle = _settle_time(ball_rows)
    if ball_settle is None or ball_settle > 1.25:
        raise AssertionError(f"ball did not settle before force pulses: {ball_settle}")
    first_ball_contact = next(
        index for index, row in enumerate(ball_rows) if row["table_contact"]
    )
    initial_ball_energy = 0.12 * 9.81 * (
        BALL_INITIAL_POSITION_M[2] - TABLE_REST_CENTER_Z_M
    )
    max_ball_energy = max(
        float(row["mechanical_energy_above_table_rest_j"])
        for row in ball_rows[first_ball_contact:]
    )
    if max_ball_energy > initial_ball_energy + 1e-9:
        raise AssertionError("ball/table contact added mechanical energy")

    block_rows = rows_by_object["block"]
    block_summary = _object_summary("block", block_rows)
    block_stop = _stop_time(block_rows, BLOCK_FORCE_STOP_SECONDS)
    if block_stop is None or block_stop > 1.80:
        raise AssertionError(f"block did not naturally stop: {block_stop}")
    if abs(block_summary["final_linear_velocity_m_s"][0]) > 0.01:
        raise AssertionError("block retained excessive final x velocity")

    cylinder_rows = rows_by_object["cylinder"]
    cylinder_summary = _object_summary("cylinder", cylinder_rows)
    post_force = [
        row for row in cylinder_rows if row["time_s"] >= CYLINDER_FORCE_STOP_SECONDS
    ]
    rows_per_100ms = round(0.100 / PHYSICS_TIMESTEP_SECONDS)
    complete_bins = len(post_force) // rows_per_100ms
    speed_means: list[float] = []
    energy_means: list[float] = []
    for bin_index in range(complete_bins):
        rows = post_force[
            bin_index * rows_per_100ms : (bin_index + 1) * rows_per_100ms
        ]
        speed_means.append(
            float(
                np.mean(
                    [
                        np.linalg.norm(row["linear_velocity_m_s"][:2])
                        for row in rows
                    ]
                )
            )
        )
        energy_means.append(
            float(
                np.mean(
                    [row["mechanical_energy_above_table_rest_j"] for row in rows]
                )
            )
        )
    speed_increases = int(np.sum(np.diff(speed_means) > 1e-4))
    energy_increases = int(np.sum(np.diff(energy_means) > 1e-6))
    vx = np.asarray(
        [float(row["linear_velocity_m_s"][0]) for row in post_force]
    )
    vx_sign_changes = int(np.sum(vx[1:] * vx[:-1] < 0.0))
    if speed_increases or energy_increases or vx_sign_changes:
        raise AssertionError(
            "cylinder post-force dissipation lost monotonicity: "
            f"speed bins={speed_increases}, energy bins={energy_increases}, "
            f"vx sign changes={vx_sign_changes}"
        )
    if not all(row["table_contact"] for row in post_force):
        raise AssertionError("cylinder lost table support after its force pulse")

    for name, rows in rows_by_object.items():
        positions = np.asarray([row["position_m"] for row in rows])
        if not np.all(np.isfinite(positions)):
            raise FloatingPointError(f"{name} trajectory contains non-finite values")
        if np.min(positions[:, 0]) < TABLE_X_BOUNDS_M[0] or np.max(
            positions[:, 0]
        ) > TABLE_X_BOUNDS_M[1]:
            raise AssertionError(f"{name} left the table x extent")
        if np.min(positions[:, 1]) < TABLE_Y_BOUNDS_M[0] or np.max(
            positions[:, 1]
        ) > TABLE_Y_BOUNDS_M[1]:
            raise AssertionError(f"{name} left the table y extent")
        final_z = float(positions[-1, 2])
        if not 0.695 <= final_z <= 0.705:
            raise AssertionError(f"{name} did not finish table-supported: z={final_z}")

    return {
        "ball": {
            **_object_summary("ball", ball_rows),
            "rebound_apexes": ball_apexes,
            "successive_clearances_strictly_decreasing": True,
            "settle_time_s": ball_settle,
            "settled_before_first_force_pulse": True,
            "initial_mechanical_energy_j": initial_ball_energy,
            "max_mechanical_after_first_contact_j": max_ball_energy,
            "contact_energy_gain": False,
        },
        "block": {
            **block_summary,
            "force_window_s": [
                BLOCK_FORCE_START_SECONDS,
                BLOCK_FORCE_STOP_SECONDS,
            ],
            "force_x_n": BLOCK_FORCE_X_N,
            "natural_stop_time_abs_vx_below_0.01_s": block_stop,
            "naturally_stopped": True,
        },
        "cylinder": {
            **cylinder_summary,
            "force_window_s": [
                CYLINDER_FORCE_START_SECONDS,
                CYLINDER_FORCE_STOP_SECONDS,
            ],
            "force_x_n": CYLINDER_FORCE_X_N,
            "post_force_contact_fraction": float(
                np.mean([row["table_contact"] for row in post_force])
            ),
            "post_force_100ms_planar_speed_mean_m_s": speed_means,
            "post_force_100ms_mechanical_energy_mean_j": energy_means,
            "speed_envelope_positive_bins_tolerance_1e-4": speed_increases,
            "energy_envelope_positive_bins_tolerance_1e-6": energy_increases,
            "post_force_vx_sign_changes": vx_sign_changes,
            "state_at_5s": "near_rest_still_dissipating",
        },
    }


def _encode_media(
    raw_video_path: Path,
    paths: DemoPaths,
    ffmpeg_tools: dict[str, str],
    repository_root: Path,
) -> None:
    ffmpeg = ffmpeg_tools["ffmpeg_path"]
    common = (
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
        "-framerate",
        str(VIDEO_FRAME_RATE_HZ),
        "-i",
        str(raw_video_path),
        "-i",
        str(paths.pressure_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    )
    _run_checked(
        common
        + (
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-g",
            "1",
            "-slicecrc",
            "1",
            "-pix_fmt",
            "bgr0",
            "-c:a",
            "pcm_f32le",
            str(paths.master_mkv),
        ),
        cwd=repository_root,
    )
    _run_checked(
        common
        + (
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(paths.playback_mp4),
        ),
        cwd=repository_root,
    )


def _probe_media(
    path: Path,
    ffprobe: str,
    repository_root: Path,
) -> dict[str, Any]:
    result = _run_checked(
        (
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ),
        cwd=repository_root,
    )
    return json.loads(result.stdout)


def _validate_media(
    paths: DemoPaths,
    raw_video_sha256: str,
    raw_audio_interleaved_sha256: str,
    ffmpeg_tools: dict[str, str],
    repository_root: Path,
    temporary_directory: Path,
) -> dict[str, Any]:
    """Decode the lossless master and prove byte equality with public sources."""

    ffmpeg = ffmpeg_tools["ffmpeg_path"]
    ffprobe = ffmpeg_tools["ffprobe_path"]
    master_probe = _probe_media(paths.master_mkv, ffprobe, repository_root)
    playback_probe = _probe_media(paths.playback_mp4, ffprobe, repository_root)
    for probe, expected_codecs in (
        (master_probe, ("ffv1", "pcm_f32le")),
        (playback_probe, ("h264", "aac")),
    ):
        codecs = {stream["codec_type"]: stream["codec_name"] for stream in probe["streams"]}
        if (codecs.get("video"), codecs.get("audio")) != expected_codecs:
            raise AssertionError(f"encoded codec contract changed: {codecs}")
        duration = float(probe["format"]["duration"])
        if abs(duration - DURATION_SECONDS) > 0.02:
            raise AssertionError(f"encoded duration changed: {duration}")
        video_stream = next(
            stream for stream in probe["streams"] if stream["codec_type"] == "video"
        )
        if (
            int(video_stream["width"]) != VIDEO_WIDTH
            or int(video_stream["height"]) != VIDEO_HEIGHT
            or video_stream["avg_frame_rate"] != f"{VIDEO_FRAME_RATE_HZ}/1"
            or int(video_stream["nb_read_frames"]) != VIDEO_FRAMES
        ):
            raise AssertionError(f"encoded video geometry changed: {video_stream}")
        audio_stream = next(
            stream for stream in probe["streams"] if stream["codec_type"] == "audio"
        )
        if int(audio_stream["sample_rate"]) != AUDIO_SAMPLE_RATE_HZ:
            raise AssertionError("encoded audio sample rate changed")

    decoded_video = temporary_directory / "decoded.rgb24"
    decoded_audio = temporary_directory / "decoded.f32le"
    _run_checked(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(paths.master_mkv),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            str(decoded_video),
        ),
        cwd=repository_root,
    )
    _run_checked(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(paths.master_mkv),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            str(decoded_audio),
        ),
        cwd=repository_root,
    )
    _run_checked(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-v",
            "error",
            "-i",
            str(paths.playback_mp4),
            "-f",
            "null",
            "-",
        ),
        cwd=repository_root,
    )
    decoded_video_sha256 = _sha256_file(decoded_video)
    decoded_audio_sha256 = _sha256_file(decoded_audio)
    if decoded_video_sha256 != raw_video_sha256:
        raise AssertionError("lossless master video differs from public eye panels")
    if decoded_audio_sha256 != raw_audio_interleaved_sha256:
        raise AssertionError("lossless master audio differs from public audition")
    if decoded_video.stat().st_size != VIDEO_FRAMES * VIDEO_WIDTH * VIDEO_HEIGHT * 3:
        raise AssertionError("decoded lossless video byte count changed")
    if decoded_audio.stat().st_size != AUDIO_SAMPLES_PER_EAR * 2 * 4:
        raise AssertionError("decoded lossless audio byte count changed")
    return {
        "master_probe": master_probe,
        "playback_probe": playback_probe,
        "master_decoded_video_sha256": decoded_video_sha256,
        "master_decoded_audio_interleaved_sha256": decoded_audio_sha256,
        "lossless_master_matches_public_sources": True,
        "playback_decode_exit_zero": True,
    }


def export_demo(
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the public-step storyboard, encode it, and return its manifest."""

    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("set MUJOCO_GL=egl before starting the exporter")
    repository_root = Path(__file__).resolve().parents[1]
    ffmpeg_tools = _validate_ffmpeg(repository_root)
    git_state = _git_state(repository_root)
    script_path = Path(__file__).resolve()
    script_sha256 = _sha256_file(script_path)
    paths = _paths(output_directory)
    _prepare_output(paths, overwrite=overwrite)

    environment = make_demo_environment()
    events: list[dict[str, Any]] = []
    audio_blocks: list[Float32Array] = []
    object_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("ball", "block", "cylinder")
    }
    original_play_transient: Any = None
    original_substep_callback: Any = None
    original_projection = audition_renderer_module.project_source_position
    projection_rows: list[dict[str, Any]] = []
    try:
        model_evidence = _validate_model(environment)
        addresses = _stage_demo_state(environment)
        staged_object_state = {
            name: {
                "qpos_address": qpos_address,
                "dof_address": dof_address,
                "position_m": environment.data.qpos[
                    qpos_address : qpos_address + 3
                ].astype(float).tolist(),
                "quaternion_wxyz": environment.data.qpos[
                    qpos_address + 3 : qpos_address + 7
                ].astype(float).tolist(),
                "velocity": environment.data.qvel[
                    dof_address : dof_address + 6
                ].astype(float).tolist(),
            }
            for name, (qpos_address, dof_address) in addresses.object_qpos_dof.items()
        }
        if environment.audition is None:
            raise RuntimeError("audition unexpectedly unavailable after staging")
        sensor = environment.audition
        renderer = sensor.renderer
        original_play_transient = renderer.play_transient

        def recording_play_transient(
            clip_id: str,
            attachment: Attachment,
            *,
            gain_db: float = 0.0,
            delay_s: float = 0.0,
            impact_energy_j: float = 0.0,
        ) -> int | None:
            voice_id = original_play_transient(
                clip_id,
                attachment,
                gain_db=gain_db,
                delay_s=delay_s,
                impact_energy_j=impact_energy_j,
            )
            events.append(
                {
                    "simulation_time_s": float(environment.data.time),
                    "clip_id": clip_id,
                    "attachment": _attachment_json(attachment),
                    "gain_db": float(gain_db),
                    "delay_s": float(delay_s),
                    "impact_energy_j": float(impact_energy_j),
                    "voice_id": voice_id,
                    "accepted": voice_id is not None,
                }
            )
            return voice_id

        def recording_projection(position_m: Float64Array) -> Float64Array:
            projected = original_projection(position_m)
            if not np.array_equal(projected, position_m):
                projection_rows.append(
                    {
                        "input_m": np.asarray(position_m).astype(float).tolist(),
                        "projected_m": projected.astype(float).tolist(),
                    }
                )
            return projected

        renderer.play_transient = recording_play_transient  # type: ignore[method-assign]
        audition_renderer_module.project_source_position = recording_projection

        root_position0 = np.asarray(
            environment.data.xpos[addresses.root_body_id]
        ).copy()
        root_rotation0 = np.asarray(
            environment.data.xmat[addresses.root_body_id]
        ).copy()
        torso_body_ids = {
            _scalar_int(environment.model.body(name).id)
            for name in (
                "mimo_location",
                "hip",
                "lower_body",
                "upper_body",
                "chest",
                "head",
            )
        }
        root_position_errors: list[float] = []
        root_rotation_errors: list[float] = []
        hip_positions: list[float] = []
        head_positions: list[float] = []
        external_contacts: list[dict[str, Any]] = []
        force_step_counts = {"block": 0, "cylinder": 0}
        max_active_transients = 0
        max_retained_transients = 0
        max_active_continuous = 0
        max_retained_continuous = 0
        dropped_continuous: set[str] = set()

        original_substep_callback = environment._substep_callback

        def recording_substep_callback() -> None:
            """Call through, then record read-only 5 ms physics diagnostics."""

            original_substep_callback()
            for name, body_id in addresses.object_body_ids.items():
                row = _body_kinematics(environment, body_id)
                row["time_s"] = float(environment.data.time)
                row["table_contact"] = _table_contact(environment, name)
                object_rows[name].append(row)
            root_position_errors.append(
                float(
                    np.max(
                        np.abs(
                            environment.data.xpos[addresses.root_body_id]
                            - root_position0
                        )
                    )
                )
            )
            root_rotation_errors.append(
                float(
                    np.max(
                        np.abs(
                            environment.data.xmat[addresses.root_body_id]
                            - root_rotation0
                        )
                    )
                )
            )
            hip_positions.append(
                float(
                    np.mean(
                        environment.data.qpos[
                            list(addresses.hip_qpos_addresses)
                        ]
                    )
                )
            )
            head_positions.append(
                float(environment.data.qpos[addresses.head_qpos_address])
            )
            external_contacts.extend(
                _external_torso_contacts(
                    environment,
                    addresses.root_body_id,
                    torso_body_ids,
                )
            )
            nonlocal max_active_transients
            nonlocal max_retained_transients
            nonlocal max_active_continuous
            nonlocal max_retained_continuous
            max_active_transients = max(
                max_active_transients,
                renderer.active_voice_count,
            )
            max_retained_transients = max(
                max_retained_transients,
                len(renderer._voices),
            )
            max_active_continuous = max(
                max_active_continuous,
                renderer.continuous_active_voice_count,
            )
            max_retained_continuous = max(
                max_retained_continuous,
                renderer.continuous_retained_voice_count,
            )
            dropped_continuous.update(renderer.continuous_dropped_stream_ids)

        environment._substep_callback = recording_substep_callback  # type: ignore[method-assign]

        left_digest = sha256()
        right_digest = sha256()
        frame_digest = sha256()
        frame_count = 0
        panel_equality_count = 0
        binocular_nonidentical_count = 0
        temporal_left_change_count = 0
        temporal_right_change_count = 0
        binocular_mean_absolute_differences: list[float] = []
        target_visibility_rows: list[dict[str, Any]] = []
        table_visibility_rows: list[dict[str, Any]] = []
        eye_standard_deviations: dict[str, list[float]] = {
            "left": [],
            "right": [],
        }
        previous_left: Uint8Array | None = None
        previous_right: Uint8Array | None = None

        with TemporaryDirectory(
            prefix="blight-binocular-physics-demo-",
            dir=paths.output_directory,
        ) as temporary_name:
            temporary_directory = Path(temporary_name)
            raw_video_path = temporary_directory / "public_eyes.rgb24"
            with raw_video_path.open("wb") as raw_video:
                for step_index in range(PUBLIC_STEPS):
                    expected_time = step_index * ENVIRONMENT_STEP_SECONDS
                    time_before = float(environment.data.time)
                    if not np.isclose(time_before, expected_time, rtol=0, atol=1e-10):
                        raise AssertionError("MuJoCo clock departed from public-step time")
                    action, forces = _apply_step_inputs(
                        environment,
                        addresses,
                        time_before,
                    )
                    force_step_counts["block"] += forces[0] != 0.0
                    force_step_counts["cylinder"] += forces[1] != 0.0
                    observation, _, terminated, truncated, _ = environment.step(action)
                    if terminated or truncated:
                        raise AssertionError(
                            f"demo ended at step {step_index}: {terminated=}, {truncated=}"
                        )
                    if not environment.observation_space.contains(observation):
                        raise AssertionError("public observation left its Gymnasium space")

                    pressure_block = observation["audition"]
                    left = observation["eye_left"]
                    right = observation["eye_right"]
                    if (
                        pressure_block.shape != (2, AUDIO_BLOCK_SAMPLES)
                        or pressure_block.dtype != np.float32
                        or not pressure_block.flags.c_contiguous
                    ):
                        raise AssertionError("public audition block contract changed")
                    if (
                        left.shape != (EYE_HEIGHT, EYE_WIDTH, 3)
                        or right.shape != (EYE_HEIGHT, EYE_WIDTH, 3)
                        or left.dtype != np.uint8
                        or right.dtype != np.uint8
                    ):
                        raise AssertionError("public binocular observation contract changed")
                    if not np.all(np.isfinite(pressure_block)):
                        raise FloatingPointError("public audition contains non-finite values")
                    audio_blocks.append(pressure_block.copy(order="C"))

                    if step_index % 2 == 1:
                        left_frame = np.ascontiguousarray(left, dtype=np.uint8)
                        right_frame = np.ascontiguousarray(right, dtype=np.uint8)
                        frame = np.ascontiguousarray(
                            np.concatenate((left_frame, right_frame), axis=1),
                            dtype=np.uint8,
                        )
                        if np.array_equal(frame[:, :EYE_WIDTH], left_frame) and np.array_equal(
                            frame[:, EYE_WIDTH:], right_frame
                        ):
                            panel_equality_count += 1
                        else:
                            raise AssertionError("video panels differ from public eye arrays")
                        if not np.array_equal(left_frame, right_frame):
                            binocular_nonidentical_count += 1
                        binocular_mean_absolute_differences.append(
                            float(
                                np.mean(
                                    np.abs(
                                        left_frame.astype(np.float64)
                                        - right_frame.astype(np.float64)
                                    )
                                )
                            )
                        )
                        eye_standard_deviations["left"].append(float(left_frame.std()))
                        eye_standard_deviations["right"].append(float(right_frame.std()))
                        if previous_left is not None and not np.array_equal(
                            previous_left,
                            left_frame,
                        ):
                            temporal_left_change_count += 1
                        if previous_right is not None and not np.array_equal(
                            previous_right,
                            right_frame,
                        ):
                            temporal_right_change_count += 1
                        previous_left = left_frame.copy()
                        previous_right = right_frame.copy()
                        target_rows, table_rows = _visibility_measurements(
                            environment,
                            frame_count,
                            left_frame,
                            right_frame,
                        )
                        target_visibility_rows.extend(target_rows)
                        table_visibility_rows.extend(table_rows)
                        contents = frame.tobytes(order="C")
                        raw_video.write(contents)
                        left_digest.update(left_frame.tobytes(order="C"))
                        right_digest.update(right_frame.tobytes(order="C"))
                        frame_digest.update(contents)
                        frame_count += 1

            if force_step_counts != {"block": 20, "cylinder": 15}:
                raise AssertionError(f"force window step counts changed: {force_step_counts}")
            if len(audio_blocks) != AUDIO_BLOCKS:
                raise AssertionError("public audition block count changed")
            if frame_count != VIDEO_FRAMES:
                raise AssertionError("public binocular frame count changed")
            if any(len(rows) != PHYSICS_SUBSTEPS for rows in object_rows.values()):
                raise AssertionError(
                    "5 ms physics diagnostic row count changed: "
                    f"{ {name: len(rows) for name, rows in object_rows.items()} }"
                )
            if panel_equality_count != VIDEO_FRAMES:
                raise AssertionError("not every panel preserved its public source bytes")
            if binocular_nonidentical_count != VIDEO_FRAMES:
                raise AssertionError("left and right eye panels became identical")
            if (
                temporal_left_change_count != VIDEO_FRAMES - 1
                or temporal_right_change_count != VIDEO_FRAMES - 1
            ):
                raise AssertionError("one or both public eye streams stopped changing")
            if min(eye_standard_deviations["left"]) <= 0.0 or min(
                eye_standard_deviations["right"]
            ) <= 0.0:
                raise AssertionError("one or both public eye streams became blank")
            expected_target_rows = 2 * sum(
                last_frame - first_frame + 1
                for first_frame, last_frame in TARGET_VISIBILITY_WINDOWS.values()
            )
            if len(target_visibility_rows) != expected_target_rows:
                raise AssertionError(
                    "target visibility row count changed: "
                    f"{len(target_visibility_rows)} != {expected_target_rows}"
                )
            if len(table_visibility_rows) != VIDEO_FRAMES * 2:
                raise AssertionError("table visibility row count changed")

            pressure = np.ascontiguousarray(
                np.concatenate(audio_blocks, axis=1),
                dtype=np.float32,
            )
            if pressure.shape != (2, AUDIO_SAMPLES_PER_EAR):
                raise AssertionError("public audition timeline shape changed")
            if not np.all(np.isfinite(pressure)):
                raise FloatingPointError("public audition timeline is non-finite")
            peak_pressure_pa = float(np.max(np.abs(pressure)))
            if peak_pressure_pa >= 1.0:
                raise AssertionError("1 Pa = 1 FS media mapping would clip")
            pressure_channels_first = np.ascontiguousarray(
                pressure,
                dtype=np.dtype("<f4"),
            )
            pressure_interleaved = np.ascontiguousarray(
                pressure.T,
                dtype=np.dtype("<f4"),
            )
            raw_audio_channels_first_sha256 = sha256(
                pressure_channels_first.tobytes(order="C")
            ).hexdigest()
            raw_audio_interleaved_sha256 = sha256(
                pressure_interleaved.tobytes(order="C")
            ).hexdigest()
            wavfile.write(
                paths.pressure_wav,
                AUDIO_SAMPLE_RATE_HZ,
                pressure_interleaved,
            )
            wav_rate, wav_pressure = wavfile.read(paths.pressure_wav)
            if (
                wav_rate != AUDIO_SAMPLE_RATE_HZ
                or wav_pressure.shape != (AUDIO_SAMPLES_PER_EAR, 2)
                or wav_pressure.dtype != np.float32
            ):
                raise AssertionError("float32 pressure WAV contract changed")
            np.testing.assert_array_equal(wav_pressure, pressure_interleaved)

            accepted_events = [event for event in events if event["accepted"]]
            rejected_events = [event for event in events if not event["accepted"]]
            if len(accepted_events) != 6 or rejected_events:
                raise AssertionError(
                    "impact admission changed: "
                    f"accepted={len(accepted_events)}, rejected={len(rejected_events)}"
                )
            if projection_rows:
                raise AssertionError(f"auditory source projection occurred: {projection_rows}")
            if dropped_continuous:
                raise AssertionError(
                    f"continuous capacity dropped streams: {sorted(dropped_continuous)}"
                )
            if max_active_transients > 8 or max_active_continuous > 4:
                raise AssertionError("auditory voice capacity exceeded its configured bounds")
            if max(root_position_errors) > 0.002:
                raise AssertionError("demo-only torso support exceeded 2 mm")
            if external_contacts:
                raise AssertionError(
                    f"torso made external contacts: {external_contacts[:5]}"
                )

            physics = _physics_evidence(object_rows)
            _encode_media(
                raw_video_path,
                paths,
                ffmpeg_tools,
                repository_root,
            )
            media_validation = _validate_media(
                paths,
                frame_digest.hexdigest(),
                raw_audio_interleaved_sha256,
                ffmpeg_tools,
                repository_root,
                temporary_directory,
            )

        manifest = {
            "schema_version": 1,
            "title": "BlightWorld binocular physical-interaction demonstration",
            "description": (
                "MIMo binocular observations with neutral hip/head posture "
                "stabilization and a demo-only physical torso support. Object "
                "motion after one-time staging is entirely MuJoCo gravity, "
                "contact, and finite forces."
            ),
            "behavior_claims": {
                "neutral_hip_head_posture_stabilization": True,
                "scripted_head_swivel": False,
                "demo_only_physical_torso_support": True,
                "object_motion_after_staging": "MuJoCo gravity/contact/finite forces",
                "walking": False,
                "autonomous_behavior": False,
                "learned_policy": False,
            },
            "listening_note": "Headphones recommended for binaural playback.",
            "media_pressure_mapping": {
                "pascal_to_digital_full_scale": "1 Pa = 1 FS",
                "normalization": False,
                "automatic_gain_control": False,
                "time_warp": False,
                "clipping": False,
                "peak_pressure_pa": peak_pressure_pa,
            },
            "configuration": {
                "seed": DEMO_SEED,
                "duration_seconds": DURATION_SECONDS,
                "public_environment_steps": PUBLIC_STEPS,
                "environment_step_seconds": ENVIRONMENT_STEP_SECONDS,
                "physics_diagnostic_interval_seconds": PHYSICS_TIMESTEP_SECONDS,
                "physics_diagnostic_rows_per_object": PHYSICS_SUBSTEPS,
                "vision": {
                    "source": "public observation eye_left and eye_right",
                    "eye_shape": [EYE_HEIGHT, EYE_WIDTH, 3],
                    "composite_layout": "eye_left then eye_right",
                    "composite_shape": [VIDEO_HEIGHT, VIDEO_WIDTH, 3],
                    "frame_rate_hz": VIDEO_FRAME_RATE_HZ,
                    "frame_count": VIDEO_FRAMES,
                    "target_visibility_windows_inclusive": {
                        target: list(window)
                        for target, window in TARGET_VISIBILITY_WINDOWS.items()
                    },
                    "minimum_mask_pixels": TARGET_MINIMUM_MASK_PIXELS,
                    "target_angle_limits_degrees": {
                        "absolute_horizontal": 31.0,
                        "absolute_vertical": 23.0,
                    },
                    "table_absolute_vertical_limit_degrees": 14.0,
                },
                "audition": {
                    "profile": "AuditionConfig.quality_48k",
                    "continuous": "ContinuousSoundConfig",
                    "source": "public observation audition",
                    "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
                    "block_shape": [2, AUDIO_BLOCK_SAMPLES],
                    "block_count": AUDIO_BLOCKS,
                    "timeline_shape": [2, AUDIO_SAMPLES_PER_EAR],
                },
                "one_time_stage": staged_object_state,
                "post_stage_object_qpos_qvel_writes": 0,
                "countergravity_after_staging": False,
                "forces": {
                    "block": {
                        "body": "block",
                        "vector_n": [BLOCK_FORCE_X_N, 0.0, 0.0],
                        "start_s_inclusive": BLOCK_FORCE_START_SECONDS,
                        "stop_s_exclusive": BLOCK_FORCE_STOP_SECONDS,
                        "public_step_count": force_step_counts["block"],
                    },
                    "cylinder": {
                        "body": "cylinder",
                        "vector_n": [CYLINDER_FORCE_X_N, 0.0, 0.0],
                        "start_s_inclusive": CYLINDER_FORCE_START_SECONDS,
                        "stop_s_exclusive": CYLINDER_FORCE_STOP_SECONDS,
                        "public_step_count": force_step_counts["cylinder"],
                    },
                },
                "neutral_posture_stabilization": {
                    "label": "neutral hip/head posture stabilization",
                    "hip": {
                        "actuator": "act:hip_bend",
                        "actuator_id": addresses.hip_actuator_id,
                        "action_index": addresses.hip_action_index,
                        "reference_rad": addresses.hip_reference_rad,
                        "torque_formula": (
                            "bias[d1]+bias[d2]+10*(ref-mean(q1,q2))"
                            "-4.5*mean(v1,v2)"
                        ),
                        "actuator_gear": HIP_ACTUATOR_GEAR,
                    },
                    "head": {
                        "actuator": "act:head_tilt",
                        "actuator_id": addresses.head_actuator_id,
                        "action_index": addresses.head_action_index,
                        "reference_rad": addresses.head_reference_rad,
                        "torque_formula": "bias[d]+1*(ref-q)-0.20*v",
                        "actuator_gear": HEAD_ACTUATOR_GEAR,
                    },
                    "action_bounds_clipped": True,
                    "other_actions_zero": True,
                },
            },
            "model": model_evidence,
            "provenance": {
                "script_path": str(script_path.relative_to(repository_root)),
                "script_sha256": script_sha256,
                "git": git_state,
                "software": {
                    "python": os.sys.version.split()[0],
                    "blight_world": metadata.version("blight-world"),
                    "mimo": metadata.version("MIMo"),
                    "mujoco": mujoco.__version__,
                    "numpy": np.__version__,
                    "ffmpeg": ffmpeg_tools,
                },
            },
            "public_source_validation": {
                "every_observation_in_space": True,
                "panel_byte_equal_frame_count": panel_equality_count,
                "left_right_nonidentical_frame_count": binocular_nonidentical_count,
                "left_temporally_changed_transitions": temporal_left_change_count,
                "right_temporally_changed_transitions": temporal_right_change_count,
                "binocular_mean_absolute_difference_range": [
                    min(binocular_mean_absolute_differences),
                    max(binocular_mean_absolute_differences),
                ],
                "left_standard_deviation_range": [
                    min(eye_standard_deviations["left"]),
                    max(eye_standard_deviations["left"]),
                ],
                "right_standard_deviation_range": [
                    min(eye_standard_deviations["right"]),
                    max(eye_standard_deviations["right"]),
                ],
                "visibility": {
                    "target_rows": target_visibility_rows,
                    "table_rows": table_visibility_rows,
                    "all_scheduled_targets_visible_in_both_eyes": True,
                    "table_in_both_eye_views_every_frame": True,
                    "target_summary": {
                        target: {
                            eye_name: {
                                "frame_count": len(rows),
                                "minimum_mask_pixels": min(
                                    int(row["mask_pixels"]) for row in rows
                                ),
                                "minimum_depth_m": min(
                                    float(row["depth_m"]) for row in rows
                                ),
                                "maximum_absolute_horizontal_degrees": max(
                                    abs(float(row["horizontal_degrees"]))
                                    for row in rows
                                ),
                                "maximum_absolute_vertical_degrees": max(
                                    abs(float(row["vertical_degrees"]))
                                    for row in rows
                                ),
                            }
                            for eye_name in ("eye_left", "eye_right")
                            for rows in [
                                [
                                    row
                                    for row in target_visibility_rows
                                    if row["target"] == target
                                    and row["eye"] == eye_name
                                ]
                            ]
                        }
                        for target in TARGET_VISIBILITY_WINDOWS
                    },
                    "table_summary": {
                        eye_name: {
                            "frame_count": len(rows),
                            "minimum_depth_m": min(
                                float(row["depth_m"]) for row in rows
                            ),
                            "maximum_absolute_vertical_degrees": max(
                                abs(float(row["vertical_degrees"]))
                                for row in rows
                            ),
                        }
                        for eye_name in ("eye_left", "eye_right")
                        for rows in [
                            [
                                row
                                for row in table_visibility_rows
                                if row["eye"] == eye_name
                            ]
                        ]
                    },
                },
            },
            "raw_sources": {
                "video_rgb24_frame_major_sha256": frame_digest.hexdigest(),
                "left_eye_uint8_frame_major_sha256": left_digest.hexdigest(),
                "right_eye_uint8_frame_major_sha256": right_digest.hexdigest(),
                "audio_pressure_pa_channels_first_float32_sha256": (
                    raw_audio_channels_first_sha256
                ),
                "audio_pressure_pa_interleaved_float32_sha256": (
                    raw_audio_interleaved_sha256
                ),
            },
            "physics_validation": {
                "support": {
                    "maximum_root_position_error_m": max(root_position_errors),
                    "maximum_root_rotation_matrix_element_error": max(
                        root_rotation_errors
                    ),
                    "external_torso_contact_count": len(external_contacts),
                },
                "neutral_posture_qpos_rad": {
                    "hip_bend_mean": {
                        "reference": addresses.hip_reference_rad,
                        "minimum": min(hip_positions),
                        "maximum": max(hip_positions),
                    },
                    "head_tilt": {
                        "reference": addresses.head_reference_rad,
                        "minimum": min(head_positions),
                        "maximum": max(head_positions),
                    },
                },
                "objects": physics,
            },
            "audition_validation": {
                "impact_events": events,
                "accepted_impact_count": len(accepted_events),
                "rejected_impact_count": len(rejected_events),
                "source_projection_count": len(projection_rows),
                "max_active_transient_voices": max_active_transients,
                "max_retained_transient_voices": max_retained_transients,
                "max_active_continuous_voices": max_active_continuous,
                "max_retained_continuous_voices": max_retained_continuous,
                "continuous_dropped_stream_ids": sorted(dropped_continuous),
            },
            "media_validation": media_validation,
            "artifacts": {
                "lossless_master": {
                    "path": paths.master_mkv.name,
                    "sha256": _sha256_file(paths.master_mkv),
                    "video": "FFV1 640x240 50 fps",
                    "audio": "pcm_f32le 48 kHz stereo",
                },
                "easy_play": {
                    "path": paths.playback_mp4.name,
                    "sha256": _sha256_file(paths.playback_mp4),
                    "video": "H.264 CRF 18 640x240 50 fps",
                    "audio": "AAC 192 kb/s 48 kHz stereo",
                },
                "pressure_wav": {
                    "path": paths.pressure_wav.name,
                    "sha256": _sha256_file(paths.pressure_wav),
                    "format": "IEEE float32 48 kHz stereo pressure in pascals",
                },
            },
        }
        paths.manifest_json.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest
    finally:
        audition_renderer_module.project_source_position = original_projection
        if original_substep_callback is not None:
            environment._substep_callback = original_substep_callback  # type: ignore[method-assign]
        if environment.audition is not None and original_play_transient is not None:
            environment.audition.renderer.play_transient = original_play_transient
        environment.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/binocular_physics_demo"),
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    manifest = export_demo(
        arguments.output_directory,
        overwrite=arguments.overwrite,
    )
    paths = _paths(arguments.output_directory)
    print(f"wrote {paths.master_mkv}")
    print(f"wrote {paths.playback_mp4}")
    print(f"wrote {paths.pressure_wav}")
    print(f"wrote {paths.manifest_json}")
    print("peak pressure Pa:", manifest["media_pressure_mapping"]["peak_pressure_pa"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

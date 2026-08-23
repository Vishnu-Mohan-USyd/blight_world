"""Export a deterministic external-camera diagnostic of BlightWorld audition.

The eight-second storyboard uses kinematically constrained validation fixtures
and a scripted head/right-arm pose with the production scene's current contact
physics.  It is not a physical-behavior demonstration, autonomous behavior, or
learned behavior.  Audio is taken only from the public
``observation["audition"]`` transport and is written without gain control,
normalization, or clipping; video comes from an explicitly external camera.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import json
from math import cos, log10, pi, sin
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

from blight_world import (
    AuditionConfig,
    BlightWorldEnv,
    BodyAttachment,
    ContinuousSoundConfig,
    SiteAttachment,
    StaticAttachment,
)
from mimo_audition.sources import Attachment


FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
Uint8Array = NDArray[np.uint8]

DEMO_SEED = 1_401
DURATION_SECONDS = 8.0
PHYSICS_TIMESTEP_SECONDS = 0.005
ENVIRONMENT_STEP_SECONDS = 0.010
VIDEO_FRAME_RATE_HZ = 50
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 360
AUDIO_SAMPLE_RATE_HZ = 48_000
PHYSICS_SUBSTEPS = 1_600
VIDEO_FRAMES = 400
AUDIO_BLOCKS = 800
AUDIO_SAMPLES_PER_EAR = 384_000
SAMPLES_PER_FRAME = 960
EXPECTED_PEAK_PRESSURE_PA = 0.8566
EXPECTED_PUBLIC_AUDIO_SHA256 = (
    "d51749cec3e3e76739f8be7077dea40cf153bcf11a213d6cd9bd8ddf4ef1229b"
)

CAMERA_LOOKAT = (0.55, 0.0, 0.55)
CAMERA_DISTANCE = 2.2
CAMERA_AZIMUTH_DEGREES = 135.0
CAMERA_ELEVATION_DEGREES = -25.0

BALL_HELD_POSITION_M = (1.20, 0.0, 1.10)
BLOCK_INITIAL_POSITION_M = (0.65, 0.38, 0.699)
CYLINDER_INITIAL_POSITION_M = (0.65, -0.38, 0.699)
MIMO_ROOT_POSITION_M = (0.0, 0.0, 0.4)
IDENTITY_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)

BALL_RELEASE_TIME_SECONDS = 1.0
BLOCK_DRAG_START_SECONDS = 2.2
BLOCK_DRAG_STOP_SECONDS = 3.4
CYLINDER_DRAG_START_SECONDS = 4.4
CYLINDER_DRAG_STOP_SECONDS = 5.6
DRAG_SPEED_M_S = 0.30

EXPECTED_SCRAPE_WINDOWS = {
    "scrape:6:11": (2.215, 3.405),
    "scrape:6:13": (4.415, 5.605),
}
EXPECTED_BALL_IMPACTS = (
    (1.290, "impact_wood_rubber_2", 0.464927093468),
    (1.460, "impact_wood_rubber_0", 0.035354958453),
    (1.510, "impact_wood_rubber_2", 0.002989637751),
    (1.535, "impact_wood_rubber_2", 0.000500020446),
)
IMPACT_TIME_TOLERANCE_SECONDS = 1e-12
IMPACT_ENERGY_TOLERANCE_J = 5e-13

POSE_JOINTS = {
    "head_swivel": "robot:head_swivel",
    "right_shoulder_horizontal": "robot:right_shoulder_horizontal",
    "right_shoulder_ad_ab": "robot:right_shoulder_ad_ab",
    "right_elbow": "robot:right_elbow",
}


@dataclass(frozen=True, slots=True)
class DemoPaths:
    """Final artifact paths for one export."""

    output_directory: Path
    master_mkv: Path
    playback_mp4: Path
    pressure_wav: Path
    manifest_json: Path


def _scalar_int(value: Any) -> int:
    return int(np.asarray(value).item())


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_checked(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one external command and retain useful failure evidence."""

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


def _validate_ffmpeg(repository_root: Path) -> dict[str, str]:
    """Fail closed unless every required codec and container is available."""

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise FileNotFoundError("ffmpeg and ffprobe must both be available on PATH")
    version = _run_checked((ffmpeg, "-hide_banner", "-version"), cwd=repository_root)
    encoders = _run_checked((ffmpeg, "-hide_banner", "-encoders"), cwd=repository_root)
    muxers = _run_checked((ffmpeg, "-hide_banner", "-muxers"), cwd=repository_root)
    for encoder in ("ffv1", "libx264", "pcm_f32le", "aac"):
        if encoder not in encoders.stdout:
            raise RuntimeError(f"required ffmpeg encoder is unavailable: {encoder}")
    for muxer in ("matroska", "mp4", "wav"):
        if muxer not in muxers.stdout:
            raise RuntimeError(f"required ffmpeg muxer is unavailable: {muxer}")
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
        ("git", "status", "--porcelain=v1"),
        cwd=repository_root,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain": status,
    }


def _paths(output_directory: Path) -> DemoPaths:
    output = output_directory.resolve()
    return DemoPaths(
        output,
        output / "blight_world_demo_master.mkv",
        output / "blight_world_demo.mp4",
        output / "blight_world_demo_pressure_pa.wav",
        output / "blight_world_demo_validation.json",
    )


def _prepare_output(paths: DemoPaths, *, overwrite: bool) -> None:
    paths.output_directory.mkdir(parents=True, exist_ok=True)
    targets = (
        paths.master_mkv,
        paths.playback_mp4,
        paths.pressure_wav,
        paths.manifest_json,
    )
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing demo artifacts: {joined}")
    for path in existing:
        path.unlink()


def pose_values(time_seconds: float) -> dict[str, float]:
    """Return the frozen scripted pose in radians at one simulation time."""

    if not np.isfinite(time_seconds) or not 0.0 <= time_seconds <= DURATION_SECONDS:
        raise ValueError("time_seconds must be finite and within the demo")
    ramp = 0.5 - 0.5 * cos(pi * min(time_seconds / 0.8, 1.0))
    return {
        "head_swivel": ramp * 0.35 * sin(2.0 * pi * time_seconds / 4.0),
        "right_shoulder_horizontal": ramp * 0.30,
        "right_shoulder_ad_ab": ramp
        * (0.42 + 0.12 * sin(2.0 * pi * time_seconds / 1.6)),
        "right_elbow": ramp
        * (-0.70 + 0.12 * sin(2.0 * pi * time_seconds / 1.2)),
    }


def storyboard_phase(time_seconds: float) -> str:
    """Return the single primary storyboard phase at one substep start."""

    if time_seconds < BALL_RELEASE_TIME_SECONDS:
        return "establish_and_hold_ball"
    if time_seconds < BLOCK_DRAG_START_SECONDS:
        return "ball_release_and_impact"
    if time_seconds < BLOCK_DRAG_STOP_SECONDS:
        return "block_drag"
    if time_seconds < CYLINDER_DRAG_START_SECONDS:
        return "block_release"
    if time_seconds < CYLINDER_DRAG_STOP_SECONDS:
        return "cylinder_drag"
    return "cylinder_release_and_room_tail"


def _free_joint_addresses(
    environment: BlightWorldEnv,
    body_name: str,
) -> tuple[int, int]:
    joint_name = "mimo_orientation" if body_name == "mimo_location" else f"{body_name}_joint"
    joint = environment.model.joint(joint_name)
    return _scalar_int(joint.qposadr), _scalar_int(joint.dofadr)


def _write_free_pose(
    environment: BlightWorldEnv,
    body_name: str,
    position_m: tuple[float, float, float],
    *,
    zero_velocity: bool,
) -> None:
    qpos_address, dof_address = _free_joint_addresses(environment, body_name)
    environment.data.qpos[qpos_address : qpos_address + 3] = position_m
    environment.data.qpos[qpos_address + 3 : qpos_address + 7] = (
        IDENTITY_QUATERNION_WXYZ
    )
    if zero_velocity:
        environment.data.qvel[dof_address : dof_address + 6] = 0.0


def _prepare_demo_state(environment: BlightWorldEnv) -> dict[str, tuple[int, int]]:
    environment.reset(seed=DEMO_SEED)
    environment.set_state(
        np.asarray(environment.init_qpos, dtype=np.float64).copy(),
        np.zeros_like(environment.init_qvel, dtype=np.float64),
    )
    _write_free_pose(environment, "ball", BALL_HELD_POSITION_M, zero_velocity=True)
    _write_free_pose(
        environment,
        "block",
        BLOCK_INITIAL_POSITION_M,
        zero_velocity=True,
    )
    _write_free_pose(
        environment,
        "cylinder",
        CYLINDER_INITIAL_POSITION_M,
        zero_velocity=True,
    )
    _write_free_pose(
        environment,
        "mimo_location",
        MIMO_ROOT_POSITION_M,
        zero_velocity=True,
    )
    environment.data.time = 0.0
    environment.data.ctrl.fill(0.0)
    environment.data.qfrc_applied.fill(0.0)
    environment.data.xfrc_applied.fill(0.0)
    environment.data.qacc_warmstart.fill(0.0)
    for value in pose_values(0.0).values():
        if value != 0.0:
            raise AssertionError("scripted pose must begin from the neutral state")
    mujoco.mj_forward(environment.model, environment.data)
    if environment.audition is None:
        raise RuntimeError("audition demo requires an enabled auditory sensor")
    environment.audition.reset(DEMO_SEED)
    return {
        name: _free_joint_addresses(environment, name)
        for name in ("ball", "block", "cylinder", "mimo_location")
    }


def _apply_mimo_countergravity(environment: BlightWorldEnv) -> None:
    root_id = _scalar_int(environment.model.body("mimo_location").id)
    for body_id in range(environment.model.nbody):
        if _scalar_int(environment.model.body_rootid[body_id]) == root_id:
            environment.data.xfrc_applied[body_id, 2] = (
                environment.model.body_mass[body_id] * 9.81
            )


def _clamp_table_object(
    environment: BlightWorldEnv,
    addresses: tuple[int, int],
    *,
    initial_position_m: tuple[float, float, float],
    time_seconds: float,
    drag_start_seconds: float,
    drag_stop_seconds: float,
) -> None:
    qpos_address, dof_address = addresses
    environment.data.qpos[qpos_address + 1 : qpos_address + 3] = (
        initial_position_m[1],
        initial_position_m[2],
    )
    environment.data.qpos[qpos_address + 3 : qpos_address + 7] = (
        IDENTITY_QUATERNION_WXYZ
    )
    environment.data.qvel[dof_address : dof_address + 6] = 0.0
    if time_seconds < drag_start_seconds:
        environment.data.qpos[qpos_address] = initial_position_m[0]
    elif time_seconds < drag_stop_seconds:
        environment.data.qvel[dof_address] = DRAG_SPEED_M_S


def _apply_substep_fixture(
    environment: BlightWorldEnv,
    addresses: dict[str, tuple[int, int]],
    time_seconds: float,
) -> None:
    """Apply the exact pose/object controls before one 5 ms interval."""

    environment.data.ctrl.fill(0.0)
    environment.data.qfrc_applied.fill(0.0)
    environment.data.xfrc_applied.fill(0.0)
    mimo_qpos, mimo_dof = addresses["mimo_location"]
    environment.data.qpos[mimo_qpos : mimo_qpos + 3] = MIMO_ROOT_POSITION_M
    environment.data.qpos[mimo_qpos + 3 : mimo_qpos + 7] = (
        IDENTITY_QUATERNION_WXYZ
    )
    environment.data.qvel[mimo_dof : mimo_dof + 6] = 0.0
    for key, joint_name in POSE_JOINTS.items():
        joint = environment.model.joint(joint_name)
        environment.data.qpos[_scalar_int(joint.qposadr)] = pose_values(time_seconds)[key]
        environment.data.qvel[_scalar_int(joint.dofadr)] = 0.0
    _apply_mimo_countergravity(environment)

    if time_seconds < BALL_RELEASE_TIME_SECONDS:
        _write_free_pose(
            environment,
            "ball",
            BALL_HELD_POSITION_M,
            zero_velocity=True,
        )
        ball_id = _scalar_int(environment.model.body("ball").id)
        environment.data.xfrc_applied[ball_id, 2] = (
            environment.model.body_mass[ball_id] * 9.81
        )

    _clamp_table_object(
        environment,
        addresses["block"],
        initial_position_m=BLOCK_INITIAL_POSITION_M,
        time_seconds=time_seconds,
        drag_start_seconds=BLOCK_DRAG_START_SECONDS,
        drag_stop_seconds=BLOCK_DRAG_STOP_SECONDS,
    )
    _clamp_table_object(
        environment,
        addresses["cylinder"],
        initial_position_m=CYLINDER_INITIAL_POSITION_M,
        time_seconds=time_seconds,
        drag_start_seconds=CYLINDER_DRAG_START_SECONDS,
        drag_stop_seconds=CYLINDER_DRAG_STOP_SECONDS,
    )
    mujoco.mj_forward(environment.model, environment.data)


def _attachment_json(attachment: Attachment) -> dict[str, Any]:
    if isinstance(attachment, StaticAttachment):
        return {
            "kind": "static",
            "world_position_m": list(attachment.world_position_m),
        }
    if isinstance(attachment, BodyAttachment):
        return {
            "kind": "body",
            "body_name": attachment.body_name,
            "local_position_m": list(attachment.local_position_m),
        }
    if isinstance(attachment, SiteAttachment):
        return {"kind": "site", "site_name": attachment.site_name}
    raise TypeError("unsupported attachment")


def _ild_db(pressure: Float32Array, start_seconds: float, stop_seconds: float) -> float:
    start = round(start_seconds * AUDIO_SAMPLE_RATE_HZ)
    stop = round(stop_seconds * AUDIO_SAMPLE_RATE_HZ)
    window = np.asarray(pressure[:, start:stop], dtype=np.float64)
    rms = np.sqrt(np.mean(window * window, axis=1))
    if window.shape[1] == 0 or np.any(rms <= 0) or not np.all(np.isfinite(rms)):
        raise AssertionError("ILD window must contain finite binaural energy")
    return float(20.0 * log10(float(rms[0] / rms[1])))


def _hash_process_stdout(command: Sequence[str], *, cwd: Path) -> tuple[str, int]:
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise AssertionError("subprocess pipes were not created")
    digest = sha256()
    byte_count = 0
    while True:
        chunk = process.stdout.read(1 << 20)
        if not chunk:
            break
        digest.update(chunk)
        byte_count += len(chunk)
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"decode command failed with exit {return_code}: {rendered}\n{stderr}"
        )
    return digest.hexdigest(), byte_count


def _ffprobe(path: Path, ffprobe: str, repository_root: Path) -> dict[str, Any]:
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
) -> dict[str, Any]:
    ffmpeg = ffmpeg_tools["ffmpeg_path"]
    ffprobe = ffmpeg_tools["ffprobe_path"]
    master_probe = _ffprobe(paths.master_mkv, ffprobe, repository_root)
    playback_probe = _ffprobe(paths.playback_mp4, ffprobe, repository_root)
    master_streams = master_probe["streams"]
    playback_streams = playback_probe["streams"]
    master_video = next(stream for stream in master_streams if stream["codec_type"] == "video")
    master_audio = next(stream for stream in master_streams if stream["codec_type"] == "audio")
    playback_video = next(
        stream for stream in playback_streams if stream["codec_type"] == "video"
    )
    playback_audio = next(
        stream for stream in playback_streams if stream["codec_type"] == "audio"
    )
    if not (
        master_video["codec_name"] == "ffv1"
        and int(master_video["width"]) == VIDEO_WIDTH
        and int(master_video["height"]) == VIDEO_HEIGHT
        and master_video["avg_frame_rate"] == f"{VIDEO_FRAME_RATE_HZ}/1"
        and int(master_video["nb_read_frames"]) == VIDEO_FRAMES
        and master_audio["codec_name"] == "pcm_f32le"
        and int(master_audio["sample_rate"]) == AUDIO_SAMPLE_RATE_HZ
        and int(master_audio["channels"]) == 2
    ):
        raise AssertionError("lossless master stream contract changed")
    if not (
        playback_video["codec_name"] == "h264"
        and int(playback_video["width"]) == VIDEO_WIDTH
        and int(playback_video["height"]) == VIDEO_HEIGHT
        and playback_video["avg_frame_rate"] == f"{VIDEO_FRAME_RATE_HZ}/1"
        and int(playback_video["nb_read_frames"]) == VIDEO_FRAMES
        and playback_audio["codec_name"] == "aac"
        and int(playback_audio["sample_rate"]) == AUDIO_SAMPLE_RATE_HZ
        and int(playback_audio["channels"]) == 2
    ):
        raise AssertionError("easy-play stream contract changed")

    decoded_video_sha256, decoded_video_bytes = _hash_process_stdout(
        (
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(paths.master_mkv),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ),
        cwd=repository_root,
    )
    decoded_audio_sha256, decoded_audio_bytes = _hash_process_stdout(
        (
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(paths.master_mkv),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "2",
            "-ar",
            str(AUDIO_SAMPLE_RATE_HZ),
            "-",
        ),
        cwd=repository_root,
    )
    expected_video_bytes = VIDEO_FRAMES * VIDEO_WIDTH * VIDEO_HEIGHT * 3
    expected_audio_bytes = AUDIO_SAMPLES_PER_EAR * 2 * np.dtype(np.float32).itemsize
    if (
        decoded_video_sha256 != raw_video_sha256
        or decoded_video_bytes != expected_video_bytes
        or decoded_audio_sha256 != raw_audio_interleaved_sha256
        or decoded_audio_bytes != expected_audio_bytes
    ):
        raise AssertionError("lossless master did not reproduce raw sources exactly")
    return {
        "master_ffprobe": master_probe,
        "playback_ffprobe": playback_probe,
        "master_decoded_video_sha256": decoded_video_sha256,
        "master_decoded_video_bytes": decoded_video_bytes,
        "master_decoded_audio_interleaved_sha256": decoded_audio_sha256,
        "master_decoded_audio_bytes": decoded_audio_bytes,
        "lossless_master_matches_raw_sources": True,
    }


def _encode_media(
    raw_video_path: Path,
    paths: DemoPaths,
    ffmpeg_tools: dict[str, str],
    repository_root: Path,
) -> None:
    ffmpeg = ffmpeg_tools["ffmpeg_path"]
    common_inputs = (
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
        common_inputs
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
        common_inputs
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


def export_demo(
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the frozen storyboard, encode media, and return its manifest."""

    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("set MUJOCO_GL=egl before starting the exporter")
    repository_root = Path(__file__).resolve().parents[1]
    ffmpeg_tools = _validate_ffmpeg(repository_root)
    git_state = _git_state(repository_root)
    script_sha256 = _sha256_file(Path(__file__).resolve())
    paths = _paths(output_directory)
    _prepare_output(paths, overwrite=overwrite)
    configuration = {
        "seed": DEMO_SEED,
        "duration_seconds": DURATION_SECONDS,
        "physics_timestep_seconds": PHYSICS_TIMESTEP_SECONDS,
        "environment_step_seconds": ENVIRONMENT_STEP_SECONDS,
        "physics_substeps": PHYSICS_SUBSTEPS,
        "video": {
            "width": VIDEO_WIDTH,
            "height": VIDEO_HEIGHT,
            "frame_rate_hz": VIDEO_FRAME_RATE_HZ,
            "frame_count": VIDEO_FRAMES,
            "samples_per_frame": SAMPLES_PER_FRAME,
            "camera": {
                "type": "external_free",
                "lookat_m": list(CAMERA_LOOKAT),
                "distance_m": CAMERA_DISTANCE,
                "azimuth_degrees": CAMERA_AZIMUTH_DEGREES,
                "elevation_degrees": CAMERA_ELEVATION_DEGREES,
            },
        },
        "audio": {
            "profile": "AuditionConfig.quality_48k",
            "continuous": "ContinuousSoundConfig",
            "sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
            "channels": 2,
            "public_observation_blocks": AUDIO_BLOCKS,
            "samples_per_ear": AUDIO_SAMPLES_PER_EAR,
            "transport": "environment._get_obs()['audition']",
        },
        "eye_observation": {"width": 64, "height": 48},
        "storyboard": [
            {"start_s": 0.0, "stop_s": 1.0, "phase": "establish_and_hold_ball"},
            {"start_s": 1.0, "stop_s": 2.2, "phase": "ball_release_and_impact"},
            {"start_s": 2.2, "stop_s": 3.4, "phase": "block_drag", "speed_m_s": 0.30},
            {"start_s": 3.4, "stop_s": 4.4, "phase": "block_release"},
            {"start_s": 4.4, "stop_s": 5.6, "phase": "cylinder_drag", "speed_m_s": 0.30},
            {"start_s": 5.6, "stop_s": 8.0, "phase": "cylinder_release_and_room_tail"},
        ],
        "behavior_label": (
            "kinematically_constrained_external_camera_audition_diagnostic"
        ),
        "contact_physics": "current production scene contact pairs",
    }
    environment = BlightWorldEnv(
        width=64,
        height=48,
        audition=AuditionConfig.quality_48k(
            continuous=ContinuousSoundConfig()
        ),
    )
    visual_renderer: mujoco.Renderer | None = None
    original_play_transient: Any = None
    old_offwidth = int(environment.model.vis.global_.offwidth)
    old_offheight = int(environment.model.vis.global_.offheight)
    audio_blocks: list[Float32Array] = []
    events: list[dict[str, Any]] = []
    scrape_active_times: dict[str, list[float]] = {
        stream_id: [] for stream_id in EXPECTED_SCRAPE_WINDOWS
    }
    frame_digest = sha256()
    unique_frame_hashes: set[str] = set()
    raw_video_bytes = 0
    object_initial: dict[str, list[float]] = {}
    object_final: dict[str, list[float]] = {}
    try:
        addresses = _prepare_demo_state(environment)
        for name in ("ball", "block", "cylinder"):
            qpos_address, _ = addresses[name]
            object_initial[name] = [
                float(value)
                for value in environment.data.qpos[qpos_address : qpos_address + 3]
            ]
        environment.model.vis.global_.offwidth = max(old_offwidth, VIDEO_WIDTH)
        environment.model.vis.global_.offheight = max(old_offheight, VIDEO_HEIGHT)
        visual_renderer = mujoco.Renderer(
            environment.model,
            height=VIDEO_HEIGHT,
            width=VIDEO_WIDTH,
        )
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(environment.model, camera)
        camera.lookat[:] = CAMERA_LOOKAT
        camera.distance = CAMERA_DISTANCE
        camera.azimuth = CAMERA_AZIMUTH_DEGREES
        camera.elevation = CAMERA_ELEVATION_DEGREES

        if environment.audition is None:
            raise RuntimeError("audition demo requires enabled audition")
        sensor = environment.audition
        original_play_transient = sensor.renderer.play_transient

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
                    "renderer_sample_cursor": sensor.renderer.sample_cursor,
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

        sensor.renderer.play_transient = recording_play_transient  # type: ignore[method-assign]
        with TemporaryDirectory(
            prefix="blight-audition-demo-",
            dir=paths.output_directory,
        ) as temporary_directory:
            raw_video_path = Path(temporary_directory) / "frames.rgb24"
            with raw_video_path.open("wb") as raw_video:
                for substep in range(PHYSICS_SUBSTEPS):
                    expected_time_seconds = substep * PHYSICS_TIMESTEP_SECONDS
                    if not np.isclose(
                        environment.data.time,
                        expected_time_seconds,
                        rtol=0,
                        atol=1e-10,
                    ):
                        raise AssertionError("MuJoCo clock departed from storyboard")
                    time_seconds = float(environment.data.time)
                    _apply_substep_fixture(
                        environment,
                        addresses,
                        time_seconds,
                    )
                    if substep % 4 == 0:
                        visual_renderer.update_scene(environment.data, camera=camera)
                        frame = np.ascontiguousarray(
                            visual_renderer.render(),
                            dtype=np.uint8,
                        )
                        if frame.shape != (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
                            raise AssertionError("external renderer frame shape changed")
                        contents = frame.tobytes(order="C")
                        raw_video.write(contents)
                        frame_digest.update(contents)
                        unique_frame_hashes.add(sha256(contents).hexdigest())
                        raw_video_bytes += len(contents)

                    sensor.before_substep()
                    mujoco.mj_step(environment.model, environment.data)
                    mujoco.mj_rnePostConstraint(environment.model, environment.data)
                    sensor.capture_substep()
                    for diagnostic in sensor.scrape_diagnostics:
                        if (
                            diagnostic.stream_id in scrape_active_times
                            and diagnostic.phase == "active"
                        ):
                            scrape_active_times[diagnostic.stream_id].append(
                                float(environment.data.time)
                            )
                    if substep % 2 == 1:
                        sensor.finish_step()
                        observation = environment._get_obs()
                        if not environment.observation_space.contains(observation):
                            raise AssertionError("demo observation left its Gym space")
                        block = observation["audition"]
                        if (
                            block.shape != (2, 480)
                            or block.dtype != np.float32
                            or not block.flags.c_contiguous
                        ):
                            raise AssertionError("public audition block contract changed")
                        audio_blocks.append(block.copy(order="C"))

            if len(unique_frame_hashes) < 10:
                raise AssertionError("scripted visual sequence did not visibly change")
            if raw_video_bytes != VIDEO_FRAMES * VIDEO_WIDTH * VIDEO_HEIGHT * 3:
                raise AssertionError("raw video byte count changed")
            if len(audio_blocks) != AUDIO_BLOCKS:
                raise AssertionError("public audition block count changed")
            pressure = np.ascontiguousarray(
                np.concatenate(audio_blocks, axis=1),
                dtype=np.float32,
            )
            if pressure.shape != (2, AUDIO_SAMPLES_PER_EAR):
                raise AssertionError("public pressure timeline shape changed")
            if not np.all(np.isfinite(pressure)):
                raise FloatingPointError("demo pressure contains non-finite values")
            peak_pressure_pa = float(np.max(np.abs(pressure)))
            if round(peak_pressure_pa, 4) != EXPECTED_PEAK_PRESSURE_PA:
                raise AssertionError(
                    "demo peak pressure changed: "
                    f"expected {EXPECTED_PEAK_PRESSURE_PA}, got {peak_pressure_pa}"
                )
            if peak_pressure_pa >= 1.0:
                raise AssertionError("1 Pa = 1 FS diagnostic media would clip")

            scrape_windows: dict[str, Any] = {}
            for stream_id, expected_window in EXPECTED_SCRAPE_WINDOWS.items():
                times = scrape_active_times[stream_id]
                if not times:
                    raise AssertionError(f"missing expected scrape stream {stream_id}")
                observed = (min(times), max(times))
                if not np.allclose(observed, expected_window, rtol=0, atol=1e-12):
                    raise AssertionError(
                        f"scrape window changed for {stream_id}: {observed}"
                    )
                scrape_windows[stream_id] = {
                    "first_active_time_s": observed[0],
                    "last_active_time_s": observed[1],
                    "active_substeps": len(times),
                }
            ball_events = [
                event
                for event in events
                if event["clip_id"].startswith("impact_wood_rubber_")
            ]
            if len(ball_events) != len(EXPECTED_BALL_IMPACTS) or not all(
                event["accepted"] for event in ball_events
            ):
                raise AssertionError(
                    "accepted ball/table impact count changed: "
                    f"{[(event['clip_id'], event['accepted']) for event in ball_events]}"
                )
            for event, (expected_time, expected_clip, expected_energy) in zip(
                ball_events,
                EXPECTED_BALL_IMPACTS,
                strict=True,
            ):
                if event["clip_id"] != expected_clip:
                    raise AssertionError(
                        "ball/table impact clip changed: "
                        f"expected {expected_clip}, got {event['clip_id']}"
                    )
                if not np.isclose(
                    event["simulation_time_s"],
                    expected_time,
                    rtol=0,
                    atol=IMPACT_TIME_TOLERANCE_SECONDS,
                ):
                    raise AssertionError(
                        "ball/table impact time changed: "
                        f"expected {expected_time}, got {event['simulation_time_s']}"
                    )
                if not np.isclose(
                    event["impact_energy_j"],
                    expected_energy,
                    rtol=0,
                    atol=IMPACT_ENERGY_TOLERANCE_J,
                ):
                    raise AssertionError(
                        "ball/table impact energy changed: "
                        f"expected {expected_energy}, got {event['impact_energy_j']}"
                    )

            pressure_little_endian = np.ascontiguousarray(
                pressure,
                dtype=np.dtype("<f4"),
            )
            pressure_interleaved = np.ascontiguousarray(
                pressure.T,
                dtype=np.dtype("<f4"),
            )
            raw_audio_channels_first_sha256 = sha256(
                pressure_little_endian.tobytes(order="C")
            ).hexdigest()
            if raw_audio_channels_first_sha256 != EXPECTED_PUBLIC_AUDIO_SHA256:
                raise AssertionError(
                    "public audition waveform changed: "
                    f"expected {EXPECTED_PUBLIC_AUDIO_SHA256}, "
                    f"got {raw_audio_channels_first_sha256}"
                )
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
                or wav_pressure.dtype != np.float32
                or wav_pressure.shape != (AUDIO_SAMPLES_PER_EAR, 2)
            ):
                raise AssertionError("float32 pressure WAV contract changed")
            np.testing.assert_array_equal(wav_pressure, pressure_interleaved)
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
            )

        for name in ("ball", "block", "cylinder"):
            qpos_address, _ = addresses[name]
            object_final[name] = [
                float(value)
                for value in environment.data.qpos[qpos_address : qpos_address + 3]
            ]
        ild_windows = {
            "ball_impact_1.28_1.80s": _ild_db(pressure, 1.28, 1.80),
            "block_scrape_2.40_3.30s": _ild_db(pressure, 2.40, 3.30),
            "cylinder_scrape_4.60_5.50s": _ild_db(pressure, 4.60, 5.50),
            "full_0_8s": _ild_db(pressure, 0.0, 8.0),
        }
        if (
            ild_windows["block_scrape_2.40_3.30s"]
            * ild_windows["cylinder_scrape_4.60_5.50s"]
            >= 0
        ):
            raise AssertionError("left/right scrape ILD signs did not reverse")
        manifest = {
            "schema_version": 1,
            "title": "BlightWorld external-camera audition diagnostic",
            "description": (
                "Kinematically constrained validation fixtures, an external "
                "camera, a scripted head/right-arm pose, and current production "
                "contact physics. This is an audition diagnostic, not autonomous, "
                "learned, or naturally driven physical behavior."
            ),
            "listening_note": "Headphones recommended for binaural playback.",
            "media_pressure_mapping": {
                "pascal_to_digital_full_scale": "1 Pa = 1 FS",
                "normalization": False,
                "automatic_gain_control": False,
                "clipping": False,
                "peak_pressure_pa": peak_pressure_pa,
            },
            "configuration": configuration,
            "provenance": {
                "script_path": str(Path(__file__).resolve().relative_to(repository_root)),
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
            "raw_sources": {
                "video_rgb24_frame_major_sha256": frame_digest.hexdigest(),
                "video_bytes": raw_video_bytes,
                "unique_frame_hashes": len(unique_frame_hashes),
                "audio_pressure_pa_channels_first_float32_sha256": (
                    raw_audio_channels_first_sha256
                ),
                "audio_pressure_pa_interleaved_float32_sha256": (
                    raw_audio_interleaved_sha256
                ),
                "audio_samples_per_ear": pressure.shape[1],
            },
            "storyboard_evidence": {
                "object_initial_positions_m": object_initial,
                "object_final_positions_m": object_final,
                "impact_events": events,
                "expected_ball_impact_time_s": 1.290,
                "scrape_windows": scrape_windows,
                "ild_db_left_over_right": ild_windows,
            },
            "media_validation": media_validation,
            "artifacts": {
                "lossless_master": {
                    "path": paths.master_mkv.name,
                    "sha256": _sha256_file(paths.master_mkv),
                    "video": "FFV1 640x360 50 fps",
                    "audio": "pcm_f32le 48 kHz stereo",
                },
                "easy_play": {
                    "path": paths.playback_mp4.name,
                    "sha256": _sha256_file(paths.playback_mp4),
                    "video": "H.264 CRF 18 640x360 50 fps",
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
        if environment.audition is not None and original_play_transient is not None:
            environment.audition.renderer.play_transient = original_play_transient
        try:
            if visual_renderer is not None:
                visual_renderer.close()
        finally:
            environment.model.vis.global_.offwidth = old_offwidth
            environment.model.vis.global_.offheight = old_offheight
            environment.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/audition_demo"),
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
    print(
        "peak pressure Pa:",
        manifest["media_pressure_mapping"]["peak_pressure_pa"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

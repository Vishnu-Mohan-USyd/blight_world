"""Export one uninterrupted public-observation crawl/ball demonstration.

The 41.8-second episode uses the validated free-ball crawl fixture and the
exact scripted ``UnsupportedBellyCrawlController`` action sequence.  Video is
formed only by concatenating MIMo's public left/right eye observations.  Audio
is formed only from public stereo pressure blocks.  The episode has one reset,
no cuts, overlays, external camera, external forces, runtime state writes, or
synthetic sound injection.

The controller is an open-loop scripted belly/army crawl.  This artifact is
not evidence of turning, picking, lifting, learned behavior, or autonomy.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO

import mimoEnv
import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from blight_world.crawl_ball_fixture import (
    CRAWL_BALL_CONTACT_GEOM_NAMES,
    CrawlBallContact,
    CrawlBallFixtureEnv,
)
from blight_world.locomotion_skills import UnsupportedBellyCrawlController
from mimo_audition import AuditionConfig, ContinuousSoundConfig


Float32Array = NDArray[np.float32]
Uint8Array = NDArray[np.uint8]

DEMO_SEED = 1_401
DURATION_SECONDS = 41.8
PUBLIC_STEP_SECONDS = 0.010
PHYSICS_STEP_SECONDS = 0.005
PUBLIC_STEPS = 4_180
PHYSICS_SUBSTEPS = 8_360
EYE_WIDTH = 320
EYE_HEIGHT = 240
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 240
VIDEO_FRAME_RATE_HZ = 50
VIDEO_FRAMES = 2_090
AUDIO_SAMPLE_RATE_HZ = 48_000
AUDIO_BLOCK_SAMPLES = 480
AUDIO_SAMPLES_PER_EAR = 2_006_400
MEDIA_MONITOR_GAIN = 0.04
CONTACT_PUBLIC_STEP = 2_851
CONTACT_VIDEO_FRAME = CONTACT_PUBLIC_STEP // 2

EXPECTED_SCENE_SHA256 = (
    "819783d00b8983ae26053e814a3af50439f1ac0f9ad223f081f309265455c50f"
)
EXPECTED_ACTION_SHA256 = (
    "46d8fde52bf2bd78c0a4d2a9791a2e056194761816422be58118ee02464fff67"
)
EXPECTED_QPOS_SHA256 = (
    "e5379268f771e4f05578297a6df5257e1380e381056bc852e916f9f0a3d7e4ec"
)
EXPECTED_BALL_TRAJECTORY_SHA256 = (
    "a2123ee5d9c7ad2f100ff6e59569f641770e618a346ad4930e00631f3d645b54"
)
EXPECTED_AUDIO_TRANSPORT_SHA256 = (
    "79fb1a31c8d4672b17b7b4f9fd407c316aa42e936eca3affa00d08bb0cd5d7a5"
)
EXPECTED_CONTACT_LEFT_EYE_SHA256 = (
    "23ad078ea1ffc360a0fcdf77beb74f3df86639cf71b3e1bdbd3866b776891153"
)
EXPECTED_CONTACT_RIGHT_EYE_SHA256 = (
    "5bc700fae754596b28cb9a55f1f129d13a340507bad72fb41e6b9d4fd3191c13"
)
EXPECTED_CONTACT_LEFT_RED_PIXELS = 2_270
EXPECTED_CONTACT_RIGHT_RED_PIXELS = 2_163
EXPECTED_AUDIO_NONZERO_BLOCKS = 3_665
EXPECTED_AUDIO_PEAK_PA = 12.987203598022461
EXPECTED_AUDIO_LEFT_RMS_PA = 0.359957768664457
EXPECTED_AUDIO_RIGHT_RMS_PA = 0.3523857205074939
EXPECTED_BALL_FINAL_POSITION_M = (
    0.8282788213116993,
    0.35641593947774347,
    0.024552576685847535,
)
EXPECTED_BALL_PLANAR_DISPLACEMENT_M = 0.3279796105546315


@dataclass(frozen=True, slots=True)
class DemoPaths:
    """Canonical output paths owned by this exporter."""

    output_directory: Path
    playback_mp4: Path
    pressure_wav: Path
    manifest_json: Path


def _paths(output_directory: Path) -> DemoPaths:
    output = output_directory.resolve()
    return DemoPaths(
        output_directory=output,
        playback_mp4=output / "continuous_embodied_demo.mp4",
        pressure_wav=output / "continuous_embodied_demo_pressure_pa.wav",
        manifest_json=output / "continuous_embodied_demo_manifest.json",
    )


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
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _run_bytes_checked(command: Sequence[str], *, cwd: Path) -> bytes:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {result.returncode}: {' '.join(command)}\n"
            f"stderr:\n{result.stderr.decode(errors='replace')}"
        )
    return result.stdout


def _validate_ffmpeg(repository_root: Path) -> dict[str, str]:
    """Fail closed unless the required executable codecs are available."""

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise FileNotFoundError("ffmpeg and ffprobe must both be available on PATH")
    version = _run_checked((ffmpeg, "-hide_banner", "-version"), cwd=repository_root)
    encoders = _run_checked((ffmpeg, "-hide_banner", "-encoders"), cwd=repository_root)
    for encoder in ("libx264", "aac"):
        if encoder not in encoders.stdout:
            raise RuntimeError(f"required ffmpeg encoder is unavailable: {encoder}")
    return {
        "ffmpeg_path": str(Path(ffmpeg).resolve()),
        "ffprobe_path": str(Path(ffprobe).resolve()),
        "ffmpeg_version": version.stdout.splitlines()[0],
    }


def _git_state(repository_root: Path) -> dict[str, Any]:
    commit = _run_checked(("git", "rev-parse", "HEAD"), cwd=repository_root)
    branch = _run_checked(
        ("git", "branch", "--show-current"), cwd=repository_root
    )
    status = _run_checked(
        ("git", "status", "--porcelain=v1"), cwd=repository_root
    )
    rows = status.stdout.splitlines()
    return {
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip(),
        "dirty": bool(rows),
        "status_porcelain": rows,
    }


def _prepare_output(paths: DemoPaths, *, overwrite: bool) -> None:
    """Resolve owned targets without touching unrelated directory contents."""

    paths.output_directory.mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for path in (paths.playback_mp4, paths.pressure_wav, paths.manifest_json)
        if path.exists()
    ]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing demo artifacts: {joined}")


def _red_pixel_count(image: Uint8Array) -> int:
    mask = (
        (image[..., 0] > 150)
        & (image[..., 1] < 140)
        & (image[..., 2] < 140)
    )
    return int(np.count_nonzero(mask))


def _model_snapshot(environment: CrawlBallFixtureEnv) -> dict[str, NDArray[Any]]:
    return {
        "body_pos": environment.model.body_pos.copy(),
        "body_quat": environment.model.body_quat.copy(),
        "geom_pos": environment.model.geom_pos.copy(),
        "geom_quat": environment.model.geom_quat.copy(),
        "geom_size": environment.model.geom_size.copy(),
        "geom_friction": environment.model.geom_friction.copy(),
        "eq_active0": environment.model.eq_active0.copy(),
        "gravity": environment.model.opt.gravity.copy(),
    }


def _assert_model_unchanged(
    environment: CrawlBallFixtureEnv,
    snapshot: dict[str, NDArray[Any]],
) -> None:
    for name, expected in snapshot.items():
        actual = (
            environment.model.opt.gravity
            if name == "gravity"
            else getattr(environment.model, name)
        )
        if not np.array_equal(actual, expected):
            raise AssertionError(f"runtime model field changed: {name}")


def _start_video_encoder(
    output_path: Path,
    ffmpeg: str,
    repository_root: Path,
) -> subprocess.Popen[bytes]:
    command = (
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
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    )
    return subprocess.Popen(
        command,
        cwd=repository_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _finish_video_encoder(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is None or process.stderr is None:
        raise RuntimeError("video encoder pipes were not created")
    process.stdin.close()
    stderr = process.stderr.read()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"raw-video encoder failed with exit {return_code}: "
            f"{stderr.decode(errors='replace')}"
        )


def _stop_video_encoder(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    if process.stdin is not None:
        process.stdin.close()
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _encode_playback(
    video_only_path: Path,
    pressure_wav_path: Path,
    playback_path: Path,
    ffmpeg: str,
    repository_root: Path,
) -> None:
    """Mux video and public pressure with one fixed monitor gain."""

    _run_checked(
        (
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_only_path),
            "-i",
            str(pressure_wav_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-af",
            f"volume={MEDIA_MONITOR_GAIN:.17g}",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            str(AUDIO_SAMPLE_RATE_HZ),
            "-ac",
            "2",
            "-t",
            f"{DURATION_SECONDS:.3f}",
            "-movflags",
            "+faststart",
            str(playback_path),
        ),
        cwd=repository_root,
    )


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
    playback_path: Path,
    ffmpeg_tools: dict[str, str],
    repository_root: Path,
) -> dict[str, Any]:
    probe = _ffprobe(playback_path, ffmpeg_tools["ffprobe_path"], repository_root)
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
    if not (
        video["codec_name"] == "h264"
        and int(video["width"]) == VIDEO_WIDTH
        and int(video["height"]) == VIDEO_HEIGHT
        and video["avg_frame_rate"] == f"{VIDEO_FRAME_RATE_HZ}/1"
        and int(video["nb_read_frames"]) == VIDEO_FRAMES
        and audio["codec_name"] == "aac"
        and int(audio["sample_rate"]) == AUDIO_SAMPLE_RATE_HZ
        and int(audio["channels"]) == 2
    ):
        raise AssertionError("playback stream contract changed")
    duration = float(probe["format"]["duration"])
    if abs(duration - DURATION_SECONDS) > 0.02:
        raise AssertionError(f"playback duration changed: {duration}")
    _run_checked(
        (
            ffmpeg_tools["ffmpeg_path"],
            "-v",
            "error",
            "-i",
            str(playback_path),
            "-f",
            "null",
            "-",
        ),
        cwd=repository_root,
    )
    decoded_audio_bytes = _run_bytes_checked(
        (
            ffmpeg_tools["ffmpeg_path"],
            "-v",
            "error",
            "-i",
            str(playback_path),
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
    decoded = np.frombuffer(decoded_audio_bytes, dtype=np.dtype("<f4"))
    if decoded.size == 0 or decoded.size % 2 != 0 or not np.all(np.isfinite(decoded)):
        raise AssertionError("decoded playback audio is malformed")
    decoded_peak = float(np.max(np.abs(decoded)))
    if decoded_peak >= 1.0:
        raise AssertionError(f"fixed-gain playback audio clipped: {decoded_peak}")
    return {
        "ffprobe": probe,
        "duration_seconds": duration,
        "video_frames": int(video["nb_read_frames"]),
        "decoded_audio_samples_per_channel": decoded.size // 2,
        "decoded_audio_peak_full_scale": decoded_peak,
        "decoded_audio_clipped": False,
        "full_decode_exit_code": 0,
    }


def _validate_contacts(records: tuple[CrawlBallContact, ...]) -> dict[str, Any]:
    expected = (
        (2851, 5702, "geom:left_ffdistal1", 0.2453847408010164),
        (2851, 5702, "geom:left_mfdistal1", 0.27134175246687636),
        (2851, 5703, "geom:left_ffdistal1", 0.10444521108857753),
        (2851, 5703, "geom:left_mfdistal1", 0.09753414144702192),
        (2852, 5704, "geom:left_mfdistal1", 0.04491341623436752),
    )
    if len(records) != len(expected):
        raise AssertionError(f"crawl-ball contact count changed: {len(records)}")
    for record, (step, substep, geom, normal_force) in zip(
        records, expected, strict=True
    ):
        if (
            record.public_step_index != step
            or record.physics_substep_index != substep
            or record.mimo_geom_name != geom
            or record.ball_geom_name != "ball_geom"
            or abs(record.normal_force_n - normal_force) > 1e-12
            or record.mimo_geom_name not in CRAWL_BALL_CONTACT_GEOM_NAMES
        ):
            raise AssertionError(f"crawl-ball contact trace changed: {record}")
        nominal_time = (substep + 1) * PHYSICS_STEP_SECONDS
        if abs(record.time_seconds - nominal_time) > 5e-11:
            raise AssertionError(f"contact substep clock changed: {record}")
    return {
        "count": len(records),
        "first_public_step": records[0].public_step_index,
        "first_nominal_time_seconds": (
            records[0].physics_substep_index + 1
        )
        * PHYSICS_STEP_SECONDS,
        "maximum_normal_force_n": max(record.normal_force_n for record in records),
        "pre_active_contact_count": 0,
        "post_active_contact_count": 0,
        "allowlisted_mimo_geoms": sorted(
            {record.mimo_geom_name for record in records}
        ),
        "records": [asdict(record) for record in records],
    }


def export_demo(
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run, validate, encode, and atomically publish the continuous episode."""

    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("set MUJOCO_GL=egl before starting the exporter")
    if PUBLIC_STEPS != int(round(DURATION_SECONDS / PUBLIC_STEP_SECONDS)):
        raise AssertionError("public episode clock is inconsistent")
    if VIDEO_FRAMES != PUBLIC_STEPS // 2:
        raise AssertionError("video frame clock is inconsistent")
    if AUDIO_SAMPLES_PER_EAR != PUBLIC_STEPS * AUDIO_BLOCK_SAMPLES:
        raise AssertionError("audio sample clock is inconsistent")
    if EXPECTED_AUDIO_PEAK_PA * MEDIA_MONITOR_GAIN >= 1.0:
        raise AssertionError("predeclared monitor gain would clip the known source")

    repository_root = Path(__file__).resolve().parents[1]
    paths = _paths(output_directory)
    _prepare_output(paths, overwrite=overwrite)
    ffmpeg_tools = _validate_ffmpeg(repository_root)
    git_state = _git_state(repository_root)
    script_sha256 = _sha256_file(Path(__file__).resolve())

    environment = CrawlBallFixtureEnv(
        width=EYE_WIDTH,
        height=EYE_HEIGHT,
        audition=AuditionConfig.quality_48k(
            continuous=ContinuousSoundConfig()
        ),
    )
    video_encoder: subprocess.Popen[bytes] | None = None
    try:
        reset_count = 0
        observation, info = environment.reset(seed=DEMO_SEED)
        reset_count += 1
        if info != {} or not environment.observation_space.contains(observation):
            raise AssertionError("initial public observation contract changed")
        scene_bytes = environment.scene_path.read_bytes()
        assets = Path(mimoEnv.__file__).resolve().parent / "assets"
        for resolved, token in (
            (assets / "mimo" / "MIMo_metav2.xml", "{{MIMO_META_XML}}"),
            (assets / "mimo" / "MIMo_modelv2.xml", "{{MIMO_MODEL_XML}}"),
            (assets, "{{MIMO_TEXTURE_DIR}}"),
        ):
            scene_bytes = scene_bytes.replace(
                escape(resolved.as_posix(), quote=True).encode("utf-8"),
                token.encode("ascii"),
            )
        if sha256(scene_bytes).hexdigest() != EXPECTED_SCENE_SHA256:
            raise AssertionError("crawl-ball materialized scene changed")
        recorder = environment.start_contact_recording()
        controller = UnsupportedBellyCrawlController(environment)
        state = controller.reset()
        ball_body_id = int(environment.model.body("ball").id)
        initial_ball_position = np.asarray(
            environment.data.xpos[ball_body_id], dtype=np.float64
        ).copy()
        model_snapshot = _model_snapshot(environment)
        mocap_pos = environment.data.mocap_pos.copy()
        mocap_quat = environment.data.mocap_quat.copy()

        action_digest = sha256()
        qpos_digest = sha256()
        ball_digest = sha256()
        audio_transport_digest = sha256()
        video_digest = sha256()
        frame_hashes: dict[str, dict[str, Any]] = {}
        unique_frame_hashes: set[str] = set()
        audio_blocks: list[Float32Array] = []
        audio_nonzero_blocks = 0
        raw_video_bytes = 0
        contact_left: Uint8Array | None = None
        contact_right: Uint8Array | None = None

        with TemporaryDirectory(
            prefix="continuous-embodied-demo-",
            dir=paths.output_directory,
        ) as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            video_only_path = temporary_directory / "video_only.mp4"
            temporary_wav_path = temporary_directory / paths.pressure_wav.name
            temporary_playback_path = temporary_directory / paths.playback_mp4.name
            temporary_manifest_path = temporary_directory / paths.manifest_json.name
            video_encoder = _start_video_encoder(
                video_only_path,
                ffmpeg_tools["ffmpeg_path"],
                repository_root,
            )
            if video_encoder.stdin is None:
                raise RuntimeError("video encoder stdin was not created")
            video_stream: BinaryIO = video_encoder.stdin

            for step_index in range(PUBLIC_STEPS):
                qpos_before = environment.data.qpos.copy()
                qvel_before = environment.data.qvel.copy()
                xfrc_before = environment.data.xfrc_applied.copy()
                action, state, _ = controller.compose(environment, state)
                if not np.array_equal(environment.data.qpos, qpos_before):
                    raise AssertionError("controller wrote qpos")
                if not np.array_equal(environment.data.qvel, qvel_before):
                    raise AssertionError("controller wrote qvel")
                if not np.array_equal(environment.data.xfrc_applied, xfrc_before):
                    raise AssertionError("controller wrote applied forces")
                if not environment.action_space.contains(action):
                    raise AssertionError("controller produced an invalid action")
                action_digest.update(
                    np.ascontiguousarray(action, dtype=np.float32).tobytes(order="C")
                )

                observation, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    raise AssertionError(
                        f"episode ended at step {step_index}: {terminated=}, {truncated=}"
                    )
                if not environment.observation_space.contains(observation):
                    raise AssertionError(f"invalid public observation at {step_index}")
                if np.any(environment.data.xfrc_applied != 0.0):
                    raise AssertionError("runtime applied force became nonzero")
                _assert_model_unchanged(environment, model_snapshot)
                if not np.array_equal(environment.data.mocap_pos, mocap_pos):
                    raise AssertionError("runtime mocap position changed")
                if not np.array_equal(environment.data.mocap_quat, mocap_quat):
                    raise AssertionError("runtime mocap orientation changed")

                qpos_digest.update(
                    np.ascontiguousarray(
                        environment.data.qpos, dtype=np.float64
                    ).tobytes(order="C")
                )
                ball_digest.update(
                    np.ascontiguousarray(
                        environment.data.xpos[ball_body_id], dtype=np.float64
                    ).tobytes(order="C")
                )
                audio_block = observation["audition"]
                if (
                    audio_block.shape != (2, AUDIO_BLOCK_SAMPLES)
                    or audio_block.dtype != np.float32
                    or not audio_block.flags.c_contiguous
                    or not np.all(np.isfinite(audio_block))
                ):
                    raise AssertionError(f"invalid public audio block at {step_index}")
                audio_copy = np.ascontiguousarray(audio_block, dtype=np.float32).copy()
                audio_blocks.append(audio_copy)
                audio_transport_digest.update(audio_copy.tobytes(order="C"))
                audio_nonzero_blocks += int(np.any(audio_copy != 0.0))

                if step_index % 2 == 1:
                    left = np.ascontiguousarray(
                        observation["eye_left"], dtype=np.uint8
                    )
                    right = np.ascontiguousarray(
                        observation["eye_right"], dtype=np.uint8
                    )
                    if left.shape != (EYE_HEIGHT, EYE_WIDTH, 3) or right.shape != (
                        EYE_HEIGHT,
                        EYE_WIDTH,
                        3,
                    ):
                        raise AssertionError("public eye shape changed")
                    stereo = np.ascontiguousarray(
                        np.concatenate((left, right), axis=1), dtype=np.uint8
                    )
                    contents = stereo.tobytes(order="C")
                    video_stream.write(contents)
                    video_digest.update(contents)
                    frame_sha = sha256(contents).hexdigest()
                    unique_frame_hashes.add(frame_sha)
                    raw_video_bytes += len(contents)
                    frame_index = step_index // 2
                    if frame_index in {0, CONTACT_VIDEO_FRAME, VIDEO_FRAMES - 1}:
                        frame_hashes[str(frame_index)] = {
                            "frame_index": frame_index,
                            "public_step_index": step_index,
                            "rgb24_sha256": frame_sha,
                        }
                    if step_index == CONTACT_PUBLIC_STEP:
                        contact_left = left.copy()
                        contact_right = right.copy()

            _finish_video_encoder(video_encoder)
            video_encoder = None

            if reset_count != 1 or len(audio_blocks) != PUBLIC_STEPS:
                raise AssertionError("episode reset/audio-block count changed")
            if raw_video_bytes != VIDEO_FRAMES * VIDEO_WIDTH * VIDEO_HEIGHT * 3:
                raise AssertionError("raw public video byte count changed")
            if len(frame_hashes) != 3 or len(unique_frame_hashes) < 10:
                raise AssertionError("source-frame evidence is incomplete")
            if action_digest.hexdigest() != EXPECTED_ACTION_SHA256:
                raise AssertionError("crawl action hash changed")
            if qpos_digest.hexdigest() != EXPECTED_QPOS_SHA256:
                raise AssertionError("fixture qpos hash changed")
            if ball_digest.hexdigest() != EXPECTED_BALL_TRAJECTORY_SHA256:
                raise AssertionError("free-ball trajectory hash changed")
            if audio_transport_digest.hexdigest() != EXPECTED_AUDIO_TRANSPORT_SHA256:
                raise AssertionError("public audition transport hash changed")
            if audio_nonzero_blocks != EXPECTED_AUDIO_NONZERO_BLOCKS:
                raise AssertionError("public audition nonzero-block count changed")

            contact_validation = _validate_contacts(recorder.records)
            if contact_left is None or contact_right is None:
                raise AssertionError("contact public eye frame was not captured")
            left_sha = sha256(contact_left.tobytes(order="C")).hexdigest()
            right_sha = sha256(contact_right.tobytes(order="C")).hexdigest()
            left_red = _red_pixel_count(contact_left)
            right_red = _red_pixel_count(contact_right)
            if (
                left_sha != EXPECTED_CONTACT_LEFT_EYE_SHA256
                or right_sha != EXPECTED_CONTACT_RIGHT_EYE_SHA256
                or left_sha == right_sha
                or left_red != EXPECTED_CONTACT_LEFT_RED_PIXELS
                or right_red != EXPECTED_CONTACT_RIGHT_RED_PIXELS
            ):
                raise AssertionError("binocular contact-frame visibility changed")

            final_ball_position = np.asarray(
                environment.data.xpos[ball_body_id], dtype=np.float64
            ).copy()
            if not np.allclose(
                final_ball_position,
                np.asarray(EXPECTED_BALL_FINAL_POSITION_M),
                rtol=0,
                atol=1e-12,
            ):
                raise AssertionError("free-ball final position changed")
            planar_displacement = float(
                np.linalg.norm(
                    final_ball_position[:2] - initial_ball_position[:2]
                )
            )
            if abs(planar_displacement - EXPECTED_BALL_PLANAR_DISPLACEMENT_M) > 1e-12:
                raise AssertionError("free-ball displacement changed")

            pressure = np.ascontiguousarray(
                np.concatenate(audio_blocks, axis=1), dtype=np.float32
            )
            if pressure.shape != (2, AUDIO_SAMPLES_PER_EAR):
                raise AssertionError("public pressure sample count changed")
            peak_pressure_pa = float(np.max(np.abs(pressure)))
            rms_pressure_pa = np.sqrt(
                np.mean(pressure.astype(np.float64) ** 2, axis=1)
            )
            if (
                abs(peak_pressure_pa - EXPECTED_AUDIO_PEAK_PA) > 1e-12
                or abs(float(rms_pressure_pa[0]) - EXPECTED_AUDIO_LEFT_RMS_PA) > 1e-12
                or abs(float(rms_pressure_pa[1]) - EXPECTED_AUDIO_RIGHT_RMS_PA) > 1e-12
            ):
                raise AssertionError("public pressure metrics changed")
            gained_peak = peak_pressure_pa * MEDIA_MONITOR_GAIN
            if gained_peak >= 1.0:
                raise AssertionError("fixed monitor gain clips the source peak")

            pressure_channels_first = np.ascontiguousarray(
                pressure, dtype=np.dtype("<f4")
            )
            pressure_interleaved = np.ascontiguousarray(
                pressure.T, dtype=np.dtype("<f4")
            )
            pressure_channels_first_sha = sha256(
                pressure_channels_first.tobytes(order="C")
            ).hexdigest()
            pressure_interleaved_sha = sha256(
                pressure_interleaved.tobytes(order="C")
            ).hexdigest()
            wavfile.write(
                temporary_wav_path,
                AUDIO_SAMPLE_RATE_HZ,
                pressure_interleaved,
            )
            wav_rate, wav_pressure = wavfile.read(temporary_wav_path)
            if (
                wav_rate != AUDIO_SAMPLE_RATE_HZ
                or wav_pressure.dtype != np.float32
                or wav_pressure.shape != (AUDIO_SAMPLES_PER_EAR, 2)
                or not np.array_equal(wav_pressure, pressure_interleaved)
            ):
                raise AssertionError("lossless public pressure WAV changed")

            _encode_playback(
                video_only_path,
                temporary_wav_path,
                temporary_playback_path,
                ffmpeg_tools["ffmpeg_path"],
                repository_root,
            )
            media_validation = _validate_media(
                temporary_playback_path,
                ffmpeg_tools,
                repository_root,
            )

            manifest: dict[str, Any] = {
                "schema_version": 1,
                "title": "BlightWorld continuous embodied crawl/ball demonstration",
                "description": (
                    "One uninterrupted scripted actuator-only belly/army crawl "
                    "with genuine finger contact against a free sphere. This is "
                    "not a turn, pick, lift, learned policy, or autonomous behavior."
                ),
                "listening_note": "Headphones recommended for binaural playback.",
                "configuration": {
                    "seed": DEMO_SEED,
                    "duration_seconds": DURATION_SECONDS,
                    "reset_count": reset_count,
                    "public_environment_steps": PUBLIC_STEPS,
                    "physics_substeps": PHYSICS_SUBSTEPS,
                    "public_step_seconds": PUBLIC_STEP_SECONDS,
                    "video_frames": VIDEO_FRAMES,
                    "video_width": VIDEO_WIDTH,
                    "video_height": VIDEO_HEIGHT,
                    "video_frame_rate_hz": VIDEO_FRAME_RATE_HZ,
                    "audio_sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
                    "audio_samples_per_ear": AUDIO_SAMPLES_PER_EAR,
                    "behavior_label": "scripted_actuator_only_belly_army_crawl",
                    "camera_source": "public_eye_left_plus_public_eye_right",
                    "audio_source": "public_audition_pressure_only",
                    "cuts_or_overlays": False,
                    "external_camera": False,
                    "external_forces": False,
                    "runtime_state_or_model_writes": False,
                    "synthetic_sound_injection": False,
                },
                "media_pressure_mapping": {
                    "raw_wav_units": "pascals",
                    "raw_wav_gain": 1.0,
                    "playback_fixed_monitor_gain": MEDIA_MONITOR_GAIN,
                    "gain_selected_before_rendering": True,
                    "automatic_gain_control": False,
                    "normalization": False,
                    "time_varying_gain": False,
                    "raw_peak_pressure_pa": peak_pressure_pa,
                    "fixed_gain_peak_full_scale_before_aac": gained_peak,
                    "fixed_gain_clips": False,
                },
                "provenance": {
                    "script_path": str(
                        Path(__file__).resolve().relative_to(repository_root)
                    ),
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
                    "scene_sha256": EXPECTED_SCENE_SHA256,
                    "action_float32_step_major_sha256": action_digest.hexdigest(),
                    "qpos_float64_post_step_sha256": qpos_digest.hexdigest(),
                    "ball_xpos_float64_post_step_sha256": ball_digest.hexdigest(),
                    "audition_block_transport_sha256": (
                        audio_transport_digest.hexdigest()
                    ),
                    "audition_pressure_channels_first_sha256": (
                        pressure_channels_first_sha
                    ),
                    "audition_pressure_interleaved_sha256": (
                        pressure_interleaved_sha
                    ),
                    "audition_nonzero_blocks": audio_nonzero_blocks,
                    "audition_rms_pa_left_right": [
                        float(rms_pressure_pa[0]),
                        float(rms_pressure_pa[1]),
                    ],
                    "raw_video_rgb24_frame_major_sha256": video_digest.hexdigest(),
                    "raw_video_bytes": raw_video_bytes,
                    "unique_source_frame_hashes": len(unique_frame_hashes),
                    "source_frame_hashes": frame_hashes,
                    "contact_frame": {
                        "video_frame_index": CONTACT_VIDEO_FRAME,
                        "public_step_index": CONTACT_PUBLIC_STEP,
                        "eye_left_sha256": left_sha,
                        "eye_right_sha256": right_sha,
                        "eye_left_red_pixels": left_red,
                        "eye_right_red_pixels": right_red,
                        "binocular_images_equal": False,
                    },
                },
                "physics_validation": {
                    "contacts": contact_validation,
                    "ball_initial_position_m": initial_ball_position.tolist(),
                    "ball_final_position_m": final_ball_position.tolist(),
                    "ball_planar_displacement_m": planar_displacement,
                    "runtime_qpos_qvel_writes": 0,
                    "runtime_model_equality_mocap_writes": 0,
                    "runtime_xfrc_nonzero_steps": 0,
                },
                "media_validation": media_validation,
                "artifacts": {
                    "playback_mp4": {
                        "path": paths.playback_mp4.name,
                        "sha256": _sha256_file(temporary_playback_path),
                        "video": "H.264 CRF 18, 640x240, 50 fps",
                        "audio": "AAC 192 kb/s, 48 kHz stereo, fixed gain 0.04",
                    },
                    "pressure_wav": {
                        "path": paths.pressure_wav.name,
                        "sha256": _sha256_file(temporary_wav_path),
                        "format": "IEEE float32, 48 kHz stereo pressure in pascals",
                    },
                },
            }
            temporary_manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_playback_path.replace(paths.playback_mp4)
            temporary_wav_path.replace(paths.pressure_wav)
            temporary_manifest_path.replace(paths.manifest_json)
            return manifest
    finally:
        _stop_video_encoder(video_encoder)
        environment.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/vishnu/neuroips_outputs/continuous_embodied_demo"),
    )
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    manifest = export_demo(arguments.output, overwrite=arguments.overwrite)
    paths = _paths(arguments.output)
    print(f"wrote {paths.playback_mp4}")
    print(f"wrote {paths.pressure_wav}")
    print(f"wrote {paths.manifest_json}")
    print(
        "source hashes:",
        manifest["public_source_validation"]["raw_video_rgb24_frame_major_sha256"],
        manifest["public_source_validation"]["audition_block_transport_sha256"],
    )


if __name__ == "__main__":
    main()

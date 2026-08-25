"""Stream the calibrated constraint-assisted three-object episode.

Root travel is carried by an explicitly disclosed six-DOF mocap weld and each
pickup by one target-specific mocap grip weld. This is not free walking or a
physical grasp. No zero-action fallback, object teleport, external force,
synthetic sound, cut, or external camera is substituted. Video uses only the
two public MIMo eye observations and audio only public binaural pressure.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import import_module, metadata
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO, Protocol

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from blight_world.three_object_scenario import (
    ControllerHandoffPendingError,
    PICKUP_MECHANISM,
    ROOT_MOTION_MECHANISM,
    SCENARIO_BODY_NAMES,
    SCENARIO_DURATION_SECONDS,
    SCENARIO_TOTAL_STEPS,
    ScenarioContext,
    ScenarioControllerHooks,
    ScenarioEvent,
    ScenarioEventKind,
    ScenarioObject,
    ScenarioPhase,
    ScenarioSchedule,
    ThreeObjectScenarioEnv,
    dispatch_controller,
    next_object,
    require_controller,
)
from blight_world.constraint_assisted_three_object import (
    ConstraintAssistedThreeObjectController,
    constraint_assisted_three_object_controller_factory,
)
from mimo_audition import AuditionConfig
from export_continuous_embodied_demo import (
    _git_state,
    _run_bytes_checked,
    _run_checked,
    _sha256_file,
    _validate_ffmpeg,
)


Float32Array = NDArray[np.float32]

DEFAULT_OUTPUT_DIRECTORY = Path(
    "/home/vishnu/neuroips_outputs/three_object_walk_pick_throw"
)
EYE_WIDTH = 320
EYE_HEIGHT = 240
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 240
VIDEO_FRAME_RATE_HZ = 50
AUDIO_SAMPLE_RATE_HZ = 48_000
AUDIO_BLOCK_SAMPLES = 480
VIDEO_FRAMES = 2_250
AUDIO_SAMPLES_PER_EAR = 2_160_000
DEFAULT_MONITOR_GAIN = 0.04

# Each gate is an observed 320x240 lower bound from the proven candidate.
# The subject offset is cyclic relative to the current cycle target.
SEGMENTATION_VISIBILITY_GATES = (
    ("approach", 299, 0, 896),
    ("held_lift", 649, 0, 1_383),
    ("first_other_object_look", 849, 1, 71),
    ("second_other_object_look", 999, 2, 15),
    ("release", 1_200, 0, 859),
    ("floor_impact", 1_237, 0, 138),
    ("settle", 1_449, 0, 35),
)

EXPECTED_ACTION_SHA256 = (
    "9c5efe7aea08c439f1e63a05564ef2685887c2b6b8ab227d3cb6b4a4596644b4"
)
EXPECTED_QPOS_SHA256 = (
    "0d4fc3e66e7bfc9fffc5d83bad98f1df910a9eef0adb88b9b3cf227ee0dbc9f5"
)
EXPECTED_OBJECT_XPOS_SHA256 = (
    "efa8c9b105e16fe049303c28397d8311b5e26bdbfd97e0f3543d5745edbac3c0"
)
EXPECTED_AUDIO_SHA256 = (
    "1ea8c4667cc7a3e879c38f4a7647e39941d6fee604e982bda975d2e0aaa9920c"
)


class ControllerFactory(Protocol):
    """Factory resolved only after the dedicated MuJoCo model is loaded."""

    def __call__(
        self,
        environment: ThreeObjectScenarioEnv,
        schedule: ScenarioSchedule,
    ) -> ScenarioControllerHooks:
        """Construct name-resolved controller hooks for this model instance."""


@dataclass(frozen=True, slots=True)
class ExportPaths:
    """Only the three artifact paths owned by this exporter."""

    output_directory: Path
    playback_mp4: Path
    pressure_wav: Path
    manifest_json: Path


def output_paths(output_directory: Path) -> ExportPaths:
    """Resolve canonical paths without touching the filesystem."""

    output = output_directory.resolve()
    return ExportPaths(
        output_directory=output,
        playback_mp4=output / "three_object_walk_pick_throw.mp4",
        pressure_wav=output / "three_object_walk_pick_throw_pressure_pa.wav",
        manifest_json=output / "three_object_walk_pick_throw_manifest.json",
    )


def planned_phase_records(
    schedule: ScenarioSchedule | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return JSON-ready planned phase/event times before controller execution."""

    active_schedule = ScenarioSchedule.default() if schedule is None else schedule
    return tuple(
        {
            "cycle_index": window.cycle_index,
            "phase": window.phase.value,
            "target": window.target.value,
            "gaze_schedule_reference": "planned_gaze_windows",
            "start_public_step": window.start_step,
            "stop_public_step_exclusive": window.stop_step,
            "start_time_seconds": (
                window.start_step * active_schedule.public_step_seconds
            ),
            "stop_time_seconds": (
                window.stop_step * active_schedule.public_step_seconds
            ),
        }
        for window in active_schedule.windows
    )


def planned_gaze_records() -> tuple[dict[str, Any], ...]:
    """Return the four exact, half-open gaze windows in each 15 s cycle."""

    local_windows = (
        ("current_target_acquire_and_lift", 0, 700, 0),
        ("first_other_object", 700, 850, 1),
        ("second_other_object", 850, 1_000, 2),
        ("current_target_release_and_follow", 1_000, 1_500, 0),
    )
    records: list[dict[str, Any]] = []
    for cycle_index, target in enumerate(ScenarioObject):
        cycle_start = cycle_index * 1_500
        subjects = (target, next_object(target), next_object(next_object(target)))
        for label, local_start, local_stop, subject_index in local_windows:
            records.append(
                {
                    "cycle_index": cycle_index,
                    "label": label,
                    "subject": subjects[subject_index].value,
                    "start_public_step": cycle_start + local_start,
                    "stop_public_step_exclusive": cycle_start + local_stop,
                    "start_time_seconds": (cycle_start + local_start) * 0.01,
                    "stop_time_seconds": (cycle_start + local_stop) * 0.01,
                }
            )
    return tuple(records)


def build_plan_manifest(
    schedule: ScenarioSchedule | None = None,
) -> dict[str, Any]:
    """Describe the calibrated storyboard and its disclosed constraints."""

    active_schedule = ScenarioSchedule.default() if schedule is None else schedule
    return {
        "schema_version": 2,
        "status": "calibrated_constraint_assisted_controller_available",
        "title": "Constraint-assisted three-object stepping and throw scenario",
        "description": (
            "One uninterrupted binocular and binaural episode. Root motion is "
            "constraint-carried stepping, not free walking; pickup is assisted "
            "by a six-DOF mocap grip weld, not a physical hand grasp. Released "
            "objects are unconstrained and reach measured floor impacts."
        ),
        "output_directory": str(DEFAULT_OUTPUT_DIRECTORY),
        "clock": {
            "public_step_seconds": active_schedule.public_step_seconds,
            "public_steps": active_schedule.total_steps,
            "duration_seconds": active_schedule.duration_seconds,
            "video_frame_rate_hz": VIDEO_FRAME_RATE_HZ,
            "video_frames": active_schedule.total_steps // 2,
            "audio_sample_rate_hz": AUDIO_SAMPLE_RATE_HZ,
            "audio_samples_per_ear": (
                active_schedule.total_steps * AUDIO_BLOCK_SAMPLES
            ),
        },
        "media_sources": {
            "video": "public_eye_left_plus_public_eye_right",
            "audio": "public_audition_pressure_only",
            "cuts": False,
            "overlays": False,
            "external_camera": False,
        },
        "physics_contract": {
            "initial_object_staging": "scene_xml_only",
            "runtime_object_qpos_qvel_writes": False,
            "runtime_external_forces": False,
            "root_motion_mechanism": ROOT_MOTION_MECHANISM,
            "pickup_mechanism": PICKUP_MECHANISM,
            "always_active_root_mocap_weld": "demo_root_support",
            "one_active_grip_weld_maximum": True,
            "grip_active_local_time_seconds": [6.0, 12.0],
            "post_release_objects_unconstrained": True,
            "synthetic_sound_injection": False,
        },
        "controller_hooks": [
            "compose_locomotion",
            "compose_grasp",
            "compose_gaze",
            "compose_release",
            "compose_settle",
        ],
        "planned_phase_events": list(planned_phase_records(active_schedule)),
        "planned_gaze_windows": list(planned_gaze_records()),
    }


def _prepare_output(paths: ExportPaths, *, overwrite: bool) -> None:
    paths.output_directory.mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for path in (paths.playback_mp4, paths.pressure_wav, paths.manifest_json)
        if path.exists()
    ]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite exporter-owned files: {joined}")


def _model_snapshot(environment: ThreeObjectScenarioEnv) -> dict[str, NDArray[Any]]:
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
    environment: ThreeObjectScenarioEnv,
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


def _assert_allowed_constraint_state(
    environment: ThreeObjectScenarioEnv,
    controller: ConstraintAssistedThreeObjectController,
    context: ScenarioContext,
    baseline_eq_active: NDArray[np.integer],
) -> int:
    """Audit exactly one root weld and the scheduled target grip weld."""

    root_id = controller.root_support_equality_id
    grip_ids = controller.grip_equality_ids
    allowed_ids = {root_id, *grip_ids.values()}
    if int(environment.data.eq_active[root_id]) != 1:
        raise AssertionError("demo_root_support must remain active")
    local_step = context.step_index % 1_500
    expected_target = context.target if 600 <= local_step < 1_200 else None
    active_grips = 0
    for target, equality_id in grip_ids.items():
        expected = int(target is expected_target)
        actual = int(environment.data.eq_active[equality_id])
        if actual != expected:
            raise AssertionError(
                f"grip equality schedule changed for {target.value}: {actual}"
            )
        active_grips += actual
    for equality_id, expected in enumerate(baseline_eq_active):
        if equality_id in allowed_ids:
            continue
        if int(environment.data.eq_active[equality_id]) != int(expected):
            raise AssertionError(f"unapproved equality changed: {equality_id}")
    if active_grips > 1:
        raise AssertionError("more than one demo grip weld is active")
    if environment.model.nmocap != 2:
        raise AssertionError("demo must contain exactly the root and grip mocap bodies")
    if not np.all(np.isfinite(environment.data.mocap_pos)) or not np.all(
        np.isfinite(environment.data.mocap_quat)
    ):
        raise AssertionError("mocap state became non-finite")
    return active_grips


def _cyclic_subject(
    target: ScenarioObject,
    offset: int,
) -> ScenarioObject:
    subject = target
    for _ in range(offset):
        subject = next_object(subject)
    return subject


def _segmentation_metrics(
    segmentation: NDArray[np.integer],
    geom_id: int,
) -> dict[str, Any]:
    """Measure one MuJoCo geom in an object-ID/type segmentation image."""

    if segmentation.shape != (EYE_HEIGHT, EYE_WIDTH, 2):
        raise AssertionError(
            f"segmentation shape changed: {segmentation.shape}"
        )
    mask = (segmentation[..., 0] == geom_id) & (
        segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    )
    y, x = np.nonzero(mask)
    return {
        "geom_pixels": int(x.size),
        "bbox_xyxy": (
            None
            if x.size == 0
            else [int(x.min()), int(y.min()), int(x.max()), int(y.max())]
        ),
        "centroid_xy_fraction": (
            None
            if x.size == 0
            else [
                float((x.mean() + 0.5) / EYE_WIDTH),
                float((y.mean() + 0.5) / EYE_HEIGHT),
            ]
        ),
    }


def capture_segmentation_visibility_gate(
    renderer: mujoco.Renderer,
    environment: ThreeObjectScenarioEnv,
    context: ScenarioContext,
    *,
    label: str,
    local_step: int,
    subject_offset: int,
    minimum_pixels_per_eye: int,
) -> dict[str, Any]:
    """Capture and enforce one real-geom gate through both public eye cameras."""

    if context.step_index % 1_500 != local_step:
        raise ValueError("visibility gate does not match the scenario clock")
    subject = _cyclic_subject(context.target, subject_offset)
    geom_id = int(
        environment.model.geom(f"demo_{subject.value}_geom").id
    )
    eye_records: dict[str, Any] = {}
    for eye in ("eye_left", "eye_right"):
        renderer.update_scene(environment.data, camera=eye)
        segmentation = np.asarray(renderer.render()).copy()
        metrics = _segmentation_metrics(segmentation, geom_id)
        pixels = int(metrics["geom_pixels"])
        if pixels < minimum_pixels_per_eye:
            raise AssertionError(
                f"{label} {subject.value} is not segmentation-visible in "
                f"{eye}: {pixels} < {minimum_pixels_per_eye} pixels"
            )
        eye_records[eye] = metrics
    return {
        "cycle_index": context.cycle_index,
        "label": label,
        "subject": subject.value,
        "post_public_step_index": context.step_index,
        "data_time_seconds": float(environment.data.time),
        "minimum_pixels_per_eye": minimum_pixels_per_eye,
        "source": "mujoco_geom_segmentation_through_public_eye_cameras",
        "eyes": eye_records,
    }


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
    *,
    monitor_gain: float,
    duration_seconds: float,
    ffmpeg: str,
    repository_root: Path,
) -> None:
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
            f"volume={monitor_gain:.17g}",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            str(AUDIO_SAMPLE_RATE_HZ),
            "-ac",
            "2",
            "-t",
            f"{duration_seconds:.3f}",
            "-movflags",
            "+faststart",
            str(playback_path),
        ),
        cwd=repository_root,
    )


def _validate_media(
    playback_path: Path,
    *,
    expected_duration_seconds: float,
    expected_frames: int,
    ffmpeg_tools: dict[str, str],
    repository_root: Path,
) -> dict[str, Any]:
    probe_result = _run_checked(
        (
            ffmpeg_tools["ffprobe_path"],
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(playback_path),
        ),
        cwd=repository_root,
    )
    probe = json.loads(probe_result.stdout)
    video = next(row for row in probe["streams"] if row["codec_type"] == "video")
    audio = next(row for row in probe["streams"] if row["codec_type"] == "audio")
    if not (
        video["codec_name"] == "h264"
        and int(video["width"]) == VIDEO_WIDTH
        and int(video["height"]) == VIDEO_HEIGHT
        and video["avg_frame_rate"] == f"{VIDEO_FRAME_RATE_HZ}/1"
        and int(video["nb_read_frames"]) == expected_frames
        and audio["codec_name"] == "aac"
        and int(audio["sample_rate"]) == AUDIO_SAMPLE_RATE_HZ
        and int(audio["channels"]) == 2
    ):
        raise AssertionError("encoded media stream contract changed")
    duration = float(probe["format"]["duration"])
    if abs(duration - expected_duration_seconds) > 0.02:
        raise AssertionError(f"encoded duration changed: {duration}")
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
    decoded_bytes = _run_bytes_checked(
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
    decoded = np.frombuffer(decoded_bytes, dtype=np.dtype("<f4"))
    if decoded.size == 0 or decoded.size % 2 or not np.all(np.isfinite(decoded)):
        raise AssertionError("decoded playback audio is malformed")
    decoded_peak = float(np.max(np.abs(decoded)))
    if decoded_peak >= 1.0:
        raise AssertionError(f"fixed-gain playback audio clipped: {decoded_peak}")
    return {
        "duration_seconds": duration,
        "video_frames": int(video["nb_read_frames"]),
        "decoded_audio_samples_per_channel": decoded.size // 2,
        "decoded_audio_peak_full_scale": decoded_peak,
        "decoded_audio_clipped": False,
        "full_decode_exit_code": 0,
        "ffprobe": probe,
    }


def _event_record(event: ScenarioEvent) -> dict[str, Any]:
    return {
        "kind": event.kind.value,
        "step_index": event.step_index,
        "time_seconds": event.time_seconds,
        "cycle_index": event.cycle_index,
        "target": event.target.value,
        "subject": None if event.subject is None else event.subject.value,
        "metrics": dict(event.metrics),
    }


def _validate_step_events(
    events: tuple[ScenarioEvent, ...],
    context: ScenarioContext,
    data_time_seconds: float,
) -> None:
    allowed_phases = {
        ScenarioEventKind.STATION_REACHED: {
            ScenarioPhase.GO_TO_TARGET,
            ScenarioPhase.PICK_TARGET,
        },
        ScenarioEventKind.GRIP_CONSTRAINT_ACTIVATED: {
            ScenarioPhase.PICK_TARGET
        },
        ScenarioEventKind.HELD_OBJECT_INSPECTED: {
            ScenarioPhase.PICK_TARGET
        },
        ScenarioEventKind.NEXT_OBJECT_GLANCE_MEASURED: {
            ScenarioPhase.INSPECT_HELD_TARGET
        },
        ScenarioEventKind.SECOND_OTHER_OBJECT_GLANCE_MEASURED: {
            ScenarioPhase.GLANCE_AT_NEXT_OBJECT
        },
        ScenarioEventKind.GRIP_CONSTRAINT_RELEASED: {
            ScenarioPhase.THROW_TO_FLOOR
        },
        ScenarioEventKind.FLOOR_IMPACT: {
            ScenarioPhase.THROW_TO_FLOOR,
            ScenarioPhase.SETTLE,
        },
        ScenarioEventKind.PHASE_FAILED: set(ScenarioPhase),
    }
    for event in events:
        if (
            event.step_index != context.step_index
            or event.cycle_index != context.cycle_index
            or event.target is not context.target
            or abs(event.time_seconds - data_time_seconds) > 5e-11
        ):
            raise AssertionError(f"controller event clock/context mismatch: {event}")
        if context.phase not in allowed_phases[event.kind]:
            raise AssertionError(f"controller event emitted in wrong phase: {event}")
        if event.kind is ScenarioEventKind.HELD_OBJECT_INSPECTED:
            if event.subject is not context.target:
                raise AssertionError(f"held inspection subject mismatch: {event}")
        elif event.kind is ScenarioEventKind.NEXT_OBJECT_GLANCE_MEASURED:
            if event.subject is not next_object(context.target):
                raise AssertionError(f"next glance subject mismatch: {event}")
        elif event.kind is ScenarioEventKind.SECOND_OTHER_OBJECT_GLANCE_MEASURED:
            if event.subject is not next_object(next_object(context.target)):
                raise AssertionError(f"second other glance subject mismatch: {event}")
        elif event.subject is not None:
            raise AssertionError(f"non-look event cannot define a subject: {event}")
        if event.kind is ScenarioEventKind.PHASE_FAILED:
            raise RuntimeError(f"controller reported physical phase failure: {event}")


def validate_event_coverage(
    events: Sequence[ScenarioEvent],
) -> dict[str, Any]:
    """Require physical success milestones before publishing showcase media."""

    coverage: dict[str, Any] = {}
    for cycle_index, target in enumerate(ScenarioObject):
        cycle_events = tuple(
            event
            for event in events
            if event.cycle_index == cycle_index and event.target is target
        )
        kinds = {event.kind for event in cycle_events}
        required = {
            ScenarioEventKind.STATION_REACHED,
            ScenarioEventKind.GRIP_CONSTRAINT_ACTIVATED,
            ScenarioEventKind.HELD_OBJECT_INSPECTED,
            ScenarioEventKind.NEXT_OBJECT_GLANCE_MEASURED,
            ScenarioEventKind.SECOND_OTHER_OBJECT_GLANCE_MEASURED,
            ScenarioEventKind.GRIP_CONSTRAINT_RELEASED,
            ScenarioEventKind.FLOOR_IMPACT,
        }
        missing = required - kinds
        held_subjects = {
            event.subject
            for event in cycle_events
            if event.kind is ScenarioEventKind.HELD_OBJECT_INSPECTED
        }
        next_subjects = {
            event.subject
            for event in cycle_events
            if event.kind is ScenarioEventKind.NEXT_OBJECT_GLANCE_MEASURED
        }
        second_other_subjects = {
            event.subject
            for event in cycle_events
            if event.kind
            is ScenarioEventKind.SECOND_OTHER_OBJECT_GLANCE_MEASURED
        }
        if (
            missing
            or held_subjects != {target}
            or next_subjects != {next_object(target)}
            or second_other_subjects
            != {next_object(next_object(target))}
        ):
            raise AssertionError(
                f"cycle {cycle_index} lacks physical event coverage: "
                f"missing={sorted(kind.value for kind in missing)}, "
                f"held_subjects={held_subjects}, next_subjects={next_subjects}, "
                f"second_other_subjects={second_other_subjects}"
            )
        coverage[str(cycle_index)] = {
            "target": target.value,
            "required_events_present": sorted(kind.value for kind in required),
            "held_inspection_subject": target.value,
            "next_glance_subject": next_object(target).value,
            "second_other_glance_subject": next_object(
                next_object(target)
            ).value,
            "event_count": len(cycle_events),
        }
    return coverage


def _load_controller_factory(specification: str) -> ControllerFactory:
    """Load ``module:attribute`` without inventing a built-in fallback."""

    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("controller factory must use module:attribute syntax")
    candidate = getattr(import_module(module_name), attribute_name)
    if not callable(candidate):
        raise TypeError("controller factory attribute must be callable")
    return candidate


def export_scenario(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    controller_factory: ControllerFactory | None = (
        constraint_assisted_three_object_controller_factory
    ),
    monitor_gain: float = DEFAULT_MONITOR_GAIN,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run and publish one validated uninterrupted public-sensor episode."""

    if controller_factory is None:
        raise ControllerHandoffPendingError(
            "export requires a validated controller factory; no active fallback exists"
        )
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("set MUJOCO_GL=egl before starting the exporter")
    monitor_gain = float(monitor_gain)
    if not isfinite(monitor_gain) or monitor_gain <= 0.0:
        raise ValueError("monitor_gain must be finite and positive")

    schedule = ScenarioSchedule.default()
    if (
        schedule.total_steps != SCENARIO_TOTAL_STEPS
        or schedule.duration_seconds != SCENARIO_DURATION_SECONDS
        or schedule.total_steps // 2 != VIDEO_FRAMES
        or schedule.total_steps * AUDIO_BLOCK_SAMPLES != AUDIO_SAMPLES_PER_EAR
    ):
        raise AssertionError("export clock disagrees with the frozen schedule")

    repository_root = Path(__file__).resolve().parents[1]
    paths = output_paths(output_directory)
    _prepare_output(paths, overwrite=overwrite)
    ffmpeg_tools = _validate_ffmpeg(repository_root)
    git_state = _git_state(repository_root)
    script_sha256 = _sha256_file(Path(__file__).resolve())

    environment = ThreeObjectScenarioEnv(
        width=EYE_WIDTH,
        height=EYE_HEIGHT,
        audition=AuditionConfig.quality_48k(),
    )
    video_encoder: subprocess.Popen[bytes] | None = None
    segmentation_renderer: mujoco.Renderer | None = None
    try:
        observation, info = environment.reset(seed=1_401)
        if info != {} or not environment.observation_space.contains(observation):
            raise AssertionError("initial public observation contract changed")
        hooks = require_controller(controller_factory(environment, schedule))
        if not isinstance(hooks, ConstraintAssistedThreeObjectController):
            raise TypeError(
                "exporter requires ConstraintAssistedThreeObjectController audit API"
            )
        controller_state = hooks.reset(environment, schedule, 1_401)
        segmentation_renderer = mujoco.Renderer(
            environment.model,
            height=EYE_HEIGHT,
            width=EYE_WIDTH,
        )
        segmentation_renderer.enable_segmentation_rendering()

        model_snapshot = _model_snapshot(environment)
        baseline_eq_active = environment.data.eq_active.copy()
        object_body_ids = {
            candidate: int(environment.model.body(SCENARIO_BODY_NAMES[candidate]).id)
            for candidate in ScenarioObject
        }
        initial_object_positions = {
            candidate: environment.data.xpos[body_id].copy()
            for candidate, body_id in object_body_ids.items()
        }

        action_digest = sha256()
        qpos_digest = sha256()
        video_digest = sha256()
        audio_digest = sha256()
        object_digest = sha256()
        object_digests = {candidate: sha256() for candidate in ScenarioObject}
        frame_hashes: dict[str, dict[str, Any]] = {}
        event_frame_hashes: dict[str, dict[str, Any]] = {}
        pending_event_frames: list[ScenarioEvent] = []
        audio_blocks: list[Float32Array] = []
        events: list[ScenarioEvent] = []
        segmentation_visibility: list[dict[str, Any]] = []
        raw_video_bytes = 0
        maximum_root_position_error_m = 0.0
        maximum_active_grip_welds = 0

        with TemporaryDirectory(
            prefix="three-object-walk-pick-throw-",
            dir=paths.output_directory,
        ) as temporary_directory_name:
            temporary_directory = Path(temporary_directory_name)
            video_only_path = temporary_directory / "video_only.mp4"
            temporary_wav = temporary_directory / paths.pressure_wav.name
            temporary_mp4 = temporary_directory / paths.playback_mp4.name
            temporary_manifest = temporary_directory / paths.manifest_json.name
            video_encoder = _start_video_encoder(
                video_only_path,
                ffmpeg_tools["ffmpeg_path"],
                repository_root,
            )
            if video_encoder.stdin is None:
                raise RuntimeError("video encoder stdin was not created")
            video_stream: BinaryIO = video_encoder.stdin

            for step_index in range(schedule.total_steps):
                context = schedule.context_at(step_index)
                qpos_before = environment.data.qpos.copy()
                qvel_before = environment.data.qvel.copy()
                xfrc_before = environment.data.xfrc_applied.copy()
                controller_step = dispatch_controller(
                    hooks,
                    environment,
                    controller_state,
                    context,
                )
                if not np.array_equal(environment.data.qpos, qpos_before):
                    raise AssertionError("controller wrote qpos")
                if not np.array_equal(environment.data.qvel, qvel_before):
                    raise AssertionError("controller wrote qvel")
                if not np.array_equal(environment.data.xfrc_applied, xfrc_before):
                    raise AssertionError("controller wrote applied forces")
                _validate_step_events(
                    controller_step.events,
                    context,
                    float(environment.data.time),
                )
                events.extend(controller_step.events)
                pending_event_frames.extend(controller_step.events)
                controller_state = controller_step.state
                action = np.ascontiguousarray(
                    controller_step.action, dtype=np.float32
                )
                if not environment.action_space.contains(action):
                    raise AssertionError("controller produced an invalid action")
                active_grips = _assert_allowed_constraint_state(
                    environment,
                    hooks,
                    context,
                    baseline_eq_active,
                )
                maximum_active_grip_welds = max(
                    maximum_active_grip_welds, active_grips
                )
                action_digest.update(action.tobytes(order="C"))

                observation, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    raise AssertionError(f"episode ended at public step {step_index}")
                if not environment.observation_space.contains(observation):
                    raise AssertionError(f"invalid public observation at {step_index}")
                if np.any(environment.data.xfrc_applied != 0.0):
                    raise AssertionError("runtime applied force became nonzero")
                _assert_model_unchanged(environment, model_snapshot)
                active_grips = _assert_allowed_constraint_state(
                    environment,
                    hooks,
                    context,
                    baseline_eq_active,
                )
                maximum_active_grip_welds = max(
                    maximum_active_grip_welds, active_grips
                )

                diagnostics = dict(controller_step.diagnostics)
                root_target = np.asarray(
                    (
                        diagnostics["root_target_x_m"],
                        diagnostics["root_target_y_m"],
                        diagnostics["root_target_z_m"],
                    ),
                    dtype=np.float64,
                )
                maximum_root_position_error_m = max(
                    maximum_root_position_error_m,
                    float(
                        np.linalg.norm(
                            np.asarray(environment.data.body("mimo_location").xpos)
                            - root_target
                        )
                    ),
                )

                qpos_digest.update(
                    np.ascontiguousarray(
                        environment.data.qpos, dtype=np.float64
                    ).tobytes(order="C")
                )
                object_positions: list[NDArray[np.float64]] = []
                for candidate, body_id in object_body_ids.items():
                    position = np.asarray(
                        environment.data.xpos[body_id], dtype=np.float64
                    )
                    object_positions.append(position)
                    object_digests[candidate].update(
                        np.ascontiguousarray(position).tobytes(order="C")
                    )
                object_digest.update(
                    np.ascontiguousarray(
                        np.concatenate(object_positions), dtype=np.float64
                    ).tobytes(order="C")
                )

                local_step = step_index % 1_500
                for (
                    label,
                    gate_local_step,
                    subject_offset,
                    minimum_pixels,
                ) in SEGMENTATION_VISIBILITY_GATES:
                    if local_step == gate_local_step:
                        segmentation_visibility.append(
                            capture_segmentation_visibility_gate(
                                segmentation_renderer,
                                environment,
                                context,
                                label=label,
                                local_step=gate_local_step,
                                subject_offset=subject_offset,
                                minimum_pixels_per_eye=minimum_pixels,
                            )
                        )

                audio = np.ascontiguousarray(
                    observation["audition"], dtype=np.float32
                )
                if (
                    audio.shape != (2, AUDIO_BLOCK_SAMPLES)
                    or not np.all(np.isfinite(audio))
                ):
                    raise AssertionError(f"invalid public audition block at {step_index}")
                audio_blocks.append(audio.copy())
                audio_digest.update(audio.tobytes(order="C"))

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
                    raw_video_bytes += len(contents)
                    frame_index = step_index // 2
                    if frame_index in {0, VIDEO_FRAMES - 1} or any(
                        step_index in {window.start_step, window.start_step + 1}
                        for window in schedule.windows
                    ):
                        frame_hashes[str(frame_index)] = {
                            "frame_index": frame_index,
                            "public_step_index": step_index,
                            "rgb24_sha256": sha256(contents).hexdigest(),
                        }
                    for event in pending_event_frames:
                        key = f"{event.cycle_index}:{event.kind.value}"
                        event_frame_hashes[key] = {
                            "event": _event_record(event),
                            "frame_index": frame_index,
                            "public_step_index": step_index,
                            "stereo_rgb24_sha256": sha256(contents).hexdigest(),
                            "eye_left_rgb24_sha256": sha256(
                                left.tobytes(order="C")
                            ).hexdigest(),
                            "eye_right_rgb24_sha256": sha256(
                                right.tobytes(order="C")
                            ).hexdigest(),
                            "binocular_images_equal": bool(
                                np.array_equal(left, right)
                            ),
                        }
                    pending_event_frames.clear()

            _finish_video_encoder(video_encoder)
            video_encoder = None
            coverage = validate_event_coverage(events)
            if len(audio_blocks) != schedule.total_steps:
                raise AssertionError("public audio block count changed")
            if raw_video_bytes != VIDEO_FRAMES * VIDEO_WIDTH * VIDEO_HEIGHT * 3:
                raise AssertionError("raw public video byte count changed")
            if pending_event_frames:
                raise AssertionError("an event has no following public video frame")
            if len(segmentation_visibility) != (
                len(SEGMENTATION_VISIBILITY_GATES) * len(ScenarioObject)
            ):
                raise AssertionError("segmentation visibility gate count changed")
            if action_digest.hexdigest() != EXPECTED_ACTION_SHA256:
                raise AssertionError("calibrated action hash changed")
            if qpos_digest.hexdigest() != EXPECTED_QPOS_SHA256:
                raise AssertionError("calibrated qpos hash changed")
            if object_digest.hexdigest() != EXPECTED_OBJECT_XPOS_SHA256:
                raise AssertionError("calibrated object trajectory hash changed")
            actual_audio_sha256 = audio_digest.hexdigest()
            if actual_audio_sha256 != EXPECTED_AUDIO_SHA256:
                raise AssertionError(
                    "calibrated public audition hash changed: "
                    f"expected {EXPECTED_AUDIO_SHA256}, got {actual_audio_sha256}"
                )
            if maximum_active_grip_welds != 1:
                raise AssertionError("grip-weld occupancy changed")
            if maximum_root_position_error_m > 0.0041:
                raise AssertionError("root mocap-weld tracking error changed")
            if getattr(controller_state, "saturation_rows", None) != 0:
                raise AssertionError("calibrated controller saturated")

            pressure = np.ascontiguousarray(
                np.concatenate(audio_blocks, axis=1), dtype=np.float32
            )
            if pressure.shape != (2, AUDIO_SAMPLES_PER_EAR):
                raise AssertionError("public pressure sample count changed")
            peak_pressure_pa = float(np.max(np.abs(pressure)))
            gained_peak = monitor_gain * peak_pressure_pa
            if gained_peak >= 1.0:
                raise AssertionError(
                    f"predeclared monitor gain clips measured source: {gained_peak}"
                )
            pressure_interleaved = np.ascontiguousarray(
                pressure.T, dtype=np.dtype("<f4")
            )
            wavfile.write(temporary_wav, AUDIO_SAMPLE_RATE_HZ, pressure_interleaved)
            wav_rate, wav_pressure = wavfile.read(temporary_wav)
            if (
                wav_rate != AUDIO_SAMPLE_RATE_HZ
                or wav_pressure.dtype != np.float32
                or wav_pressure.shape != (AUDIO_SAMPLES_PER_EAR, 2)
                or not np.array_equal(wav_pressure, pressure_interleaved)
            ):
                raise AssertionError("lossless public pressure WAV changed")

            _encode_playback(
                video_only_path,
                temporary_wav,
                temporary_mp4,
                monitor_gain=monitor_gain,
                duration_seconds=schedule.duration_seconds,
                ffmpeg=ffmpeg_tools["ffmpeg_path"],
                repository_root=repository_root,
            )
            media_validation = _validate_media(
                temporary_mp4,
                expected_duration_seconds=schedule.duration_seconds,
                expected_frames=VIDEO_FRAMES,
                ffmpeg_tools=ffmpeg_tools,
                repository_root=repository_root,
            )

            final_object_positions = {
                candidate: environment.data.xpos[body_id].copy()
                for candidate, body_id in object_body_ids.items()
            }
            manifest: dict[str, Any] = {
                **build_plan_manifest(schedule),
                "status": "completed_constraint_assisted_event_validated",
                "controller": {"controller_id": hooks.controller_id},
                "actual_events": [_event_record(event) for event in events],
                "event_coverage": coverage,
                "configuration": {
                    "seed": 1_401,
                    "reset_count": 1,
                    "public_environment_steps": schedule.total_steps,
                    "video_frames": VIDEO_FRAMES,
                    "audio_samples_per_ear": AUDIO_SAMPLES_PER_EAR,
                    "camera_source": "public_eye_left_plus_public_eye_right",
                    "audio_source": "public_audition_pressure_only",
                    "monitor_gain_selected_before_rendering": monitor_gain,
                    "automatic_gain_control": False,
                    "normalization": False,
                    "cuts_or_overlays": False,
                    "root_motion_mechanism": ROOT_MOTION_MECHANISM,
                    "pickup_mechanism": PICKUP_MECHANISM,
                    "free_walking": False,
                    "physical_hand_grasp": False,
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
                    "action_float32_step_major_sha256": action_digest.hexdigest(),
                    "qpos_float64_post_step_sha256": qpos_digest.hexdigest(),
                    "combined_object_xpos_float64_post_step_sha256": (
                        object_digest.hexdigest()
                    ),
                    "object_xpos_float64_post_step_sha256": {
                        candidate.value: digest.hexdigest()
                        for candidate, digest in object_digests.items()
                    },
                    "audition_block_transport_sha256": audio_digest.hexdigest(),
                    "audition_pressure_channels_first_sha256": sha256(
                        pressure.tobytes(order="C")
                    ).hexdigest(),
                    "raw_video_rgb24_frame_major_sha256": video_digest.hexdigest(),
                    "raw_video_bytes": raw_video_bytes,
                    "source_frame_hashes": frame_hashes,
                    "event_frame_hashes": event_frame_hashes,
                    "segmentation_visibility_gates": segmentation_visibility,
                    "raw_peak_pressure_pa": peak_pressure_pa,
                    "fixed_gain_peak_full_scale_before_aac": gained_peak,
                },
                "physics_validation": {
                    "initial_object_positions_m": {
                        candidate.value: position.tolist()
                        for candidate, position in initial_object_positions.items()
                    },
                    "final_object_positions_m": {
                        candidate.value: position.tolist()
                        for candidate, position in final_object_positions.items()
                    },
                    "runtime_qpos_qvel_writes": 0,
                    "runtime_object_qpos_qvel_writes": 0,
                    "runtime_mocap_writes": (
                        "demo_root_anchor and demo_grip_anchor only"
                    ),
                    "runtime_equality_writes": (
                        "demo grip weld activation only; root weld always active"
                    ),
                    "runtime_xfrc_nonzero_steps": 0,
                    "fake_object_teleports": 0,
                    "maximum_active_grip_welds": maximum_active_grip_welds,
                    "maximum_root_position_error_m": (
                        maximum_root_position_error_m
                    ),
                    "action_saturation_rows": 0,
                    "objects_unconstrained_after_release": True,
                },
                "media_validation": media_validation,
                "artifacts": {
                    "playback_mp4": {
                        "path": paths.playback_mp4.name,
                        "sha256": _sha256_file(temporary_mp4),
                        "format": "H.264 CRF18 plus AAC192k stereo",
                    },
                    "pressure_wav": {
                        "path": paths.pressure_wav.name,
                        "sha256": _sha256_file(temporary_wav),
                        "format": "IEEE float32 48kHz stereo pressure in pascals",
                    },
                },
            }
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_wav.replace(paths.pressure_wav)
            temporary_mp4.replace(paths.playback_mp4)
            temporary_manifest.replace(paths.manifest_json)
            return manifest
    finally:
        _stop_video_encoder(video_encoder)
        if segmentation_renderer is not None:
            segmentation_renderer.close()
        environment.close()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="directory for exporter-owned MP4, WAV, and manifest files",
    )
    parser.add_argument(
        "--controller-factory",
        help=(
            "optional validated controller factory in module:attribute form; "
            "defaults to the calibrated constraint-assisted controller"
        ),
    )
    parser.add_argument(
        "--monitor-gain",
        type=float,
        default=DEFAULT_MONITOR_GAIN,
        help="single fixed playback gain selected before the episode",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the deterministic storyboard without running simulation",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    if arguments.plan_only:
        print(json.dumps(build_plan_manifest(), indent=2, sort_keys=True))
        return
    factory = (
        constraint_assisted_three_object_controller_factory
        if arguments.controller_factory is None
        else _load_controller_factory(arguments.controller_factory)
    )
    manifest = export_scenario(
        arguments.output,
        controller_factory=factory,
        monitor_gain=arguments.monitor_gain,
        overwrite=arguments.overwrite,
    )
    paths = output_paths(arguments.output)
    print(f"wrote {paths.playback_mp4}")
    print(f"wrote {paths.pressure_wav}")
    print(f"wrote {paths.manifest_json}")
    print(json.dumps(manifest["event_coverage"], sort_keys=True))


if __name__ == "__main__":
    main()

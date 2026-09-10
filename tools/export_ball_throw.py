#!/usr/bin/env python3
"""Export one short binocular, stereo-audio assisted ball throw.

The unrecorded pre-roll uses the validated three-object controller only until
cycle-zero local step 699, where the ball is visibly held.  The recorded five
seconds keep that disclosed hand-relative mocap weld through an
actuator-driven arm swing, disable it at release, and then leave the ball to
MuJoCo.  Video contains the two unmodified public eye observations.  No pickup
is shown or claimed, and no object qpos, qvel, applied force, or world-space
throw offset is written during the recorded segment.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from blight_world.assisted_ball_throw import (
    BALL_THROW_IMPACT_CLIP_ID,
    BALL_THROW_IMPACT_SOURCE_CLIP_ID,
    MAXIMUM_HAND_RELATIVE_ERROR_M,
    MAXIMUM_RELEASE_VELOCITY_ERROR_M_S,
    PRE_SWING_PUBLIC_STEPS,
    RECORDED_PUBLIC_STEPS,
    RELEASE_RECORDED_STEP,
    AssistedBallThrowController,
    BallThrowState,
    derive_single_hit_rubber_clip,
    onset_cluster_starts,
    prepare_held_ball,
)
from blight_world.three_object_scenario import (
    SCENARIO_BODY_NAMES,
    SCENARIO_SEED,
    ScenarioObject,
    ThreeObjectScenarioEnv,
)
from mimo_audition import AuditionConfig

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


DEFAULT_OUTPUT_DIRECTORY = Path("/home/vishnu/neuroips_outputs/ball_throw")
EYE_WIDTH = 320
EYE_HEIGHT = 240
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 240
VIDEO_FRAME_RATE_HZ = 50
AUDIO_SAMPLE_RATE_HZ = 48_000
AUDIO_BLOCK_SAMPLES = 480
DURATION_SECONDS = 5.0
VIDEO_FRAMES = RECORDED_PUBLIC_STEPS // 2
AUDIO_SAMPLES_PER_EAR = RECORDED_PUBLIC_STEPS * AUDIO_BLOCK_SAMPLES
DEFAULT_MONITOR_GAIN = 0.40
MINIMUM_HAND_FORWARD_EXCURSION_M = 0.08
MINIMUM_BODY_LOCAL_FORWARD_SPEED_M_S = 0.80
MAXIMUM_BODY_LOCAL_LATERAL_SPEED_M_S = 0.08
MINIMUM_BODY_LOCAL_VERTICAL_SPEED_M_S = 0.09
MAXIMUM_BODY_LOCAL_VERTICAL_SPEED_M_S = 0.30
MINIMUM_NON_HAND_BODY_CLEARANCE_M = 0.020
MINIMUM_HORIZONTAL_FLIGHT_M = 0.25
MINIMUM_AIR_TIME_SECONDS = 0.35
EVIDENCE_PUBLIC_STEPS = {
    "held": 1,
    "windup": 45,
    "swing": 95,
    "release": 99,
    "flight": 113,
}
OPEN_FLOOR_HIDDEN_GEOM_NAMES = (
    "demo_table_top",
    "demo_block_geom",
    "demo_cylinder_geom",
)
EYE_INTERIOR_GEOM_NAMES = (
    "head",
    "geom:left_eye1",
    "geom:right_eye1",
)


Float32Array = NDArray[np.float32]
Uint8Array = NDArray[np.uint8]
FrameSink = Callable[[bytes], None]


@dataclass(frozen=True, slots=True)
class ThrowPaths:
    """Exporter-owned canonical files."""

    output_directory: Path
    playback_mp4: Path
    pressure_wav: Path
    manifest_json: Path


@dataclass(frozen=True, slots=True)
class ImpactSchedulingRecord:
    """One physical hard/rubber event routed to the demo single-hit clip."""

    time_seconds: float
    recorded_sample_cursor: int
    detected_clip_id: str
    selected_clip_id: str
    gain_db: float
    impact_energy_j: float
    admitted: bool


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Bounded in-memory evidence from one recorded throw segment."""

    pressure_channels_first: Float32Array
    frame_count: int
    raw_video_sha256: str
    first_frame_sha256: str
    release_frame_sha256: str
    final_frame_sha256: str
    left_right_equal_frame_count: int
    final_state: BallThrowState
    release_velocity: dict[str, float]
    pre_swing_maximum_hand_relative_error_m: float
    hand_forward_excursion_m: float
    horizontal_flight_m: float
    air_time_seconds: float
    impact_records: tuple[ImpactSchedulingRecord, ...]
    impact_audio_correlation: dict[str, float]
    action_sha256: str
    qpos_sha256: str
    ball_position_sha256: str
    audio_transport_sha256: str
    ball_final_position_m: tuple[float, float, float]
    recorded_start_simulation_time_s: float
    evidence_visibility: dict[str, dict[str, int]]
    first_floor_contact_time_seconds: float
    pre_floor_other_ball_contacts: tuple[tuple[str, float], ...]


def output_paths(output_directory: Path) -> ThrowPaths:
    """Resolve the three files owned by this exporter."""

    output = output_directory.resolve()
    return ThrowPaths(
        output_directory=output,
        playback_mp4=output / "ball_throw.mp4",
        pressure_wav=output / "ball_throw_pressure_pa.wav",
        manifest_json=output / "ball_throw_manifest.json",
    )


def _prepare_output(paths: ThrowPaths, *, overwrite: bool) -> None:
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


def _configure_open_floor(
    environment: ThreeObjectScenarioEnv,
) -> dict[int, tuple[int, int, NDArray[np.float32]]]:
    """Hide and disable only the table and two unused task objects."""

    snapshots: dict[int, tuple[int, int, NDArray[np.float32]]] = {}
    for geom_name in OPEN_FLOOR_HIDDEN_GEOM_NAMES:
        geom_id = int(environment.model.geom(geom_name).id)
        snapshots[geom_id] = (
            int(environment.model.geom_contype[geom_id]),
            int(environment.model.geom_conaffinity[geom_id]),
            environment.model.geom_rgba[geom_id].copy(),
        )
        environment.model.geom_contype[geom_id] = 0
        environment.model.geom_conaffinity[geom_id] = 0
        environment.model.geom_rgba[geom_id, 3] = 0.0
    mujoco.mj_forward(environment.model, environment.data)
    return snapshots


def _restore_open_floor_geoms(
    environment: ThreeObjectScenarioEnv,
    snapshots: dict[int, tuple[int, int, NDArray[np.float32]]],
) -> None:
    for geom_id, (contype, conaffinity, rgba) in snapshots.items():
        environment.model.geom_contype[geom_id] = contype
        environment.model.geom_conaffinity[geom_id] = conaffinity
        environment.model.geom_rgba[geom_id] = rgba


def _install_single_hit_clip(
    environment: ThreeObjectScenarioEnv,
    recorded_start_time_s: float,
) -> tuple[list[ImpactSchedulingRecord], Callable[..., int | None]]:
    """Register and route physical ball-floor events without changing assets."""

    if environment.audition is None:
        raise AssertionError("ball-throw export requires audition")
    renderer = environment.audition.renderer
    source = renderer.clips.get(BALL_THROW_IMPACT_SOURCE_CLIP_ID)
    derived = derive_single_hit_rubber_clip(source)
    if onset_cluster_starts(derived, source.sample_rate_hz) != (0,):
        starts = onset_cluster_starts(derived, source.sample_rate_hz)
        if len(starts) != 1:
            raise AssertionError(f"single-hit clip has {len(starts)} onset clusters")
    renderer.clips.register(
        BALL_THROW_IMPACT_CLIP_ID,
        derived,
        reference_peak_pressure_pa_at_1m=(
            source.reference_peak_pressure_pa_at_1m
        ),
    )
    records: list[ImpactSchedulingRecord] = []
    original = renderer.play_transient
    recorded_start_cursor = renderer.sample_cursor

    def route_transient(
        clip_id: str,
        attachment: object,
        *,
        gain_db: float = 0.0,
        delay_s: float = 0.0,
        impact_energy_j: float = 0.0,
    ) -> int | None:
        selected = (
            BALL_THROW_IMPACT_CLIP_ID
            if clip_id.startswith("impact_hard_rubber_")
            else clip_id
        )
        voice_id = original(
            selected,
            attachment,  # type: ignore[arg-type]
            gain_db=gain_db,
            delay_s=delay_s,
            impact_energy_j=impact_energy_j,
        )
        records.append(
            ImpactSchedulingRecord(
                time_seconds=(
                    float(environment.data.time) - recorded_start_time_s
                ),
                recorded_sample_cursor=(
                    renderer.sample_cursor - recorded_start_cursor
                ),
                detected_clip_id=clip_id,
                selected_clip_id=selected,
                gain_db=float(gain_db),
                impact_energy_j=float(impact_energy_j),
                admitted=voice_id is not None,
            )
        )
        return voice_id

    renderer.play_transient = route_transient  # type: ignore[method-assign]
    return records, original


def _event_waveform_correlation(
    pressure: Float32Array,
    record: ImpactSchedulingRecord,
) -> dict[str, float]:
    """Require public waveform energy shortly after the physical event."""

    event_sample = record.recorded_sample_cursor
    window_start = max(0, event_sample)
    window_stop = min(
        pressure.shape[1],
        event_sample + round(0.25 * AUDIO_SAMPLE_RATE_HZ),
    )
    window = np.max(np.abs(pressure[:, window_start:window_stop]), axis=0)
    peak = float(np.max(window)) if window.size else 0.0
    if not np.isfinite(peak) or peak <= 1e-7:
        raise AssertionError("physical impact has no correlated public waveform")
    crossings = np.flatnonzero(window >= 0.05 * peak)
    if crossings.size == 0:
        raise AssertionError("impact waveform has no measurable onset")
    onset_offset = float(crossings[0] / AUDIO_SAMPLE_RATE_HZ)
    peak_offset = float(np.argmax(window) / AUDIO_SAMPLE_RATE_HZ)
    if not 0.0 <= onset_offset <= 0.20 or not 0.0 <= peak_offset <= 0.25:
        raise AssertionError("impact event and public waveform are not time correlated")
    return {
        "event_time_seconds": record.time_seconds,
        "waveform_onset_after_event_seconds": onset_offset,
        "waveform_peak_after_event_seconds": peak_offset,
        "correlated_peak_pressure_pa": peak,
    }


def _public_binocular_frame(
    observation: dict[str, NDArray[Any]],
    *,
    width: int,
    height: int,
) -> tuple[Uint8Array, Uint8Array, Uint8Array]:
    """Validate and join the exact public left/right RGB observations."""

    expected_shape = (height, width, 3)
    panels: list[Uint8Array] = []
    for eye_name in ("eye_left", "eye_right"):
        panel = np.asarray(observation[eye_name])
        if panel.shape != expected_shape or panel.dtype != np.uint8:
            raise AssertionError(
                f"public {eye_name} changed: shape={panel.shape}, "
                f"dtype={panel.dtype}"
            )
        panels.append(np.ascontiguousarray(panel, dtype=np.uint8))
    left, right = panels
    stereo = np.ascontiguousarray(
        np.concatenate((left, right), axis=1), dtype=np.uint8
    )
    return left, right, stereo


def _segmentation_pixels(
    segmentation: NDArray[Any],
    geom_ids: tuple[int, ...],
) -> int:
    """Count pixels belonging to any named MuJoCo geom."""

    geom_layer = np.asarray(segmentation)[..., 0]
    object_layer = np.asarray(segmentation)[..., 1]
    return int(
        np.count_nonzero(
            np.isin(geom_layer, geom_ids)
            & (object_layer == int(mujoco.mjtObj.mjOBJ_GEOM))
        )
    )


def _colored_object_pixels(image: Uint8Array) -> int:
    return int(
        np.count_nonzero(
            (image[..., 0] > 180)
            & (image[..., 1] < 120)
            & (image[..., 2] < 120)
        )
    )


def run_episode(
    *,
    width: int,
    height: int,
    frame_sink: FrameSink | None = None,
) -> EpisodeResult:
    """Run one pre-rolled, five-second public-sensor ball throw."""

    environment = ThreeObjectScenarioEnv(
        width=width,
        height=height,
        audition=AuditionConfig.quality_48k(),
    )
    controller: AssistedBallThrowController | None = None
    original_play_transient: Callable[..., int | None] | None = None
    segmentation_renderer: mujoco.Renderer | None = None
    open_floor_snapshots: dict[
        int, tuple[int, int, NDArray[np.float32]]
    ] = {}
    original_substep_callback: Callable[[], None] | None = None
    try:
        observation, info = environment.reset(seed=SCENARIO_SEED)
        if info != {} or not environment.observation_space.contains(observation):
            raise AssertionError("initial public observation contract changed")
        preparation = prepare_held_ball(environment)
        open_floor_snapshots = _configure_open_floor(environment)
        controller = AssistedBallThrowController(environment, preparation)
        state = controller.initial_state()
        recorded_start_time_s = float(environment.data.time)
        recorded_start_cursor = environment.audition.renderer.sample_cursor  # type: ignore[union-attr]
        impact_records, original_play_transient = _install_single_hit_clip(
            environment,
            recorded_start_time_s,
        )
        contact_audit: dict[str, Any] = {
            "active": False,
            "first_floor_time": None,
            "first_floor_position": None,
            "other": [],
        }
        original_substep_callback = environment._substep_callback
        ball_geom_for_audit = int(environment.model.geom("demo_ball_geom").id)
        ball_body_id = int(
            environment.model.body(SCENARIO_BODY_NAMES[ScenarioObject.BALL]).id
        )
        floor_geom_for_audit = int(environment.model.geom("floor").id)

        def audit_substep_contacts() -> None:
            original_substep_callback()  # type: ignore[misc]
            if not contact_audit["active"]:
                return
            for contact in environment.data.contact:
                if int(contact.efc_address) < 0:
                    continue
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                if ball_geom_for_audit not in (geom1, geom2):
                    continue
                other = geom2 if geom1 == ball_geom_for_audit else geom1
                relative_time = float(environment.data.time) - recorded_start_time_s
                if other == floor_geom_for_audit:
                    if contact_audit["first_floor_time"] is None:
                        contact_audit["first_floor_time"] = relative_time
                        contact_audit["first_floor_position"] = np.asarray(
                            environment.data.xpos[ball_body_id],
                            dtype=np.float64,
                        ).copy()
                    contact_audit["active"] = False
                    continue
                name = mujoco.mj_id2name(
                    environment.model, mujoco.mjtObj.mjOBJ_GEOM, other
                )
                contact_audit["other"].append(
                    (name if name is not None else f"geom:{other}", relative_time)
                )

        environment._substep_callback = audit_substep_contacts  # type: ignore[method-assign]
        segmentation_renderer = mujoco.Renderer(
            environment.model,
            height=height,
            width=width,
        )
        segmentation_renderer.enable_segmentation_rendering()

        ball_joint = environment.data.joint("demo_ball_joint")
        eye_interior_geom_ids = tuple(
            int(environment.model.geom(name).id)
            for name in EYE_INTERIOR_GEOM_NAMES
        )
        hand_forward_positions_m: list[float] = []
        release_ball_position: NDArray[np.float64] | None = None
        release_time_seconds: float | None = None
        release_body_local_velocity: NDArray[np.float64] | None = None
        release_non_hand_clearance_m: float | None = None
        pre_swing_max_error = 0.0
        action_digest = sha256()
        qpos_digest = sha256()
        ball_digest = sha256()
        audio_digest = sha256()
        video_digest = sha256()
        first_frame_sha = ""
        release_frame_sha = ""
        final_frame_sha = ""
        left_right_equal = 0
        frame_count = 0
        audio_blocks: list[Float32Array] = []
        evidence_visibility: dict[str, dict[str, int]] = {}

        for step_index in range(RECORDED_PUBLIC_STEPS):
            if step_index <= RELEASE_RECORDED_STEP:
                upper_body = environment.data.body("upper_body")
                upper_body_rotation = np.asarray(
                    upper_body.xmat, dtype=np.float64
                ).reshape(3, 3)
                hand_forward_positions_m.append(
                    float(
                        (
                            upper_body_rotation.T
                            @ (
                                np.asarray(
                                    environment.data.body("right_hand").xpos,
                                    dtype=np.float64,
                                )
                                - np.asarray(upper_body.xpos, dtype=np.float64)
                            )
                        )[0]
                    )
                )
            object_qpos_before = np.asarray(ball_joint.qpos).copy()
            object_qvel_before = np.asarray(ball_joint.qvel).copy()
            xfrc_before = environment.data.xfrc_applied.copy()
            command = controller.compose(state)
            if step_index == RELEASE_RECORDED_STEP:
                upper_body_rotation = np.asarray(
                    environment.data.body("upper_body").xmat,
                    dtype=np.float64,
                ).reshape(3, 3)
                release_body_local_velocity = upper_body_rotation.T @ np.asarray(
                    ball_joint.qvel[:3], dtype=np.float64
                )
                release_ball_position = np.asarray(
                    environment.data.xpos[ball_body_id], dtype=np.float64
                ).copy()
                release_time_seconds = (
                    float(environment.data.time) - recorded_start_time_s
                )
                mimo_body_id = int(environment.model.body("mimo_location").id)
                from_to = np.zeros(6, dtype=np.float64)
                release_non_hand_clearance_m = min(
                    float(
                        mujoco.mj_geomDistance(
                            environment.model,
                            environment.data,
                            controller.ball_geom_id,
                            geom_id,
                            1.0,
                            from_to,
                        )
                    )
                    for geom_id in range(environment.model.ngeom)
                    if int(
                        environment.model.body_rootid[
                            int(environment.model.geom_bodyid[geom_id])
                        ]
                    )
                    == mimo_body_id
                    and geom_id not in controller.hand_geom_ids
                )
            if not np.array_equal(ball_joint.qpos, object_qpos_before):
                raise AssertionError("throw controller wrote ball qpos")
            if (
                step_index != RELEASE_RECORDED_STEP
                and not np.array_equal(ball_joint.qvel, object_qvel_before)
            ):
                raise AssertionError("throw controller wrote ball qvel")
            if not np.array_equal(environment.data.xfrc_applied, xfrc_before):
                raise AssertionError("throw controller wrote applied force")
            if not environment.action_space.contains(command.action):
                raise AssertionError(f"invalid throw action at step {step_index}")
            action_digest.update(command.action.tobytes(order="C"))
            state = command.state
            if step_index == RELEASE_RECORDED_STEP:
                contact_audit["active"] = True

            observation, _, terminated, truncated, _ = environment.step(
                command.action
            )
            if terminated or truncated:
                raise AssertionError(f"throw episode ended at step {step_index}")
            if not environment.observation_space.contains(observation):
                raise AssertionError(f"invalid public observation at {step_index}")
            if np.any(environment.data.xfrc_applied != 0.0):
                raise AssertionError("runtime applied force became nonzero")
            state = controller.after_step(state)
            if not state.released:
                observation = environment._get_obs()
                if not environment.observation_space.contains(observation):
                    raise AssertionError(
                        f"invalid synchronized public observation at {step_index}"
                    )
            if step_index < PRE_SWING_PUBLIC_STEPS:
                pre_swing_max_error = max(
                    pre_swing_max_error,
                    controller.hand_relative_error_m(),
                )
            qpos_digest.update(
                np.ascontiguousarray(
                    environment.data.qpos, dtype=np.float64
                ).tobytes(order="C")
            )
            ball_position = np.asarray(
                environment.data.xpos[ball_body_id], dtype=np.float64
            )
            ball_digest.update(
                np.ascontiguousarray(ball_position).tobytes(order="C")
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
                    observation,
                    width=width,
                    height=height,
                )
                left_right_equal += int(np.array_equal(left, right))
                contents = stereo.tobytes(order="C")
                if frame_sink is not None:
                    frame_sink(contents)
                frame_sha = sha256(contents).hexdigest()
                video_digest.update(contents)
                if frame_count == 0:
                    first_frame_sha = frame_sha
                if step_index in {
                    RELEASE_RECORDED_STEP,
                    RELEASE_RECORDED_STEP + 1,
                }:
                    release_frame_sha = frame_sha
                for label, evidence_step in EVIDENCE_PUBLIC_STEPS.items():
                    if step_index == evidence_step:
                        segmentations: dict[str, NDArray[Any]] = {}
                        for eye_name in ("eye_left", "eye_right"):
                            segmentation_renderer.update_scene(
                                environment.data, camera=eye_name
                            )
                            segmentations[eye_name] = np.asarray(
                                segmentation_renderer.render()
                            ).copy()
                        left_segmentation = segmentations["eye_left"]
                        right_segmentation = segmentations["eye_right"]
                        evidence_visibility[label] = {
                            "public_step": step_index,
                            "left_ball_pixels": _segmentation_pixels(
                                left_segmentation, (controller.ball_geom_id,)
                            ),
                            "right_ball_pixels": _segmentation_pixels(
                                right_segmentation, (controller.ball_geom_id,)
                            ),
                            "left_red_pixels": _colored_object_pixels(left),
                            "right_red_pixels": _colored_object_pixels(right),
                            "left_hand_pixels": _segmentation_pixels(
                                left_segmentation, controller.hand_geom_ids
                            ),
                            "right_hand_pixels": _segmentation_pixels(
                                right_segmentation, controller.hand_geom_ids
                            ),
                            "left_eye_interior_pixels": _segmentation_pixels(
                                left_segmentation, eye_interior_geom_ids
                            ),
                            "right_eye_interior_pixels": _segmentation_pixels(
                                right_segmentation, eye_interior_geom_ids
                            ),
                        }
                final_frame_sha = frame_sha
                frame_count += 1

        if frame_count != RECORDED_PUBLIC_STEPS // 2:
            raise AssertionError("recorded video frame count changed")
        if set(evidence_visibility) != set(EVIDENCE_PUBLIC_STEPS):
            raise AssertionError("public-eye evidence frames are incomplete")
        for label, record in evidence_visibility.items():
            if min(record["left_ball_pixels"], record["right_ball_pixels"]) <= 0:
                raise AssertionError(f"ball is absent from {label} public eye frame")
            if label in {"held", "swing", "release"} and min(
                record["left_hand_pixels"], record["right_hand_pixels"]
            ) <= 0:
                raise AssertionError(f"hand is absent from {label} public eye frame")
            if record["left_eye_interior_pixels"] != 0 or record[
                "right_eye_interior_pixels"
            ] != 0:
                raise AssertionError(
                    f"head/eye interior is visible in {label} public eye evidence: "
                    f"{record}"
                )
        pressure = np.ascontiguousarray(
            np.concatenate(audio_blocks, axis=1), dtype=np.float32
        )
        if pressure.shape != (2, AUDIO_SAMPLES_PER_EAR):
            raise AssertionError("recorded public pressure shape changed")
        if not np.all(np.isfinite(pressure)) or not np.any(pressure != 0.0):
            raise AssertionError("recorded stereo audio is not finite and nonzero")
        if state.maximum_hand_relative_error_m > MAXIMUM_HAND_RELATIVE_ERROR_M:
            raise AssertionError(
                "ball did not remain within 8 mm of its synchronized held center: "
                f"{state.maximum_hand_relative_error_m}"
            )
        if pre_swing_max_error > MAXIMUM_HAND_RELATIVE_ERROR_M:
            raise AssertionError(
                f"synchronized ball moved relative to hand: {pre_swing_max_error}"
            )
        if len(hand_forward_positions_m) != RELEASE_RECORDED_STEP + 1:
            raise AssertionError("arm-motion samples are incomplete")
        hand_forward_excursion = float(
            hand_forward_positions_m[-1] - min(hand_forward_positions_m)
        )
        if hand_forward_excursion < MINIMUM_HAND_FORWARD_EXCURSION_M:
            raise AssertionError(
                "wind-up-to-release hand excursion is too small: "
                f"{hand_forward_excursion}"
            )
        velocity = controller.release_velocity_metrics(state)
        if (
            release_body_local_velocity is None
            or release_non_hand_clearance_m is None
        ):
            raise AssertionError("body-local release evidence is incomplete")
        forward, lateral, vertical = map(float, release_body_local_velocity)
        velocity.update(
            {
                "upper_body_local_forward_velocity_m_s": forward,
                "upper_body_local_lateral_velocity_m_s": lateral,
                "upper_body_local_vertical_velocity_m_s": vertical,
                "minimum_non_hand_body_clearance_m": (
                    release_non_hand_clearance_m
                ),
            }
        )
        if (
            velocity["pre_release_velocity_error_m_s"]
            > MAXIMUM_RELEASE_VELOCITY_ERROR_M_S
        ):
            raise AssertionError(f"release velocity initialization failed: {velocity}")
        if not (
            forward >= MINIMUM_BODY_LOCAL_FORWARD_SPEED_M_S
            and abs(lateral) <= MAXIMUM_BODY_LOCAL_LATERAL_SPEED_M_S
            and MINIMUM_BODY_LOCAL_VERTICAL_SPEED_M_S
            <= vertical
            <= MAXIMUM_BODY_LOCAL_VERTICAL_SPEED_M_S
            and release_non_hand_clearance_m
            >= MINIMUM_NON_HAND_BODY_CLEARANCE_M
        ):
            raise AssertionError(
                "throw is not straight and safely clear in the upper-body frame: "
                f"{velocity}"
            )
        if state.held_pose_sync_count != RELEASE_RECORDED_STEP:
            raise AssertionError(
                f"held-pose synchronization count changed: {state.held_pose_sync_count}"
            )
        if state.release_velocity_write_count != 1:
            raise AssertionError(
                "release velocity must be initialized exactly once: "
                f"{state.release_velocity_write_count}"
            )
        if not state.collisions_restored or state.collision_restore_step is None:
            raise AssertionError("ball collision mask was not restored after clearance")
        first_floor_time = contact_audit["first_floor_time"]
        first_floor_position = contact_audit["first_floor_position"]
        if (
            first_floor_time is None
            or first_floor_position is None
            or release_ball_position is None
            or release_time_seconds is None
        ):
            raise AssertionError("released ball produced no solved floor contact")
        horizontal_flight = float(
            np.linalg.norm(
                (
                    np.asarray(first_floor_position, dtype=np.float64)
                    - release_ball_position
                )[:2]
            )
        )
        air_time = float(first_floor_time - release_time_seconds)
        if (
            horizontal_flight < MINIMUM_HORIZONTAL_FLIGHT_M
            or air_time < MINIMUM_AIR_TIME_SECONDS
        ):
            raise AssertionError(
                "released ball lacks throw displacement or air time: "
                f"horizontal={horizontal_flight}, air_time={air_time}"
            )
        pre_floor_other_contacts = tuple(contact_audit["other"])
        if pre_floor_other_contacts:
            raise AssertionError(
                "ball contacted a non-floor geom before floor: "
                f"{pre_floor_other_contacts}"
            )
        admitted_impacts = tuple(
            record
            for record in impact_records
            if record.admitted
            and RELEASE_RECORDED_STEP * 0.01
            <= record.time_seconds
            <= float(first_floor_time) + 1e-9
        )
        if len(admitted_impacts) != 1:
            raise AssertionError("released ball produced no admitted floor impact")
        if any(
            record.selected_clip_id != BALL_THROW_IMPACT_CLIP_ID
            for record in admitted_impacts
        ):
            raise AssertionError("ball impact used the wrong demo clip")
        correlation = _event_waveform_correlation(pressure, admitted_impacts[0])
        if left_right_equal:
            raise AssertionError("binocular source frames became identical")
        if int(environment.data.eq_active[controller.grip_equality_id]) != 0:
            raise AssertionError("ball grip weld remained active after release")
        if environment.audition.renderer.sample_cursor - recorded_start_cursor != (
            AUDIO_SAMPLES_PER_EAR
        ):
            raise AssertionError("recorded audition clock changed")

        return EpisodeResult(
            pressure_channels_first=pressure,
            frame_count=frame_count,
            raw_video_sha256=video_digest.hexdigest(),
            first_frame_sha256=first_frame_sha,
            release_frame_sha256=release_frame_sha,
            final_frame_sha256=final_frame_sha,
            left_right_equal_frame_count=left_right_equal,
            final_state=state,
            release_velocity=velocity,
            pre_swing_maximum_hand_relative_error_m=pre_swing_max_error,
            hand_forward_excursion_m=hand_forward_excursion,
            horizontal_flight_m=horizontal_flight,
            air_time_seconds=air_time,
            impact_records=tuple(impact_records),
            impact_audio_correlation=correlation,
            action_sha256=action_digest.hexdigest(),
            qpos_sha256=qpos_digest.hexdigest(),
            ball_position_sha256=ball_digest.hexdigest(),
            audio_transport_sha256=audio_digest.hexdigest(),
            ball_final_position_m=tuple(map(float, environment.data.xpos[ball_body_id])),
            recorded_start_simulation_time_s=recorded_start_time_s,
            evidence_visibility=evidence_visibility,
            first_floor_contact_time_seconds=float(first_floor_time),
            pre_floor_other_ball_contacts=pre_floor_other_contacts,
        )
    finally:
        if segmentation_renderer is not None:
            segmentation_renderer.close()
        if environment.audition is not None and original_play_transient is not None:
            environment.audition.renderer.play_transient = (  # type: ignore[method-assign]
                original_play_transient
            )
        if controller is not None:
            controller.close()
        if original_substep_callback is not None:
            environment._substep_callback = original_substep_callback  # type: ignore[method-assign]
        if open_floor_snapshots:
            _restore_open_floor_geoms(environment, open_floor_snapshots)
        environment.close()


def _result_record(result: EpisodeResult) -> dict[str, Any]:
    return {
        "clock": {
            "duration_seconds": DURATION_SECONDS,
            "recorded_public_steps": RECORDED_PUBLIC_STEPS,
            "video_frames": result.frame_count,
            "audio_samples_per_ear": result.pressure_channels_first.shape[1],
            "pre_roll_public_steps": 700,
        },
        "mechanism": {
            "label": "actuator_driven_assisted_ball_throw_only",
            "pickup_shown_or_claimed": False,
            "held_ball_mechanism": "collision_suppressed_hand_relative_mocap_weld",
            "world_space_throw_offset": False,
            "held_pose_qpos_synchronization_for_rendering": True,
            "held_pose_sync_count": result.final_state.held_pose_sync_count,
            "release_qvel_initialized_from_hand_once": True,
            "release_velocity_write_count": (
                result.final_state.release_velocity_write_count
            ),
            "post_release_object_qpos_qvel_writes": False,
            "runtime_object_applied_force": False,
            "release_then_mujoco_free_flight": True,
            "open_floor_hidden_disabled_geoms": list(
                OPEN_FLOOR_HIDDEN_GEOM_NAMES
            ),
        },
        "release": {
            "recorded_step": RELEASE_RECORDED_STEP,
            "time_seconds": RELEASE_RECORDED_STEP * 0.01,
            **result.release_velocity,
            "collision_restore_step": result.final_state.collision_restore_step,
            "maximum_hand_relative_error_m": (
                result.final_state.maximum_hand_relative_error_m
            ),
            "pre_swing_maximum_hand_relative_error_m": (
                result.pre_swing_maximum_hand_relative_error_m
            ),
            "hand_forward_excursion_from_windup_m": (
                result.hand_forward_excursion_m
            ),
            "horizontal_free_flight_m": result.horizontal_flight_m,
            "air_time_seconds": result.air_time_seconds,
            "saturation_rows": result.final_state.saturation_rows,
        },
        "impact_audio": {
            "selected_clip_id": BALL_THROW_IMPACT_CLIP_ID,
            "derived_from_clip_id": BALL_THROW_IMPACT_SOURCE_CLIP_ID,
            "derived_clip_onset_clusters": 1,
            "records": [asdict(record) for record in result.impact_records],
            "event_waveform_correlation": result.impact_audio_correlation,
            "first_floor_contact_time_seconds": (
                result.first_floor_contact_time_seconds
            ),
            "pre_floor_other_ball_contacts": list(
                result.pre_floor_other_ball_contacts
            ),
        },
        "public_sources": {
            "video": "public_eye_left_plus_public_eye_right",
            "camera_overrides": False,
            "evidence_visibility": result.evidence_visibility,
            "audio": "audition_float32_pressure_pa",
            "binocular_equal_frame_count": result.left_right_equal_frame_count,
            "raw_video_sha256": result.raw_video_sha256,
            "first_frame_sha256": result.first_frame_sha256,
            "release_frame_sha256": result.release_frame_sha256,
            "final_frame_sha256": result.final_frame_sha256,
            "audio_transport_sha256": result.audio_transport_sha256,
            "pressure_channels_first_sha256": sha256(
                result.pressure_channels_first.tobytes(order="C")
            ).hexdigest(),
            "raw_peak_pressure_pa": float(
                np.max(np.abs(result.pressure_channels_first))
            ),
            "raw_rms_pressure_pa": float(
                np.sqrt(np.mean(result.pressure_channels_first.astype(np.float64) ** 2))
            ),
        },
        "physics_hashes": {
            "action_sha256": result.action_sha256,
            "qpos_sha256": result.qpos_sha256,
            "ball_position_sha256": result.ball_position_sha256,
            "ball_final_position_m": result.ball_final_position_m,
        },
    }


def run_smoke() -> dict[str, Any]:
    """Run the bounded physics/audio path without encoding media."""

    result = run_episode(width=128, height=96)
    return _result_record(result)


def export_ball_throw(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    overwrite: bool = False,
    monitor_gain: float = DEFAULT_MONITOR_GAIN,
) -> dict[str, Any]:
    """Run, validate, encode, and atomically publish the short clip."""

    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("set MUJOCO_GL=egl before starting the exporter")
    paths = output_paths(output_directory)
    _prepare_output(paths, overwrite=overwrite)
    repository_root = Path(__file__).resolve().parents[1]
    ffmpeg_tools = _validate_ffmpeg(repository_root)
    video_encoder: subprocess.Popen[bytes] | None = None
    with TemporaryDirectory(
        prefix="ball-throw-",
        dir=paths.output_directory,
    ) as temporary_directory_name:
        temporary_directory = Path(temporary_directory_name)
        video_only = temporary_directory / "video_only.mp4"
        temporary_wav = temporary_directory / paths.pressure_wav.name
        temporary_mp4 = temporary_directory / paths.playback_mp4.name
        temporary_manifest = temporary_directory / paths.manifest_json.name
        try:
            video_encoder = _start_video_encoder(
                video_only,
                ffmpeg_tools["ffmpeg_path"],
                repository_root,
            )
            if video_encoder.stdin is None:
                raise RuntimeError("video encoder stdin was not created")
            stream: BinaryIO = video_encoder.stdin
            result = run_episode(
                width=EYE_WIDTH,
                height=EYE_HEIGHT,
                frame_sink=stream.write,
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
                "title": "BlightWorld short assisted ball throw",
                "seed": SCENARIO_SEED,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        print(json.dumps(run_smoke(), indent=2, sort_keys=True))
        return
    manifest = export_ball_throw(arguments.output, overwrite=arguments.overwrite)
    paths = output_paths(arguments.output)
    print(f"wrote {paths.playback_mp4}")
    print(f"wrote {paths.pressure_wav}")
    print(f"wrote {paths.manifest_json}")
    print(json.dumps(manifest["result"]["release"], sort_keys=True))


if __name__ == "__main__":
    main()

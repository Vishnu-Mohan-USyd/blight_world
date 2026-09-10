"""Reproducible live-MuJoCo validation profile for MIMo audition.

Every reported waveform is produced by stepping :class:`BlightWorldEnv`.
The fixtures deliberately isolate physical/acoustic invariants; they are not
an agent, gait controller, learned policy, or scrape/rolling sound model.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import hashlib
import json
from math import cos, log10, radians, sin
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from blight_world import (
    AuditionConfig,
    AudioSourceSpec,
    BlightWorldEnv,
    BodyAttachment,
    StaticAttachment,
)
from mimo_audition.contacts import ImpactEvent
from mimo_audition.renderer import BinauralRenderer
from mimo_audition.sources import Attachment


FloatArray = NDArray[np.float64]
ActionFactory = Callable[[int, BlightWorldEnv], NDArray[np.floating]]
ForceCallback = Callable[[int, BlightWorldEnv], None]
TrajectoryCallback = Callable[[int, BlightWorldEnv, dict[str, NDArray[Any]]], None]

SAMPLE_RATE_HZ = 48_000
PROFILE_SEED = 1_401
PARKED_OBJECT_POSITIONS = {
    "block": (-1.8, 1.5, 0.099),
    "ball": (-1.8, 0.0, 0.099),
    "cylinder": (-1.8, -1.5, 0.099),
}
FREE_JOINT_LAYOUT = {
    "block": (0, 0),
    "ball": (7, 6),
    "cylinder": (14, 12),
    "mimo_location": (21, 18),
}


def _scalar_int(value: Any) -> int:
    """Convert a one-element MuJoCo named-view scalar without warnings."""

    return int(np.asarray(value).item())


def _json_float(value: Any) -> float:
    """Return a finite built-in float for deterministic JSON serialization."""

    result = float(np.asarray(value).item())
    if not np.isfinite(result):
        raise ValueError("profile measurement must be finite")
    return result


def _set_free_joint(
    qpos: FloatArray,
    qvel: FloatArray,
    name: str,
    position_m: Sequence[float],
    *,
    quaternion_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    linear_velocity_m_s: Sequence[float] = (0.0, 0.0, 0.0),
    angular_velocity_rad_s: Sequence[float] = (0.0, 0.0, 0.0),
) -> None:
    """Write one verified free-joint pose and velocity into state vectors."""

    qpos_address, qvel_address = FREE_JOINT_LAYOUT[name]
    qpos[qpos_address : qpos_address + 3] = position_m
    qpos[qpos_address + 3 : qpos_address + 7] = quaternion_wxyz
    qvel[qvel_address : qvel_address + 3] = linear_velocity_m_s
    qvel[qvel_address + 3 : qvel_address + 6] = angular_velocity_rad_s


def _prepare_state(
    environment: BlightWorldEnv,
    *,
    seed: int,
    objects: dict[str, dict[str, Sequence[float]]] | None = None,
    mimo: dict[str, Sequence[float]] | None = None,
    joint_positions_rad: dict[str, float] | None = None,
) -> None:
    """Park unused objects, apply fixture state, and seed resting contacts."""

    environment.reset(seed=seed)
    qpos = np.asarray(environment.init_qpos, dtype=np.float64).copy()
    qvel = np.zeros_like(environment.init_qvel, dtype=np.float64)
    for name, position in PARKED_OBJECT_POSITIONS.items():
        _set_free_joint(qpos, qvel, name, position)
    for name, state in (objects or {}).items():
        _set_free_joint(
            qpos,
            qvel,
            name,
            state["position_m"],
            quaternion_wxyz=state.get("quaternion_wxyz", (1.0, 0.0, 0.0, 0.0)),
            linear_velocity_m_s=state.get("linear_velocity_m_s", (0.0, 0.0, 0.0)),
            angular_velocity_rad_s=state.get(
                "angular_velocity_rad_s", (0.0, 0.0, 0.0)
            ),
        )
    if mimo is not None:
        _set_free_joint(
            qpos,
            qvel,
            "mimo_location",
            mimo["position_m"],
            quaternion_wxyz=mimo.get(
                "quaternion_wxyz", (1.0, 0.0, 0.0, 0.0)
            ),
            linear_velocity_m_s=mimo.get(
                "linear_velocity_m_s", (0.0, 0.0, 0.0)
            ),
            angular_velocity_rad_s=mimo.get(
                "angular_velocity_rad_s", (0.0, 0.0, 0.0)
            ),
        )
    for joint_name, value in (joint_positions_rad or {}).items():
        address = _scalar_int(environment.model.joint(joint_name).qposadr)
        qpos[address] = value

    environment.set_state(qpos, qvel)
    environment.data.time = 0.0
    environment.data.ctrl.fill(0)
    environment.data.qfrc_applied.fill(0)
    environment.data.xfrc_applied.fill(0)
    environment.data.qacc_warmstart.fill(0)
    mujoco.mj_forward(environment.model, environment.data)
    if environment.audition is None:
        raise RuntimeError("profile requires audition")
    environment.audition.reset(seed)


def _apply_countergravity(
    environment: BlightWorldEnv,
    *,
    object_names: Sequence[str] = (),
) -> None:
    """Cancel gravity on every MIMo body and selected airborne objects."""

    environment.data.xfrc_applied.fill(0)
    mimo_root = _scalar_int(environment.model.body("mimo_location").id)
    for body_id in range(environment.model.nbody):
        if _scalar_int(environment.model.body_rootid[body_id]) == mimo_root:
            environment.data.xfrc_applied[body_id, 2] = (
                environment.model.body_mass[body_id] * 9.81
            )
    for name in object_names:
        body_id = _scalar_int(environment.model.body(name).id)
        environment.data.xfrc_applied[body_id, 2] += (
            environment.model.body_mass[body_id] * 9.81
        )


def _material_pair(clip_id: str) -> str:
    """Extract the canonical material family from an impact clip identifier."""

    stem, separator, variant = clip_id.rpartition("_")
    if not separator or not variant.isdigit() or not stem.startswith("impact_"):
        raise ValueError(f"unexpected impact clip identifier: {clip_id}")
    return stem.removeprefix("impact_")


def _peak_reserved_dry_voices(renderer: BinauralRenderer) -> int:
    """Measure maximum render-block occupancy across retained dry intervals."""

    changes: dict[int, int] = {}
    for voice in renderer._voices:
        interval = renderer._voice_dry_block_interval(voice)
        if interval is None:
            continue
        start_block, end_block = interval
        changes[start_block] = changes.get(start_block, 0) + 1
        if end_block is not None:
            changes[end_block] = changes.get(end_block, 0) - 1
    occupancy = 0
    peak = 0
    for block in sorted(changes):
        occupancy += changes[block]
        peak = max(peak, occupancy)
    return peak


def _run_steps(
    environment: BlightWorldEnv,
    step_count: int,
    *,
    action_factory: ActionFactory | None = None,
    force_callback: ForceCallback | None = None,
    trajectory_callback: TrajectoryCallback | None = None,
) -> tuple[FloatArray, list[dict[str, Any]], int]:
    """Step the live world while capturing pressure and impact admissions."""

    if environment.audition is None:
        raise RuntimeError("profile requires audition")
    renderer = environment.audition.renderer
    original_play_transient = renderer.play_transient
    events: list[dict[str, Any]] = []
    peak_occupancy = _peak_reserved_dry_voices(renderer)

    def recording_play_transient(
        clip_id: str,
        attachment: Attachment,
        *,
        gain_db: float = 0.0,
        delay_s: float = 0.0,
        impact_energy_j: float = 0.0,
    ) -> int | None:
        nonlocal peak_occupancy
        source_position, _ = environment.audition.scene.attachment_pose(attachment)
        listener_position, head_rotation = environment.audition.scene.listener_pose()
        offset = source_position - listener_position
        distance_m = float(np.linalg.norm(offset))
        direction_head = head_rotation.T @ (offset / distance_m)
        hrtf_index = renderer.hrtf_bank.nearest_index(direction_head)
        voice_id = original_play_transient(
            clip_id,
            attachment,
            gain_db=gain_db,
            delay_s=delay_s,
            impact_energy_j=impact_energy_j,
        )
        events.append(
            {
                "time_s": _json_float(environment.data.time),
                "clip_id": clip_id,
                "material_pair": _material_pair(clip_id),
                "energy_j": _json_float(impact_energy_j),
                "gain_db": _json_float(gain_db),
                "attachment_body": getattr(attachment, "body_name", None),
                "world_position_m": [float(value) for value in source_position],
                "listener_distance_m": distance_m,
                "direction_head_xyz": [float(value) for value in direction_head],
                "hrtf_index": int(hrtf_index),
                "accepted": voice_id is not None,
                "voice_id": voice_id,
            }
        )
        peak_occupancy = max(peak_occupancy, _peak_reserved_dry_voices(renderer))
        return voice_id

    renderer.play_transient = recording_play_transient  # type: ignore[method-assign]
    blocks: list[FloatArray] = []
    try:
        for step_index in range(step_count):
            if force_callback is not None:
                force_callback(step_index, environment)
            action = (
                np.zeros(environment.action_space.shape, dtype=np.float32)
                if action_factory is None
                else action_factory(step_index, environment)
            )
            observation, *_ = environment.step(action)
            pressure = np.asarray(observation["audition"], dtype=np.float64)
            if pressure.shape != (2, 480) or not np.all(np.isfinite(pressure)):
                raise AssertionError("live audition block violated shape/finite contract")
            blocks.append(pressure.copy())
            peak_occupancy = max(
                peak_occupancy,
                _peak_reserved_dry_voices(renderer),
            )
            if trajectory_callback is not None:
                trajectory_callback(step_index, environment, observation)
    finally:
        del renderer.play_transient
    return np.concatenate(blocks, axis=1), events, peak_occupancy


def _waveform_sha256(pressure: NDArray[np.floating]) -> str:
    values = np.ascontiguousarray(pressure, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _audio_metrics(
    pressure: NDArray[np.floating],
    *,
    event_time_s: float | None = None,
    early_window_s: float = 0.020,
) -> dict[str, Any]:
    """Measure binaural onset and early pressure without assuming symmetry."""

    values = np.asarray(pressure, dtype=np.float64)
    if values.shape[0] != 2 or values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("pressure must be finite channels-first stereo")
    peaks = np.max(np.abs(values), axis=1)
    if np.any(peaks <= 0):
        raise AssertionError("profile waveform must be nonzero in both ears")
    onsets = []
    for channel in range(2):
        threshold = max(float(peaks[channel]) * 1e-4, 1e-12)
        indices = np.flatnonzero(np.abs(values[channel]) >= threshold)
        if indices.size == 0:
            raise AssertionError("profile waveform has no measurable onset")
        onsets.append(int(indices[0]))
    first = min(onsets)
    stop = min(values.shape[1], first + round(early_window_s * SAMPLE_RATE_HZ))
    early = values[:, first:stop]
    rms = np.sqrt(np.mean(np.square(early), axis=1))
    if np.any(rms <= 0):
        raise AssertionError("profile early window must have nonzero RMS")
    metrics: dict[str, Any] = {
        "left_onset_sample": onsets[0],
        "right_onset_sample": onsets[1],
        "left_rms_pa": float(rms[0]),
        "right_rms_pa": float(rms[1]),
        "left_peak_pa": float(peaks[0]),
        "right_peak_pa": float(peaks[1]),
        "ild_db_left_over_right": float(20 * log10(rms[0] / rms[1])),
        "itd_us_right_minus_left": float(
            (onsets[1] - onsets[0]) * 1e6 / SAMPLE_RATE_HZ
        ),
        "waveform_sha256": _waveform_sha256(values),
    }
    if event_time_s is not None:
        metrics["arrival_after_event_ms"] = float(
            1_000 * (first / SAMPLE_RATE_HZ - event_time_s)
        )
    return metrics


def _complete_response_binaural_metrics(
    pressure: NDArray[np.floating],
) -> dict[str, float]:
    """Measure energy ILD and sub-sample ITD over a complete response.

    Complete-response energy avoids the phase sensitivity of a short RMS
    window.  ITD is the peak of normalized right-versus-left cross-correlation
    with a three-point parabolic interpolation around the discrete maximum.
    """

    values = np.asarray(pressure, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("pressure must be finite channels-first stereo")
    channel_energy = np.sum(np.square(values), axis=1)
    if np.any(channel_energy <= 0):
        raise AssertionError("complete response must have energy in both ears")

    normalization = float(np.sqrt(channel_energy[0] * channel_energy[1]))
    correlation = np.correlate(values[1], values[0], mode="full") / normalization
    peak_index = int(np.argmax(correlation))
    fractional_offset = 0.0
    if 0 < peak_index < correlation.size - 1:
        left_value, center_value, right_value = correlation[
            peak_index - 1 : peak_index + 2
        ]
        denominator = left_value - 2.0 * center_value + right_value
        if denominator != 0.0:
            fractional_offset = float(
                0.5 * (left_value - right_value) / denominator
            )
    lag_samples = (
        peak_index - (values.shape[1] - 1) + fractional_offset
    )
    return {
        "complete_energy_ild_db_left_over_right": float(
            10.0 * log10(channel_energy[0] / channel_energy[1])
        ),
        "xcorr_itd_samples_right_minus_left": float(lag_samples),
        "xcorr_itd_us_right_minus_left": float(
            lag_samples * 1e6 / SAMPLE_RATE_HZ
        ),
        "xcorr_peak": float(correlation[peak_index]),
    }


def _first_event(
    events: Sequence[dict[str, Any]],
    expected_material: str,
) -> dict[str, Any]:
    matches = [
        event
        for event in events
        if event["material_pair"] == expected_material and event["accepted"]
    ]
    if not matches:
        raise AssertionError(f"missing accepted {expected_material} impact")
    return matches[0]


def _drop_trial(
    environment: BlightWorldEnv,
    *,
    object_name: str,
    position_xy_m: tuple[float, float],
    center_z_m: float,
    expected_material: str,
    seed: int,
    step_count: int = 8,
) -> dict[str, Any]:
    """Run one verified vertical drop and return event/audio measurements."""

    _prepare_state(
        environment,
        seed=seed,
        objects={
            object_name: {
                "position_m": (*position_xy_m, center_z_m),
                "linear_velocity_m_s": (0.0, 0.0, -2.0),
            }
        },
    )
    pressure, events, peak_occupancy = _run_steps(environment, step_count)
    event = _first_event(events, expected_material)
    return {
        "event": event,
        "audio": _audio_metrics(pressure, event_time_s=event["time_s"]),
        "event_count": len(events),
        "event_material_families": sorted(
            {item["material_pair"] for item in events if item["accepted"]}
        ),
        "peak_reserved_voices": peak_occupancy,
    }


def profile_directional_drops() -> dict[str, Any]:
    """Measure equal-energy front/left/right ball-table drop cues."""

    environment = BlightWorldEnv(
        width=16,
        height=12,
        audition=AuditionConfig.quality_48k(),
    )
    positions = {
        "front": (0.81, 0.0),
        "left": (0.702820323, 0.4),
        "right": (0.702820323, -0.4),
    }
    try:
        first = {
            name: _drop_trial(
                environment,
                object_name="ball",
                position_xy_m=position,
                center_z_m=0.73,
                expected_material="wood_rubber",
                seed=PROFILE_SEED,
            )
            for name, position in positions.items()
        }
        repeated = {
            name: _drop_trial(
                environment,
                object_name="ball",
                position_xy_m=position,
                center_z_m=0.73,
                expected_material="wood_rubber",
                seed=PROFILE_SEED,
            )
            for name, position in positions.items()
        }
    finally:
        environment.close()

    energies = [result["event"]["energy_j"] for result in first.values()]
    if not np.allclose(energies, energies[0], rtol=0, atol=2e-9):
        raise AssertionError("equal drop fixtures produced unequal contact energy")
    left_ild = first["left"]["audio"]["ild_db_left_over_right"]
    right_ild = first["right"]["audio"]["ild_db_left_over_right"]
    if not left_ild * right_ild < 0:
        raise AssertionError("left/right drop cues did not reverse ILD sign")
    repeat_exact = all(first[name] == repeated[name] for name in positions)
    if not repeat_exact:
        raise AssertionError("directional drop replay was not exact")
    return {
        "description": "Equal-energy ball/table drops; adult HRTF asymmetry allowed.",
        "positions_xy_m": {name: list(value) for name, value in positions.items()},
        "trials": first,
        "equal_energy": True,
        "left_right_ild_sign_reversal": True,
        "repeat_exact": True,
    }


def _register_impulse_source(
    environment: BlightWorldEnv,
    source_id: str,
    world_position_m: FloatArray,
) -> None:
    if environment.audition is None:
        raise RuntimeError("profile requires audition")
    environment.audition.register_source(
        AudioSourceSpec(
            source_id,
            "profile_impulse",
            StaticAttachment(tuple(float(value) for value in world_position_m)),
        )
    )


def profile_direct_path_distance() -> dict[str, Any]:
    """Isolate propagation delay and inverse-distance pressure in live steps."""

    config = AuditionConfig.quality_48k(
        enable_occlusion=False,
        enable_first_order_reflections=False,
        enable_late_reverb=False,
    )
    environment = BlightWorldEnv(width=16, height=12, audition=config)
    distances_m = np.asarray((1.0, 1.2, 1.5), dtype=np.float64)
    try:
        _prepare_state(environment, seed=PROFILE_SEED)
        if environment.audition is None:
            raise RuntimeError("profile requires audition")
        impulse = np.zeros(4_096, dtype=np.float64)
        impulse[0] = 1.0
        environment.audition.clips.register(
            "profile_impulse",
            impulse,
            reference_peak_pressure_pa_at_1m=1.0,
        )
        listener, rotation = environment.audition.scene.listener_pose()
        source_geometry = []
        for index, distance in enumerate(distances_m):
            position = listener + rotation @ np.asarray((distance, 0.0, 0.0))
            _register_impulse_source(environment, f"ray_{index}", position)
            direction_head = rotation.T @ ((position - listener) / distance)
            source_geometry.append(
                {
                    "direction_head_xyz": [float(value) for value in direction_head],
                    "hrtf_index": int(
                        environment.audition.renderer.hrtf_bank.nearest_index(
                            direction_head
                        )
                    ),
                }
            )

        measurements = []
        for index, distance in enumerate(distances_m):
            _prepare_state(environment, seed=PROFILE_SEED)
            environment.audition.play(f"ray_{index}")
            pressure, events, _ = _run_steps(environment, 4)
            if events:
                raise AssertionError("fixed-ray trial unexpectedly produced impacts")
            metrics = _audio_metrics(pressure)
            metrics.update(_complete_response_binaural_metrics(pressure))
            energy_rms = float(np.sqrt(np.mean(np.square(pressure))))
            measurements.append(
                {
                    "distance_m": float(distance),
                    "whole_signal_rms_pa": energy_rms,
                    **source_geometry[index],
                    "audio": metrics,
                }
            )
    finally:
        environment.close()

    pressure_db = 20 * np.log10(
        [measurement["whole_signal_rms_pa"] for measurement in measurements]
    )
    slope_db_per_decade = float(np.polyfit(np.log10(distances_m), pressure_db, 1)[0])
    if not -20.2 < slope_db_per_decade < -19.8:
        raise AssertionError("direct-only distance fit departed from inverse distance")
    directions = np.asarray(
        [measurement["direction_head_xyz"] for measurement in measurements]
    )
    hrtf_indices = {measurement["hrtf_index"] for measurement in measurements}
    if not np.array_equal(directions, np.repeat(directions[:1], 3, axis=0)):
        raise AssertionError("fixed-ray direction changed with distance")
    if len(hrtf_indices) != 1:
        raise AssertionError("fixed-ray HRTF index changed with distance")
    ild = np.asarray(
        [
            measurement["audio"]["complete_energy_ild_db_left_over_right"]
            for measurement in measurements
        ]
    )
    itd_samples = np.asarray(
        [
            measurement["audio"]["xcorr_itd_samples_right_minus_left"]
            for measurement in measurements
        ]
    )
    if np.ptp(ild) > 0.20 or np.ptp(itd_samples) > 0.01:
        raise AssertionError("fixed-direction direct ray changed ITD/ILD with distance")
    return {
        "description": "Direct-only fixed-head impulse; excludes room and occlusion.",
        "measurements": measurements,
        "fit_db_per_decade": slope_db_per_decade,
        "stable_ild": True,
        "stable_itd": True,
    }


def profile_near_far_drops() -> dict[str, Any]:
    """Compare equal-energy near/far impacts in the complete room renderer."""

    environment = BlightWorldEnv(
        width=16,
        height=12,
        audition=AuditionConfig.quality_48k(),
    )
    positions = {"near": (0.71, 0.0), "far": (1.21, 0.0)}
    try:
        trials = {
            name: _drop_trial(
                environment,
                object_name="ball",
                position_xy_m=position,
                center_z_m=0.73,
                expected_material="wood_rubber",
                seed=PROFILE_SEED,
            )
            for name, position in positions.items()
        }
    finally:
        environment.close()

    near_event = trials["near"]["event"]
    far_event = trials["far"]["event"]
    if not np.isclose(
        near_event["energy_j"],
        far_event["energy_j"],
        rtol=0,
        atol=2e-9,
    ):
        raise AssertionError("near/far drop fixtures produced unequal contact energy")
    arrival_delta_ms = float(
        trials["far"]["audio"]["arrival_after_event_ms"]
        - trials["near"]["audio"]["arrival_after_event_ms"]
    )
    if arrival_delta_ms <= 0:
        raise AssertionError("far physical impact did not arrive after near impact")
    return {
        "description": (
            "Equal-energy full-room drops. Reflections and the FDN may make total "
            "energy nonmonotonic; direct-path 1/r is measured separately."
        ),
        "positions_xy_m": {name: list(value) for name, value in positions.items()},
        "trials": trials,
        "equal_energy": True,
        "far_minus_near_arrival_ms": arrival_delta_ms,
    }


def _object_collision_trial(
    environment: BlightWorldEnv,
    *,
    moving_object: str,
    stationary_object: str,
    lateral_position_m: float,
    expected_material: str,
    seed: int,
) -> dict[str, Any]:
    """Run one verified, table-supported object/object collision fixture."""

    _prepare_state(
        environment,
        seed=seed,
        objects={
            moving_object: {
                "position_m": (0.75, lateral_position_m, 0.699),
                "linear_velocity_m_s": (2.0, 0.0, 0.0),
            },
            stationary_object: {
                "position_m": (1.0, lateral_position_m, 0.699),
            },
        },
    )
    pressure, events, peak_occupancy = _run_steps(environment, 10)
    event = _first_event(events, expected_material)
    return {
        "event": event,
        "audio": _audio_metrics(pressure, event_time_s=event["time_s"]),
        "event_count": len(events),
        "event_material_families": sorted(
            {item["material_pair"] for item in events if item["accepted"]}
        ),
        "peak_reserved_voices": peak_occupancy,
    }


def _foot_landing_trial(
    environment: BlightWorldEnv,
    *,
    roll_deg: float,
    seed: int,
) -> dict[str, Any]:
    """Run a countergravity-assisted tilted-foot landing fixture."""

    angle = radians(roll_deg)
    _prepare_state(
        environment,
        seed=seed,
        mimo={
            "position_m": (0.0, 0.0, 0.36),
            "quaternion_wxyz": (cos(angle / 2), sin(angle / 2), 0.0, 0.0),
            "linear_velocity_m_s": (0.0, 0.0, -0.5),
        },
    )

    def countergravity(_: int, env: BlightWorldEnv) -> None:
        _apply_countergravity(env)

    assert environment.audition is not None
    detector = environment.audition._impacts
    original_detect = detector.detect
    physical_events: list[tuple[float, ImpactEvent]] = []

    def recording_detect() -> tuple[ImpactEvent, ...]:
        detected = original_detect()
        physical_events.extend((float(environment.data.time), item) for item in detected)
        return detected

    detector.detect = recording_detect  # type: ignore[method-assign]
    try:
        pressure, events, peak_occupancy = _run_steps(
            environment,
            9,
            force_callback=countergravity,
        )
    finally:
        detector.detect = original_detect  # type: ignore[method-assign]
    event = _first_event(events, "hard_soft")
    physical_event = next(
        item for time_seconds, item in physical_events
        if time_seconds == event["time_s"]
        and item.clip_id == event["clip_id"]
        and item.energy_j == event["energy_j"]
    )
    return {
        "event": event,
        "physical_body": physical_event.attachment.body_name,
        "audio": _audio_metrics(pressure, event_time_s=event["time_s"]),
        "event_count": len(events),
        "event_material_families": sorted(
            {item["material_pair"] for item in events if item["accepted"]}
        ),
        "peak_reserved_voices": peak_occupancy,
    }


def _forced_body_trial(
    environment: BlightWorldEnv,
    *,
    root_position_m: tuple[float, float, float],
    forced_body: str,
    force_xyz_n: tuple[float, float, float],
    expected_material: str,
    seed: int,
    step_count: int,
    joint_positions_rad: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run a labeled countergravity force fixture against room geometry."""

    _prepare_state(
        environment,
        seed=seed,
        mimo={"position_m": root_position_m},
        joint_positions_rad=joint_positions_rad,
    )
    body_id = _scalar_int(environment.model.body(forced_body).id)

    def apply_force(_: int, env: BlightWorldEnv) -> None:
        _apply_countergravity(env)
        env.data.xfrc_applied[body_id, :3] += force_xyz_n

    pressure, events, peak_occupancy = _run_steps(
        environment,
        step_count,
        force_callback=apply_force,
    )
    event = _first_event(events, expected_material)
    return {
        "event": event,
        "audio": _audio_metrics(pressure, event_time_s=event["time_s"]),
        "event_count": len(events),
        "event_material_families": sorted(
            {item["material_pair"] for item in events if item["accepted"]}
        ),
        "peak_reserved_voices": peak_occupancy,
    }


def _object_limb_trial(
    environment: BlightWorldEnv,
    *,
    object_name: str,
    position_m: tuple[float, float, float],
    linear_velocity_m_s: tuple[float, float, float],
    expected_material: str,
    seed: int,
) -> dict[str, Any]:
    """Run a moving-object/MIMo-limb contact under countergravity."""

    _prepare_state(
        environment,
        seed=seed,
        objects={
            object_name: {
                "position_m": position_m,
                "linear_velocity_m_s": linear_velocity_m_s,
            }
        },
    )

    def countergravity(_: int, env: BlightWorldEnv) -> None:
        _apply_countergravity(env, object_names=(object_name,))

    pressure, events, peak_occupancy = _run_steps(
        environment,
        9,
        force_callback=countergravity,
    )
    event = _first_event(events, expected_material)
    return {
        "event": event,
        "audio": _audio_metrics(pressure, event_time_s=event["time_s"]),
        "event_count": len(events),
        "event_material_families": sorted(
            {item["material_pair"] for item in events if item["accepted"]}
        ),
        "peak_reserved_voices": peak_occupancy,
    }


def _run_material_recipe(
    environment: BlightWorldEnv,
    recipe_name: str,
    *,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    """Dispatch one source-verified physical material recipe."""

    drop_recipes = {
        "floor_ball": ("ball", (1.8, 0.0), 0.13, "hard_rubber"),
        "floor_block": ("block", (1.8, 0.0), 0.13, "hard_wood"),
        "floor_cylinder": ("cylinder", (1.8, 0.0), 0.13, "hard_plastic"),
        "table_ball": ("ball", (1.0, -0.3), 0.73, "wood_rubber"),
        "table_block": ("block", (1.0, 0.3), 0.73, "wood_wood"),
        "table_cylinder": ("cylinder", (1.0, 0.0), 0.73, "wood_plastic"),
    }
    if recipe_name in drop_recipes:
        object_name, position, height, material = drop_recipes[recipe_name]
        return material, _drop_trial(
            environment,
            object_name=object_name,
            position_xy_m=position,
            center_z_m=height,
            expected_material=material,
            seed=seed,
        )
    object_recipes = {
        "ball_to_block": ("ball", "block", 0.3, "wood_rubber"),
        "ball_to_cylinder": ("ball", "cylinder", 0.0, "rubber_plastic"),
        "block_to_cylinder": ("block", "cylinder", 0.0, "wood_plastic"),
    }
    if recipe_name in object_recipes:
        moving, stationary, lateral, material = object_recipes[recipe_name]
        return material, _object_collision_trial(
            environment,
            moving_object=moving,
            stationary_object=stationary,
            lateral_position_m=lateral,
            expected_material=material,
            seed=seed,
        )
    if recipe_name == "foot_floor":
        return "hard_soft", _foot_landing_trial(
            environment,
            roll_deg=-10.0,
            seed=seed,
        )
    if recipe_name == "forearm_table":
        return "wood_soft", _forced_body_trial(
            environment,
            root_position_m=(0.47, 0.0, 0.4),
            forced_body="left_lower_arm",
            force_xyz_n=(15.0, 0.0, 0.0),
            expected_material="wood_soft",
            seed=seed,
            step_count=14,
        )
    if recipe_name == "ball_lower_leg":
        return "rubber_soft", _object_limb_trial(
            environment,
            object_name="ball",
            position_m=(0.005, 0.20, 0.214),
            linear_velocity_m_s=(0.0, -1.0, 0.0),
            expected_material="rubber_soft",
            seed=seed,
        )
    if recipe_name == "cylinder_hand":
        return "plastic_soft", _object_limb_trial(
            environment,
            object_name="cylinder",
            position_m=(0.002, 0.28, 0.444),
            linear_velocity_m_s=(0.0, -1.0, 0.0),
            expected_material="plastic_soft",
            seed=seed,
        )
    raise ValueError(f"unknown material recipe: {recipe_name}")


def profile_material_matrix() -> dict[str, Any]:
    """Exercise all reachable impact material families through physics."""

    recipe_names = (
        "floor_ball",
        "floor_block",
        "floor_cylinder",
        "table_ball",
        "table_block",
        "table_cylinder",
        "ball_to_block",
        "ball_to_cylinder",
        "block_to_cylinder",
        "foot_floor",
        "forearm_table",
        "ball_lower_leg",
        "cylinder_hand",
    )
    expected_families = {
        "hard_wood",
        "hard_rubber",
        "hard_plastic",
        "hard_soft",
        "wood_wood",
        "wood_rubber",
        "wood_plastic",
        "wood_soft",
        "rubber_plastic",
        "rubber_soft",
        "plastic_soft",
    }
    expected_event_counts = {
        "floor_ball": 1,
        "floor_block": 1,
        "floor_cylinder": 1,
        "table_ball": 1,
        "table_block": 1,
        "table_cylinder": 1,
        "ball_to_block": 2,
        "ball_to_cylinder": 2,
        "block_to_cylinder": 4,
        "foot_floor": 3,
        "forearm_table": 1,
        "ball_lower_leg": 2,
        "cylinder_hand": 1,
    }
    expected_event_families = {
        "floor_ball": ["hard_rubber"],
        "floor_block": ["hard_wood"],
        "floor_cylinder": ["hard_plastic"],
        "table_ball": ["wood_rubber"],
        "table_block": ["wood_wood"],
        "table_cylinder": ["wood_plastic"],
        "ball_to_block": ["wood_rubber", "wood_wood"],
        "ball_to_cylinder": ["rubber_plastic", "wood_plastic"],
        "block_to_cylinder": ["wood_plastic", "wood_wood"],
        "foot_floor": ["hard_soft"],
        "forearm_table": ["wood_soft"],
        "ball_lower_leg": ["rubber_soft"],
        "cylinder_hand": ["plastic_soft"],
    }
    environment = BlightWorldEnv(
        width=16,
        height=12,
        audition=AuditionConfig.quality_48k(),
    )
    try:
        first: dict[str, dict[str, Any]] = {}
        repeated: dict[str, dict[str, Any]] = {}
        recipe_families: dict[str, str] = {}
        for recipe_name in recipe_names:
            material, result = _run_material_recipe(
                environment,
                recipe_name,
                seed=PROFILE_SEED,
            )
            recipe_families[recipe_name] = material
            first[recipe_name] = result
        for recipe_name in recipe_names:
            material, result = _run_material_recipe(
                environment,
                recipe_name,
                seed=PROFILE_SEED,
            )
            if material != recipe_families[recipe_name]:
                raise AssertionError("material recipe changed family on replay")
            repeated[recipe_name] = result

        _prepare_state(environment, seed=PROFILE_SEED)

        def resting_countergravity(_: int, env: BlightWorldEnv) -> None:
            _apply_countergravity(env, object_names=tuple(PARKED_OBJECT_POSITIONS))

        resting_pressure, resting_events, resting_peak = _run_steps(
            environment,
            6,
            force_callback=resting_countergravity,
        )
    finally:
        environment.close()

    if set(recipe_families.values()) != expected_families:
        raise AssertionError("physical recipes did not cover every material family")
    if first != repeated:
        raise AssertionError("material recipe replay was not exact")
    for recipe_name, expected_family in recipe_families.items():
        result = first[recipe_name]
        if result["event"]["material_pair"] != expected_family:
            raise AssertionError(f"{recipe_name} produced the wrong material family")
        if expected_family not in result["event_material_families"]:
            raise AssertionError(f"{recipe_name} omitted its target material family")
        if result["event_material_families"] != expected_event_families[recipe_name]:
            raise AssertionError(f"{recipe_name} changed its material-family trace")
        if result["event_count"] != expected_event_counts[recipe_name]:
            raise AssertionError(f"{recipe_name} changed its impact-event count")
    if resting_events or np.any(resting_pressure):
        raise AssertionError("seeded resting contacts produced sound")
    return {
        "description": (
            "Thirteen physical fixtures cover all eleven reachable unordered "
            "material families; duplicate-family fixtures exercise distinct contacts."
        ),
        "recipes": first,
        "recipe_material_families": recipe_families,
        "covered_material_families": sorted(expected_families),
        "repeat_exact": True,
        "resting_control": {
            "event_count": 0,
            "waveform_peak_pa": 0.0,
            "peak_reserved_voices": resting_peak,
        },
    }


def profile_mirrored_feet() -> dict[str, Any]:
    """Compare verified left/right tilted-foot landing fixtures."""

    environment = BlightWorldEnv(
        width=16,
        height=12,
        audition=AuditionConfig.quality_48k(),
    )
    try:
        trials = {
            "left_first": _foot_landing_trial(
                environment,
                roll_deg=-10.0,
                seed=PROFILE_SEED,
            ),
            "right_first": _foot_landing_trial(
                environment,
                roll_deg=10.0,
                seed=PROFILE_SEED,
            ),
        }
        repeated = {
            "left_first": _foot_landing_trial(
                environment,
                roll_deg=-10.0,
                seed=PROFILE_SEED,
            ),
            "right_first": _foot_landing_trial(
                environment,
                roll_deg=10.0,
                seed=PROFILE_SEED,
            ),
        }
    finally:
        environment.close()

    if "left" not in str(trials["left_first"]["physical_body"]):
        raise AssertionError("negative-roll fixture did not land left-first")
    if "right" not in str(trials["right_first"]["physical_body"]):
        raise AssertionError("positive-roll fixture did not land right-first")
    ild_left = trials["left_first"]["audio"]["ild_db_left_over_right"]
    ild_right = trials["right_first"]["audio"]["ild_db_left_over_right"]
    if not ild_left * ild_right < 0:
        raise AssertionError("mirrored foot fixtures did not reverse ILD sign")
    if trials != repeated:
        raise AssertionError("mirrored foot replay was not exact")
    return {
        "description": (
            "Countergravity-assisted force-free landing fixtures, not a gait policy."
        ),
        "roll_deg": {"left_first": -10.0, "right_first": 10.0},
        "trials": trials,
        "cue_sign_reversal": True,
        "repeat_exact": True,
    }


def profile_hand_forearm_fixtures() -> dict[str, Any]:
    """Measure labeled hand/table and forearm/table force fixtures."""

    environment = BlightWorldEnv(
        width=16,
        height=12,
        audition=AuditionConfig.quality_48k(),
    )
    try:
        hand = _forced_body_trial(
            environment,
            root_position_m=(0.36, 0.0, 0.4),
            forced_body="left_hand",
            force_xyz_n=(15.0, 0.0, 0.0),
            expected_material="wood_soft",
            seed=PROFILE_SEED,
            step_count=10,
            joint_positions_rad={"robot:left_elbow": radians(-70.5)},
        )
        forearm = _forced_body_trial(
            environment,
            root_position_m=(0.47, 0.0, 0.4),
            forced_body="left_lower_arm",
            force_xyz_n=(15.0, 0.0, 0.0),
            expected_material="wood_soft",
            seed=PROFILE_SEED,
            step_count=14,
        )
    finally:
        environment.close()
    return {
        "description": (
            "Countergravity plus 15 N external-force fixtures; not learned reaching."
        ),
        "hand_table": hand,
        "forearm_table": forearm,
    }


def _block_ild_db(pressure: NDArray[np.floating]) -> float:
    """Return block-energy ILD in decibels for finite nonzero stereo."""

    values = np.asarray(pressure, dtype=np.float64)
    energy = np.sum(np.square(values), axis=1)
    if values.ndim != 2 or values.shape[0] != 2 or np.any(energy <= 0):
        raise AssertionError("spatial profile block must contain stereo energy")
    return float(10.0 * log10(energy[0] / energy[1]))


def profile_moving_ball_crossing() -> dict[str, Any]:
    """Track binaural cues for a physically stepped looping ball source."""

    environment = BlightWorldEnv(
        width=16,
        height=12,
        audition=AuditionConfig.quality_48k(),
    )
    attachment = BodyAttachment("ball")
    trajectory: list[dict[str, Any]] = []
    try:
        _prepare_state(
            environment,
            seed=PROFILE_SEED,
            objects={
                "ball": {
                    "position_m": (1.0, -0.8, 1.0),
                    "linear_velocity_m_s": (0.0, 0.8, 0.0),
                }
            },
        )
        if environment.audition is None:
            raise RuntimeError("profile requires audition")
        environment.audition.register_source(
            AudioSourceSpec(
                "crossing_ball_rattle",
                "rattle",
                attachment,
                loop=True,
            )
        )
        environment.audition.play("crossing_ball_rattle")

        def countergravity(_: int, env: BlightWorldEnv) -> None:
            _apply_countergravity(env, object_names=("ball",))

        def record(
            step_index: int,
            env: BlightWorldEnv,
            observation: dict[str, NDArray[Any]],
        ) -> None:
            if env.audition is None:
                raise RuntimeError("profile requires audition")
            source_position, _ = env.audition.scene.attachment_pose(attachment)
            listener, rotation = env.audition.scene.listener_pose()
            offset = source_position - listener
            distance = float(np.linalg.norm(offset))
            direction = rotation.T @ (offset / distance)
            trajectory.append(
                {
                    "time_s": (step_index + 1) * 0.010,
                    "ball_y_m": float(source_position[1]),
                    "azimuth_deg": float(
                        np.degrees(np.arctan2(direction[1], direction[0]))
                    ),
                    "distance_m": distance,
                    "hrtf_index": int(
                        env.audition.renderer.hrtf_bank.nearest_index(direction)
                    ),
                    "ild_db_left_over_right": _block_ild_db(
                        observation["audition"]
                    ),
                }
            )

        pressure, events, peak = _run_steps(
            environment,
            200,
            force_callback=countergravity,
            trajectory_callback=record,
        )
    finally:
        environment.close()

    if events:
        raise AssertionError("airborne crossing fixture unexpectedly produced impacts")
    if not trajectory[0]["ball_y_m"] < 0 < trajectory[-1]["ball_y_m"]:
        raise AssertionError("ball did not physically cross the listener midline")
    if not trajectory[0]["azimuth_deg"] < 0 < trajectory[-1]["azimuth_deg"]:
        raise AssertionError("moving source azimuth did not cross zero")
    first_ild = float(
        np.mean([sample["ild_db_left_over_right"] for sample in trajectory[:10]])
    )
    last_ild = float(
        np.mean([sample["ild_db_left_over_right"] for sample in trajectory[-10:]])
    )
    if not first_ild * last_ild < 0:
        raise AssertionError("moving source did not reverse its binaural cue sign")
    near_midline = [
        sample for sample in trajectory if abs(sample["azimuth_deg"]) <= 10.0
    ]
    midline_ild = np.asarray(
        [sample["ild_db_left_over_right"] for sample in near_midline]
    )
    max_midline_step = float(np.max(np.abs(np.diff(midline_ild))))
    return {
        "description": (
            "Two-second MuJoCo trajectory with countergravity and a looping "
            "body-attached rattle."
        ),
        "start": trajectory[0],
        "midpoint": min(trajectory, key=lambda item: abs(item["azimuth_deg"])),
        "end": trajectory[-1],
        "mean_endpoint_ild_db": {"start": first_ild, "end": last_ild},
        "zero_crossing_max_adjacent_ild_step_db": max_midline_step,
        "trajectory": trajectory,
        "audio": _audio_metrics(pressure),
        "impact_event_count": 0,
        "peak_reserved_voices": peak,
    }


def profile_head_yaw() -> dict[str, Any]:
    """Measure a fixed world source at three static head-yaw angles."""

    environment = BlightWorldEnv(
        width=16,
        height=12,
        audition=AuditionConfig.quality_48k(),
    )
    try:
        _prepare_state(environment, seed=PROFILE_SEED)
        if environment.audition is None:
            raise RuntimeError("profile requires audition")
        neutral_listener, neutral_rotation = environment.audition.scene.listener_pose()
        source_position = neutral_listener + neutral_rotation @ np.asarray(
            (1.0, 0.0, 0.0)
        )
        source = AudioSourceSpec(
            "fixed_yaw_chime",
            "chime",
            StaticAttachment(tuple(float(value) for value in source_position)),
            loop=True,
        )
        environment.audition.register_source(source)
        measurements: dict[str, dict[str, Any]] = {}
        for yaw_deg in (-30, 0, 30):
            _prepare_state(
                environment,
                seed=PROFILE_SEED,
                joint_positions_rad={"robot:head_swivel": radians(yaw_deg)},
            )
            environment.audition.play("fixed_yaw_chime")
            pressure, events, peak = _run_steps(environment, 10)
            if events:
                raise AssertionError("head-yaw fixture unexpectedly produced impacts")
            listener, rotation = environment.audition.scene.listener_pose()
            offset = source_position - listener
            distance = float(np.linalg.norm(offset))
            direction = rotation.T @ (offset / distance)
            measurements[str(yaw_deg)] = {
                "yaw_deg": yaw_deg,
                "direction_head_xyz": [float(value) for value in direction],
                "azimuth_deg": float(
                    np.degrees(np.arctan2(direction[1], direction[0]))
                ),
                "hrtf_index": int(
                    environment.audition.renderer.hrtf_bank.nearest_index(direction)
                ),
                "audio": _audio_metrics(pressure),
                "complete_binaural": _complete_response_binaural_metrics(pressure),
                "peak_reserved_voices": peak,
            }
    finally:
        environment.close()

    negative_ild = measurements["-30"]["complete_binaural"][
        "complete_energy_ild_db_left_over_right"
    ]
    positive_ild = measurements["30"]["complete_binaural"][
        "complete_energy_ild_db_left_over_right"
    ]
    if not negative_ild * positive_ild < 0:
        raise AssertionError("opposite head yaw did not reverse binaural cue sign")
    if len({item["hrtf_index"] for item in measurements.values()}) != 3:
        raise AssertionError("head-yaw fixtures did not select distinct HRTF directions")
    return {
        "description": "Static head swivel with a fixed world-frame looping source.",
        "source_world_position_m": [float(value) for value in source_position],
        "measurements": measurements,
        "cue_sign_reversal": True,
    }


def _steady_pressure_metrics(pressure: NDArray[np.floating]) -> dict[str, float]:
    """Measure steady RMS and high-frequency energy from the final 40 ms."""

    values = np.asarray(pressure, dtype=np.float64)[:, -4 * 480 :]
    rms = float(np.sqrt(np.mean(np.square(values))))
    spectrum = np.fft.rfft(values, axis=1)
    frequencies = np.fft.rfftfreq(values.shape[1], d=1.0 / SAMPLE_RATE_HZ)
    high_frequency_rms = float(
        np.sqrt(np.mean(np.square(np.abs(spectrum[:, frequencies >= 4_000]))))
        / values.shape[1]
    )
    if rms <= 0 or high_frequency_rms <= 0:
        raise AssertionError("steady occlusion fixture must contain pressure energy")
    return {
        "rms_pa": rms,
        "high_frequency_rms_pa": high_frequency_rms,
        "ild_db_left_over_right": _block_ild_db(values),
    }


def profile_block_occlusion() -> dict[str, Any]:
    """Measure a real block moving into and out of a fixed direct ray."""

    config = AuditionConfig.quality_48k(
        enable_first_order_reflections=False,
        enable_late_reverb=False,
    )
    environment = BlightWorldEnv(width=16, height=12, audition=config)
    try:
        _prepare_state(
            environment,
            seed=PROFILE_SEED,
            objects={"block": {"position_m": (1.0, 0.3, 0.7)}},
        )
        if environment.audition is None:
            raise RuntimeError("profile requires audition")
        time_s = np.arange(round(0.25 * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
        tone = 0.1 * np.sin(2 * np.pi * 6_000 * time_s)
        environment.audition.clips.register(
            "profile_occlusion_tone",
            tone,
            reference_peak_pressure_pa_at_1m=1.0,
        )
        listener, _ = environment.audition.scene.listener_pose()
        block_position = np.asarray(environment.data.body("block").xpos).copy()
        source_position = listener + 1.5 * (block_position - listener)
        source = AudioSourceSpec(
            "occlusion_tone",
            "profile_occlusion_tone",
            StaticAttachment(tuple(float(value) for value in source_position)),
            loop=True,
        )
        environment.audition.register_source(source)
        measurements: dict[str, dict[str, Any]] = {}
        for name, block_xy in (("blocked", (1.0, 0.3)), ("clear", (1.0, -0.3))):
            _prepare_state(
                environment,
                seed=PROFILE_SEED,
                objects={
                    "block": {
                        "position_m": (block_xy[0], block_xy[1], 0.7),
                    }
                },
            )
            listener_now, _ = environment.audition.scene.listener_pose()
            ray_blocked = environment.audition.scene.direct_path_occluded(
                listener_now,
                source_position,
                -1,
            )
            environment.audition.play("occlusion_tone")

            def countergravity(_: int, env: BlightWorldEnv) -> None:
                _apply_countergravity(env)

            pressure, events, peak = _run_steps(
                environment,
                20,
                force_callback=countergravity,
            )
            if events:
                raise AssertionError("occlusion fixture unexpectedly produced impacts")
            measurements[name] = {
                "block_position_xy_m": list(block_xy),
                "ray_occluded": ray_blocked,
                "steady": _steady_pressure_metrics(pressure),
                "waveform_sha256": _waveform_sha256(pressure),
                "peak_reserved_voices": peak,
            }
    finally:
        environment.close()

    if not measurements["blocked"]["ray_occluded"]:
        raise AssertionError("verified block fixture did not occlude the source ray")
    if measurements["clear"]["ray_occluded"]:
        raise AssertionError("moved block continued to occlude the source ray")
    blocked_rms = measurements["blocked"]["steady"]["rms_pa"]
    clear_rms = measurements["clear"]["steady"]["rms_pa"]
    blocked_high = measurements["blocked"]["steady"]["high_frequency_rms_pa"]
    clear_high = measurements["clear"]["steady"]["high_frequency_rms_pa"]
    rms_ratio = float(blocked_rms / clear_rms)
    high_frequency_ratio = float(blocked_high / clear_high)
    if not 0 < rms_ratio < 0.5 or not 0 < high_frequency_ratio < rms_ratio:
        raise AssertionError("occlusion did not attenuate and low-pass the direct path")
    return {
        "description": (
            "Direct-only looping 6 kHz fixture; the same MuJoCo block moves off-ray."
        ),
        "source_world_position_m": [float(value) for value in source_position],
        "measurements": measurements,
        "blocked_over_clear_rms": rms_ratio,
        "blocked_over_clear_high_frequency": high_frequency_ratio,
    }


def _interaction_stress_trial(active: bool) -> dict[str, Any]:
    """Run the exact deterministic 60-step passive or active interaction trace."""

    environment = BlightWorldEnv(
        width=16,
        height=12,
        audition=AuditionConfig.quality_48k(),
    )
    final_rgb: dict[str, str] = {}
    try:
        environment.reset(seed=PROFILE_SEED)

        def actions(
            step_index: int,
            env: BlightWorldEnv,
        ) -> NDArray[np.floating]:
            phase = 0.17 * step_index + 0.37 * np.arange(90)
            return np.clip(
                0.28 * np.sin(phase),
                env.action_space.low,
                env.action_space.high,
            )

        def capture_rgb(
            _: int,
            __: BlightWorldEnv,
            observation: dict[str, NDArray[Any]],
        ) -> None:
            for eye_name in ("eye_left", "eye_right"):
                eye = np.ascontiguousarray(observation[eye_name])
                final_rgb[eye_name] = hashlib.sha256(eye.tobytes()).hexdigest()

        pressure, events, peak = _run_steps(
            environment,
            60,
            action_factory=actions if active else None,
            trajectory_callback=capture_rgb,
        )
        if environment.audition is None:
            raise RuntimeError("profile requires audition")
        cursor = environment.audition.renderer.sample_cursor
        final_voice_count = len(environment.audition.renderer._voices)
    finally:
        environment.close()
    accepted = [event for event in events if event["accepted"]]
    rejected = [event for event in events if not event["accepted"]]
    return {
        "mode": "active_sinusoidal" if active else "passive_zero_action",
        "sample_cursor": cursor,
        "audio": _audio_metrics(pressure),
        "event_count": len(events),
        "accepted_event_count": len(accepted),
        "rejected_event_count": len(rejected),
        "peak_reserved_voices": peak,
        "final_retained_voice_count": final_voice_count,
        "foot_or_toe_events": [
            event
            for event in events
            if any(
                token in str(event["attachment_body"])
                for token in ("foot", "toe")
            )
        ],
        "events": events,
        "final_rgb_sha256": final_rgb,
    }


def profile_interaction_stress() -> dict[str, Any]:
    """Validate passive and seed-1401 active voice-admission behavior."""

    passive = _interaction_stress_trial(False)
    active = _interaction_stress_trial(True)
    active_repeat = _interaction_stress_trial(True)
    if passive["sample_cursor"] != 28_800 or active["sample_cursor"] != 28_800:
        raise AssertionError("60-step acoustic sample cursor was not exact")
    if (
        passive["event_count"],
        passive["accepted_event_count"],
        passive["rejected_event_count"],
        passive["peak_reserved_voices"],
        passive["final_retained_voice_count"],
    ) != (7, 7, 0, 7, 0):
        raise AssertionError("passive interaction trace changed")
    if (
        active["event_count"],
        active["accepted_event_count"],
        active["rejected_event_count"],
        active["peak_reserved_voices"],
    ) != (11, 8, 3, 8):
        raise AssertionError("active interaction voice allocation changed")
    if active != active_repeat:
        raise AssertionError("active interaction replay was not exact")
    return {
        "description": (
            "Exact 60-step passive control and deterministic 90-actuator sinusoid."
        ),
        "passive": passive,
        "active": active,
        "active_repeat_exact": True,
        "max_voice_capacity": 8,
    }


def run_profile() -> dict[str, Any]:
    """Run the complete reproducible live-world validation profile."""

    return {
        "schema_version": 1,
        "seed": PROFILE_SEED,
        "environment": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "physics_timestep_s": 0.005,
            "frame_skip": 2,
            "observation_shape": [2, 480],
        },
        "directional_drops": profile_directional_drops(),
        "near_far_drops": profile_near_far_drops(),
        "direct_path_distance": profile_direct_path_distance(),
        "material_matrix": profile_material_matrix(),
        "mirrored_feet": profile_mirrored_feet(),
        "hand_forearm_fixtures": profile_hand_forearm_fixtures(),
        "moving_ball_crossing": profile_moving_ball_crossing(),
        "head_yaw": profile_head_yaw(),
        "block_occlusion": profile_block_occlusion(),
        "interaction_stress": profile_interaction_stress(),
        "scope": {
            "modelled": [
                "impact onsets",
                "positioned and moving point sources",
                "adult H10 HRTF cues",
                "direct distance and delay",
                "occlusion",
                "first-order reflections and late reverb",
            ],
            "not_modelled": [
                "scrape sounds",
                "rolling sounds",
                "a gait policy",
                "a learned reaching policy",
            ],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a deterministic human-readable summary of measured JSON."""

    directional = report["directional_drops"]["trials"]
    direct = report["direct_path_distance"]
    near_far = report["near_far_drops"]
    feet = report["mirrored_feet"]["trials"]
    material = report["material_matrix"]
    moving = report["moving_ball_crossing"]
    yaw = report["head_yaw"]["measurements"]
    occlusion = report["block_occlusion"]
    stress = report["interaction_stress"]
    lines = [
        "# Audition live-world validation profile",
        "",
        (
            "This report was generated by `tools/profile_audition_world.py` from "
            "deterministic MuJoCo stepping in `BlightWorldEnv` (seed 1401)."
        ),
        "",
        "## Spatial impact cues",
        "",
        "| Fixture | Energy (J) | Distance (m) | ILD L/R (dB) | Arrival (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("front", "left", "right"):
        trial = directional[name]
        lines.append(
            "| {name} | {energy:.9f} | {distance:.6f} | {ild:+.3f} | "
            "{arrival:.3f} |".format(
                name=name,
                energy=trial["event"]["energy_j"],
                distance=trial["event"]["listener_distance_m"],
                ild=trial["audio"]["ild_db_left_over_right"],
                arrival=trial["audio"]["arrival_after_event_ms"],
            )
        )
    lines.extend(
        [
            "",
            (
                "The equal-energy lateral fixtures reverse ILD sign. Adult-HRTF "
                "left/right magnitudes are not assumed symmetric."
            ),
            "",
            "## Distance",
            "",
            (
                f"The direct-only fixed-ray pressure fit is "
                f"**{direct['fit_db_per_decade']:.4f} dB/decade**; its direction "
                "and HRTF index remain fixed."
            ),
            (
                f"The full-room far-minus-near arrival difference is "
                f"**{near_far['far_minus_near_arrival_ms']:.4f} ms**. Direct-path "
                "pressure follows 1/r, while reflected plus late-reverb energy is "
                "allowed to be nonmonotonic."
            ),
            "",
            "## Physical material coverage",
            "",
            "| Recipe | Material family | Event time (s) | Energy (J) |",
            "|---|---|---:|---:|",
        ]
    )
    for recipe_name, family in material["recipe_material_families"].items():
        event = material["recipes"][recipe_name]["event"]
        lines.append(
            f"| {recipe_name} | {family} | {event['time_s']:.3f} | "
            f"{event['energy_j']:.9f} |"
        )
    lines.extend(
        [
            "",
            (
                "All eleven reachable unordered material families produced finite, "
                "nonzero stereo pressure and repeated exactly. Seeded resting contacts "
                "produced no event and zero pressure."
            ),
            "",
            "## MIMo contact fixtures",
            "",
            (
                "The -10°/+10° tilted landings selected "
                f"`{feet['left_first']['physical_body']}` and "
                f"`{feet['right_first']['physical_body']}` respectively, "
                "with reversed binaural cue signs. The hand and forearm cases use "
                "countergravity and a labeled 15 N external-force fixture; they are "
                "not learned behavior."
            ),
            "",
            "## Moving sources and head rotation",
            "",
            "| Crossing sample | Ball y (m) | Azimuth (°) | ILD L/R (dB) | HRTF index |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("start", "midpoint", "end"):
        sample = moving[name]
        lines.append(
            f"| {name} | {sample['ball_y_m']:+.3f} | "
            f"{sample['azimuth_deg']:+.3f} | "
            f"{sample['ild_db_left_over_right']:+.3f} | "
            f"{sample['hrtf_index']} |"
        )
    lines.extend(
        [
            "",
            (
                "The body-attached looping source crossed the midline through actual "
                "MuJoCo stepping. Its ten-block endpoint means were "
                f"{moving['mean_endpoint_ild_db']['start']:+.3f} and "
                f"{moving['mean_endpoint_ild_db']['end']:+.3f} dB; the maximum "
                "adjacent ILD change within ±10° was "
                f"{moving['zero_crossing_max_adjacent_ild_step_db']:.3f} dB."
            ),
            "",
            "| Head yaw (°) | Source azimuth in head frame (°) | ILD L/R (dB) | HRTF index |",
            "|---:|---:|---:|---:|",
        ]
    )
    for yaw_name in ("-30", "0", "30"):
        sample = yaw[yaw_name]
        lines.append(
            f"| {sample['yaw_deg']} | {sample['azimuth_deg']:+.3f} | "
            f"{sample['complete_binaural']['complete_energy_ild_db_left_over_right']:+.3f} | "
            f"{sample['hrtf_index']} |"
        )
    lines.extend(
        [
            "",
            "## Geometric occlusion",
            "",
            (
                "With the real block on the source ray, the ray flag was "
                f"`{str(occlusion['measurements']['blocked']['ray_occluded']).lower()}`; "
                "moving the block to the other side changed it to "
                f"`{str(occlusion['measurements']['clear']['ray_occluded']).lower()}`. "
                "Blocked/clear steady RMS was "
                f"**{occlusion['blocked_over_clear_rms']:.4f}×** and its ≥4 kHz "
                "spectral measure was "
                f"**{occlusion['blocked_over_clear_high_frequency']:.4f}×**."
            ),
            "",
            "## Interaction and voice capacity",
            "",
            "| Trace | Events | Accepted | Rejected | Peak dry voices | Final retained |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in ("passive", "active"):
        sample = stress[mode]
        lines.append(
            f"| {mode} | {sample['event_count']} | "
            f"{sample['accepted_event_count']} | "
            f"{sample['rejected_event_count']} | "
            f"{sample['peak_reserved_voices']} | "
            f"{sample['final_retained_voice_count']} |"
        )
    lines.extend(
        [
            "",
            (
                "Both 60-step traces advanced exactly 28,800 acoustic samples. The "
                "seed-1401 90-actuator sinusoid replayed exactly and never exceeded "
                "the eight-voice dry capacity."
            ),
            "",
            "## Scope",
            "",
            (
                "The framework models positioned sources, impact onsets, adult H10 "
                "binaural cues, distance/delay, occlusion, first-order reflections, "
                "and a late tail. It does **not** synthesize scraping or rolling, and "
                "this profile contains no gait or learned interaction policy."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the live profile and emit deterministic JSON and Markdown."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for the complete machine-readable report",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="optional path for the human-readable measured summary",
    )
    arguments = parser.parse_args(argv)
    report = run_profile()
    json_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    markdown_text = render_markdown(report)
    if arguments.json_output is not None:
        arguments.json_output.write_text(json_text, encoding="utf-8")
    else:
        print(json_text, end="")
    if arguments.markdown_output is not None:
        arguments.markdown_output.write_text(markdown_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

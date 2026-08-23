"""Reproducible live-MuJoCo validation profile for scrape audition.

The fixtures constrain object pose, speed, and load to isolate acoustic and
contact invariants. They are validation apparatus, not learned or autonomous
MIMo behavior.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import json
from math import log10
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
    ContinuousSoundConfig,
    StaticAttachment,
)
from mimo_audition.config import ScrapeConfig
from mimo_audition.continuous import ContinuousEmission
from mimo_audition.scrape import (
    ScrapeContact,
    ScrapeDiagnostic,
    ScrapeSynthesizer,
    ScrapeTracker,
)
from mimo_audition.sources import Attachment

if __package__:
    from tools.profile_audition_world import (
        _apply_countergravity,
        _prepare_state,
        _scalar_int,
    )
else:
    from profile_audition_world import (  # type: ignore[no-redef]
        _apply_countergravity,
        _prepare_state,
        _scalar_int,
    )


FloatArray = NDArray[np.float64]
PROFILE_SEED = 1_401
SAMPLE_RATE_HZ = 48_000
SAMPLES_PER_SUBSTEP = 240
SUPPORTED_FIXTURES = (
    ("block_floor", "block", "floor", (1.8, 0.0, 0.099), ("hard", "wood")),
    ("block_table", "block", "table_top", (0.8, 0.0, 0.699), ("wood", "wood")),
    (
        "cylinder_floor",
        "cylinder",
        "floor",
        (1.8, 0.0, 0.099),
        ("hard", "plastic"),
    ),
    (
        "cylinder_table",
        "cylinder",
        "table_top",
        (0.8, 0.0, 0.699),
        ("wood", "plastic"),
    ),
)


def _json_float(value: Any) -> float:
    result = float(np.asarray(value).item())
    if not np.isfinite(result):
        raise ValueError("profile measurement must be finite")
    return result


def _rms(values: NDArray[np.floating]) -> float:
    samples = np.asarray(values, dtype=np.float64)
    return _json_float(np.sqrt(np.mean(samples * samples)))


def _sha256_array(values: NDArray[np.floating]) -> str:
    contiguous = np.ascontiguousarray(values)
    return sha256(contiguous.tobytes()).hexdigest()


def _new_environment() -> BlightWorldEnv:
    return BlightWorldEnv(
        width=16,
        height=12,
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


def _geom_pair(
    environment: BlightWorldEnv,
    support_geom: str,
    body_name: str,
) -> tuple[int, int]:
    return tuple(
        sorted(
            (
                _scalar_int(environment.model.geom(support_geom).id),
                _scalar_int(environment.model.geom(f"{body_name}_geom").id),
            )
        )
    )


def _target_diagnostic(
    diagnostics: tuple[ScrapeDiagnostic, ...],
    pair: tuple[int, int],
) -> ScrapeDiagnostic | None:
    matches = [diagnostic for diagnostic in diagnostics if diagnostic.geom_pair == pair]
    if len(matches) > 1:
        raise AssertionError("one geom pair produced duplicate scrape diagnostics")
    return matches[0] if matches else None


@dataclass(slots=True)
class _RenderRecord:
    source_blocks: dict[str, NDArray[np.float32]]
    admitted_stream_ids: tuple[str, ...]
    dropped_stream_ids: tuple[str, ...]


@contextmanager
def _record_renderer(
    environment: BlightWorldEnv,
) -> Iterator[list[_RenderRecord]]:
    if environment.audition is None:
        raise RuntimeError("scrape profile requires audition")
    renderer = environment.audition.renderer
    original = renderer.render_substep
    records: list[_RenderRecord] = []

    def recording_render_substep(
        scene: object,
        continuous_emissions: Sequence[ContinuousEmission] = (),
    ) -> FloatArray:
        output = original(scene, continuous_emissions)  # type: ignore[arg-type]
        records.append(
            _RenderRecord(
                source_blocks={
                    emission.stream_id: emission.samples_pa_at_1m.copy()
                    for emission in continuous_emissions
                    if emission.kind.startswith("scrape:")
                },
                admitted_stream_ids=renderer.continuous_active_stream_ids,
                dropped_stream_ids=renderer.continuous_dropped_stream_ids,
            )
        )
        return output

    renderer.render_substep = recording_render_substep  # type: ignore[method-assign]
    try:
        yield records
    finally:
        renderer.render_substep = original  # type: ignore[method-assign]


def _finish_audio_step_if_ready(environment: BlightWorldEnv) -> None:
    if environment.audition is None:
        raise RuntimeError("scrape profile requires audition")
    if (
        len(environment.audition._pending_substeps)
        == environment.audition.config.spec.frame_skip
    ):
        environment.audition.finish_step()


def _supported_substep(
    environment: BlightWorldEnv,
    *,
    body_name: str,
    qpos_address: int,
    dof_address: int,
    contact_height_m: float,
    speed_m_s: float,
    extra_downward_force_n: float,
) -> tuple[ScrapeDiagnostic, ...]:
    """Advance one 5 ms constrained support-contact fixture."""

    environment.data.qpos[qpos_address + 2] = contact_height_m
    environment.data.qvel[dof_address : dof_address + 6] = 0.0
    environment.data.qvel[dof_address] = speed_m_s
    _apply_countergravity(environment)
    body_id = _scalar_int(environment.model.body(body_name).id)
    environment.data.xfrc_applied[body_id, 2] -= extra_downward_force_n
    mujoco.mj_forward(environment.model, environment.data)
    if environment.audition is None:
        raise RuntimeError("scrape profile requires audition")
    environment.audition.before_substep()
    mujoco.mj_step(environment.model, environment.data)
    mujoco.mj_rnePostConstraint(environment.model, environment.data)
    environment.audition.capture_substep()
    _finish_audio_step_if_ready(environment)
    return environment.audition.scrape_diagnostics


def _prepare_supported_fixture(
    environment: BlightWorldEnv,
    *,
    body_name: str,
    position_m: tuple[float, float, float],
    seed: int,
) -> tuple[int, int]:
    _prepare_state(
        environment,
        seed=seed,
        objects={body_name: {"position_m": position_m}},
    )
    return _free_joint_addresses(environment, body_name)


def _spectral_summary(samples: FloatArray) -> dict[str, float]:
    centered = samples - np.mean(samples)
    windowed = centered * np.hanning(centered.size)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(centered.size, 1.0 / SAMPLE_RATE_HZ)
    total = float(np.sum(spectrum))
    if total <= 0:
        raise AssertionError("scrape spectrum has zero energy")
    centroid = float(np.sum(frequencies * spectrum) / total)
    peak = float(frequencies[int(np.argmax(spectrum))])
    low = float(np.sum(spectrum[frequencies < 2_000.0]))
    high = float(np.sum(spectrum[frequencies >= 4_000.0]))
    return {
        "spectral_centroid_hz": _json_float(centroid),
        "spectral_peak_hz": _json_float(peak),
        "high_over_low_energy": _json_float(high / max(low, np.finfo(float).tiny)),
    }


def _clock_artifact_metrics(samples: FloatArray) -> dict[str, float]:
    differences = np.diff(samples)
    difference_rms = _rms(differences)
    result: dict[str, float] = {}
    for period_ms, period_samples in ((5, 240), (10, 480)):
        boundaries = np.arange(period_samples, samples.size, period_samples)
        jumps = samples[boundaries] - samples[boundaries - 1]
        result[f"boundary_{period_ms}ms_jump_over_all_difference_rms"] = (
            _json_float(_rms(jumps) / difference_rms)
        )
    rms = _rms(samples)
    time = np.arange(samples.size, dtype=np.float64) / SAMPLE_RATE_HZ
    for frequency_hz in (100, 200):
        coherent = abs(
            np.sum(samples * np.exp(-2j * np.pi * frequency_hz * time))
        ) / samples.size
        result[f"coherent_{frequency_hz}hz_over_rms"] = _json_float(
            coherent / rms
        )
    lag = SAMPLES_PER_SUBSTEP
    result["lag_5ms_correlation"] = _json_float(
        np.corrcoef(samples[:-lag], samples[lag:])[0, 1]
    )
    return result


def _run_supported_cell(
    environment: BlightWorldEnv,
    *,
    fixture_name: str,
    body_name: str,
    support_geom: str,
    position_m: tuple[float, float, float],
    material_pair: tuple[str, str],
    speed_m_s: float,
    extra_downward_force_n: float,
    seed: int = PROFILE_SEED,
) -> tuple[dict[str, Any], FloatArray, FloatArray, tuple[ScrapeDiagnostic, ...]]:
    qpos_address, dof_address = _prepare_supported_fixture(
        environment,
        body_name=body_name,
        position_m=position_m,
        seed=seed,
    )
    pair = _geom_pair(environment, support_geom, body_name)
    if environment.audition is None or not isinstance(
        environment.audition._scrapes,
        ScrapeTracker,
    ):
        raise RuntimeError("supported scrape profile requires a contact tracker")
    sensor = environment.audition
    tracker = sensor._scrapes
    diagnostics: list[ScrapeDiagnostic] = []
    observation_blocks: list[NDArray[np.float32]] = []
    with _record_renderer(environment) as records:
        for substep in range(38):
            controls = _supported_substep(
                environment,
                body_name=body_name,
                qpos_address=qpos_address,
                dof_address=dof_address,
                contact_height_m=position_m[2],
                speed_m_s=0.0 if substep < 8 else speed_m_s,
                extra_downward_force_n=extra_downward_force_n,
            )
            target = _target_diagnostic(controls, pair)
            if substep >= 18:
                measurement = tracker._measurements().get(pair)
                if measurement is None:
                    raise AssertionError("supported target measurement disappeared")
                if target is None or target.phase != "active":
                    raise AssertionError("supported target scrape was not continuously active")
                if not (
                    measurement.material_pair
                    == target.material_pair
                    == material_pair
                ):
                    raise AssertionError("supported target material identity changed")
                if (
                    target.normal_load_n != measurement.normal_load_n
                    or target.tangential_speed_m_s
                    != measurement.tangential_speed_m_s
                ):
                    raise AssertionError(
                        "scrape diagnostic diverged from solved contact controls"
                    )
                stream_id = target.stream_id
                record = records[-1]
                if stream_id not in record.source_blocks:
                    raise AssertionError("target scrape source block was not rendered")
                if stream_id not in record.admitted_stream_ids:
                    raise AssertionError("target scrape stream was not admitted")
                if stream_id in record.dropped_stream_ids:
                    raise AssertionError("target scrape stream was capacity-dropped")
                diagnostics.append(target)
            if substep % 2 == 1:
                time_before = float(environment.data.time)
                cursor_before = sensor.renderer.sample_cursor
                render_count_before = len(records)
                observation = environment._get_obs()
                if not environment.observation_space.contains(observation):
                    raise AssertionError("post-render observation left its Gym space")
                audition = observation.get("audition")
                if (
                    not isinstance(audition, np.ndarray)
                    or audition.shape != (2, 480)
                    or audition.dtype != np.float32
                    or not audition.flags.c_contiguous
                ):
                    raise AssertionError("public audition observation contract changed")
                if (
                    float(environment.data.time) != time_before
                    or sensor.renderer.sample_cursor != cursor_before
                    or len(records) != render_count_before
                ):
                    raise AssertionError("observation read advanced simulation or audio")
                observation_blocks.append(audition.copy(order="C"))

    if (
        len(records) != 38
        or len(diagnostics) != 20
        or len(observation_blocks) != 19
    ):
        raise AssertionError("supported fixture produced the wrong substep count")
    stream_id = f"scrape:{pair[0]}:{pair[1]}"
    source_blocks = [record.source_blocks[stream_id] for record in records[18:]]
    source = np.concatenate(source_blocks).astype(np.float64)
    received = np.concatenate(observation_blocks[9:], axis=1)
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(received)):
        raise AssertionError("supported scrape produced non-finite pressure")
    row = {
        "fixture": fixture_name,
        "body": body_name,
        "support_geom": support_geom,
        "stream_id": stream_id,
        "material_pair": "_".join(material_pair),
        "speed_m_s": speed_m_s,
        "extra_downward_force_n": extra_downward_force_n,
        "median_normal_load_n": _json_float(
            np.median([value.normal_load_n for value in diagnostics])
        ),
        "median_friction_power_w": _json_float(
            np.median([value.friction_power_w for value in diagnostics])
        ),
        "source_rms_pa_at_1m": _rms(source),
        "received_left_rms_pa": _rms(received[0]),
        "received_right_rms_pa": _rms(received[1]),
        "received_stereo_rms_pa": _rms(received),
        "received_transport": "gym_observation_float32",
        "post_render_observation_reads": len(observation_blocks),
        "steady_observation_blocks": len(observation_blocks[9:]),
        "public_observation_conformant": True,
        "observation_reads_idempotent": True,
        "solved_measurement_present_every_analyzed_substep": True,
        "target_source_block_present_every_analyzed_substep": True,
        "target_material_verified": True,
        "target_stream_admitted_every_analyzed_substep": True,
        "diagnostic_matches_solved_contact_controls": True,
        "median_centroid_m": [
            _json_float(component)
            for component in np.median(
                np.asarray([value.world_position_m for value in diagnostics]),
                axis=0,
            )
        ],
    }
    return row, source, received, tuple(diagnostics)


def _profile_scrape_matrix(environment: BlightWorldEnv) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    canonical_sources: dict[str, FloatArray] = {}
    for fixture_name, body_name, support, position, material_pair in SUPPORTED_FIXTURES:
        body_id = _scalar_int(environment.model.body(body_name).id)
        body_weight_n = _json_float(environment.model.body_mass[body_id] * 9.81)
        fixture_rows: list[dict[str, Any]] = []
        for extra_load in (0.0, body_weight_n):
            for speed in (0.2, 0.4, 0.8):
                row, source, _, _ = _run_supported_cell(
                    environment,
                    fixture_name=fixture_name,
                    body_name=body_name,
                    support_geom=support,
                    position_m=position,
                    material_pair=material_pair,
                    speed_m_s=speed,
                    extra_downward_force_n=extra_load,
                )
                rows.append(row)
                fixture_rows.append(row)
                if extra_load == 0.0 and speed == 0.4:
                    canonical_sources["_".join(material_pair)] = source
        for extra_load in (0.0, body_weight_n):
            powers = [
                row["median_friction_power_w"]
                for row in fixture_rows
                if row["extra_downward_force_n"] == extra_load
            ]
            if not powers[0] < powers[1] < powers[2]:
                raise AssertionError("friction power did not increase with speed")
        for speed in (0.2, 0.4, 0.8):
            samples = [row for row in fixture_rows if row["speed_m_s"] == speed]
            if not (
                samples[1]["median_normal_load_n"]
                > samples[0]["median_normal_load_n"]
                and samples[1]["median_friction_power_w"]
                > samples[0]["median_friction_power_w"]
            ):
                raise AssertionError("load did not increase scrape control power")

    spectral = {
        material: {
            **_spectral_summary(source),
            **_clock_artifact_metrics(source),
        }
        for material, source in sorted(canonical_sources.items())
    }
    return {
        "cell_count": len(rows),
        "all_contacts_continuous": True,
        "all_target_streams_admitted": True,
        "public_observation_transport_valid": True,
        "power_monotonic_with_speed_and_load": True,
        "cells": rows,
        "material_source_spectra": spectral,
    }


def profile_scrape_matrix() -> dict[str, Any]:
    """Run the complete 24-cell live object/support/speed/load matrix."""

    environment = _new_environment()
    try:
        return _profile_scrape_matrix(environment)
    finally:
        environment.close()


@contextmanager
def _record_impacts(
    environment: BlightWorldEnv,
) -> Iterator[list[dict[str, Any]]]:
    if environment.audition is None:
        raise RuntimeError("scrape profile requires audition")
    renderer = environment.audition.renderer
    original = renderer.play_transient
    events: list[dict[str, Any]] = []

    def recording_play_transient(
        clip_id: str,
        attachment: Attachment,
        *,
        gain_db: float = 0.0,
        delay_s: float = 0.0,
        impact_energy_j: float = 0.0,
    ) -> int | None:
        voice_id = original(
            clip_id,
            attachment,
            gain_db=gain_db,
            delay_s=delay_s,
            impact_energy_j=impact_energy_j,
        )
        events.append(
            {
                "clip_id": clip_id,
                "energy_j": _json_float(impact_energy_j),
                "voice_id": voice_id,
                "time_s": _json_float(environment.data.time),
            }
        )
        return voice_id

    renderer.play_transient = recording_play_transient  # type: ignore[method-assign]
    try:
        yield events
    finally:
        renderer.play_transient = original  # type: ignore[method-assign]


def _unconstrained_substep(
    environment: BlightWorldEnv,
) -> tuple[ScrapeDiagnostic, ...]:
    _apply_countergravity(environment)
    mujoco.mj_forward(environment.model, environment.data)
    if environment.audition is None:
        raise RuntimeError("scrape profile requires audition")
    environment.audition.before_substep()
    mujoco.mj_step(environment.model, environment.data)
    mujoco.mj_rnePostConstraint(environment.model, environment.data)
    environment.audition.capture_substep()
    _finish_audio_step_if_ready(environment)
    return environment.audition.scrape_diagnostics


def _profile_reversal(environment: BlightWorldEnv) -> dict[str, Any]:
    qpos_address, dof_address = _prepare_supported_fixture(
        environment,
        body_name="block",
        position_m=(1.8, 0.0, 0.099),
        seed=PROFILE_SEED,
    )
    pair = _geom_pair(environment, "floor", "block")
    positive: list[ScrapeDiagnostic] = []
    negative: list[ScrapeDiagnostic] = []
    with _record_renderer(environment) as records:
        for substep in range(48):
            if substep < 8:
                speed = 0.0
            elif substep < 28:
                speed = 0.4
            else:
                speed = -0.4
            diagnostics = _supported_substep(
                environment,
                body_name="block",
                qpos_address=qpos_address,
                dof_address=dof_address,
                contact_height_m=0.099,
                speed_m_s=speed,
                extra_downward_force_n=0.0,
            )
            target = _target_diagnostic(diagnostics, pair)
            if 18 <= substep < 28 and target is not None:
                positive.append(target)
            if 38 <= substep and target is not None:
                negative.append(target)
    if len(positive) != 10 or len(negative) != 10:
        raise AssertionError("reversal fixture lost its target stream")
    positive_velocity = np.median(
        np.asarray([value.tangential_velocity_world_m_s for value in positive]),
        axis=0,
    )
    negative_velocity = np.median(
        np.asarray([value.tangential_velocity_world_m_s for value in negative]),
        axis=0,
    )
    positive_norm = float(np.linalg.norm(positive_velocity))
    negative_norm = float(np.linalg.norm(negative_velocity))
    cosine = float(
        np.dot(positive_velocity, negative_velocity)
        / (positive_norm * negative_norm)
    )
    if not np.isclose(positive_norm, negative_norm, rtol=0, atol=1e-12):
        raise AssertionError("reversal changed controlled tangential speed")
    if not np.isclose(cosine, -1.0, rtol=0, atol=1e-12):
        raise AssertionError("reversal did not invert the tangential vector")
    stream_id = positive[0].stream_id
    positive_source = np.concatenate(
        [record.source_blocks[stream_id] for record in records[18:28]]
    )
    negative_source = np.concatenate(
        [record.source_blocks[stream_id] for record in records[38:48]]
    )
    return {
        "stream_id": stream_id,
        "material_pair": "_".join(positive[0].material_pair),
        "positive_velocity_world_m_s": [
            _json_float(value) for value in positive_velocity
        ],
        "negative_velocity_world_m_s": [
            _json_float(value) for value in negative_velocity
        ],
        "speed_norm_ratio": _json_float(negative_norm / positive_norm),
        "direction_cosine": _json_float(cosine),
        "positive_source_rms_pa_at_1m": _rms(positive_source),
        "negative_source_rms_pa_at_1m": _rms(negative_source),
    }


def _profile_stop(environment: BlightWorldEnv) -> dict[str, Any]:
    qpos_address, dof_address = _prepare_supported_fixture(
        environment,
        body_name="block",
        position_m=(1.8, 0.0, 0.099),
        seed=PROFILE_SEED,
    )
    pair = _geom_pair(environment, "floor", "block")
    phases: list[str | None] = []
    final_samples: list[float | None] = []
    with _record_renderer(environment) as records:
        for substep in range(36):
            diagnostics = _supported_substep(
                environment,
                body_name="block",
                qpos_address=qpos_address,
                dof_address=dof_address,
                contact_height_m=0.099,
                speed_m_s=0.0 if substep < 8 or substep >= 28 else 0.4,
                extra_downward_force_n=0.0,
            )
            if substep >= 28:
                target = _target_diagnostic(diagnostics, pair)
                phases.append(None if target is None else target.phase)
                stream_id = f"scrape:{pair[0]}:{pair[1]}"
                source = records[-1].source_blocks.get(stream_id)
                final_samples.append(
                    None if source is None else _json_float(source[-1])
                )
    expected = ["release"] * 6 + [None, None]
    if phases != expected or final_samples[5] != 0.0:
        raise AssertionError("stop fixture did not complete its exact release")
    return {
        "phase_by_substep": phases,
        "release_substeps": 6,
        "last_release_sample_pa_at_1m": final_samples[5],
        "absent_after_release": final_samples[6:] == [None, None],
    }


def _profile_lift(environment: BlightWorldEnv) -> dict[str, Any]:
    qpos_address, dof_address = _prepare_supported_fixture(
        environment,
        body_name="block",
        position_m=(1.8, 0.0, 0.099),
        seed=PROFILE_SEED,
    )
    pair = _geom_pair(environment, "floor", "block")
    for substep in range(28):
        _supported_substep(
            environment,
            body_name="block",
            qpos_address=qpos_address,
            dof_address=dof_address,
            contact_height_m=0.099,
            speed_m_s=0.0 if substep < 8 else 0.4,
            extra_downward_force_n=0.0,
        )
    phases: list[str | None] = []
    physical_pair_present: list[bool] = []
    if environment.audition is None or environment.audition._scrapes is None:
        raise RuntimeError("lift fixture requires scrape tracking")
    tracker = environment.audition._scrapes
    with _record_renderer(environment):
        for substep in range(8):
            if substep == 0:
                environment.data.qpos[qpos_address + 2] = 0.30
                environment.data.qvel[dof_address : dof_address + 6] = 0.0
                environment.data.qvel[dof_address + 2] = 1.0
            diagnostics = _unconstrained_substep(environment)
            target = _target_diagnostic(diagnostics, pair)
            phases.append(None if target is None else target.phase)
            physical_pair_present.append(pair in tracker._measurements())
    expected = ["active"] + ["release"] * 6 + [None]
    if phases != expected or any(physical_pair_present):
        raise AssertionError("lift dropout/release lifecycle changed")
    return {
        "physical_pair_present_by_substep": physical_pair_present,
        "phase_by_substep": phases,
        "dropout_bridge_substeps": 1,
        "release_substeps": 6,
    }


def _profile_drop(environment: BlightWorldEnv) -> dict[str, Any]:
    _prepare_state(
        environment,
        seed=PROFILE_SEED,
        objects={
            "block": {
                "position_m": (1.8, 0.0, 0.13),
                "linear_velocity_m_s": (0.0, 0.0, -2.0),
            }
        },
    )
    qpos_address, dof_address = _free_joint_addresses(environment, "block")
    pair = _geom_pair(environment, "floor", "block")
    phases: list[str | None] = []
    with _record_renderer(environment), _record_impacts(environment) as impacts:
        for _ in range(12):
            if impacts:
                diagnostics = _supported_substep(
                    environment,
                    body_name="block",
                    qpos_address=qpos_address,
                    dof_address=dof_address,
                    contact_height_m=0.099,
                    speed_m_s=0.4,
                    extra_downward_force_n=0.0,
                )
            else:
                diagnostics = _unconstrained_substep(environment)
            target = _target_diagnostic(diagnostics, pair)
            phases.append(None if target is None else target.phase)
    if len(impacts) != 1 or not impacts[0]["clip_id"].startswith(
        "impact_hard_wood_"
    ):
        raise AssertionError("drop fixture did not emit exactly one target impact")
    active_indices = [index for index, phase in enumerate(phases) if phase == "active"]
    if not active_indices:
        raise AssertionError("drop fixture never transitioned into scrape")
    return {
        "impact_count": len(impacts),
        "impact": impacts[0],
        "scrape_phase_by_substep": phases,
        "first_active_scrape_substep": active_indices[0],
    }


def _profile_scrape_transitions(environment: BlightWorldEnv) -> dict[str, Any]:
    return {
        "reversal": _profile_reversal(environment),
        "stop": _profile_stop(environment),
        "lift": _profile_lift(environment),
        "drop": _profile_drop(environment),
    }


def profile_scrape_transitions() -> dict[str, Any]:
    """Run reversal, stop, lift, and impact-to-scrape live fixtures."""

    environment = _new_environment()
    try:
        return _profile_scrape_transitions(environment)
    finally:
        environment.close()


def _ild_db(received: FloatArray) -> float:
    return _json_float(20.0 * log10(_rms(received[0]) / _rms(received[1])))


def _profile_mirrors(environment: BlightWorldEnv) -> dict[str, Any]:
    fixtures: dict[str, Any] = {}
    strict_mirrors = {"block_floor", "block_table", "cylinder_floor"}
    for fixture_name, body_name, support, position, material_pair in SUPPORTED_FIXTURES:
        sides: dict[str, Any] = {}
        controls: dict[str, tuple[ScrapeDiagnostic, ...]] = {}
        for side, lateral_m in (("left", 0.35), ("right", -0.35)):
            mirrored_position = (position[0], lateral_m, position[2])
            row, source, received, diagnostics = _run_supported_cell(
                environment,
                fixture_name=fixture_name,
                body_name=body_name,
                support_geom=support,
                position_m=mirrored_position,
                material_pair=material_pair,
                speed_m_s=0.4,
                extra_downward_force_n=0.0,
            )
            controls[side] = diagnostics
            sides[side] = {
                "centroid_m": row["median_centroid_m"],
                "source_rms_pa_at_1m": _rms(source),
                "received_left_rms_pa": _rms(received[0]),
                "received_right_rms_pa": _rms(received[1]),
                "ild_db": _ild_db(received),
            }
        if not (
            sides["left"]["centroid_m"][1] > 0.25
            and sides["right"]["centroid_m"][1] < -0.25
        ):
            raise AssertionError("mirrored contact centroids left their hemifields")
        cue_reversal = sides["left"]["ild_db"] * sides["right"]["ild_db"] < 0
        if not cue_reversal:
            raise AssertionError("mirrored full-room ILD did not reverse sign")
        roundoff_control_mirror = fixture_name in strict_mirrors
        if roundoff_control_mirror:
            for field in (
                "normal_load_n",
                "tangential_speed_m_s",
                "friction_power_w",
            ):
                np.testing.assert_array_max_ulp(
                    [getattr(value, field) for value in controls["left"]],
                    [getattr(value, field) for value in controls["right"]],
                    maxulp=16,
                )
        fixtures[fixture_name] = {
            "material_pair": "_".join(material_pair),
            "left": sides["left"],
            "right": sides["right"],
            "ild_sign_reversal": cue_reversal,
            "control_mirror_within_float64_roundoff": roundoff_control_mirror,
            "control_mirror_max_ulp_allowed": (
                16 if roundoff_control_mirror else None
            ),
        }
    return {
        "fixtures": fixtures,
        "adult_hrtf_asymmetry_allowed": True,
        "cylinder_table_contact_manifold_not_claimed_mirror_exact": True,
    }


def profile_mirrored_scrapes() -> dict[str, Any]:
    """Run left/right full-room binaural scrape fixtures."""

    environment = _new_environment()
    try:
        return _profile_mirrors(environment)
    finally:
        environment.close()


def _synthetic_capacity_contacts() -> tuple[ScrapeContact, ...]:
    """Return four labeled controls for the renderer-pool capacity fixture.

    This fixture isolates scheduling from MuJoCo contact generation. The live
    matrix above is the physical-contact validation; these controls exercise
    the independently bounded clip and continuous pools with calibrated scrape
    blocks from the real synthesizer.
    """

    definitions = (
        ((0, 11), ("hard", "wood"), BodyAttachment("block")),
        ((6, 12), ("wood", "rubber"), BodyAttachment("ball")),
        ((0, 13), ("hard", "plastic"), BodyAttachment("cylinder")),
        ((6, 11), ("wood", "wood"), BodyAttachment("block")),
    )
    speed_m_s = 0.2
    contacts: list[ScrapeContact] = []
    for index, (pair, materials, attachment) in enumerate(definitions):
        force_n = 0.5 + 0.1 * index
        contacts.append(
            ScrapeContact(
                stream_id=f"scrape:{pair[0]}:{pair[1]}",
                geom_pair=pair,
                material_pair=materials,  # type: ignore[arg-type]
                attachment=attachment,
                world_position_m=(1.0, 0.0, 0.7),
                normal_load_n=1.0,
                tangential_force_world_n=(force_n, 0.0, 0.0),
                tangential_velocity_world_m_s=(speed_m_s, 0.0, 0.0),
                tangential_speed_m_s=speed_m_s,
                friction_power_w=force_n * speed_m_s,
            )
        )
    return tuple(contacts)


def _profile_replay_and_capacity(environment: BlightWorldEnv) -> dict[str, Any]:
    fixture = SUPPORTED_FIXTURES[0]
    fixture_name, body_name, support, position, material_pair = fixture
    first_row, first_source, first_received, _ = _run_supported_cell(
        environment,
        fixture_name=fixture_name,
        body_name=body_name,
        support_geom=support,
        position_m=position,
        material_pair=material_pair,
        speed_m_s=0.4,
        extra_downward_force_n=0.0,
    )
    replay_row, replay_source, replay_received, _ = _run_supported_cell(
        environment,
        fixture_name=fixture_name,
        body_name=body_name,
        support_geom=support,
        position_m=position,
        material_pair=material_pair,
        speed_m_s=0.4,
        extra_downward_force_n=0.0,
    )
    np.testing.assert_array_equal(first_source, replay_source)
    np.testing.assert_array_equal(first_received, replay_received)
    if first_row != replay_row:
        raise AssertionError("same-side seeded scrape controls did not replay exactly")

    _prepare_state(
        environment,
        seed=PROFILE_SEED,
        objects={
            "block": {"position_m": (1.0, 0.3, 0.699)},
            "ball": {"position_m": (1.0, -0.3, 0.699)},
            "cylinder": {"position_m": (1.0, 0.0, 0.699)},
        },
    )
    if environment.audition is None:
        raise RuntimeError("scrape profile requires audition")
    sensor = environment.audition
    source_id = "scrape_profile_capacity_clip"
    sensor.register_source(
        AudioSourceSpec(
            source_id,
            "chime",
            StaticAttachment((1.0, 0.0, 0.8)),
            loop=True,
        )
    )
    for _ in range(8):
        sensor.play(source_id)
    synthesizer = ScrapeSynthesizer(
        sensor.config.spec,
        sensor.config.continuous.scrape,  # type: ignore[union-attr]
        seed=PROFILE_SEED,
    )
    emissions = synthesizer.render(_synthetic_capacity_contacts())
    capacity_output = sensor.renderer.render_substep(sensor.scene, emissions)
    if not np.all(np.isfinite(capacity_output)) or not np.any(capacity_output):
        raise AssertionError("combined clip/scrape capacity fixture was not audible")
    if sensor.renderer.active_voice_count != 8:
        raise AssertionError("capacity fixture did not retain eight clip voices")
    if sensor.renderer.continuous_active_voice_count != 4:
        raise AssertionError("capacity fixture did not admit four scrape streams")

    return {
        "same_side_seeded_replay_byte_exact": True,
        "source_waveform_sha256": _sha256_array(first_source),
        "received_waveform_sha256": _sha256_array(first_received),
        "control_row": first_row,
        "capacity_fixture_kind": "direct_renderer_with_scrape_synthesizer",
        "clip_active_voices": sensor.renderer.active_voice_count,
        "continuous_active_voices": sensor.renderer.continuous_active_voice_count,
        "continuous_retained_voices": (
            sensor.renderer.continuous_retained_voice_count
        ),
        "continuous_stream_ids": list(
            sensor.renderer.continuous_active_stream_ids
        ),
        "combined_output_rms_pa": _rms(capacity_output),
        "combined_output_finite": True,
    }


def profile_scrape_replay_and_capacity() -> dict[str, Any]:
    """Verify seeded replay and independent eight-plus-four voice capacity."""

    environment = _new_environment()
    try:
        return _profile_replay_and_capacity(environment)
    finally:
        environment.close()


def _load_benchmark_summary(path: Path) -> dict[str, Any]:
    """Load and validate one completed standalone streaming benchmark."""

    contents = path.read_bytes()
    parsed = json.loads(contents)
    if not isinstance(parsed, dict) or parsed.get("schema_version") != 1:
        raise ValueError("unsupported benchmark report schema")
    if (
        parsed.get("benchmark")
        != "audition_clip_plus_four_scrape_streams"
        or parsed.get("non_ci") is not True
    ):
        raise ValueError("unexpected benchmark identity")
    rates = parsed.get("rates")
    if not isinstance(rates, dict) or set(rates) != {"24000", "48000"}:
        raise ValueError("benchmark must contain both supported sample rates")

    expected_durations = {"24000": 600.0, "48000": 60.0}
    summaries: dict[str, Any] = {}
    for rate_text, simulated_seconds in expected_durations.items():
        row = rates[rate_text]
        if not isinstance(row, dict):
            raise ValueError("benchmark rate entry must be an object")
        rate = int(rate_text)
        latency = row.get("substep_latency_ms")
        memory = row.get("memory_kib")
        if not isinstance(latency, dict) or not isinstance(memory, dict):
            raise ValueError("benchmark rate lacks latency or memory metrics")
        numeric_values = (
            row.get("wall_seconds"),
            row.get("simulated_seconds_per_wall_second"),
            latency.get("p50"),
            latency.get("p95"),
            latency.get("p99"),
            latency.get("maximum"),
        )
        if not np.all(np.isfinite(numeric_values)):
            raise ValueError("benchmark contains non-finite timing metrics")
        if (
            row.get("simulated_seconds") != simulated_seconds
            or row.get("sample_cursor") != round(rate * simulated_seconds)
            or row.get("all_output_finite") is not True
            or row.get("max_clip_active_voices") != 8
            or row.get("max_continuous_active_voices") != 4
            or row.get("max_scrape_synthesizer_states") != 4
            or row.get("realtime_claimed") is not False
        ):
            raise ValueError("benchmark rate violates the frozen run contract")
        summaries[rate_text] = {
            "simulated_seconds": row["simulated_seconds"],
            "physics_substeps": row["physics_substeps"],
            "sample_cursor": row["sample_cursor"],
            "accepted_impacts": row["accepted_impacts"],
            "rejected_impacts": row["rejected_impacts"],
            "all_output_finite": row["all_output_finite"],
            "max_clip_active_voices": row["max_clip_active_voices"],
            "max_clip_retained_states": row["max_clip_retained_states"],
            "max_continuous_active_voices": row[
                "max_continuous_active_voices"
            ],
            "max_continuous_retained_states": row[
                "max_continuous_retained_states"
            ],
            "max_scrape_synthesizer_states": row[
                "max_scrape_synthesizer_states"
            ],
            "memory_kib": memory,
            "substep_latency_ms": latency,
            "wall_seconds": row["wall_seconds"],
            "simulated_seconds_per_wall_second": row[
                "simulated_seconds_per_wall_second"
            ],
            "p99_below_5ms": row["p99_below_5ms"],
            "realtime_claimed": row["realtime_claimed"],
        }
    return {
        "artifact_sha256": sha256(contents).hexdigest(),
        "invocation": (
            "MUJOCO_GL=egl .venv/bin/python "
            "tools/benchmark_audition_streaming.py --duration-24k 600 "
            "--duration-48k 60 --warmup 2 --progress-every 5 "
            "--json-output /tmp/blight-audition-benchmark-full.json"
        ),
        "seed": parsed.get("seed"),
        "non_ci": True,
        "rates": summaries,
    }


def run_profile(
    streaming_benchmark: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the complete deterministic scrape validation profile."""

    environment = _new_environment()
    try:
        matrix = _profile_scrape_matrix(environment)
        transitions = _profile_scrape_transitions(environment)
        mirrors = _profile_mirrors(environment)
        replay_capacity = _profile_replay_and_capacity(environment)
    finally:
        environment.close()
    report = {
        "schema_version": 1,
        "profile": "blight_world_live_scrape_audition",
        "seed": PROFILE_SEED,
        "software": {
            "blight_world": metadata.version("blight-world"),
            "mimo": metadata.version("MIMo"),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "timing": {
            "physics_substep_seconds": 0.005,
            "environment_step_seconds": 0.010,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "samples_per_substep": SAMPLES_PER_SUBSTEP,
        },
        "fixture_contract": {
            "kinematically_constrained_validation_fixtures": True,
            "learned_or_autonomous_behavior": False,
            "continuous_sound_scope": "sliding_scrape_only",
            "excluded_mechanisms": ["rolling", "clothing", "air"],
        },
        "scrape_matrix": matrix,
        "transitions": transitions,
        "mirrored_binaural": mirrors,
        "replay_and_capacity": replay_capacity,
    }
    if streaming_benchmark is not None:
        report["streaming_benchmark"] = streaming_benchmark
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render the measured report without rerunning physics."""

    matrix = report["scrape_matrix"]
    transitions = report["transitions"]
    mirrors = report["mirrored_binaural"]
    replay = report["replay_and_capacity"]
    software = report["software"]
    lines = [
        "# Live-world scrape audition validation profile",
        "",
        (
            "This report was generated by stepping the real BlightWorld MuJoCo "
            f"model with seed **{report['seed']}** at 48 kHz. The fixtures "
            "kinematically constrain object height, velocity, and applied load to "
            "isolate contact-to-audio behavior; they are not learned or autonomous "
            "MIMo behavior."
        ),
        "",
        (
            f"Software: blight-world {software['blight_world']}, MIMo "
            f"{software['mimo']}, MuJoCo {software['mujoco']}, NumPy "
            f"{software['numpy']}."
        ),
        "",
        "## Live 24-cell scrape matrix",
        "",
        (
            f"All **{matrix['cell_count']}** object/support/speed/load cells kept "
            "their target contact continuously solved. Median friction power rose "
            "with controlled speed and added normal load in every fixture."
        ),
        "",
        (
            "Each cell made 19 idempotent post-render `_get_obs()` reads that "
            "remained inside the full observation space. Received metrics use the "
            "ten steady `observation['audition']` blocks (substeps 18–37) exactly "
            "as public C-contiguous float32 Gym data, not an internal renderer block."
        ),
        "",
        (
            "| Fixture | Material | Speed (m/s) | Extra load (N) | Normal load "
            "(N) | Friction power (W) | Source RMS Pa @ 1 m | Received stereo RMS "
            "(Pa) |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in matrix["cells"]:
        lines.append(
            f"| {row['fixture']} | {row['material_pair']} | "
            f"{row['speed_m_s']:.1f} | {row['extra_downward_force_n']:.6f} | "
            f"{row['median_normal_load_n']:.6f} | "
            f"{row['median_friction_power_w']:.6f} | "
            f"{row['source_rms_pa_at_1m']:.8f} | "
            f"{row['received_stereo_rms_pa']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Material texture and clock-boundary measures",
            "",
            (
                "The source-domain summaries use the canonical 0.4 m/s, unloaded "
                "trace for each reachable surface/object pair. Boundary ratios "
                "compare the 5/10 ms boundary jump RMS with all adjacent-sample "
                "difference RMS; coherent terms report 100/200 Hz components "
                "relative to waveform RMS."
            ),
            "",
            (
                "| Material | Spectral centroid (Hz) | Peak (Hz) | High/low "
                "energy | 5 ms boundary ratio | 10 ms boundary ratio | 100 Hz/RMS "
                "| 200 Hz/RMS |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for material, sample in matrix["material_source_spectra"].items():
        lines.append(
            f"| {material} | {sample['spectral_centroid_hz']:.3f} | "
            f"{sample['spectral_peak_hz']:.3f} | "
            f"{sample['high_over_low_energy']:.6f} | "
            f"{sample['boundary_5ms_jump_over_all_difference_rms']:.6f} | "
            f"{sample['boundary_10ms_jump_over_all_difference_rms']:.6f} | "
            f"{sample['coherent_100hz_over_rms']:.6f} | "
            f"{sample['coherent_200hz_over_rms']:.6f} |"
        )
    reversal = transitions["reversal"]
    stop = transitions["stop"]
    lift = transitions["lift"]
    drop = transitions["drop"]
    lines.extend(
        [
            "",
            "## State transitions",
            "",
            (
                "A velocity reversal retained one stream and reversed its full "
                f"tangential vector (cosine **{reversal['direction_cosine']:.12f}**, "
                f"speed ratio **{reversal['speed_norm_ratio']:.12f}**)."
            ),
            "",
            (
                f"A constrained stop emitted exactly {stop['release_substeps']} "
                "5 ms release blocks and then disappeared. A lifted contact used "
                f"one dropout bridge plus {lift['release_substeps']} release blocks."
            ),
            "",
            (
                f"The drop-to-slide fixture emitted exactly {drop['impact_count']} "
                f"impact (`{drop['impact']['clip_id']}` at "
                f"{drop['impact']['time_s']:.3f} s) before the persistent scrape "
                f"became active at substep {drop['first_active_scrape_substep']}."
            ),
            "",
            "## Mirrored full-room binaural cues",
            "",
            (
                "| Fixture | Left-source ILD L/R (dB) | Right-source ILD L/R "
                "(dB) | Cue sign reversed | Control mirror contract |"
            ),
            "|---|---:|---:|---|---|",
        ]
    )
    for fixture_name, fixture in mirrors["fixtures"].items():
        mirror_contract = (
            "within float64 roundoff (≤16 ULP)"
            if fixture["control_mirror_within_float64_roundoff"]
            else "not claimed: convex contact-manifold asymmetry"
        )
        lines.append(
            f"| {fixture_name} | {fixture['left']['ild_db']:+.6f} | "
            f"{fixture['right']['ild_db']:+.6f} | "
            f"{str(fixture['ild_sign_reversal']).lower()} | {mirror_contract} |"
        )
    lines.extend(
        [
            "",
            (
                "Adult H10 HRTF asymmetry is allowed: reflected source positions "
                "must reverse ILD sign, not magnitude. Block-floor, block-table, "
                "and cylinder-floor contact controls match within measured float64 "
                "roundoff (16 ULP bound). Cylinder-table is excluded from scalar "
                "mirror equality because MuJoCo selects an asymmetric single convex "
                "contact manifold; its contact hemifields and cue reversal still pass."
            ),
            "",
            "## Replay and bounded voice pools",
            "",
            (
                "The same-side seeded live scrape replayed byte-exactly. Source "
                f"SHA-256: `{replay['source_waveform_sha256']}`; received SHA-256: "
                f"`{replay['received_waveform_sha256']}`."
            ),
            "",
            (
                f"A labeled direct-renderer capacity fixture rendered "
                f"{replay['clip_active_voices']} looping clip voices and "
                f"{replay['continuous_active_voices']} independently bounded scrape "
                "streams concurrently with finite nonzero output. It isolates the "
                "scheduler; the 24-cell matrix above supplies the live-contact evidence."
            ),
        ]
    )
    benchmark = report.get("streaming_benchmark")
    if benchmark is not None:
        lines.extend(
            [
                "",
                "## Bounded-state streaming benchmark",
                "",
                (
                    "The standalone non-CI benchmark held four scrape streams while "
                    "scheduling dense eight-impact bursts once per simulated second. "
                    f"Its JSON artifact SHA-256 is `{benchmark['artifact_sha256']}`."
                ),
                "",
                (
                    "| Rate | Simulated duration | Substeps | Accepted/rejected "
                    "impacts | Max clip/continuous/synth states | Warmup→end RSS "
                    "(KiB) | p99 substep (ms) | Wall time (s) |"
                ),
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rate_text, sample in benchmark["rates"].items():
            memory = sample["memory_kib"]
            lines.append(
                f"| {rate_text} | {sample['simulated_seconds']:.0f} s | "
                f"{sample['physics_substeps']} | "
                f"{sample['accepted_impacts']}/{sample['rejected_impacts']} | "
                f"{sample['max_clip_active_voices']}/"
                f"{sample['max_continuous_active_voices']}/"
                f"{sample['max_scrape_synthesizer_states']} | "
                f"{memory['after_warmup_rss']}→{memory['end_rss']} | "
                f"{sample['substep_latency_ms']['p99']:.3f} | "
                f"{sample['wall_seconds']:.3f} |"
            )
        lines.extend(
            [
                "",
                (
                    "Both rates retained bounded clip, renderer-continuous, and "
                    "synthesizer state and produced only finite output. Their p99 "
                    "latencies exceeded the 5 ms physics budget, so this reference "
                    "Python implementation makes **no real-time claim**."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Scope and exclusions",
            "",
            (
                "This validation covers persistent sliding scrape synthesis, contact "
                "hysteresis/release, positioned full-room binaural rendering, and "
                "independent clip/continuous capacity. Rolling, clothing, and air "
                "sounds are not implemented. The constrained fixtures are engineering "
                "experiments and do not demonstrate gait, manipulation skill, or an "
                "agent policy."
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
        help="optional path for the measured human-readable report",
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        help="optional completed standalone benchmark JSON to incorporate",
    )
    arguments = parser.parse_args(argv)
    benchmark = (
        None
        if arguments.benchmark_json is None
        else _load_benchmark_summary(arguments.benchmark_json)
    )
    report = run_profile(benchmark)
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

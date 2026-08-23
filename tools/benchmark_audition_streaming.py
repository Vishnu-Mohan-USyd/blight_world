"""Bounded-state long-run benchmark for clip plus continuous audition.

The default run advances ten simulated minutes at 24 kHz and one simulated
minute at 48 kHz. It is intentionally a standalone, non-CI benchmark. Timing
numbers describe this process and machine only; the report never labels the
renderer real-time merely because one percentile crosses a threshold.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import resource
import sys
from time import perf_counter_ns
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mimo_audition import (
    AuditionConfig,
    BodyAttachment,
    ContinuousSoundConfig,
    StaticAttachment,
)
from mimo_audition.renderer import BinauralRenderer
from mimo_audition.scrape import ScrapeContact, ScrapeSynthesizer
from mimo_audition.sources import Attachment, SiteAttachment


FloatArray = NDArray[np.float64]
PHYSICS_SUBSTEP_SECONDS = 0.005
BENCHMARK_SEED = 1_401


class _BenchmarkScene:
    """Small deterministic scene implementing the renderer's public protocol."""

    _BODY_POSITIONS = {
        "block": np.asarray((1.0, 0.30, 0.70), dtype=np.float64),
        "ball": np.asarray((1.0, -0.30, 0.70), dtype=np.float64),
        "cylinder": np.asarray((1.2, 0.10, 0.70), dtype=np.float64),
        "table": np.asarray((0.8, -0.10, 0.65), dtype=np.float64),
    }

    def listener_pose(self) -> tuple[FloatArray, FloatArray]:
        return (
            np.asarray((0.0, 0.0, 0.70), dtype=np.float64),
            np.eye(3, dtype=np.float64),
        )

    def attachment_pose(self, attachment: Attachment) -> tuple[FloatArray, int]:
        if isinstance(attachment, StaticAttachment):
            return np.asarray(attachment.world_position_m, dtype=np.float64), -1
        if isinstance(attachment, SiteAttachment):
            raise ValueError("benchmark scene has no named sites")
        try:
            origin = self._BODY_POSITIONS[attachment.body_name]
        except KeyError as error:
            raise ValueError(
                f"benchmark scene has no body {attachment.body_name}"
            ) from error
        return (
            origin + np.asarray(attachment.local_position_m, dtype=np.float64),
            -1,
        )

    def direct_path_occluded(
        self,
        listener_position_m: FloatArray,
        source_position_m: FloatArray,
        source_body_id: int,
    ) -> bool:
        del listener_position_m, source_position_m, source_body_id
        return False


def _scrape_contacts() -> tuple[ScrapeContact, ...]:
    definitions = (
        ((0, 11), ("hard", "wood"), "block", (1.0, 0.30, 0.70), 0.10),
        ((6, 12), ("wood", "rubber"), "ball", (1.0, -0.30, 0.70), 0.12),
        ((0, 13), ("hard", "plastic"), "cylinder", (1.2, 0.10, 0.70), 0.14),
        ((6, 11), ("wood", "wood"), "table", (0.8, -0.10, 0.65), 0.16),
    )
    result: list[ScrapeContact] = []
    speed_m_s = 0.20
    for pair, materials, body_name, position, power_w in definitions:
        force_n = power_w / speed_m_s
        result.append(
            ScrapeContact(
                stream_id=f"scrape:{pair[0]}:{pair[1]}",
                geom_pair=pair,
                material_pair=materials,  # type: ignore[arg-type]
                attachment=BodyAttachment(body_name),
                world_position_m=position,
                normal_load_n=1.0,
                tangential_force_world_n=(0.0, force_n, 0.0),
                tangential_velocity_world_m_s=(0.0, speed_m_s, 0.0),
                tangential_speed_m_s=speed_m_s,
                friction_power_w=power_w,
            )
        )
    return tuple(result)


def _linux_memory_kib() -> tuple[int, int]:
    """Return current and high-water RSS in KiB without another dependency."""

    status_path = Path("/proc/self/status")
    if status_path.is_file():
        values: dict[str, int] = {}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            name, separator, remainder = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM"}:
                values[name] = int(remainder.strip().split()[0])
        if "VmRSS" in values and "VmHWM" in values:
            return values["VmRSS"], values["VmHWM"]
    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum, maximum


def _audition_config(sample_rate_hz: int) -> AuditionConfig:
    factory = (
        AuditionConfig.training_24k
        if sample_rate_hz == 24_000
        else AuditionConfig.quality_48k
    )
    return factory(continuous=ContinuousSoundConfig(max_continuous_voices=4))


def benchmark_rate(
    sample_rate_hz: int,
    *,
    simulated_seconds: float,
    warmup_seconds: float = 2.0,
    impact_burst_period_seconds: float = 1.0,
    progress_every_simulated_seconds: float | None = None,
) -> dict[str, Any]:
    """Run one deterministic bounded-state streaming benchmark.

    Four independently seeded scrape streams render every 5 ms. Eight impact
    one-shots are scheduled together once per configured simulated interval.
    Reported latency covers synthesis, scheduling, and rendering for each
    measured substep after warmup.
    """

    if sample_rate_hz not in (24_000, 48_000):
        raise ValueError("sample_rate_hz must be 24000 or 48000")
    for name, value in (
        ("simulated_seconds", simulated_seconds),
        ("warmup_seconds", warmup_seconds),
        ("impact_burst_period_seconds", impact_burst_period_seconds),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if warmup_seconds >= simulated_seconds:
        raise ValueError("warmup_seconds must be below simulated_seconds")
    if (
        progress_every_simulated_seconds is not None
        and progress_every_simulated_seconds <= 0
    ):
        raise ValueError("progress interval must be positive")

    config = _audition_config(sample_rate_hz)
    renderer = BinauralRenderer(config)
    scene = _BenchmarkScene()
    synthesizer = ScrapeSynthesizer(
        config.spec,
        config.continuous.scrape,  # type: ignore[union-attr]
        seed=BENCHMARK_SEED,
    )
    contacts = _scrape_contacts()
    total_substeps = round(simulated_seconds / PHYSICS_SUBSTEP_SECONDS)
    warmup_substeps = round(warmup_seconds / PHYSICS_SUBSTEP_SECONDS)
    burst_period_substeps = max(
        1,
        round(impact_burst_period_seconds / PHYSICS_SUBSTEP_SECONDS),
    )
    progress_period_substeps = (
        None
        if progress_every_simulated_seconds is None
        else max(
            1,
            round(
                progress_every_simulated_seconds / PHYSICS_SUBSTEP_SECONDS
            ),
        )
    )
    latencies_ms = np.empty(total_substeps - warmup_substeps, dtype=np.float64)
    initial_rss_kib, initial_hwm_kib = _linux_memory_kib()
    warmup_rss_kib = initial_rss_kib
    warmup_hwm_kib = initial_hwm_kib
    accepted_impacts = 0
    rejected_impacts = 0
    max_clip_active = 0
    max_clip_retained = 0
    max_continuous_active = 0
    max_continuous_retained = 0
    max_synthesizer_states = 0
    all_output_finite = True
    latency_index = 0
    loop_start_ns = perf_counter_ns()

    for substep in range(total_substeps):
        start_ns = perf_counter_ns()
        if substep % burst_period_substeps == 0:
            for impact_index in range(8):
                voice_id = renderer.play_transient(
                    "impact_hard_wood_1",
                    StaticAttachment((1.0, 0.0, 0.70)),
                    impact_energy_j=0.01 + impact_index * 0.001,
                )
                if voice_id is None:
                    rejected_impacts += 1
                else:
                    accepted_impacts += 1
            max_clip_active = max(max_clip_active, renderer.active_voice_count)
        emissions = synthesizer.render(contacts)
        output = renderer.render_substep(scene, emissions)
        elapsed_ms = (perf_counter_ns() - start_ns) / 1_000_000.0
        if substep >= warmup_substeps:
            latencies_ms[latency_index] = elapsed_ms
            latency_index += 1
        if not np.all(np.isfinite(output)):
            all_output_finite = False
        max_clip_active = max(max_clip_active, renderer.active_voice_count)
        max_clip_retained = max(max_clip_retained, len(renderer._voices))
        max_continuous_active = max(
            max_continuous_active,
            renderer.continuous_active_voice_count,
        )
        max_continuous_retained = max(
            max_continuous_retained,
            renderer.continuous_retained_voice_count,
        )
        max_synthesizer_states = max(
            max_synthesizer_states,
            len(synthesizer.stream_ids),
        )
        if substep + 1 == warmup_substeps:
            warmup_rss_kib, warmup_hwm_kib = _linux_memory_kib()
        if (
            progress_period_substeps is not None
            and (substep + 1) % progress_period_substeps == 0
        ):
            print(
                f"{sample_rate_hz} Hz: "
                f"{(substep + 1) * PHYSICS_SUBSTEP_SECONDS:.1f}/"
                f"{simulated_seconds:.1f} simulated seconds",
                file=sys.stderr,
                flush=True,
            )

    loop_wall_seconds = (perf_counter_ns() - loop_start_ns) / 1_000_000_000.0
    end_rss_kib, end_hwm_kib = _linux_memory_kib()
    if latency_index != latencies_ms.size:
        raise AssertionError("benchmark latency sample count changed")
    if renderer.sample_cursor != round(simulated_seconds * sample_rate_hz):
        raise AssertionError("benchmark acoustic clock drifted")
    if not all_output_finite:
        raise FloatingPointError("benchmark produced non-finite output")
    if max_clip_active > 8 or max_continuous_active > 4:
        raise AssertionError("benchmark exceeded a configured voice pool")
    percentiles = np.percentile(latencies_ms, (50, 95, 99))
    return {
        "sample_rate_hz": sample_rate_hz,
        "simulated_seconds": simulated_seconds,
        "warmup_seconds": warmup_seconds,
        "physics_substeps": total_substeps,
        "measured_substeps": int(latencies_ms.size),
        "sample_cursor": renderer.sample_cursor,
        "impact_burst_period_seconds": impact_burst_period_seconds,
        "accepted_impacts": accepted_impacts,
        "rejected_impacts": rejected_impacts,
        "all_output_finite": all_output_finite,
        "max_clip_active_voices": max_clip_active,
        "max_clip_retained_states": max_clip_retained,
        "max_continuous_active_voices": max_continuous_active,
        "max_continuous_retained_states": max_continuous_retained,
        "max_scrape_synthesizer_states": max_synthesizer_states,
        "memory_kib": {
            "initial_rss": initial_rss_kib,
            "after_warmup_rss": warmup_rss_kib,
            "end_rss": end_rss_kib,
            "initial_high_water": initial_hwm_kib,
            "after_warmup_high_water": warmup_hwm_kib,
            "peak_high_water": end_hwm_kib,
            "end_minus_warmup_rss": end_rss_kib - warmup_rss_kib,
        },
        "substep_latency_ms": {
            "p50": float(percentiles[0]),
            "p95": float(percentiles[1]),
            "p99": float(percentiles[2]),
            "maximum": float(np.max(latencies_ms)),
        },
        "wall_seconds": loop_wall_seconds,
        "simulated_seconds_per_wall_second": (
            simulated_seconds / loop_wall_seconds
        ),
        "p99_below_5ms": bool(percentiles[2] < 5.0),
        "realtime_claimed": False,
        "realtime_claim_note": (
            "No real-time claim; this is one process/machine measurement."
        ),
    }


def run_benchmarks(
    *,
    duration_24k_seconds: float = 600.0,
    duration_48k_seconds: float = 60.0,
    warmup_seconds: float = 2.0,
    progress_every_simulated_seconds: float | None = 30.0,
) -> dict[str, Any]:
    """Run the default long 24 kHz and shorter 48 kHz benchmarks."""

    return {
        "schema_version": 1,
        "benchmark": "audition_clip_plus_four_scrape_streams",
        "seed": BENCHMARK_SEED,
        "non_ci": True,
        "configuration": {
            "duration_24k_seconds": duration_24k_seconds,
            "duration_48k_seconds": duration_48k_seconds,
            "warmup_seconds": warmup_seconds,
            "impact_burst_period_seconds": 1.0,
            "continuous_streams": 4,
            "impacts_per_burst": 8,
            "physics_substep_seconds": PHYSICS_SUBSTEP_SECONDS,
        },
        "rates": {
            "24000": benchmark_rate(
                24_000,
                simulated_seconds=duration_24k_seconds,
                warmup_seconds=warmup_seconds,
                progress_every_simulated_seconds=(
                    progress_every_simulated_seconds
                ),
            ),
            "48000": benchmark_rate(
                48_000,
                simulated_seconds=duration_48k_seconds,
                warmup_seconds=min(warmup_seconds, duration_48k_seconds / 2.0),
                progress_every_simulated_seconds=(
                    progress_every_simulated_seconds
                ),
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-24k", type=float, default=600.0)
    parser.add_argument("--duration-48k", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--progress-every", type=float, default=30.0)
    parser.add_argument("--json-output", type=Path)
    arguments = parser.parse_args(argv)
    result = run_benchmarks(
        duration_24k_seconds=arguments.duration_24k,
        duration_48k_seconds=arguments.duration_48k,
        warmup_seconds=arguments.warmup,
        progress_every_simulated_seconds=arguments.progress_every,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.json_output is None:
        print(text, end="")
    else:
        arguments.json_output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

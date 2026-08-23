"""Deterministic unit checks for persistent scrape contact tracking."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mimo_audition.config import AuditionConfig, ScrapeConfig
from mimo_audition.scrape import (
    ScrapeContact,
    ScrapeSynthesizer,
    _aggregate_contact_points,
    _ContactPoint,
    _ScrapeGate,
    _tangential_force_world,
)
from mimo_audition.sources import BodyAttachment


def _measurement(
    *,
    load: float = 0.2,
    speed: float = 0.04,
    pair: tuple[int, int] = (2, 7),
) -> ScrapeContact:
    return ScrapeContact(
        stream_id=f"scrape:{pair[0]}:{pair[1]}",
        geom_pair=pair,
        material_pair=("hard", "wood"),
        attachment=BodyAttachment("body"),
        world_position_m=(1.0, 0.0, 0.0),
        normal_load_n=load,
        tangential_force_world_n=(0.0, 1.5, 0.0),
        tangential_velocity_world_m_s=(0.0, speed, 0.0),
        tangential_speed_m_s=speed,
        friction_power_w=1.5 * speed,
    )


def _synthesis_contact(
    stream_id: str = "scrape:2:7",
    *,
    power_w: float = 0.1,
    attachment: BodyAttachment | None = None,
) -> ScrapeContact:
    speed = 0.1
    force = power_w / speed
    return ScrapeContact(
        stream_id=stream_id,
        geom_pair=(2, 7),
        material_pair=("hard", "wood"),
        attachment=attachment or BodyAttachment("body"),
        world_position_m=(1.0, 0.0, 0.0),
        normal_load_n=1.0,
        tangential_force_world_n=(0.0, force, 0.0),
        tangential_velocity_world_m_s=(0.0, speed, 0.0),
        tangential_speed_m_s=speed,
        friction_power_w=power_w,
    )


def test_contact_force_frame_and_load_weighted_aggregation() -> None:
    """Friction axes rotate to world space before deterministic aggregation."""

    frame = np.asarray(
        (
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    contact_force = np.asarray((5.0, 2.0, -3.0, 0.0, 0.0, 0.0))
    np.testing.assert_array_equal(
        _tangential_force_world(frame, contact_force),
        np.asarray((2.0, 0.0, -3.0)),
    )

    aggregate = _aggregate_contact_points(
        (
            _ContactPoint(
                world_position_m=(0.0, 0.0, 0.0),
                normal_load_n=1.0,
                tangential_force_world_n=(2.0, 0.0, 0.0),
                tangential_velocity_world_m_s=(0.0, 1.0, 0.0),
            ),
            _ContactPoint(
                world_position_m=(4.0, 0.0, 0.0),
                normal_load_n=3.0,
                tangential_force_world_n=(0.0, 3.0, 0.0),
                tangential_velocity_world_m_s=(0.0, 3.0, 0.0),
            ),
        )
    )
    centroid, load, force, velocity, speed, power = aggregate
    assert centroid == pytest.approx((3.0, 0.0, 0.0))
    assert load == 4.0
    assert force == pytest.approx((2.0, 3.0, 0.0))
    assert velocity == pytest.approx((0.0, 2.5, 0.0))
    assert speed == 2.5
    assert power == pytest.approx(np.sqrt(13.0) * 2.5)


def test_scrape_control_rejects_inconsistent_or_nonfinite_si_values() -> None:
    """Malformed geometry, units, and derived controls fail at construction."""

    with pytest.raises(ValueError, match="geom_pair"):
        replace(_synthesis_contact(), geom_pair=(7, 2))
    with pytest.raises(ValueError, match="finite"):
        replace(_synthesis_contact(), friction_power_w=float("nan"))
    with pytest.raises(ValueError, match="speed must match"):
        replace(_synthesis_contact(), tangential_speed_m_s=0.2)
    with pytest.raises(ValueError, match="force times speed"):
        replace(_synthesis_contact(), friction_power_w=0.2)


def test_scrape_gate_persistence_hysteresis_and_dropout() -> None:
    """Admission needs two contacts, holds hysteresis, and bridges one dropout."""

    gate = _ScrapeGate(ScrapeConfig())
    strong = _measurement()
    assert gate.update({strong.geom_pair: strong}) == ()
    assert gate.update({strong.geom_pair: strong}) == (strong,)

    hysteresis = _measurement(load=0.075, speed=0.015)
    assert gate.update({hysteresis.geom_pair: hysteresis}) == (hysteresis,)
    assert gate.update({}) == (hysteresis,)
    assert gate.update({}) == ()

    assert gate.update({strong.geom_pair: strong}) == ()
    assert gate.update({strong.geom_pair: strong}) == (strong,)
    below_off = _measurement(load=0.049, speed=0.04)
    assert gate.update({below_off.geom_pair: below_off}) == ()


def test_scrape_gate_orders_pairs_and_resets_exactly() -> None:
    """Pair ordering is deterministic and reset erases persistence."""

    gate = _ScrapeGate(ScrapeConfig(minimum_contact_substeps=1))
    later = _measurement(pair=(9, 10))
    earlier = _measurement(pair=(1, 12))
    assert gate.update(
        {later.geom_pair: later, earlier.geom_pair: earlier}
    ) == (earlier, later)
    gate.reset()
    assert gate.update({}) == ()


@pytest.mark.parametrize(
    ("config", "sample_count"),
    (
        (AuditionConfig.training_24k(), 120),
        (AuditionConfig.quality_48k(), 240),
    ),
)
def test_scrape_synthesis_has_exact_rate_shape_and_finite_pressure(
    config: AuditionConfig,
    sample_count: int,
) -> None:
    """Both fixed rates emit exact immutable float32 pressure substeps."""

    synthesizer = ScrapeSynthesizer(config.spec, ScrapeConfig(), seed=11)
    emission = synthesizer.render((_synthesis_contact(),))[0]
    assert emission.samples_pa_at_1m.shape == (sample_count,)
    assert emission.samples_pa_at_1m.dtype == np.float32
    assert not emission.samples_pa_at_1m.flags.writeable
    assert np.all(np.isfinite(emission.samples_pa_at_1m))
    assert np.max(np.abs(emission.samples_pa_at_1m)) > 0


def test_fourfold_friction_power_doubles_steady_source_rms() -> None:
    """The square-root pressure calibration maps 4x power to 2x RMS."""

    spec = AuditionConfig.quality_48k().spec
    low = ScrapeSynthesizer(spec, ScrapeConfig(), seed=17)
    high = ScrapeSynthesizer(spec, ScrapeConfig(), seed=17)
    low_block = np.zeros(240, dtype=np.float32)
    high_block = np.zeros(240, dtype=np.float32)
    for _ in range(20):
        low_block = low.render((_synthesis_contact(power_w=0.025),))[0].samples_pa_at_1m
        high_block = high.render((_synthesis_contact(power_w=0.1),))[0].samples_pa_at_1m
    low_rms = float(np.sqrt(np.mean(low_block.astype(np.float64) ** 2)))
    high_rms = float(np.sqrt(np.mean(high_block.astype(np.float64) ** 2)))
    assert high_rms / low_rms == pytest.approx(2.0, rel=1e-6)


def test_stream_rng_is_independent_of_contact_order_and_neighbours() -> None:
    """Stable BLAKE2 streams do not consume or reorder each other's texture."""

    spec = AuditionConfig.quality_48k().spec
    first = ScrapeSynthesizer(spec, ScrapeConfig(), seed=23)
    second = ScrapeSynthesizer(spec, ScrapeConfig(), seed=23)
    contact_a = _synthesis_contact("a")
    contact_b = _synthesis_contact("b", power_w=0.04)
    together = {
        emission.stream_id: emission.samples_pa_at_1m
        for emission in first.render((contact_b, contact_a))
    }
    alone = second.render((contact_a,))[0].samples_pa_at_1m
    np.testing.assert_array_equal(together["a"], alone)


def test_scrape_release_is_finite_and_reset_replays_exactly() -> None:
    """A vanished contact emits 30 ms to zero, disappears, and resets exactly."""

    spec = AuditionConfig.quality_48k().spec
    synthesizer = ScrapeSynthesizer(spec, ScrapeConfig(), seed=31)
    contact = _synthesis_contact()
    active = [synthesizer.render((contact,))[0].samples_pa_at_1m for _ in range(2)]
    release = [synthesizer.render(())[0].samples_pa_at_1m for _ in range(6)]
    assert release[-1][-1] == 0.0
    assert synthesizer.stream_ids == ()
    assert synthesizer.render(()) == ()

    synthesizer.reset(31)
    replay = [synthesizer.render((contact,))[0].samples_pa_at_1m for _ in range(2)]
    for expected, actual in zip(active, replay, strict=True):
        np.testing.assert_array_equal(expected, actual)


def test_persistent_texture_has_no_repeated_block_or_clock_line() -> None:
    """Independent samples avoid a 5 ms seam and coherent 100/200 Hz lines."""

    spec = AuditionConfig.quality_48k().spec
    synthesizer = ScrapeSynthesizer(spec, ScrapeConfig(), seed=47)
    contact = _synthesis_contact()
    blocks = [
        synthesizer.render((contact,))[0].samples_pa_at_1m.astype(np.float64)
        for _ in range(80)
    ]
    assert all(not np.array_equal(blocks[0], block) for block in blocks[1:])
    pressure = np.concatenate(blocks[4:])
    lag = spec.samples_per_substep
    correlation = float(np.corrcoef(pressure[:-lag], pressure[lag:])[0, 1])
    assert abs(correlation) < 0.2
    rms = float(np.sqrt(np.mean(pressure * pressure)))
    time = np.arange(pressure.size, dtype=np.float64) / spec.sample_rate_hz
    for frequency_hz in (100.0, 200.0):
        coherent_amplitude = abs(
            np.sum(pressure * np.exp(-2j * np.pi * frequency_hz * time))
        ) / pressure.size
        assert coherent_amplitude / rms < 0.1

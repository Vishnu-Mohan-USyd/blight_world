"""Unit tests for audition's public, simulation-independent contracts."""

from __future__ import annotations

import numpy as np
import pytest

from mimo_audition import (
    AuditionConfig,
    AudioSourceSpec,
    BodyAttachment,
    ContinuousEmission,
    ContinuousSoundConfig,
    ScrapeConfig,
    SiteAttachment,
    StaticAttachment,
)


@pytest.mark.parametrize(
    ("config", "expected_shape", "expected_substep"),
    (
        (AuditionConfig.quality_48k(), (2, 480), 240),
        (AuditionConfig.training_24k(), (2, 240), 120),
    ),
)
def test_profiles_preserve_mimo_timing(
    config: AuditionConfig,
    expected_shape: tuple[int, int],
    expected_substep: int,
) -> None:
    """Both profiles map exactly onto two 5 ms physics substeps."""

    assert config.spec.observation_shape == expected_shape
    assert config.spec.samples_per_substep == expected_substep
    assert config.spec.physics_timestep_s == 0.005
    assert config.spec.frame_skip == 2


def test_source_specifications_are_immutable_and_normalized() -> None:
    """Attachments carry explicit finite world/body/site coordinates."""

    static = StaticAttachment((1, 2, 3))
    body = BodyAttachment("ball", (0, 0.1, 0))
    site = SiteAttachment("torque_neck")
    assert static.world_position_m == (1.0, 2.0, 3.0)
    assert body.local_position_m == (0.0, 0.1, 0.0)
    assert site.site_name == "torque_neck"

    source = AudioSourceSpec("bell", "chime", body, loop=True, gain_db=-3)
    assert source.source_id == "bell"
    assert source.gain_db == -3


def test_continuous_configuration_defaults_and_hysteresis_are_validated() -> None:
    """Continuous and future scrape settings fail closed at construction."""

    assert AuditionConfig().continuous is None
    continuous = ContinuousSoundConfig()
    assert continuous.max_continuous_voices == 4
    assert continuous.unexpected_disappearance_fade_seconds == 0.005
    assert continuous.scrape == ScrapeConfig()
    assert AuditionConfig(continuous=continuous).continuous is continuous

    with pytest.raises(ValueError, match="cannot exceed 32"):
        ContinuousSoundConfig(max_continuous_voices=33)
    with pytest.raises(ValueError, match="positive integer"):
        ContinuousSoundConfig(max_continuous_voices=0)
    with pytest.raises(ValueError, match="non-negative"):
        ContinuousSoundConfig(unexpected_disappearance_fade_seconds=-0.001)
    with pytest.raises(ValueError, match="non-negative"):
        ContinuousSoundConfig(unexpected_disappearance_fade_seconds=float("nan"))
    with pytest.raises(ValueError, match="normal_force_off"):
        ScrapeConfig(normal_force_on_n=0.1, normal_force_off_n=0.2)
    with pytest.raises(ValueError, match="slip_speed_off"):
        ScrapeConfig(slip_speed_on_m_s=0.01, slip_speed_off_m_s=0.02)
    with pytest.raises(ValueError, match="reference pressure"):
        ScrapeConfig(
            reference_rms_pressure_pa_at_1m=0.06,
            max_rms_pressure_pa_at_1m=0.05,
        )
    with pytest.raises(ValueError, match="attack_seconds"):
        ScrapeConfig(attack_seconds=0.0)


@pytest.mark.parametrize("sample_count", (120, 240))
def test_continuous_emission_copies_exact_finite_substep_block(
    sample_count: int,
) -> None:
    """Emission storage is immutable float32 and independent of its caller."""

    samples = np.linspace(-0.1, 0.1, sample_count, dtype=np.float32)
    expected = samples.copy()
    emission = ContinuousEmission(
        "contact:1:2",
        "scrape",
        BodyAttachment("ball"),
        samples,
        priority=2,
    )
    samples.fill(0)

    np.testing.assert_array_equal(emission.samples_pa_at_1m, expected)
    assert emission.samples_pa_at_1m.dtype == np.float32
    assert emission.samples_pa_at_1m.shape == (sample_count,)
    assert emission.samples_pa_at_1m.flags.c_contiguous
    assert not emission.samples_pa_at_1m.flags.owndata
    assert not emission.samples_pa_at_1m.flags.writeable
    assert not np.shares_memory(emission.samples_pa_at_1m, samples)
    assert samples.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        emission.samples_pa_at_1m.setflags(write=True)
    with pytest.raises(ValueError, match="read-only"):
        emission.samples_pa_at_1m[0] = np.float32(1.0)
    np.testing.assert_array_equal(emission.samples_pa_at_1m, expected)
    assert samples.flags.writeable
    assert emission.priority == 2.0


@pytest.mark.parametrize(
    "samples",
    (
        np.zeros(119, dtype=np.float32),
        np.zeros((1, 120), dtype=np.float32),
        np.full(120, np.nan, dtype=np.float32),
    ),
)
def test_continuous_emission_rejects_invalid_blocks(samples: np.ndarray) -> None:
    """Malformed or nonfinite blocks cannot enter renderer state."""

    with pytest.raises(ValueError, match="samples_pa_at_1m"):
        ContinuousEmission(
            "stream",
            "scrape",
            StaticAttachment((1.0, 0.0, 0.0)),
            samples,
        )


@pytest.mark.parametrize("sample_rate_hz", (0, 44_100, 96_000))
def test_unsupported_sample_rates_fail_closed(sample_rate_hz: int) -> None:
    """Only the two validated block rates are accepted."""

    with pytest.raises(ValueError, match="24000 or 48000"):
        AuditionConfig(sample_rate_hz=sample_rate_hz)


@pytest.mark.parametrize(
    "attachment",
    (
        StaticAttachment,
        lambda value: BodyAttachment("body", value),
    ),
)
def test_invalid_positions_are_rejected(attachment: object) -> None:
    """NaN and malformed coordinates never reach MuJoCo transforms."""

    with pytest.raises(ValueError):
        attachment((0.0, float("nan"), 0.0))  # type: ignore[operator]

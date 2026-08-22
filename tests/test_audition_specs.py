"""Unit tests for audition's public, simulation-independent contracts."""

from __future__ import annotations

import pytest

from mimo_audition import (
    AuditionConfig,
    AudioSourceSpec,
    BodyAttachment,
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

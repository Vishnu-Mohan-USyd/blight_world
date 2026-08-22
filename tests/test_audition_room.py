"""Tests for deterministic early and late room acoustics."""

from __future__ import annotations

import numpy as np
import pytest

from mimo_audition.room import (
    FDN_DELAYS,
    FeedbackDelayNetwork,
    first_order_image_sources,
    normalized_sylvester_8,
    project_source_position,
    sabine_rt60_s,
)


def test_six_first_order_images_match_room_planes() -> None:
    """Every floor, ceiling, and wall contributes one image source."""

    images = first_order_image_sources(np.asarray((1.0, 0.25, 0.7)))
    assert len(images) == 6
    assert {image.surface for image in images} == {
        "wall_x_min",
        "wall_x_max",
        "wall_y_min",
        "wall_y_max",
        "floor",
        "ceiling",
    }
    floor = next(image for image in images if image.surface == "floor")
    ceiling = next(image for image in images if image.surface == "ceiling")
    np.testing.assert_array_equal(floor.world_position_m, (1.0, 0.25, -0.7))
    np.testing.assert_array_equal(ceiling.world_position_m, (1.0, 0.25, 4.3))
    assert floor.pressure_reflection == pytest.approx(np.sqrt(0.9))
    assert ceiling.pressure_reflection == pytest.approx(np.sqrt(0.7))


def test_source_boundary_projection_is_narrow_and_exact() -> None:
    """Contact-scale offsets project to a plane while larger errors fail."""

    in_bounds = np.asarray((1.0, 0.25, 0.7), dtype=np.float64)
    np.testing.assert_array_equal(project_source_position(in_bounds), in_bounds)
    np.testing.assert_array_equal(
        project_source_position(np.asarray((-2.5, 2.5, 0.0))),
        (-2.5, 2.5, 0.0),
    )
    np.testing.assert_array_equal(
        project_source_position(np.asarray((0.044, -0.076, -0.005074))),
        (0.044, -0.076, 0.0),
    )
    np.testing.assert_array_equal(
        project_source_position(np.asarray((0.0, 0.0, -0.010))),
        (0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="more than 10 mm"):
        project_source_position(np.asarray((0.0, 0.0, -0.010001)))


def test_frozen_fdn_matrix_and_delays() -> None:
    """The renderer uses the approved orthonormal matrix and prime delays."""

    matrix = normalized_sylvester_8()
    canonical_signs = np.asarray(
        (
            (1, 1, 1, 1, 1, 1, 1, 1),
            (1, -1, 1, -1, 1, -1, 1, -1),
            (1, 1, -1, -1, 1, 1, -1, -1),
            (1, -1, -1, 1, 1, -1, -1, 1),
            (1, 1, 1, 1, -1, -1, -1, -1),
            (1, -1, 1, -1, -1, 1, -1, 1),
            (1, 1, -1, -1, -1, -1, 1, 1),
            (1, -1, -1, 1, -1, 1, 1, -1),
        ),
        dtype=np.float64,
    )
    assert matrix.shape == (8, 8)
    assert matrix.dtype == np.float64
    assert not matrix.flags.writeable
    np.testing.assert_array_equal(matrix, canonical_signs / np.sqrt(8.0))
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(8), atol=1e-15)
    np.testing.assert_array_equal(
        FDN_DELAYS[48_000],
        (1_499, 1_601, 1_747, 1_861, 1_999, 2_131, 2_269, 2_381),
    )
    np.testing.assert_array_equal(
        FDN_DELAYS[24_000],
        (743, 797, 877, 929, 997, 1_063, 1_129, 1_193),
    )
    assert sabine_rt60_s() == pytest.approx(0.503125)


@pytest.mark.parametrize("sample_rate_hz", (24_000, 48_000))
def test_fdn_impulse_has_predelay_and_stereo_decay(sample_rate_hz: int) -> None:
    """A dry impulse creates delayed, finite, decorrelated stereo reverberation."""

    fdn = FeedbackDelayNetwork(sample_rate_hz)
    sample_count = round(0.16 * sample_rate_hz)
    impulse = np.zeros(sample_count, dtype=np.float64)
    impulse[0] = 1.0
    output = fdn.process(impulse)
    first_possible = fdn.predelay_samples + int(np.min(fdn.delays))
    assert np.count_nonzero(output[:, :first_possible]) == 0
    assert np.count_nonzero(output[:, first_possible:]) > 0
    assert np.all(np.isfinite(output))
    assert not np.array_equal(output[0], output[1])
    assert np.max(np.abs(output)) < 1.0


def test_fdn_reset_restores_silence() -> None:
    """Episode reset removes all late-tail state exactly."""

    fdn = FeedbackDelayNetwork(24_000)
    impulse = np.zeros(2_000)
    impulse[0] = 1.0
    fdn.process(impulse)
    fdn.reset()
    assert np.count_nonzero(fdn.process(np.zeros(2_000))) == 0

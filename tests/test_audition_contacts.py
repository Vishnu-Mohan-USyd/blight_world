"""Tests for energy- and material-dependent physical contact sound."""

from __future__ import annotations

import numpy as np
import pytest

from mimo_audition.clips import ClipLibrary
from mimo_audition.contacts import impact_amplitude


def test_impact_energy_controls_bounded_pressure_gain() -> None:
    """Contact intensity follows the documented square-root energy mapping."""

    assert impact_amplitude(0.0) == 0.25
    assert impact_amplitude(0.0025) == pytest.approx(0.5)
    assert impact_amplitude(0.0100) == pytest.approx(1.0)
    assert impact_amplitude(0.0400) == pytest.approx(2.0)
    assert impact_amplitude(1.0) == 4.0
    with pytest.raises(ValueError, match="finite and non-negative"):
        impact_amplitude(float("nan"))
    with pytest.raises(ValueError, match="finite and non-negative"):
        impact_amplitude(-0.01)


def test_material_pair_variants_are_distinct_dry_excitations() -> None:
    """Seeded variant selection changes timbre without changing calibration."""

    library = ClipLibrary(48_000)
    variants = [library.get(f"impact_hard_wood_{index}") for index in range(3)]
    assert {clip.reference_peak_pressure_pa_at_1m for clip in variants} == {0.2}
    for first, second in ((0, 1), (0, 2), (1, 2)):
        overlap = min(variants[first].samples.size, variants[second].samples.size)
        assert not np.array_equal(
            variants[first].samples[:overlap],
            variants[second].samples[:overlap],
        )

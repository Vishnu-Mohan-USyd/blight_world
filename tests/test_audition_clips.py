"""Tests for calibrated and checksummed dry clip assets."""

from __future__ import annotations

import numpy as np
import pytest

from mimo_audition.clips import ClipLibrary


@pytest.mark.parametrize("sample_rate_hz", (24_000, 48_000))
def test_packaged_clip_bank_is_complete_and_loadable(sample_rate_hz: int) -> None:
    """Both rates contain all 33 impacts and four named effects."""

    library = ClipLibrary(sample_rate_hz)
    assert len(library.clip_ids) == 37
    for clip_id in ("impact_hard_wood_0", "impact_plastic_soft_2", "chime"):
        clip = library.get(clip_id)
        assert clip.sample_rate_hz == sample_rate_hz
        assert clip.samples.ndim == 1
        assert clip.samples.size > sample_rate_hz // 20
        assert np.all(np.isfinite(clip.samples))
        assert 0.9 < np.max(np.abs(clip.samples)) <= 1.0
        assert clip.reference_peak_pressure_pa_at_1m == 0.2


def test_programmatic_clip_registration_is_immutable() -> None:
    """Downstream environments can add a deterministic clip without mutation."""

    library = ClipLibrary(24_000)
    samples = np.linspace(-0.5, 0.5, 32)
    library.register("user", samples, reference_peak_pressure_pa_at_1m=0.1)
    assert samples.flags.writeable
    samples[:] = 0
    clip = library.get("user")
    assert clip.samples[0] == pytest.approx(-0.5)
    assert not np.shares_memory(clip.samples, samples)
    assert clip.samples.flags.owndata
    assert clip.samples.flags.c_contiguous
    assert not clip.samples.flags.writeable
    with pytest.raises(ValueError, match="already registered"):
        library.register("user", np.ones(8))

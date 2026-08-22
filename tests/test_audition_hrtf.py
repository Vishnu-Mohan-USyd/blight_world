"""Tests for paired HRTF loading and direction transitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mimo_audition.hrtf import (
    HrtfBank,
    PairedHrtfConvolver,
    load_hrtf_bank,
    load_packaged_hrtf,
)


def _analytic_bank() -> HrtfBank:
    directions = np.asarray(
        ((1, 0, 0), (0, 1, 0), (0, -1, 0)),
        dtype=np.float64,
    )
    impulses = np.zeros((3, 2, 8), dtype=np.float64)
    impulses[0, :, 1] = 1.0
    impulses[1, 0, 1] = 1.0
    impulses[1, 1, 3] = 0.5
    impulses[2, 0, 3] = 0.5
    impulses[2, 1, 1] = 1.0
    return HrtfBank(directions, impulses, 48_000, 1.0, "fixture", "test")


def test_nearest_direction_keeps_ears_paired() -> None:
    """One spherical lookup selects both channels from the same measurement."""

    bank = _analytic_bank()
    assert bank.nearest_index(np.asarray((1.0, 0.01, 0.0))) == 0
    assert bank.nearest_index(np.asarray((0.0, 1.0, 0.0))) == 1
    assert bank.nearest_index(np.asarray((0.0, -1.0, 0.0))) == 2


def test_loader_validates_metadata_and_derives_24k(tmp_path: Path) -> None:
    """The portable NPZ representation remains paired after decimation."""

    bank = _analytic_bank()
    path = tmp_path / "fixture.npz"
    np.savez(
        path,
        directions_xyz=bank.directions_xyz,
        hrir=bank.hrir,
        sample_rate_hz=np.asarray(48_000),
        common_delay_samples=np.asarray(1.0),
        subject=np.asarray("fixture"),
        license_name=np.asarray("test"),
    )
    loaded = load_hrtf_bank(path, 24_000)
    assert loaded.hrir.shape == (3, 2, 4)
    assert loaded.sample_rate_hz == 24_000
    assert loaded.common_delay_samples == pytest.approx(0.5)


def test_hrtf_direction_change_crossfades_over_one_substep() -> None:
    """Direction changes blend for the frozen 5 ms transition interval."""

    bank = _analytic_bank()
    convolver = PairedHrtfConvolver(bank, crossfade_samples=4)
    impulse = np.zeros(8, dtype=np.float64)
    impulse[0] = 1.0
    front = convolver.process(impulse, 0)
    assert front.shape == (2, 8)
    np.testing.assert_array_equal(front[0], front[1])

    constant = np.ones(8, dtype=np.float64)
    transitioned = convolver.process(constant, 1)
    assert np.all(np.isfinite(transitioned))
    assert not np.array_equal(transitioned[0], transitioned[1])


def test_loader_fails_closed_without_required_arrays(tmp_path: Path) -> None:
    """A malformed HRTF archive cannot silently become a fallback bank."""

    path = tmp_path / "bad.npz"
    np.savez(path, hrir=np.zeros((1, 2, 2)))
    with pytest.raises(ValueError, match="missing fields"):
        load_hrtf_bank(path)


def test_packaged_sadie_h10_bank_is_real_and_paired() -> None:
    """The runtime asset is the complete verified H10 measurement bank."""

    bank = load_packaged_hrtf(48_000)
    assert bank.subject == "SADIE II H10"
    assert bank.license_name == "Apache-2.0"
    assert bank.direction_count == 2_114
    assert bank.hrir.shape == (2_114, 2, 256)
    assert bank.common_delay_samples == 113.0
    assert not np.array_equal(bank.hrir[:, 0], bank.hrir[:, 1])

"""Numerical tests for audition DSP primitives."""

from __future__ import annotations

import numpy as np
import pytest

from mimo_audition.dsp import (
    AttackReleaseSmoother,
    FractionalDelayLine,
    OnePoleLowpass,
    downsample_48k_to_24k,
    fractional_delay_reference,
    fractional_delay_table,
)


def test_fractional_delay_table_contract() -> None:
    """The production table has the frozen dimensions and unity DC gain."""

    table = fractional_delay_table()
    assert table.shape == (1_024, 32)
    assert not table.flags.writeable
    np.testing.assert_allclose(np.sum(table, axis=1), 1.0, atol=1e-12)


def test_streaming_fractional_delay_matches_integer_impulse() -> None:
    """A sufficiently long integer delay preserves impulse amplitude/time."""

    delay = FractionalDelayLine()
    signal = np.zeros(128, dtype=np.float64)
    signal[0] = 1.0
    output = delay.process(signal, 40.0)
    assert int(np.argmax(np.abs(output))) == 40
    assert output[40] == pytest.approx(1.0, abs=1e-12)


def test_production_fractional_delay_tracks_reference() -> None:
    """The tabulated path agrees with the 127-tap reference in steady state."""

    samples = np.arange(2_048, dtype=np.float64)
    signal = 0.4 * np.sin(2 * np.pi * 1_100 * samples / 48_000)
    requested_delay = 63.375
    production = FractionalDelayLine().process(signal, requested_delay)
    reference = fractional_delay_reference(signal, requested_delay)
    np.testing.assert_allclose(production[200:-200], reference[200:-200], atol=2e-4)


def test_prescribed_downsampling_rejects_above_band() -> None:
    """The deterministic 24 kHz derivation retains 2 kHz and rejects 15 kHz."""

    samples = np.arange(4_800, dtype=np.float64)
    low = np.sin(2 * np.pi * 2_000 * samples / 48_000)
    high = np.sin(2 * np.pi * 15_000 * samples / 48_000)
    low_out = downsample_48k_to_24k(low)
    high_out = downsample_48k_to_24k(high)
    assert low_out.shape == (2_400,)
    assert np.sqrt(np.mean(low_out[100:] ** 2)) > 0.65
    assert np.sqrt(np.mean(high_out[100:] ** 2)) < 0.01


def test_one_pole_filter_is_stateful_and_low_pass() -> None:
    """Adjacent blocks match one continuous low-pass evaluation."""

    impulse = np.zeros(64, dtype=np.float64)
    impulse[0] = 1.0
    split_filter = OnePoleLowpass(1_500, 48_000)
    split = np.concatenate(
        (split_filter.process(impulse[:17]), split_filter.process(impulse[17:]))
    )
    whole = OnePoleLowpass(1_500, 48_000).process(impulse)
    np.testing.assert_array_equal(split, whole)
    assert np.all(whole >= 0)
    assert whole[-1] < whole[0]


def test_occlusion_smoother_uses_slower_release() -> None:
    """Occlusion attacks in 20 ms and releases more slowly over 50 ms."""

    smoother = AttackReleaseSmoother(0.020, 0.050, 48_000)
    attack = smoother.process(True, 480)
    release = smoother.process(False, 480)
    assert 0 < attack[0] < attack[-1] < 1
    assert 0 < release[-1] < release[0]
    assert attack[-1] == pytest.approx(1 - np.exp(-0.010 / 0.020))
    assert release[-1] == pytest.approx(attack[-1] * np.exp(-0.010 / 0.050))
    assert release[-1] / attack[-1] > 1 - attack[-1]

"""Deterministic reference DSP used by the binaural renderer."""

from __future__ import annotations

from functools import lru_cache
from math import exp, floor, pi

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin


FloatArray = NDArray[np.float64]


@lru_cache(maxsize=4)
def fractional_delay_table(
    taps: int = 32,
    phases: int = 1_024,
    beta: float = 8.6,
) -> FloatArray:
    """Return normalized Kaiser-windowed sinc interpolation kernels.

    Each row interpolates a sample at one fractional phase using the same
    integer-centered neighborhood.  The returned table is read-only so it is
    safe to share among renderers.
    """

    if taps < 2 or taps % 2:
        raise ValueError("taps must be a positive even integer")
    if phases < 2:
        raise ValueError("phases must be at least two")
    offsets = np.arange(taps, dtype=np.float64) - (taps // 2 - 1)
    window = np.kaiser(taps, beta)
    table = np.empty((phases, taps), dtype=np.float64)
    for phase_index in range(phases):
        fraction = phase_index / phases
        kernel = np.sinc(offsets - fraction) * window
        table[phase_index] = kernel / np.sum(kernel)
    table.setflags(write=False)
    return table


def _interpolate_at(
    samples: FloatArray,
    sample_index: float,
    table: FloatArray,
) -> float:
    """Interpolate one zero-extended signal sample at a fractional index."""

    integer_index = floor(sample_index)
    fraction = sample_index - integer_index
    phase_index = min(int(fraction * table.shape[0]), table.shape[0] - 1)
    first_offset = -(table.shape[1] // 2 - 1)
    first_index = integer_index + first_offset
    start = max(first_index, 0)
    stop = min(first_index + table.shape[1], samples.size)
    if start >= stop:
        return 0.0
    kernel_start = start - first_index
    kernel_stop = kernel_start + stop - start
    return float(np.dot(samples[start:stop], table[phase_index, kernel_start:kernel_stop]))


def fractional_delay_reference(
    signal: NDArray[np.floating],
    delay_samples: float,
    *,
    taps: int = 127,
    beta: float = 8.6,
) -> FloatArray:
    """Apply a high-quality constant fractional delay to a finite signal.

    The output has the same length as ``signal`` and is zero-extended outside
    that interval.  This deliberately slow implementation is the numerical
    reference used to validate the 32-tap streaming path.
    """

    if not np.isfinite(delay_samples) or delay_samples < 0:
        raise ValueError("delay_samples must be finite and non-negative")
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    offsets = np.arange(taps, dtype=np.float64) - taps // 2
    window = np.kaiser(taps, beta)
    output = np.empty_like(values)
    for output_index in range(values.size):
        source_index = output_index - delay_samples
        integer_index = floor(source_index)
        fraction = source_index - integer_index
        kernel = np.sinc(offsets - fraction) * window
        kernel /= np.sum(kernel)
        sample_indices = integer_index + offsets.astype(np.int64)
        valid = (sample_indices >= 0) & (sample_indices < values.size)
        output[output_index] = np.dot(
            values[sample_indices[valid]],
            kernel[valid],
        )
    return output


class FractionalDelayLine:
    """Streaming 32-tap, 1,024-phase fractional delay line.

    The symmetric interpolator is exact for delays of at least 16 samples.
    Shorter requested delays are rendered at 16 samples, the minimum causal
    look-behind required by the production kernel.  Acoustic paths in the
    supplied room exceed that bound at both supported rates.
    """

    def __init__(self, max_delay_samples: int = 8_192) -> None:
        if max_delay_samples < 32:
            raise ValueError("max_delay_samples must be at least 32")
        self._table = fractional_delay_table()
        self._minimum_delay = self._table.shape[1] // 2
        self._max_delay = max_delay_samples
        self._history = np.zeros(0, dtype=np.float64)
        self._sample_cursor = 0

    @property
    def minimum_delay_samples(self) -> int:
        """Minimum delay supported without future input samples."""

        return self._minimum_delay

    def reset(self) -> None:
        """Clear all signal history and return the clock to zero."""

        self._history = np.zeros(0, dtype=np.float64)
        self._sample_cursor = 0

    def process(
        self,
        signal: NDArray[np.floating],
        delay_samples: float,
        end_delay_samples: float | None = None,
    ) -> FloatArray:
        """Delay a mono block, optionally interpolating delay across it."""

        values = np.asarray(signal, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        final_delay = delay_samples if end_delay_samples is None else end_delay_samples
        for value in (delay_samples, final_delay):
            if not np.isfinite(value) or value < 0 or value > self._max_delay:
                raise ValueError("delay is outside the configured delay line")
        combined = np.concatenate((self._history, values))
        combined_start = self._sample_cursor - self._history.size
        output = np.empty(values.size, dtype=np.float64)
        delays = np.linspace(delay_samples, final_delay, values.size, endpoint=True)
        for local_index, requested_delay in enumerate(delays):
            delay = max(float(requested_delay), float(self._minimum_delay))
            output_global_index = self._sample_cursor + local_index
            source_global_index = output_global_index - delay
            output[local_index] = _interpolate_at(
                combined,
                source_global_index - combined_start,
                self._table,
            )
        self._sample_cursor += values.size
        retain = min(combined.size, self._max_delay + self._table.shape[1] + 2)
        self._history = combined[-retain:].copy()
        return output


def downsample_48k_to_24k(signal: NDArray[np.floating]) -> FloatArray:
    """Low-pass at 10.8 kHz with a 127-tap Kaiser FIR, then decimate by two."""

    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    kernel = firwin(
        127,
        cutoff=10_800.0,
        window=("kaiser", 8.6),
        fs=48_000.0,
    )
    group_delay = (kernel.size - 1) // 2
    filtered = np.convolve(values, kernel, mode="full")
    compensated = filtered[group_delay : group_delay + values.size]
    return np.ascontiguousarray(compensated[::2], dtype=np.float64)


class OnePoleLowpass:
    """Stateful first-order low-pass filter for one mono stream."""

    def __init__(self, cutoff_hz: float, sample_rate_hz: int) -> None:
        if not 0 < cutoff_hz < sample_rate_hz / 2:
            raise ValueError("cutoff_hz must lie between DC and Nyquist")
        self._coefficient = exp(-2 * pi * cutoff_hz / sample_rate_hz)
        self._state = 0.0

    def reset(self) -> None:
        """Clear filter memory."""

        self._state = 0.0

    def process(self, signal: NDArray[np.floating]) -> FloatArray:
        """Filter one block while preserving state across calls."""

        values = np.asarray(signal, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        output = np.empty_like(values)
        state = self._state
        feed = 1.0 - self._coefficient
        for index, value in enumerate(values):
            state = feed * float(value) + self._coefficient * state
            output[index] = state
        self._state = state
        return output


class AttackReleaseSmoother:
    """Exponential attack/release smoothing for binary occlusion targets."""

    def __init__(self, attack_s: float, release_s: float, sample_rate_hz: int) -> None:
        if attack_s <= 0 or release_s <= 0:
            raise ValueError("attack and release times must be positive")
        self._attack = exp(-1.0 / (attack_s * sample_rate_hz))
        self._release = exp(-1.0 / (release_s * sample_rate_hz))
        self._state = 0.0

    def reset(self) -> None:
        """Return to the unoccluded state."""

        self._state = 0.0

    def process(self, occluded: bool, sample_count: int) -> FloatArray:
        """Return a per-sample occlusion mix between zero and one."""

        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        target = 1.0 if occluded else 0.0
        coefficient = self._attack if target > self._state else self._release
        output = np.empty(sample_count, dtype=np.float64)
        state = self._state
        for index in range(sample_count):
            state = target + coefficient * (state - target)
            output[index] = state
        self._state = state
        return output

"""Paired-direction HRTF loading, selection, and streaming convolution."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mimo_audition.dsp import downsample_48k_to_24k


FloatArray = NDArray[np.float64]
PACKAGED_HRTF = "assets/hrtf/sadie_h10_v2_2_48k.npz"


@dataclass(frozen=True, slots=True)
class HrtfBank:
    """A validated bank of paired left/right impulse responses."""

    directions_xyz: FloatArray
    hrir: FloatArray
    sample_rate_hz: int
    common_delay_samples: float
    subject: str
    license_name: str

    def __post_init__(self) -> None:
        directions = np.ascontiguousarray(self.directions_xyz, dtype=np.float64)
        impulses = np.ascontiguousarray(self.hrir, dtype=np.float64)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions_xyz must have shape (direction, 3)")
        if impulses.ndim != 3 or impulses.shape[:2] != (directions.shape[0], 2):
            raise ValueError("hrir must have shape (direction, 2, tap)")
        if impulses.shape[2] < 2:
            raise ValueError("each HRIR must contain at least two taps")
        if self.sample_rate_hz not in (24_000, 48_000):
            raise ValueError("HRTF sample rate must be 24000 or 48000")
        if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(impulses)):
            raise ValueError("HRTF arrays must be finite")
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 0):
            raise ValueError("HRTF directions must be non-zero")
        directions /= norms[:, None]
        directions.setflags(write=False)
        impulses.setflags(write=False)
        object.__setattr__(self, "directions_xyz", directions)
        object.__setattr__(self, "hrir", impulses)

    @property
    def direction_count(self) -> int:
        """Number of paired directions in the bank."""

        return self.directions_xyz.shape[0]

    @property
    def tap_count(self) -> int:
        """Number of FIR taps per ear and direction."""

        return self.hrir.shape[2]

    def nearest_index(self, direction_xyz: NDArray[np.floating]) -> int:
        """Return the nearest paired direction using spherical dot product."""

        direction = np.asarray(direction_xyz, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("direction_xyz must be a finite three-vector")
        norm = np.linalg.norm(direction)
        if norm <= 0:
            raise ValueError("direction_xyz must be non-zero")
        return int(np.argmax(self.directions_xyz @ (direction / norm)))

    def at_sample_rate(self, sample_rate_hz: int) -> HrtfBank:
        """Return this bank at a supported rate using the prescribed decimator."""

        if sample_rate_hz == self.sample_rate_hz:
            return self
        if self.sample_rate_hz != 48_000 or sample_rate_hz != 24_000:
            raise ValueError("only deterministic 48-to-24 kHz conversion is supported")
        impulses = np.empty(
            (self.direction_count, 2, (self.tap_count + 1) // 2),
            dtype=np.float64,
        )
        for direction_index in range(self.direction_count):
            for channel_index in range(2):
                impulses[direction_index, channel_index] = downsample_48k_to_24k(
                    self.hrir[direction_index, channel_index]
                )
        return HrtfBank(
            directions_xyz=self.directions_xyz.copy(),
            hrir=impulses,
            sample_rate_hz=24_000,
            common_delay_samples=self.common_delay_samples / 2,
            subject=self.subject,
            license_name=self.license_name,
        )


def _text_field(archive: np.lib.npyio.NpzFile, name: str) -> str:
    value = archive[name]
    if value.shape != ():
        raise ValueError(f"{name} must be scalar metadata")
    return str(value.item())


def load_hrtf_bank(path: str | Path, sample_rate_hz: int = 48_000) -> HrtfBank:
    """Load a fail-closed HRTF NPZ and optionally derive the 24 kHz bank."""

    hrtf_path = Path(path)
    if not hrtf_path.is_file():
        raise FileNotFoundError(f"HRTF bank not found: {hrtf_path}")
    with np.load(hrtf_path, allow_pickle=False) as archive:
        required = {
            "directions_xyz",
            "hrir",
            "sample_rate_hz",
            "common_delay_samples",
            "subject",
            "license_name",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"HRTF bank is missing fields: {sorted(missing)}")
        bank = HrtfBank(
            directions_xyz=archive["directions_xyz"],
            hrir=archive["hrir"],
            sample_rate_hz=int(archive["sample_rate_hz"].item()),
            common_delay_samples=float(archive["common_delay_samples"].item()),
            subject=_text_field(archive, "subject"),
            license_name=_text_field(archive, "license_name"),
        )
    return bank.at_sample_rate(sample_rate_hz)


def load_packaged_hrtf(
    sample_rate_hz: int,
    external_path: str | None = None,
) -> HrtfBank:
    """Load the configured bank, refusing synthetic or silent fallbacks."""

    if external_path is not None:
        return load_hrtf_bank(external_path, sample_rate_hz)
    resource = resources.files("mimo_audition").joinpath(*PACKAGED_HRTF.split("/"))
    if not resource.is_file():
        raise FileNotFoundError(
            "packaged SADIE H10 HRTF is unavailable; audition cannot be enabled"
        )
    with resources.as_file(resource) as path:
        return load_hrtf_bank(path, sample_rate_hz)


class PairedHrtfConvolver:
    """Streaming paired-FIR convolver with a persistent direction crossfade."""

    def __init__(self, bank: HrtfBank, crossfade_samples: int) -> None:
        if crossfade_samples <= 0:
            raise ValueError("crossfade_samples must be positive")
        self._bank = bank
        self._crossfade_samples = crossfade_samples
        self._history = np.zeros(bank.tap_count - 1, dtype=np.float64)
        self._current_index: int | None = None
        self._target_index: int | None = None
        self._fade_position = crossfade_samples

    def reset(self) -> None:
        """Clear FIR history and direction-transition state."""

        self._history.fill(0)
        self._current_index = None
        self._target_index = None
        self._fade_position = self._crossfade_samples

    def _filter(self, combined: FloatArray, direction_index: int) -> FloatArray:
        start = self._history.size
        sample_count = combined.size - start
        output = np.empty((2, sample_count), dtype=np.float64)
        for channel_index in range(2):
            full = np.convolve(
                combined,
                self._bank.hrir[direction_index, channel_index],
                mode="full",
            )
            output[channel_index] = full[start : start + sample_count]
        return output

    def process(self, mono: NDArray[np.floating], direction_index: int) -> FloatArray:
        """Spatialize one mono block using a paired HRTF direction."""

        values = np.asarray(mono, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("mono input must be one-dimensional")
        if not 0 <= direction_index < self._bank.direction_count:
            raise IndexError("direction index is outside the HRTF bank")
        combined = np.concatenate((self._history, values))
        if self._current_index is None:
            self._current_index = direction_index
            self._target_index = direction_index
            output = self._filter(combined, direction_index)
        elif direction_index == self._current_index:
            output = self._filter(combined, direction_index)
        else:
            if direction_index != self._target_index:
                self._target_index = direction_index
                self._fade_position = 0
            old_output = self._filter(combined, self._current_index)
            new_output = self._filter(combined, self._target_index)
            fade_indices = self._fade_position + np.arange(values.size)
            mix = np.minimum(fade_indices / self._crossfade_samples, 1.0)
            output = old_output * (1.0 - mix[None, :]) + new_output * mix[None, :]
            self._fade_position += values.size
            if self._fade_position >= self._crossfade_samples:
                self._current_index = self._target_index
        if self._history.size:
            self._history = combined[-self._history.size :].copy()
        return output

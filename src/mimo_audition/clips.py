"""Checksummed packaged and user-provided mono clip loading."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import json

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile


FloatArray = NDArray[np.float64]
MANIFEST_RESOURCE = "assets/audio/manifest.json"


@dataclass(frozen=True, slots=True)
class AudioClip:
    """A calibrated, immutable mono pressure-shape waveform."""

    clip_id: str
    samples: FloatArray
    sample_rate_hz: int
    reference_peak_pressure_pa_at_1m: float

    def __post_init__(self) -> None:
        values = np.array(self.samples, dtype=np.float64, order="C", copy=True)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("clip samples must be a non-empty mono array")
        if not np.all(np.isfinite(values)):
            raise ValueError("clip samples must be finite")
        if self.sample_rate_hz not in (24_000, 48_000):
            raise ValueError("clip sample rate must be 24000 or 48000")
        if self.reference_peak_pressure_pa_at_1m <= 0:
            raise ValueError("reference pressure must be positive")
        values.setflags(write=False)
        object.__setattr__(self, "samples", values)


class ClipLibrary:
    """Lazy checksummed clip bank with a programmatic registration seam."""

    def __init__(self, sample_rate_hz: int) -> None:
        if sample_rate_hz not in (24_000, 48_000):
            raise ValueError("sample_rate_hz must be 24000 or 48000")
        self.sample_rate_hz = sample_rate_hz
        manifest_resource = resources.files("mimo_audition").joinpath(
            *MANIFEST_RESOURCE.split("/")
        )
        if not manifest_resource.is_file():
            raise FileNotFoundError("packaged audition clip manifest is unavailable")
        manifest = json.loads(manifest_resource.read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or not isinstance(manifest.get("clips"), dict):
            raise ValueError("unsupported audition clip manifest")
        self._manifest: dict[str, dict[str, object]] = manifest["clips"]
        self._clips: dict[str, AudioClip] = {}

    @property
    def clip_ids(self) -> tuple[str, ...]:
        """All packaged and programmatically registered clip identifiers."""

        return tuple(sorted(set(self._manifest).union(self._clips)))

    def register(
        self,
        clip_id: str,
        samples: NDArray[np.floating],
        *,
        reference_peak_pressure_pa_at_1m: float = 0.2,
    ) -> None:
        """Register an in-memory clip already sampled at this library's rate."""

        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError("clip_id must be a non-empty string")
        if clip_id in self._manifest or clip_id in self._clips:
            raise ValueError(f"clip_id is already registered: {clip_id}")
        self._clips[clip_id] = AudioClip(
            clip_id,
            np.asarray(samples, dtype=np.float64),
            self.sample_rate_hz,
            reference_peak_pressure_pa_at_1m,
        )

    def get(self, clip_id: str) -> AudioClip:
        """Return a clip, verifying packaged bytes before first use."""

        if clip_id in self._clips:
            return self._clips[clip_id]
        try:
            metadata = self._manifest[clip_id]
            rate_metadata = metadata["files"][str(self.sample_rate_hz)]  # type: ignore[index]
            relative_path = str(rate_metadata["path"])  # type: ignore[index]
            expected_sha256 = str(rate_metadata["sha256"])  # type: ignore[index]
            reference_pressure = float(metadata["reference_peak_pressure_pa_at_1m"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid manifest entry for clip {clip_id}") from error
        resource = resources.files("mimo_audition").joinpath(
            "assets", "audio", *relative_path.split("/")
        )
        if not resource.is_file():
            raise FileNotFoundError(f"packaged audio clip is unavailable: {clip_id}")
        contents = resource.read_bytes()
        actual_sha256 = hashlib.sha256(contents).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"packaged audio clip checksum mismatch: {clip_id}")
        with resources.as_file(resource) as path:
            sample_rate_hz, pcm = wavfile.read(path)
        if sample_rate_hz != self.sample_rate_hz or pcm.ndim != 1 or pcm.dtype != np.int16:
            raise ValueError(f"unexpected WAV contract for clip {clip_id}")
        samples = np.asarray(pcm, dtype=np.float64) / 32_767.0
        clip = AudioClip(clip_id, samples, sample_rate_hz, reference_pressure)
        self._clips[clip_id] = clip
        return clip

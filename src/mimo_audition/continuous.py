"""Immutable substep emissions for persistent continuous sound streams."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from mimo_audition.sources import (
    Attachment,
    BodyAttachment,
    SiteAttachment,
    StaticAttachment,
)


Float32Array = NDArray[np.float32]
_VALID_SUBSTEP_SAMPLE_COUNTS = frozenset((120, 240))


def _identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ContinuousEmission:
    """One calibrated mono pressure block for a stable persistent stream.

    Samples are pressure in pascals at one metre for exactly one 5 ms physics
    substep. Caller-owned data is copied into read-only C-contiguous float32
    storage so later mutation cannot alter an admitted render block.
    """

    stream_id: str
    kind: str
    attachment: Attachment
    samples_pa_at_1m: Float32Array
    priority: float = 0.0

    def __post_init__(self) -> None:
        _identifier("stream_id", self.stream_id)
        _identifier("kind", self.kind)
        if not isinstance(
            self.attachment,
            (StaticAttachment, BodyAttachment, SiteAttachment),
        ):
            raise TypeError("attachment must be a supported attachment specification")
        values = np.array(
            self.samples_pa_at_1m,
            dtype=np.float32,
            order="C",
            copy=True,
        )
        if values.ndim != 1 or values.size not in _VALID_SUBSTEP_SAMPLE_COUNTS:
            raise ValueError(
                "samples_pa_at_1m must be a 120- or 240-sample mono block"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("samples_pa_at_1m must be finite")
        if not isfinite(self.priority):
            raise ValueError("priority must be finite")
        immutable_values = np.frombuffer(
            values.tobytes(order="C"),
            dtype=np.float32,
        )
        object.__setattr__(self, "samples_pa_at_1m", immutable_values)
        object.__setattr__(self, "priority", float(self.priority))

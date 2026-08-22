"""Typed sound-source descriptions independent of any one MIMo environment."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias


Position3: TypeAlias = tuple[float, float, float]


def _position3(name: str, value: Position3) -> Position3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be a three-element tuple")
    result = tuple(float(component) for component in value)
    if not all(isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result  # type: ignore[return-value]


def _identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class StaticAttachment:
    """A source fixed at a world-frame position in metres."""

    world_position_m: Position3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "world_position_m",
            _position3("world_position_m", self.world_position_m),
        )


@dataclass(frozen=True, slots=True)
class BodyAttachment:
    """A source fixed to a named body's local frame."""

    body_name: str
    local_position_m: Position3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _identifier("body_name", self.body_name)
        object.__setattr__(
            self,
            "local_position_m",
            _position3("local_position_m", self.local_position_m),
        )


@dataclass(frozen=True, slots=True)
class SiteAttachment:
    """A source located at a named MuJoCo site's world position."""

    site_name: str

    def __post_init__(self) -> None:
        _identifier("site_name", self.site_name)


Attachment: TypeAlias = StaticAttachment | BodyAttachment | SiteAttachment


@dataclass(frozen=True, slots=True)
class AudioSourceSpec:
    """Serializable definition of a playable positioned mono source."""

    source_id: str
    clip_id: str
    attachment: Attachment
    loop: bool = False
    gain_db: float = 0.0

    def __post_init__(self) -> None:
        _identifier("source_id", self.source_id)
        _identifier("clip_id", self.clip_id)
        if not isinstance(
            self.attachment,
            (StaticAttachment, BodyAttachment, SiteAttachment),
        ):
            raise TypeError("attachment must be a supported attachment specification")
        if not isfinite(self.gain_db):
            raise ValueError("gain_db must be finite")

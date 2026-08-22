"""Minimal binocular MIMo v2 world."""

from blight_world.environment import BlightWorldEnv
from mimo_audition import (
    AuditionConfig,
    AuditionSpec,
    AudioSourceSpec,
    BodyAttachment,
    SiteAttachment,
    StaticAttachment,
)

__all__ = [
    "AuditionConfig",
    "AuditionSpec",
    "AudioSourceSpec",
    "BlightWorldEnv",
    "BodyAttachment",
    "SiteAttachment",
    "StaticAttachment",
]

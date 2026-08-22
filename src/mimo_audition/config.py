"""Configuration and fixed-rate contracts for MIMo audition."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose


PHYSICS_TIMESTEP_S = 0.005
FRAME_SKIP = 2
SPEED_OF_SOUND_M_S = 343.0


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class AuditionSpec:
    """Derived immutable audio timing contract.

    Samples are blocked by environment step and stored channels-first.  The
    physics clock is deliberately fixed to MIMo v2's 5 ms substep and two
    substeps per action so enabling sound cannot change motor dynamics.
    """

    sample_rate_hz: int
    physics_timestep_s: float = PHYSICS_TIMESTEP_S
    frame_skip: int = FRAME_SKIP
    channels: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate_hz not in (24_000, 48_000):
            raise ValueError("sample_rate_hz must be 24000 or 48000")
        if not isclose(self.physics_timestep_s, PHYSICS_TIMESTEP_S, abs_tol=1e-12):
            raise ValueError("physics_timestep_s must remain 0.005")
        if self.frame_skip != FRAME_SKIP:
            raise ValueError("frame_skip must remain 2")
        if self.channels != 2:
            raise ValueError("audition is strictly binaural")
        samples = self.sample_rate_hz * self.physics_timestep_s
        if not float(samples).is_integer():
            raise ValueError("sample rate must produce an integer substep size")

    @property
    def samples_per_substep(self) -> int:
        """Number of samples rendered for one 5 ms MuJoCo substep."""

        return int(self.sample_rate_hz * self.physics_timestep_s)

    @property
    def samples_per_step(self) -> int:
        """Number of samples exposed for one 10 ms environment step."""

        return self.samples_per_substep * self.frame_skip

    @property
    def observation_shape(self) -> tuple[int, int]:
        """Channels-first waveform shape exposed through Gymnasium."""

        return self.channels, self.samples_per_step


@dataclass(frozen=True, slots=True)
class AuditionConfig:
    """Serializable configuration for the optional auditory sensor.

    The bundled H10 HRTF is an adult surrogate rather than an infant-specific
    anatomical model.  Pressure samples are in pascals and are never clipped
    or automatically gain-controlled.
    """

    sample_rate_hz: int = 48_000
    hrtf_path: str | None = None
    max_registered_sources: int = 64
    max_scheduled_events: int = 256
    max_active_voices: int = 8
    speed_of_sound_m_s: float = SPEED_OF_SOUND_M_S
    distance_floor_m: float = 0.05
    occlusion_attenuation_db: float = -9.0
    occlusion_cutoff_hz: float = 1_500.0
    occlusion_attack_s: float = 0.020
    occlusion_release_s: float = 0.050
    hrtf_crossfade_s: float = 0.005
    enable_occlusion: bool = True
    enable_first_order_reflections: bool = True
    enable_late_reverb: bool = True
    late_reverb_send: float = 0.12

    def __post_init__(self) -> None:
        AuditionSpec(self.sample_rate_hz)
        for name in (
            "max_registered_sources",
            "max_scheduled_events",
            "max_active_voices",
        ):
            _positive_int(name, getattr(self, name))
        if self.max_active_voices > self.max_registered_sources:
            raise ValueError("max_active_voices cannot exceed registered sources")
        for name in (
            "speed_of_sound_m_s",
            "distance_floor_m",
            "occlusion_cutoff_hz",
            "occlusion_attack_s",
            "occlusion_release_s",
            "hrtf_crossfade_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.occlusion_attenuation_db > 0:
            raise ValueError("occlusion_attenuation_db must be non-positive")
        if self.late_reverb_send < 0:
            raise ValueError("late_reverb_send must be non-negative")

    @classmethod
    def quality_48k(cls, **overrides: object) -> AuditionConfig:
        """Return the full-band 48 kHz profile."""

        return cls(sample_rate_hz=48_000, **overrides)

    @classmethod
    def training_24k(cls, **overrides: object) -> AuditionConfig:
        """Return the lower-cost 24 kHz training profile."""

        return cls(sample_rate_hz=24_000, **overrides)

    @property
    def spec(self) -> AuditionSpec:
        """Derived fixed-rate observation specification."""

        return AuditionSpec(self.sample_rate_hz)

"""Configuration and fixed-rate contracts for MIMo audition."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite


PHYSICS_TIMESTEP_S = 0.005
FRAME_SKIP = 2
SPEED_OF_SOUND_M_S = 343.0


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ScrapeConfig:
    """Validated scrape-generator thresholds and pressure bounds.

    These are practical engineering defaults, not measured biological
    calibration.
    """

    enabled: bool = True
    minimum_contact_substeps: int = 2
    normal_force_on_n: float = 0.10
    normal_force_off_n: float = 0.05
    slip_speed_on_m_s: float = 0.02
    slip_speed_off_m_s: float = 0.01
    attack_seconds: float = 0.010
    release_seconds: float = 0.030
    power_smoothing_seconds: float = 0.010
    reference_friction_power_w: float = 0.10
    reference_rms_pressure_pa_at_1m: float = 0.005
    max_rms_pressure_pa_at_1m: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        _positive_int("minimum_contact_substeps", self.minimum_contact_substeps)
        positive_fields = (
            "normal_force_on_n",
            "normal_force_off_n",
            "slip_speed_on_m_s",
            "slip_speed_off_m_s",
            "attack_seconds",
            "release_seconds",
            "power_smoothing_seconds",
            "reference_friction_power_w",
            "reference_rms_pressure_pa_at_1m",
            "max_rms_pressure_pa_at_1m",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.normal_force_off_n > self.normal_force_on_n:
            raise ValueError("normal_force_off_n cannot exceed normal_force_on_n")
        if self.slip_speed_off_m_s > self.slip_speed_on_m_s:
            raise ValueError("slip_speed_off_m_s cannot exceed slip_speed_on_m_s")
        if (
            self.max_rms_pressure_pa_at_1m
            < self.reference_rms_pressure_pa_at_1m
        ):
            raise ValueError(
                "max_rms_pressure_pa_at_1m cannot be below reference pressure"
            )


@dataclass(frozen=True, slots=True)
class ContinuousSoundConfig:
    """Bounded persistent-stream settings for continuous sound rendering."""

    max_continuous_voices: int = 4
    unexpected_disappearance_fade_seconds: float = 0.005
    scrape: ScrapeConfig = field(default_factory=ScrapeConfig)

    def __post_init__(self) -> None:
        _positive_int("max_continuous_voices", self.max_continuous_voices)
        if self.max_continuous_voices > 32:
            raise ValueError("max_continuous_voices cannot exceed 32")
        fade = self.unexpected_disappearance_fade_seconds
        if not isfinite(fade) or fade < 0:
            raise ValueError(
                "unexpected_disappearance_fade_seconds must be finite and non-negative"
            )
        if not isinstance(self.scrape, ScrapeConfig):
            raise TypeError("scrape must be a ScrapeConfig")


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

    Both HRTF choices are reference anatomy, not an infant-specific model.
    Native output uses the existing pressure convention; Steam Audio output
    is a source-calibrated acoustic-pressure proxy. Neither is clipped or
    automatically gain-controlled.
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
    continuous: ContinuousSoundConfig | None = None
    backend: str = "native"
    steam_audio_library_path: str | None = None
    steam_audio_ir_duration_s: float = 2.0
    steam_audio_num_rays: int = 4_096
    steam_audio_num_bounces: int = 16
    steam_audio_num_threads: int = 2

    def __post_init__(self) -> None:
        AuditionSpec(self.sample_rate_hz)
        if self.backend not in ("native", "steam_audio"):
            raise ValueError("backend must be 'native' or 'steam_audio'")
        if self.backend == "steam_audio" and self.sample_rate_hz != 48_000:
            raise ValueError("Steam Audio requires 48000 Hz and 240-sample blocks")
        if self.backend == "steam_audio" and self.hrtf_path is not None:
            raise ValueError("Steam Audio uses its built-in reference-head HRTF")
        if self.backend == "steam_audio":
            if self.speed_of_sound_m_s not in (SPEED_OF_SOUND_M_S, 340.0):
                raise ValueError("Steam Audio propagation uses the SDK's fixed 340 m/s")
            object.__setattr__(self, "speed_of_sound_m_s", 340.0)
        for name in (
            "steam_audio_num_rays",
            "steam_audio_num_bounces",
            "steam_audio_num_threads",
        ):
            _positive_int(name, getattr(self, name))
        if not isfinite(self.steam_audio_ir_duration_s) or self.steam_audio_ir_duration_s <= 0:
            raise ValueError("steam_audio_ir_duration_s must be finite and positive")
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
        if self.continuous is not None and not isinstance(
            self.continuous,
            ContinuousSoundConfig,
        ):
            raise TypeError("continuous must be a ContinuousSoundConfig or None")

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

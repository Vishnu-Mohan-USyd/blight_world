"""Persistent contact tracking and deterministic scrape sound synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from math import cos, exp, pi, sqrt
from typing import Literal

import mujoco
import numpy as np
from numpy.typing import NDArray

from mimo_audition.config import AuditionSpec, ScrapeConfig
from mimo_audition.continuous import ContinuousEmission
from mimo_audition.materials import (
    MATERIAL_ORDER,
    AcousticMaterial,
    AcousticMaterialResolver,
    canonical_material_pair,
)
from mimo_audition.sources import BodyAttachment


FloatArray = NDArray[np.float64]
GeomPair = tuple[int, int]
MaterialPair = tuple[AcousticMaterial, AcousticMaterial]
Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _ContactPoint:
    """One solved contact point expressed in world coordinates and SI units."""

    world_position_m: Vector3
    normal_load_n: float
    tangential_force_world_n: Vector3
    tangential_velocity_world_m_s: Vector3


@dataclass(frozen=True, slots=True)
class ScrapeContact:
    """One admitted persistent geom-pair contact control for synthesis."""

    stream_id: str
    geom_pair: GeomPair
    material_pair: MaterialPair
    attachment: BodyAttachment
    world_position_m: Vector3
    normal_load_n: float
    tangential_force_world_n: Vector3
    tangential_velocity_world_m_s: Vector3
    tangential_speed_m_s: float
    friction_power_w: float

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not self.stream_id.strip():
            raise ValueError("stream_id must be a non-empty string")
        pair = self.geom_pair
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pair)
            or pair[0] < 0
            or pair[0] >= pair[1]
        ):
            raise ValueError("geom_pair must be two distinct sorted non-negative IDs")
        if (
            not isinstance(self.material_pair, tuple)
            or len(self.material_pair) != 2
            or any(material not in MATERIAL_ORDER for material in self.material_pair)
            or canonical_material_pair(*self.material_pair) != self.material_pair
        ):
            raise ValueError("material_pair must use canonical acoustic order")
        if not isinstance(self.attachment, BodyAttachment):
            raise TypeError("scrape attachment must be body-local")
        vectors = (
            self.world_position_m,
            self.tangential_force_world_n,
            self.tangential_velocity_world_m_s,
        )
        if any(
            not isinstance(vector, tuple)
            or len(vector) != 3
            or not np.all(np.isfinite(vector))
            for vector in vectors
        ):
            raise ValueError("scrape vectors must be finite three-tuples")
        scalars = (
            self.normal_load_n,
            self.tangential_speed_m_s,
            self.friction_power_w,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("scrape controls must be finite")
        if self.normal_load_n <= 0:
            raise ValueError("normal_load_n must be positive")
        if self.tangential_speed_m_s < 0 or self.friction_power_w < 0:
            raise ValueError("scrape speed and power must be non-negative")
        velocity_norm = float(np.linalg.norm(self.tangential_velocity_world_m_s))
        if not np.isclose(velocity_norm, self.tangential_speed_m_s, atol=1e-12):
            raise ValueError("tangential speed must match the velocity vector")
        expected_power = (
            float(np.linalg.norm(self.tangential_force_world_n))
            * self.tangential_speed_m_s
        )
        if not np.isclose(expected_power, self.friction_power_w, atol=1e-12):
            raise ValueError("friction power must equal tangential force times speed")


@dataclass(frozen=True, slots=True)
class ScrapeDiagnostic:
    """Immutable privileged state for one generated scrape block."""

    stream_id: str
    geom_pair: GeomPair
    material_pair: MaterialPair
    phase: Literal["active", "release"]
    world_position_m: Vector3
    normal_load_n: float
    tangential_velocity_world_m_s: Vector3
    tangential_speed_m_s: float
    friction_power_w: float
    target_rms_pressure_pa_at_1m: float


@dataclass(slots=True)
class _GateState:
    consecutive_on_substeps: int = 0
    sounding: bool = False
    missing_substeps: int = 0
    latest: ScrapeContact | None = None


@dataclass(frozen=True, slots=True)
class _TextureProfile:
    lowpass_hz: float
    resonance_hz: float
    resonance_bandwidth_hz: float
    resonance_mix: float


@dataclass(slots=True)
class _ScrapeVoice:
    stream_id: str
    material_pair: MaterialPair
    attachment: BodyAttachment
    rng: np.random.Generator
    latest: ScrapeContact
    lowpass_state: float = 0.0
    resonance_y1: float = 0.0
    resonance_y2: float = 0.0
    power_ema_w: float = 0.0
    previous_input_power_w: float = 0.0
    envelope: float = 0.0
    release_start_envelope: float = 0.0
    release_total_samples: int = 0
    release_elapsed_samples: int = 0


_MATERIAL_TEXTURES: dict[AcousticMaterial, _TextureProfile] = {
    "hard": _TextureProfile(10_000.0, 6_500.0, 2_400.0, 0.45),
    "wood": _TextureProfile(6_000.0, 1_600.0, 700.0, 0.35),
    "rubber": _TextureProfile(2_300.0, 550.0, 500.0, 0.18),
    "plastic": _TextureProfile(7_500.0, 3_700.0, 1_500.0, 0.40),
    "soft": _TextureProfile(1_700.0, 380.0, 450.0, 0.12),
}


def _tangential_force_world(
    contact_frame: FloatArray,
    contact_force: FloatArray,
) -> FloatArray:
    """Transform MuJoCo's two contact-frame friction forces into world axes."""

    frame = np.asarray(contact_frame, dtype=np.float64).reshape(3, 3)
    force = np.asarray(contact_force, dtype=np.float64)
    if force.shape != (6,):
        raise ValueError("MuJoCo contact force must have six components")
    return frame[1:, :].T @ force[1:3]


def _aggregate_contact_points(
    points: tuple[_ContactPoint, ...],
) -> tuple[Vector3, float, Vector3, Vector3, float, float]:
    """Aggregate a geom pair using solved normal-load weighting.

    Returns centroid, total normal load, summed tangential force, load-weighted
    relative tangential velocity, its speed, and the friction-power proxy.
    """

    if not points:
        raise ValueError("at least one contact point is required")
    loads = np.asarray([point.normal_load_n for point in points], dtype=np.float64)
    if not np.all(np.isfinite(loads)) or np.any(loads < 0):
        raise ValueError("contact normal loads must be finite and non-negative")
    total_load = float(np.sum(loads))
    if total_load <= 0:
        raise ValueError("aggregate contact normal load must be positive")
    positions = np.asarray(
        [point.world_position_m for point in points],
        dtype=np.float64,
    )
    forces = np.asarray(
        [point.tangential_force_world_n for point in points],
        dtype=np.float64,
    )
    velocities = np.asarray(
        [point.tangential_velocity_world_m_s for point in points],
        dtype=np.float64,
    )
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(forces))
        and np.all(np.isfinite(velocities))
    ):
        raise ValueError("contact vectors must be finite")
    centroid = np.sum(positions * loads[:, None], axis=0) / total_load
    force = np.sum(forces, axis=0)
    velocity = np.sum(velocities * loads[:, None], axis=0) / total_load
    speed = float(np.linalg.norm(velocity))
    power = float(np.linalg.norm(force) * speed)
    return (
        tuple(map(float, centroid)),
        total_load,
        tuple(map(float, force)),
        tuple(map(float, velocity)),
        speed,
        power,
    )


class _ScrapeGate:
    """Apply persistence, hysteresis, and one-substep dropout tolerance."""

    def __init__(self, config: ScrapeConfig) -> None:
        self.config = config
        self._states: dict[GeomPair, _GateState] = {}

    def reset(self) -> None:
        """Discard all contact-persistence state."""

        self._states.clear()

    def update(
        self,
        measurements: dict[GeomPair, ScrapeContact],
    ) -> tuple[ScrapeContact, ...]:
        """Return controls admitted in stable geom-pair order."""

        admitted: list[ScrapeContact] = []
        all_pairs = sorted(set(self._states) | set(measurements))
        retained: dict[GeomPair, _GateState] = {}
        for pair in all_pairs:
            measurement = measurements.get(pair)
            state = self._states.get(pair, _GateState())
            if measurement is None:
                if state.sounding and state.latest is not None:
                    state.missing_substeps += 1
                    if state.missing_substeps <= 1:
                        admitted.append(state.latest)
                        retained[pair] = state
                continue

            state.latest = measurement
            state.missing_substeps = 0
            if state.sounding:
                remains_on = (
                    measurement.normal_load_n >= self.config.normal_force_off_n
                    and measurement.tangential_speed_m_s
                    >= self.config.slip_speed_off_m_s
                )
                if remains_on:
                    admitted.append(measurement)
                    retained[pair] = state
                else:
                    state.sounding = False
                    state.consecutive_on_substeps = 0
                    retained[pair] = state
                continue

            crosses_on = (
                measurement.normal_load_n >= self.config.normal_force_on_n
                and measurement.tangential_speed_m_s
                >= self.config.slip_speed_on_m_s
            )
            state.consecutive_on_substeps = (
                state.consecutive_on_substeps + 1 if crosses_on else 0
            )
            if (
                state.consecutive_on_substeps
                >= self.config.minimum_contact_substeps
            ):
                state.sounding = True
                admitted.append(measurement)
            retained[pair] = state
        self._states = retained
        return tuple(admitted)


class ScrapeTracker:
    """Measure and gate sliding contacts after each solved MuJoCo substep."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: ScrapeConfig,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config
        self._materials = AcousticMaterialResolver(model)
        self._mimo_root_id = model.body("mimo_location").id
        self._pre_velocity = np.zeros((model.nbody, 6), dtype=np.float64)
        self._pre_body_position = np.zeros((model.nbody, 3), dtype=np.float64)
        self._gate = _ScrapeGate(config)
        self.before_substep()

    def reset(self) -> None:
        """Clear persistence and snapshot the reset MuJoCo kinematics."""

        self._gate.reset()
        self.before_substep()

    def before_substep(self, kinematic_data: mujoco.MjData | None = None) -> None:
        """Capture body motion immediately before MuJoCo integration."""

        data = self.data if kinematic_data is None else kinematic_data
        for body_id in range(self.model.nbody):
            mujoco.mj_objectVelocity(
                self.model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                self._pre_velocity[body_id],
                0,
            )
        self._pre_body_position[:] = data.xpos

    def _point_velocity(
        self,
        body_id: int,
        world_position_m: FloatArray,
    ) -> FloatArray:
        if body_id == 0 or int(self.model.body_weldid[body_id]) == 0:
            return np.zeros(3, dtype=np.float64)
        angular = self._pre_velocity[body_id, :3]
        linear = self._pre_velocity[body_id, 3:]
        return linear + np.cross(
            angular,
            world_position_m - self._pre_body_position[body_id],
        )

    def _eligible_bodies(self, body_a: int, body_b: int) -> bool:
        static_a = int(self.model.body_weldid[body_a]) == 0
        static_b = int(self.model.body_weldid[body_b]) == 0
        if static_a and static_b:
            return False
        root_a = int(self.model.body_rootid[body_a])
        root_b = int(self.model.body_rootid[body_b])
        return not (root_a == root_b == self._mimo_root_id)

    def _radiating_body(self, body_a: int, body_b: int) -> int:
        dynamic = sorted(
            body_id
            for body_id in (body_a, body_b)
            if int(self.model.body_weldid[body_id]) != 0
        )
        if not dynamic:
            raise ValueError("a scrape contact requires a dynamic body")
        return dynamic[0]

    def _measurements(self) -> dict[GeomPair, ScrapeContact]:
        grouped: dict[GeomPair, list[_ContactPoint]] = {}
        bodies: dict[GeomPair, tuple[int, int]] = {}
        contact_indices = sorted(
            range(self.data.ncon),
            key=lambda index: (
                tuple(sorted(map(int, self.data.contact[index].geom))),
                index,
            ),
        )
        for contact_index in contact_indices:
            contact = self.data.contact[contact_index]
            if int(contact.efc_address) < 0:
                continue
            geom_a, geom_b = map(int, contact.geom)
            if geom_a < 0 or geom_b < 0:
                continue
            body_a = int(self.model.geom_bodyid[geom_a])
            body_b = int(self.model.geom_bodyid[geom_b])
            if not self._eligible_bodies(body_a, body_b):
                continue
            force_contact = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                self.model,
                self.data,
                contact_index,
                force_contact,
            )
            normal_load = max(float(force_contact[0]), 0.0)
            if normal_load <= 0:
                continue
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            tangential_force = _tangential_force_world(frame, force_contact)
            position = np.asarray(contact.pos, dtype=np.float64).copy()
            relative_velocity = self._point_velocity(
                body_a,
                position,
            ) - self._point_velocity(body_b, position)
            normal = frame[0]
            tangential_velocity = relative_velocity - normal * float(
                np.dot(relative_velocity, normal)
            )
            pair = tuple(sorted((geom_a, geom_b)))
            grouped.setdefault(pair, []).append(
                _ContactPoint(
                    world_position_m=tuple(map(float, position)),
                    normal_load_n=normal_load,
                    tangential_force_world_n=tuple(map(float, tangential_force)),
                    tangential_velocity_world_m_s=tuple(
                        map(float, tangential_velocity)
                    ),
                )
            )
            bodies[pair] = (body_a, body_b)

        measurements: dict[GeomPair, ScrapeContact] = {}
        for pair in sorted(grouped):
            (
                centroid,
                normal_load,
                tangential_force,
                tangential_velocity,
                speed,
                power,
            ) = _aggregate_contact_points(tuple(grouped[pair]))
            body_a, body_b = bodies[pair]
            radiating_body = self._radiating_body(body_a, body_b)
            body_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                radiating_body,
            )
            if body_name is None:
                continue
            body = self.data.body(radiating_body)
            local_position = body.xmat.reshape(3, 3).T @ (
                np.asarray(centroid, dtype=np.float64) - body.xpos
            )
            material_pair = canonical_material_pair(
                self._materials.material(pair[0]),
                self._materials.material(pair[1]),
            )
            measurements[pair] = ScrapeContact(
                stream_id=f"scrape:{pair[0]}:{pair[1]}",
                geom_pair=pair,
                material_pair=material_pair,
                attachment=BodyAttachment(
                    body_name,
                    tuple(map(float, local_position)),
                ),
                world_position_m=centroid,
                normal_load_n=normal_load,
                tangential_force_world_n=tangential_force,
                tangential_velocity_world_m_s=tangential_velocity,
                tangential_speed_m_s=speed,
                friction_power_w=power,
            )
        return measurements

    def detect(self) -> tuple[ScrapeContact, ...]:
        """Return persistent scrape controls for the solved substep."""

        return self._gate.update(self._measurements())


def _pair_texture(material_pair: MaterialPair) -> _TextureProfile:
    """Blend two engineering texture profiles symmetrically."""

    first = _MATERIAL_TEXTURES[material_pair[0]]
    second = _MATERIAL_TEXTURES[material_pair[1]]
    return _TextureProfile(
        lowpass_hz=sqrt(first.lowpass_hz * second.lowpass_hz),
        resonance_hz=sqrt(first.resonance_hz * second.resonance_hz),
        resonance_bandwidth_hz=sqrt(
            first.resonance_bandwidth_hz * second.resonance_bandwidth_hz
        ),
        resonance_mix=(first.resonance_mix + second.resonance_mix) / 2,
    )


def _stream_rng(seed: int, stream_id: str) -> np.random.Generator:
    """Create an order-independent RNG stream from a stable BLAKE2 digest."""

    payload = f"{int(seed)}\0{stream_id}".encode("utf-8")
    digest = blake2b(
        payload,
        digest_size=16,
        person=b"MIMoScrapeV1",
    ).digest()
    return np.random.default_rng(int.from_bytes(digest, byteorder="little"))


class ScrapeSynthesizer:
    """Generate calibrated persistent roughness from admitted scrape contacts.

    The material profiles are practical engineering approximations. Each
    stable geom-pair stream owns an independently derived RNG and persistent
    filter, power, and envelope state.
    """

    def __init__(
        self,
        spec: AuditionSpec,
        config: ScrapeConfig,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.config = config
        self._seed = int(seed)
        self._voices: dict[str, _ScrapeVoice] = {}
        self._diagnostics: tuple[ScrapeDiagnostic, ...] = ()

    @property
    def diagnostics(self) -> tuple[ScrapeDiagnostic, ...]:
        """Return immutable diagnostics for the latest generated substep."""

        return self._diagnostics

    @property
    def stream_ids(self) -> tuple[str, ...]:
        """Return generator states retained for active or releasing streams."""

        return tuple(sorted(self._voices))

    def reset(self, seed: int) -> None:
        """Clear all texture and envelope state for a deterministic episode."""

        self._seed = int(seed)
        self._voices.clear()
        self._diagnostics = ()

    def _new_voice(self, contact: ScrapeContact) -> _ScrapeVoice:
        return _ScrapeVoice(
            stream_id=contact.stream_id,
            material_pair=contact.material_pair,
            attachment=contact.attachment,
            rng=_stream_rng(self._seed, contact.stream_id),
            latest=contact,
            previous_input_power_w=contact.friction_power_w,
        )

    def _texture_block(self, voice: _ScrapeVoice) -> FloatArray:
        """Advance one stream's persistent material filter by one substep."""

        sample_count = self.spec.samples_per_substep
        sample_rate = self.spec.sample_rate_hz
        profile = _pair_texture(voice.material_pair)
        cutoff = min(profile.lowpass_hz, 0.45 * sample_rate)
        resonance = min(profile.resonance_hz, 0.45 * sample_rate)
        lowpass_alpha = 1.0 - exp(-2.0 * pi * cutoff / sample_rate)
        pole_radius = exp(
            -pi * profile.resonance_bandwidth_hz / sample_rate
        )
        pole_angle = 2.0 * pi * resonance / sample_rate
        pole_coefficient = 2.0 * pole_radius * cos(pole_angle)
        noise = voice.rng.standard_normal(sample_count)
        output = np.empty(sample_count, dtype=np.float64)
        lowpass_state = voice.lowpass_state
        resonance_y1 = voice.resonance_y1
        resonance_y2 = voice.resonance_y2
        for index, sample in enumerate(noise):
            lowpass_state += lowpass_alpha * (sample - lowpass_state)
            resonant = (
                (1.0 - pole_radius) * sample
                + pole_coefficient * resonance_y1
                - pole_radius * pole_radius * resonance_y2
            )
            resonance_y2 = resonance_y1
            resonance_y1 = resonant
            output[index] = (
                (1.0 - profile.resonance_mix) * lowpass_state
                + profile.resonance_mix * resonant
            )
        voice.lowpass_state = float(lowpass_state)
        voice.resonance_y1 = float(resonance_y1)
        voice.resonance_y2 = float(resonance_y2)
        rms = float(np.sqrt(np.mean(output * output)))
        if rms > 0:
            output /= rms
        return output

    def _pressure_targets(
        self,
        voice: _ScrapeVoice,
        input_power_w: float,
    ) -> FloatArray:
        """Interpolate controls and smooth friction power at audio rate."""

        sample_count = self.spec.samples_per_substep
        sample_rate = self.spec.sample_rate_hz
        power_targets = np.linspace(
            voice.previous_input_power_w,
            input_power_w,
            sample_count,
            dtype=np.float64,
        )
        coefficient = 1.0 - exp(
            -1.0
            / (sample_rate * self.config.power_smoothing_seconds)
        )
        pressure = np.empty(sample_count, dtype=np.float64)
        power_ema = voice.power_ema_w
        for index, target in enumerate(power_targets):
            power_ema += coefficient * (target - power_ema)
            pressure[index] = min(
                self.config.max_rms_pressure_pa_at_1m,
                self.config.reference_rms_pressure_pa_at_1m
                * sqrt(max(power_ema, 0.0) / self.config.reference_friction_power_w),
            )
        voice.power_ema_w = float(power_ema)
        voice.previous_input_power_w = float(input_power_w)
        return pressure

    def _envelope_block(self, voice: _ScrapeVoice, active: bool) -> FloatArray:
        """Advance the exact finite attack or release envelope."""

        sample_count = self.spec.samples_per_substep
        sample_rate = self.spec.sample_rate_hz
        envelope = np.empty(sample_count, dtype=np.float64)
        if active:
            voice.release_total_samples = 0
            voice.release_elapsed_samples = 0
            attack_samples = max(
                1,
                round(self.config.attack_seconds * sample_rate),
            )
            for index in range(sample_count):
                voice.envelope = min(1.0, voice.envelope + 1.0 / attack_samples)
                envelope[index] = voice.envelope
            return envelope

        if voice.release_total_samples == 0:
            voice.release_total_samples = max(
                1,
                round(self.config.release_seconds * sample_rate),
            )
            voice.release_elapsed_samples = 0
            voice.release_start_envelope = voice.envelope
        total = voice.release_total_samples
        for index in range(sample_count):
            if voice.release_elapsed_samples < total:
                voice.release_elapsed_samples += 1
                voice.envelope = voice.release_start_envelope * (
                    1.0 - voice.release_elapsed_samples / total
                )
            else:
                voice.envelope = 0.0
            envelope[index] = voice.envelope
        return envelope

    def render(
        self,
        contacts: tuple[ScrapeContact, ...],
    ) -> tuple[ContinuousEmission, ...]:
        """Advance active/releasing streams and return one emission per state."""

        by_stream_id: dict[str, ScrapeContact] = {}
        for contact in contacts:
            if contact.stream_id in by_stream_id:
                raise ValueError(f"duplicate scrape stream_id: {contact.stream_id}")
            by_stream_id[contact.stream_id] = contact
            voice = self._voices.get(contact.stream_id)
            if voice is None:
                self._voices[contact.stream_id] = self._new_voice(contact)
            elif voice.material_pair != contact.material_pair:
                raise ValueError(
                    f"scrape material identity changed: {contact.stream_id}"
                )
            else:
                voice.attachment = contact.attachment
                voice.latest = contact

        emissions: list[ContinuousEmission] = []
        diagnostics: list[ScrapeDiagnostic] = []
        retained: dict[str, _ScrapeVoice] = {}
        for stream_id in sorted(self._voices):
            voice = self._voices[stream_id]
            contact = by_stream_id.get(stream_id)
            active = contact is not None
            input_power = contact.friction_power_w if contact is not None else 0.0
            texture = self._texture_block(voice)
            pressure = self._pressure_targets(voice, input_power)
            envelope = self._envelope_block(voice, active)
            samples = np.asarray(texture * pressure * envelope, dtype=np.float32)
            priority = (
                input_power
                if active
                else voice.latest.friction_power_w * voice.envelope
            )
            emissions.append(
                ContinuousEmission(
                    stream_id=stream_id,
                    kind=(
                        f"scrape:{voice.material_pair[0]}_"
                        f"{voice.material_pair[1]}"
                    ),
                    attachment=voice.attachment,
                    samples_pa_at_1m=samples,
                    priority=priority,
                )
            )
            latest = voice.latest
            diagnostics.append(
                ScrapeDiagnostic(
                    stream_id=stream_id,
                    geom_pair=latest.geom_pair,
                    material_pair=voice.material_pair,
                    phase="active" if active else "release",
                    world_position_m=latest.world_position_m,
                    normal_load_n=latest.normal_load_n,
                    tangential_velocity_world_m_s=(
                        latest.tangential_velocity_world_m_s
                    ),
                    tangential_speed_m_s=latest.tangential_speed_m_s,
                    friction_power_w=latest.friction_power_w,
                    target_rms_pressure_pa_at_1m=float(pressure[-1]),
                )
            )
            if active or voice.release_elapsed_samples < voice.release_total_samples:
                retained[stream_id] = voice
        self._voices = retained
        self._diagnostics = tuple(diagnostics)
        return tuple(emissions)

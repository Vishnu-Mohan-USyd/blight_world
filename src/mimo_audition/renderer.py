"""Reference Python binaural renderer and deterministic voice scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from mimo_audition.clips import ClipLibrary
from mimo_audition.config import AuditionConfig
from mimo_audition.dsp import AttackReleaseSmoother, FractionalDelayLine, OnePoleLowpass
from mimo_audition.hrtf import HrtfBank, PairedHrtfConvolver, load_packaged_hrtf
from mimo_audition.room import (
    FeedbackDelayNetwork,
    first_order_image_sources,
    project_source_position,
)
from mimo_audition.sources import Attachment, AudioSourceSpec


FloatArray = NDArray[np.float64]


class AcousticScene(Protocol):
    """Minimal dynamic-scene interface required by the renderer."""

    def listener_pose(self) -> tuple[FloatArray, FloatArray]:
        """Return listener midpoint and world-from-head 3x3 rotation."""

    def attachment_pose(self, attachment: Attachment) -> tuple[FloatArray, int]:
        """Return source world position and its body ID, or -1 for static."""

    def direct_path_occluded(
        self,
        listener_position_m: FloatArray,
        source_position_m: FloatArray,
        source_body_id: int,
    ) -> bool:
        """Return whether an eligible geom blocks the direct path."""


@dataclass(slots=True)
class _PathState:
    delay: FractionalDelayLine
    hrtf: PairedHrtfConvolver


@dataclass(slots=True)
class _Voice:
    voice_id: int
    source: AudioSourceSpec
    start_sample: int
    stop_sample: int | None
    paths: list[_PathState]
    occlusion_smoother: AttackReleaseSmoother
    occlusion_lowpass: OnePoleLowpass
    impact_priority: float | None
    previous_listener_position_m: FloatArray | None = None
    previous_source_position_m: FloatArray | None = None
    dry_end_sample: int | None = None


class BinauralRenderer:
    """Position mono clips in a MuJoCo-compatible binaural sound field."""

    def __init__(
        self,
        config: AuditionConfig,
        *,
        hrtf_bank: HrtfBank | None = None,
        clips: ClipLibrary | None = None,
    ) -> None:
        self.config = config
        self.spec = config.spec
        self.hrtf_bank = (
            load_packaged_hrtf(config.sample_rate_hz, config.hrtf_path)
            if hrtf_bank is None
            else hrtf_bank.at_sample_rate(config.sample_rate_hz)
        )
        self.clips = ClipLibrary(config.sample_rate_hz) if clips is None else clips
        if self.clips.sample_rate_hz != config.sample_rate_hz:
            raise ValueError("clip library and renderer sample rates differ")
        crossfade_samples = round(config.hrtf_crossfade_s * config.sample_rate_hz)
        if crossfade_samples != self.spec.samples_per_substep:
            raise ValueError("HRTF crossfade must remain one 5 ms substep")
        self._crossfade_samples = crossfade_samples
        self._sources: dict[str, AudioSourceSpec] = {}
        self._voices: list[_Voice] = []
        self._next_voice_id = 1
        self._sample_cursor = 0
        self._fdn = FeedbackDelayNetwork(config.sample_rate_hz)

    @property
    def sample_cursor(self) -> int:
        """Number of acoustic samples elapsed since the latest reset."""

        return self._sample_cursor

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Registered source identifiers in deterministic order."""

        return tuple(sorted(self._sources))

    @property
    def active_voice_count(self) -> int:
        """Number of voices whose dry clip is active at the current clock."""

        return sum(self._voice_is_dry_active(voice, self._sample_cursor) for voice in self._voices)

    def _new_paths(self) -> list[_PathState]:
        path_count = 7 if self.config.enable_first_order_reflections else 1
        return [
            _PathState(
                delay=FractionalDelayLine(),
                hrtf=PairedHrtfConvolver(self.hrtf_bank, self._crossfade_samples),
            )
            for _ in range(path_count)
        ]

    def register_source(self, source: AudioSourceSpec) -> None:
        """Register one source after verifying its clip and identifier."""

        if source.source_id in self._sources:
            raise ValueError(f"source_id is already registered: {source.source_id}")
        if len(self._sources) >= self.config.max_registered_sources:
            raise RuntimeError("registered source limit reached")
        self.clips.get(source.clip_id)
        self._sources[source.source_id] = source

    def unregister_source(self, source_id: str) -> None:
        """Remove an idle source definition."""

        if any(voice.source.source_id == source_id for voice in self._voices):
            raise RuntimeError("cannot unregister a scheduled or active source")
        try:
            del self._sources[source_id]
        except KeyError as error:
            raise KeyError(f"unknown source_id: {source_id}") from error

    def play(self, source_id: str, *, delay_s: float = 0.0) -> int:
        """Schedule a source and return its stable voice identifier."""

        if not isfinite(delay_s) or delay_s < 0:
            raise ValueError("delay_s must be finite and non-negative")
        try:
            source = self._sources[source_id]
        except KeyError as error:
            raise KeyError(f"unknown source_id: {source_id}") from error
        voice_id = self._schedule_voice(source, delay_s, impact_priority=None)
        if voice_id is None:
            raise AssertionError("explicit voice admission returned no identifier")
        return voice_id

    def play_transient(
        self,
        clip_id: str,
        attachment: Attachment,
        *,
        gain_db: float = 0.0,
        delay_s: float = 0.0,
        impact_energy_j: float = 0.0,
    ) -> int | None:
        """Best-effort schedule an unregistered physical-impact one-shot.

        A stronger pending impact may replace a weaker overlapping pending
        impact. Equal-energy candidates retain arrival order. Capacity
        rejection returns ``None`` and never defers an exception to rendering.
        """

        if not isfinite(impact_energy_j) or impact_energy_j < 0:
            raise ValueError("impact_energy_j must be finite and non-negative")

        source = AudioSourceSpec(
            source_id=f"__impact_{self._next_voice_id}",
            clip_id=clip_id,
            attachment=attachment,
            loop=False,
            gain_db=gain_db,
        )
        self.clips.get(clip_id)
        return self._schedule_voice(
            source,
            delay_s,
            impact_priority=impact_energy_j,
        )

    def _dry_end_sample(self, voice: _Voice) -> int | None:
        """Return the exclusive dry-signal end, excluding retained tails."""

        dry_end = voice.dry_end_sample
        if voice.stop_sample is not None:
            dry_end = (
                voice.stop_sample
                if dry_end is None
                else min(dry_end, voice.stop_sample)
            )
        return dry_end

    def _dry_block_interval(
        self,
        start_sample: int,
        end_sample: int | None,
    ) -> tuple[int, int | None] | None:
        """Map a half-open dry sample interval to occupied render blocks."""

        if end_sample is not None and end_sample <= start_sample:
            return None
        block_size = self.spec.samples_per_substep
        start_block = start_sample // block_size
        end_block = (
            None
            if end_sample is None
            else (end_sample + block_size - 1) // block_size
        )
        return start_block, end_block

    def _voice_dry_block_interval(
        self,
        voice: _Voice,
    ) -> tuple[int, int | None] | None:
        return self._dry_block_interval(voice.start_sample, self._dry_end_sample(voice))

    @staticmethod
    def _block_intervals_overlap(
        first: tuple[int, int | None],
        second: tuple[int, int | None],
    ) -> bool:
        first_start, first_end = first
        second_start, second_end = second
        return (first_end is None or second_start < first_end) and (
            second_end is None or first_start < second_end
        )

    def _capacity_conflicts(
        self,
        candidate: tuple[int, int | None],
        excluded_voice_ids: set[int] | frozenset[int] = frozenset(),
    ) -> frozenset[int]:
        """Return voices occupying any already-full candidate render block."""

        candidate_start, candidate_end = candidate
        intervals: list[tuple[_Voice, int, int | None]] = []
        change_blocks = {candidate_start}
        for voice in self._voices:
            if voice.voice_id in excluded_voice_ids:
                continue
            interval = self._voice_dry_block_interval(voice)
            if interval is None or not self._block_intervals_overlap(candidate, interval):
                continue
            start_block, end_block = interval
            intervals.append((voice, start_block, end_block))
            change_blocks.add(max(candidate_start, start_block))
            if end_block is not None and (
                candidate_end is None or end_block < candidate_end
            ):
                change_blocks.add(end_block)

        conflicts: set[int] = set()
        for block in sorted(change_blocks):
            if candidate_end is not None and block >= candidate_end:
                continue
            active = {
                voice.voice_id
                for voice, start_block, end_block in intervals
                if start_block <= block and (end_block is None or block < end_block)
            }
            if len(active) >= self.config.max_active_voices:
                conflicts.update(active)
        return frozenset(conflicts)

    def _pending_voice_count(
        self,
        excluded_voice_ids: set[int] | frozenset[int] = frozenset(),
    ) -> int:
        """Count scheduled voices that still have a future/current dry interval."""

        return sum(
            voice.voice_id not in excluded_voice_ids
            and voice.start_sample >= self._sample_cursor
            and self._voice_dry_block_interval(voice) is not None
            for voice in self._voices
        )

    def _append_voice(
        self,
        source: AudioSourceSpec,
        start_sample: int,
        dry_end_sample: int | None,
        impact_priority: float | None,
    ) -> int:
        """Allocate renderer state after interval admission succeeds."""

        voice = _Voice(
            voice_id=self._next_voice_id,
            source=source,
            start_sample=start_sample,
            stop_sample=None,
            paths=self._new_paths(),
            occlusion_smoother=AttackReleaseSmoother(
                self.config.occlusion_attack_s,
                self.config.occlusion_release_s,
                self.config.sample_rate_hz,
            ),
            occlusion_lowpass=OnePoleLowpass(
                self.config.occlusion_cutoff_hz,
                self.config.sample_rate_hz,
            ),
            impact_priority=impact_priority,
            dry_end_sample=dry_end_sample,
        )
        self._voices.append(voice)
        self._next_voice_id += 1
        return voice.voice_id

    def _schedule_voice(
        self,
        source: AudioSourceSpec,
        delay_s: float,
        *,
        impact_priority: float | None,
    ) -> int | None:
        """Admit a complete dry interval before allocating renderer state."""

        if not isfinite(delay_s) or delay_s < 0:
            raise ValueError("delay_s must be finite and non-negative")
        start_sample = self._sample_cursor + round(delay_s * self.config.sample_rate_hz)
        clip = self.clips.get(source.clip_id)
        dry_end_sample = None if source.loop else start_sample + clip.samples.size
        candidate = self._dry_block_interval(start_sample, dry_end_sample)
        if candidate is None:
            raise AssertionError("validated non-empty clip has an empty dry interval")

        if impact_priority is None:
            if self._pending_voice_count() >= self.config.max_scheduled_events:
                raise RuntimeError("scheduled event limit reached")
            if self._capacity_conflicts(candidate):
                raise RuntimeError(
                    "active voice capacity unavailable for requested dry interval"
                )
            return self._append_voice(
                source,
                start_sample,
                dry_end_sample,
                impact_priority=None,
            )

        excluded: set[int] = set()
        while True:
            conflicts = self._capacity_conflicts(candidate, excluded)
            scheduled_full = (
                self._pending_voice_count(excluded)
                >= self.config.max_scheduled_events
            )
            if not conflicts and not scheduled_full:
                break
            eligible_ids = conflicts or frozenset(
                voice.voice_id
                for voice in self._voices
                if voice.start_sample > self._sample_cursor
            )
            removable = [
                voice
                for voice in self._voices
                if voice.voice_id in eligible_ids
                and voice.voice_id not in excluded
                and voice.start_sample > self._sample_cursor
                and voice.impact_priority is not None
                and voice.impact_priority < impact_priority
            ]
            if not removable:
                return None
            victim = min(
                removable,
                key=lambda voice: (voice.impact_priority, -voice.voice_id),
            )
            excluded.add(victim.voice_id)

        if excluded:
            self._voices = [
                voice for voice in self._voices if voice.voice_id not in excluded
            ]
        return self._append_voice(
            source,
            start_sample,
            dry_end_sample,
            impact_priority=impact_priority,
        )

    def stop(self, source_id: str, *, delay_s: float = 0.0) -> None:
        """Schedule all voices of a registered source to stop."""

        if source_id not in self._sources:
            raise KeyError(f"unknown source_id: {source_id}")
        if not isfinite(delay_s) or delay_s < 0:
            raise ValueError("delay_s must be finite and non-negative")
        stop_sample = self._sample_cursor + round(delay_s * self.config.sample_rate_hz)
        for voice in self._voices:
            if voice.source.source_id == source_id:
                voice.stop_sample = (
                    stop_sample
                    if voice.stop_sample is None
                    else min(stop_sample, voice.stop_sample)
                )

    def reset(self) -> None:
        """Clear schedule, voices, room tail, and acoustic sample clock."""

        self._voices.clear()
        self._next_voice_id = 1
        self._sample_cursor = 0
        self._fdn.reset()

    def _voice_is_dry_active(self, voice: _Voice, sample_index: int) -> bool:
        end = self._dry_end_sample(voice)
        return voice.start_sample <= sample_index and (end is None or sample_index < end)

    def _voice_block(self, voice: _Voice, sample_count: int) -> FloatArray:
        clip = self.clips.get(voice.source.clip_id)
        output = np.zeros(sample_count, dtype=np.float64)
        block_start = self._sample_cursor
        block_end = block_start + sample_count
        dry_stop = voice.dry_end_sample
        if voice.stop_sample is not None:
            dry_stop = voice.stop_sample if dry_stop is None else min(dry_stop, voice.stop_sample)
        active_start = max(block_start, voice.start_sample)
        active_stop = block_end if dry_stop is None else min(block_end, dry_stop)
        if active_start >= active_stop:
            return output
        output_start = active_start - block_start
        output_stop = active_stop - block_start
        clip_start = active_start - voice.start_sample
        sample_indices = np.arange(clip_start, clip_start + output_stop - output_start)
        if voice.source.loop:
            sample_indices %= clip.samples.size
        output[output_start:output_stop] = clip.samples[sample_indices]
        gain = 10.0 ** (voice.source.gain_db / 20.0)
        output *= clip.reference_peak_pressure_pa_at_1m * gain
        return output

    def _render_path(
        self,
        state: _PathState,
        mono: FloatArray,
        listener_position_m: FloatArray,
        head_rotation: FloatArray,
        path_position_m: FloatArray,
        previous_listener_position_m: FloatArray,
        previous_path_position_m: FloatArray,
        pressure_gain: float,
    ) -> FloatArray:
        offset = path_position_m - listener_position_m
        previous_offset = previous_path_position_m - previous_listener_position_m
        distance_m = float(np.linalg.norm(offset))
        previous_distance_m = float(np.linalg.norm(previous_offset))
        if distance_m <= 0 or previous_distance_m <= 0:
            return np.zeros((2, mono.size), dtype=np.float64)
        direction_head = head_rotation.T @ (offset / distance_m)
        direction_index = self.hrtf_bank.nearest_index(direction_head)
        delay_samples = (
            distance_m / self.config.speed_of_sound_m_s * self.config.sample_rate_hz
            - self.hrtf_bank.common_delay_samples
        )
        previous_delay_samples = (
            previous_distance_m
            / self.config.speed_of_sound_m_s
            * self.config.sample_rate_hz
            - self.hrtf_bank.common_delay_samples
        )
        delayed = state.delay.process(
            mono,
            max(previous_delay_samples, 0.0),
            max(delay_samples, 0.0),
        )
        attenuation = pressure_gain / max(distance_m, self.config.distance_floor_m)
        return state.hrtf.process(delayed * attenuation, direction_index)

    def render_substep(self, scene: AcousticScene) -> FloatArray:
        """Render exactly one 5 ms physics interval from the current scene."""

        sample_count = self.spec.samples_per_substep
        listener_position, head_rotation = scene.listener_pose()
        listener_position = np.asarray(listener_position, dtype=np.float64)
        head_rotation = np.asarray(head_rotation, dtype=np.float64).reshape(3, 3)
        current_block = self._sample_cursor // sample_count
        active_dry = sum(
            interval is not None
            and interval[0] <= current_block
            and (interval[1] is None or current_block < interval[1])
            for voice in self._voices
            if (interval := self._voice_dry_block_interval(voice)) is not None
        )
        if active_dry > self.config.max_active_voices:
            raise RuntimeError("active voice limit reached")

        output = np.zeros((2, sample_count), dtype=np.float64)
        reverb_input = np.zeros(sample_count, dtype=np.float64)
        retained_voices: list[_Voice] = []
        for voice in self._voices:
            mono = self._voice_block(voice, sample_count)
            source_position, source_body_id = scene.attachment_pose(voice.source.attachment)
            source_position = project_source_position(source_position)
            previous_listener = (
                listener_position
                if voice.previous_listener_position_m is None
                else voice.previous_listener_position_m
            )
            previous_source = (
                source_position
                if voice.previous_source_position_m is None
                else voice.previous_source_position_m
            )

            direct_mono = mono
            if self.config.enable_occlusion:
                is_occluded = scene.direct_path_occluded(
                    listener_position,
                    source_position,
                    source_body_id,
                )
                mix = voice.occlusion_smoother.process(is_occluded, sample_count)
                lowpassed = voice.occlusion_lowpass.process(mono)
                blocked_gain = 10.0 ** (self.config.occlusion_attenuation_db / 20.0)
                direct_mono = mono * (1.0 - mix) + lowpassed * blocked_gain * mix
            output += self._render_path(
                voice.paths[0],
                direct_mono,
                listener_position,
                head_rotation,
                source_position,
                previous_listener,
                previous_source,
                1.0,
            )

            if self.config.enable_first_order_reflections:
                current_images = first_order_image_sources(source_position)
                previous_images = first_order_image_sources(previous_source)
                for path_index, (current_image, previous_image) in enumerate(
                    zip(current_images, previous_images, strict=True),
                    start=1,
                ):
                    output += self._render_path(
                        voice.paths[path_index],
                        mono,
                        listener_position,
                        head_rotation,
                        current_image.world_position_m,
                        previous_listener,
                        previous_image.world_position_m,
                        current_image.pressure_reflection,
                    )

            distance = max(
                float(np.linalg.norm(source_position - listener_position)),
                self.config.distance_floor_m,
            )
            reverb_input += mono * self.config.late_reverb_send / distance
            voice.previous_listener_position_m = listener_position.copy()
            voice.previous_source_position_m = source_position.copy()

            dry_end = voice.dry_end_sample
            if voice.stop_sample is not None:
                dry_end = voice.stop_sample if dry_end is None else min(dry_end, voice.stop_sample)
            tail_samples = 8_192 + self.hrtf_bank.tap_count
            if dry_end is None or self._sample_cursor + sample_count < dry_end + tail_samples:
                retained_voices.append(voice)

        self._voices = retained_voices
        if self.config.enable_late_reverb:
            output += self._fdn.process(reverb_input)
        self._sample_cursor += sample_count
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("auditory renderer produced non-finite pressure")
        return output

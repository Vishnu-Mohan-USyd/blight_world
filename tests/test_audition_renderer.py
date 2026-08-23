"""Tests for scheduling, motion, distance, and binaural rendering."""

from __future__ import annotations

import numpy as np
import pytest

from mimo_audition import AuditionConfig, AudioSourceSpec, StaticAttachment
from mimo_audition.clips import ClipLibrary
from mimo_audition.hrtf import HrtfBank
from mimo_audition.renderer import BinauralRenderer


def _bank() -> HrtfBank:
    directions = np.asarray(((1, 0, 0), (0, 1, 0), (0, -1, 0)), dtype=np.float64)
    hrir = np.zeros((3, 2, 32), dtype=np.float64)
    hrir[0, :, 16] = 1
    hrir[1, 0, 16] = 1
    hrir[1, 1, 18] = 0.5
    hrir[2, 0, 18] = 0.5
    hrir[2, 1, 16] = 1
    return HrtfBank(directions, hrir, 48_000, 16.0, "fixture", "test")


class _Scene:
    def __init__(self, source_position: tuple[float, float, float]) -> None:
        self.source_position = np.asarray(source_position, dtype=np.float64)
        self.occluded = False

    def listener_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(3), np.eye(3)

    def attachment_pose(self, attachment: object) -> tuple[np.ndarray, int]:
        return self.source_position.copy(), -1

    def direct_path_occluded(self, *args: object) -> bool:
        return self.occluded


def _renderer(
    position: tuple[float, float, float],
    *,
    enable_occlusion: bool = False,
    loop: bool = False,
    max_active_voices: int = 8,
    clip_sample_count: int = 2_000,
) -> tuple[BinauralRenderer, _Scene]:
    config = AuditionConfig.quality_48k(
        enable_occlusion=enable_occlusion,
        enable_first_order_reflections=False,
        enable_late_reverb=False,
        max_active_voices=max_active_voices,
    )
    clips = ClipLibrary(48_000)
    clips.register(
        "tone",
        np.ones(clip_sample_count),
        reference_peak_pressure_pa_at_1m=1.0,
    )
    renderer = BinauralRenderer(config, hrtf_bank=_bank(), clips=clips)
    renderer.register_source(
        AudioSourceSpec("source", "tone", StaticAttachment(position), loop=loop)
    )
    return renderer, _Scene(position)


def test_played_source_is_channels_first_and_positioned() -> None:
    """A leftward source produces a finite, asymmetric stereo pressure block."""

    renderer, scene = _renderer((0.0, 1.0, 0.0))
    renderer.play("source")
    output = renderer.render_substep(scene)
    assert output.shape == (2, 240)
    assert output.dtype == np.float64
    assert np.all(np.isfinite(output))
    assert np.max(np.abs(output[0])) > np.max(np.abs(output[1]))
    assert renderer.sample_cursor == 240


def test_distance_attenuation_is_inverse_distance() -> None:
    """Doubling a static direct-path distance halves steady pressure."""

    near, near_scene = _renderer((1.0, 0.0, 0.0))
    far, far_scene = _renderer((2.0, 0.0, 0.0))
    near.play("source")
    far.play("source")
    near.render_substep(near_scene)
    far_first = far.render_substep(far_scene)
    assert np.count_nonzero(far_first) == 0
    near.render_substep(near_scene)
    far.render_substep(far_scene)
    near_output = near.render_substep(near_scene)
    far_output = far.render_substep(far_scene)
    near_level = np.sqrt(np.mean(near_output[:, -100:] ** 2))
    far_level = np.sqrt(np.mean(far_output[:, -100:] ** 2))
    assert far_level / near_level == pytest.approx(0.5, rel=0.03)


def test_scheduler_limits_and_reset_are_deterministic() -> None:
    """Unknown/overfull schedules fail closed and reset clears the clock."""

    renderer, scene = _renderer((1.0, 0.0, 0.0))
    with pytest.raises(KeyError, match="unknown source_id"):
        renderer.play("missing")
    renderer.play("source")
    first = renderer.render_substep(scene)
    renderer.reset()
    assert renderer.sample_cursor == 0
    assert renderer.active_voice_count == 0
    assert renderer.source_ids == ("source",)
    renderer.play("source")
    second = renderer.render_substep(scene)
    np.testing.assert_array_equal(first, second)


def test_no_clipping_or_automatic_gain_control() -> None:
    """Pressure above one pascal remains numerically represented."""

    renderer, scene = _renderer((0.25, 0.0, 0.0))
    renderer.play("source")
    output = renderer.render_substep(scene)
    assert np.max(np.abs(output)) > 1.0


def test_unregistered_transient_uses_same_positioned_voice_path() -> None:
    """Physical impacts can sound without consuming the user source registry."""

    renderer, scene = _renderer((0.0, -1.0, 0.0))
    renderer.play_transient("tone", StaticAttachment((0.0, -1.0, 0.0)))
    output = renderer.render_substep(scene)
    assert renderer.source_ids == ("source",)
    assert np.max(np.abs(output[1])) > np.max(np.abs(output[0]))


def test_direct_occlusion_smoothly_reduces_steady_pressure() -> None:
    """The renderer applies the configured blocked-path loss after attack."""

    clear, clear_scene = _renderer(
        (1.0, 0.0, 0.0), enable_occlusion=True, loop=True
    )
    blocked, blocked_scene = _renderer(
        (1.0, 0.0, 0.0), enable_occlusion=True, loop=True
    )
    blocked_scene.occluded = True
    clear.play("source")
    blocked.play("source")
    clear_output = np.zeros((2, 240))
    blocked_output = np.zeros((2, 240))
    for _ in range(20):
        clear_output = clear.render_substep(clear_scene)
        blocked_output = blocked.render_substep(blocked_scene)

    clear_level = np.sqrt(np.mean(clear_output[:, -80:] ** 2))
    blocked_level = np.sqrt(np.mean(blocked_output[:, -80:] ** 2))
    assert clear_level > 0
    assert 0.35 < blocked_level / clear_level < 0.37


def test_pending_impact_intervals_reserve_capacity_in_arrival_order() -> None:
    """Delayed impacts reserve future blocks before any dry sample starts."""

    renderer, scene = _renderer((1.0, 0.0, 0.0), max_active_voices=8)
    accepted = [
        renderer.play_transient(
            "tone",
            StaticAttachment((1.0, 0.0, 0.0)),
            delay_s=0.010,
            impact_energy_j=float(9 - index),
        )
        for index in range(8)
    ]
    rejected = renderer.play_transient(
        "tone",
        StaticAttachment((1.0, 0.0, 0.0)),
        delay_s=0.010,
        impact_energy_j=1.0,
    )

    assert accepted == list(range(1, 9))
    assert rejected is None
    assert [voice.voice_id for voice in renderer._voices] == list(range(1, 9))
    assert renderer._next_voice_id == 9
    for _ in range(4):
        output = renderer.render_substep(scene)
        assert np.all(np.isfinite(output))


def test_stronger_pending_impact_replaces_weaker_but_tie_is_stable() -> None:
    """Best-effort admission keeps the strongest impact with stable ties."""

    renderer, _ = _renderer((1.0, 0.0, 0.0), max_active_voices=1)
    weak_id = renderer.play_transient(
        "tone",
        StaticAttachment((1.0, 0.0, 0.0)),
        delay_s=0.010,
        impact_energy_j=1.0,
    )
    tied_id = renderer.play_transient(
        "tone",
        StaticAttachment((1.0, 0.0, 0.0)),
        delay_s=0.010,
        impact_energy_j=1.0,
    )
    assert weak_id == 1
    assert tied_id is None
    assert [voice.voice_id for voice in renderer._voices] == [weak_id]

    strong_id = renderer.play_transient(
        "tone",
        StaticAttachment((1.0, 0.0, 0.0)),
        delay_s=0.010,
        impact_energy_j=2.0,
    )
    assert strong_id == 2
    assert [voice.voice_id for voice in renderer._voices] == [strong_id]
    assert renderer._voices[0].impact_priority == 2.0


def test_explicit_voices_are_protected_and_overflow_is_synchronous() -> None:
    """Explicit dry intervals reject at play time and cannot be evicted."""

    protected, _ = _renderer(
        (1.0, 0.0, 0.0),
        loop=True,
        max_active_voices=1,
    )
    explicit_id = protected.play("source", delay_s=0.010)
    impact_id = protected.play_transient(
        "tone",
        StaticAttachment((1.0, 0.0, 0.0)),
        delay_s=0.010,
        impact_energy_j=100.0,
    )
    assert impact_id is None
    assert [voice.voice_id for voice in protected._voices] == [explicit_id]
    assert protected._voices[0].impact_priority is None

    occupied, _ = _renderer((1.0, 0.0, 0.0), max_active_voices=1)
    impact_id = occupied.play_transient(
        "tone",
        StaticAttachment((1.0, 0.0, 0.0)),
        delay_s=0.010,
        impact_energy_j=1.0,
    )
    with pytest.raises(RuntimeError, match="active voice capacity"):
        occupied.play("source", delay_s=0.010)
    assert [voice.voice_id for voice in occupied._voices] == [impact_id]

    same_block, _ = _renderer(
        (1.0, 0.0, 0.0),
        max_active_voices=1,
        clip_sample_count=32,
    )
    same_block.play("source")
    with pytest.raises(RuntimeError, match="active voice capacity"):
        same_block.play("source", delay_s=0.004)


def test_retained_render_tail_does_not_consume_a_dry_voice_slot() -> None:
    """A finished clip can retain FIR state while a new dry voice begins."""

    renderer, scene = _renderer(
        (1.0, 0.0, 0.0),
        max_active_voices=1,
        clip_sample_count=32,
    )
    renderer.play("source")
    renderer.render_substep(scene)
    assert len(renderer._voices) == 1
    assert renderer.active_voice_count == 0

    assert renderer.play("source") == 2
    output = renderer.render_substep(scene)
    assert np.all(np.isfinite(output))
    assert len(renderer._voices) == 2

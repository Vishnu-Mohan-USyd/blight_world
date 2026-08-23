"""Tests for scheduling, motion, distance, and binaural rendering."""

from __future__ import annotations

import numpy as np
import pytest

from mimo_audition import (
    AuditionConfig,
    AudioSourceSpec,
    ContinuousEmission,
    ContinuousSoundConfig,
    StaticAttachment,
)
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
    continuous: ContinuousSoundConfig | None = None,
) -> tuple[BinauralRenderer, _Scene]:
    config = AuditionConfig.quality_48k(
        enable_occlusion=enable_occlusion,
        enable_first_order_reflections=False,
        enable_late_reverb=False,
        max_active_voices=max_active_voices,
        continuous=continuous,
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


def _emission(
    stream_id: str,
    *,
    priority: float = 0.0,
    value: float = 1.0,
) -> ContinuousEmission:
    """Return one exact 48 kHz continuous fixture block."""

    return ContinuousEmission(
        stream_id,
        "fixture",
        StaticAttachment((1.0, 0.0, 0.0)),
        np.full(240, value, dtype=np.float32),
        priority,
    )


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


def test_continuous_stream_preserves_path_state_without_block_seam() -> None:
    """Constant adjacent emissions use one state and remain continuous at 5 ms."""

    renderer, scene = _renderer(
        (1.0, 0.0, 0.0),
        continuous=ContinuousSoundConfig(),
    )
    emission = _emission("steady")
    blocks = []
    path_identity: tuple[int, ...] | None = None
    voice_identity: int | None = None
    for _ in range(5):
        blocks.append(renderer.render_substep(scene, (emission,)))
        voice = renderer._continuous_voices["steady"]
        current_paths = tuple(id(path) for path in voice.paths)
        if path_identity is None:
            path_identity = current_paths
            voice_identity = id(voice)
        else:
            assert current_paths == path_identity
            assert id(voice) == voice_identity

    pressure = np.concatenate(blocks, axis=1)
    for boundary in (2 * 240, 3 * 240, 4 * 240):
        np.testing.assert_allclose(
            pressure[:, boundary],
            pressure[:, boundary - 1],
            rtol=0,
            atol=1e-6,
        )
    assert renderer.active_voice_count == 0
    assert renderer.continuous_active_voice_count == 1
    assert renderer.continuous_active_stream_ids == ("steady",)
    assert renderer.continuous_retained_voice_count == 1


def test_continuous_immutable_block_replays_bit_exactly() -> None:
    """Immutable emission storage leaves deterministic rendering bit-exact."""

    renderer, scene = _renderer(
        (1.0, 0.0, 0.0),
        continuous=ContinuousSoundConfig(),
    )
    caller_samples = np.linspace(-0.25, 0.25, 240, dtype=np.float32)
    emission = ContinuousEmission(
        "immutable",
        "fixture",
        StaticAttachment((1.0, 0.0, 0.0)),
        caller_samples,
    )
    first = renderer.render_substep(scene, (emission,))

    caller_samples.fill(1.0)
    renderer.reset()
    replay = renderer.render_substep(scene, (emission,))

    np.testing.assert_array_equal(first, replay)


def test_continuous_stream_tracks_moving_attachment_with_stable_state() -> None:
    """Persistent spatial state follows a source across binaural directions."""

    renderer, scene = _renderer(
        (0.0, 1.0, 0.0),
        continuous=ContinuousSoundConfig(),
    )
    emission = _emission("moving")
    left_output = np.zeros((2, 240))
    for _ in range(4):
        left_output = renderer.render_substep(scene, (emission,))
    voice = renderer._continuous_voices["moving"]
    voice_identity = id(voice)
    path_identity = tuple(id(path) for path in voice.paths)

    scene.source_position[:] = (0.0, -1.0, 0.0)
    renderer.render_substep(scene, (emission,))
    right_output = renderer.render_substep(scene, (emission,))

    assert np.max(np.abs(left_output[0])) > np.max(np.abs(left_output[1]))
    assert np.max(np.abs(right_output[1])) > np.max(np.abs(right_output[0]))
    assert id(renderer._continuous_voices["moving"]) == voice_identity
    assert tuple(
        id(path) for path in renderer._continuous_voices["moving"].paths
    ) == path_identity
    np.testing.assert_array_equal(
        renderer._continuous_voices["moving"].previous_source_position_m,
        scene.source_position,
    )


def test_continuous_priority_admission_is_deterministic_top_four() -> None:
    """Priority descending and stream ID ascending select the bounded pool."""

    renderer, scene = _renderer(
        (1.0, 0.0, 0.0),
        continuous=ContinuousSoundConfig(max_continuous_voices=4),
    )
    emissions = tuple(
        _emission(stream_id, priority=priority)
        for stream_id, priority in (
            ("z", 10),
            ("a", 10),
            ("m", 5),
            ("b", 5),
            ("c", 4),
            ("d", 1),
        )
    )
    renderer.render_substep(scene, emissions)

    assert renderer.continuous_active_stream_ids == ("a", "z", "b", "m")
    assert renderer.continuous_dropped_stream_ids == ("c", "d")
    assert renderer.continuous_active_voice_count == 4
    assert renderer.continuous_retained_voice_count == 4

    replacement = (
        _emission("c", priority=20),
        _emission("a", priority=10),
        _emission("z", priority=10),
        _emission("b", priority=5),
    )
    renderer.render_substep(scene, replacement)
    assert renderer.continuous_active_stream_ids == ("c", "a", "z", "b")
    assert renderer.continuous_active_voice_count == 4
    assert renderer.continuous_retained_voice_count == 5


def test_clip_and_continuous_capacities_are_independent() -> None:
    """Eight clip voices and four continuous streams render concurrently."""

    renderer, scene = _renderer(
        (1.0, 0.0, 0.0),
        loop=True,
        max_active_voices=8,
        continuous=ContinuousSoundConfig(max_continuous_voices=4),
    )
    for _ in range(8):
        renderer.play("source")
    output = renderer.render_substep(
        scene,
        tuple(_emission(f"stream-{index}") for index in range(4)),
    )

    assert np.all(np.isfinite(output))
    assert renderer.active_voice_count == 8
    assert renderer.continuous_active_voice_count == 4
    assert renderer.continuous_retained_voice_count == 4


def test_continuous_disappearance_releases_once_and_reset_clears_state() -> None:
    """Unexpected loss fades the last sample, retains tails, and resets exactly."""

    renderer, scene = _renderer(
        (1.0, 0.0, 0.0),
        continuous=ContinuousSoundConfig(
            unexpected_disappearance_fade_seconds=0.005
        ),
    )
    emission = _emission("vanishing", value=0.25)
    first = renderer.render_substep(scene, (emission,))
    voice = renderer._continuous_voices["vanishing"]
    assert voice.dry_end_sample is None

    released = renderer.render_substep(scene)
    voice = renderer._continuous_voices["vanishing"]
    assert voice.release_total_samples == 240
    assert voice.release_elapsed_samples == 240
    assert voice.dry_end_sample == 480
    assert renderer.continuous_active_voice_count == 0
    assert renderer.continuous_retained_voice_count == 1
    assert np.all(np.isfinite(released))

    renderer.reset()
    assert renderer.sample_cursor == 0
    assert renderer.continuous_active_voice_count == 0
    assert renderer.continuous_retained_voice_count == 0
    assert renderer.continuous_active_stream_ids == ()
    assert renderer.continuous_dropped_stream_ids == ()
    replay = renderer.render_substep(scene, (emission,))
    np.testing.assert_array_equal(first, replay)


def test_continuous_rate_identity_and_duplicate_ids_fail_closed() -> None:
    """Renderer rate and per-substep stream identity remain unambiguous."""

    renderer, scene = _renderer(
        (1.0, 0.0, 0.0),
        continuous=ContinuousSoundConfig(),
    )
    wrong_rate = ContinuousEmission(
        "wrong-rate",
        "fixture",
        StaticAttachment((1.0, 0.0, 0.0)),
        np.ones(120, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="renderer sample rate"):
        renderer.render_substep(scene, (wrong_rate,))
    duplicate = _emission("duplicate")
    with pytest.raises(ValueError, match="duplicate continuous stream_id"):
        renderer.render_substep(scene, (duplicate, duplicate))

    stable = _emission("stable-identity")
    renderer.render_substep(scene, (stable,))
    changed_kind = ContinuousEmission(
        "stable-identity",
        "different-kind",
        stable.attachment,
        stable.samples_pa_at_1m,
    )
    with pytest.raises(ValueError, match="stream identity changed"):
        renderer.render_substep(scene, (changed_kind,))


def test_continuous_24khz_block_matches_renderer_substep() -> None:
    """The training-rate pool consumes exactly 120 float32 dry samples."""

    config = AuditionConfig.training_24k(
        enable_occlusion=False,
        enable_first_order_reflections=False,
        enable_late_reverb=False,
        continuous=ContinuousSoundConfig(),
    )
    renderer = BinauralRenderer(
        config,
        hrtf_bank=_bank(),
        clips=ClipLibrary(24_000),
    )
    scene = _Scene((1.0, 0.0, 0.0))
    emission = ContinuousEmission(
        "training-rate",
        "fixture",
        StaticAttachment((1.0, 0.0, 0.0)),
        np.ones(120, dtype=np.float32),
    )

    output = renderer.render_substep(scene, (emission,))
    assert output.shape == (2, 120)
    assert output.dtype == np.float64
    assert np.all(np.isfinite(output))
    assert renderer.sample_cursor == 120
    assert renderer.continuous_active_stream_ids == ("training-rate",)


def test_disabled_continuous_path_is_exact_and_rejects_direct_input() -> None:
    """Default render calls retain exact clip output and clip-only diagnostics."""

    implicit, implicit_scene = _renderer((1.0, 0.0, 0.0), loop=True)
    explicit, explicit_scene = _renderer((1.0, 0.0, 0.0), loop=True)
    implicit.play("source")
    explicit.play("source")
    implicit_output = implicit.render_substep(implicit_scene)
    explicit_output = explicit.render_substep(explicit_scene, ())

    np.testing.assert_array_equal(implicit_output, explicit_output)
    assert implicit.active_voice_count == explicit.active_voice_count == 1
    assert implicit.continuous_active_voice_count == 0
    assert implicit.continuous_retained_voice_count == 0
    with pytest.raises(RuntimeError, match="continuous sound is disabled"):
        implicit.render_substep(implicit_scene, (_emission("disabled"),))

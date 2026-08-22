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
) -> tuple[BinauralRenderer, _Scene]:
    config = AuditionConfig.quality_48k(
        enable_occlusion=enable_occlusion,
        enable_first_order_reflections=False,
        enable_late_reverb=False,
    )
    clips = ClipLibrary(48_000)
    clips.register("tone", np.ones(2_000), reference_peak_pressure_pa_at_1m=1.0)
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

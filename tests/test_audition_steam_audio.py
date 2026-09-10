"""Numerical checks using the actual standalone Steam Audio 4.8.1 SDK."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mimo_audition import (AuditionConfig, AudioSourceSpec, ContinuousEmission,
                          ContinuousSoundConfig, StaticAttachment)
from mimo_audition.clips import ClipLibrary
from mimo_audition.mujoco_scene import MujocoAcousticScene
from mimo_audition.steam_audio import SteamAudioRenderer, _coordinate_space


@pytest.fixture
def acoustic_room():
    library = Path(os.environ.get("STEAM_AUDIO_LIBRARY", Path(__file__).resolve().parents[1] /
                                 "third_party/steam_audio/lib/linux-x64/libphonon.so"))
    if not library.is_file():
        pytest.skip("standalone Steam Audio SDK is not installed")
    model = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep=".005"/><worldbody>
          <geom name="floor" type="plane" size="3 3 .1"/>
          <geom name="ceiling" type="box" pos="0 0 3" size="3 3 .1"/>
          <geom name="front" type="box" pos="3 0 1.5" size=".1 3 1.5"/>
          <geom name="back" type="box" pos="-3 0 1.5" size=".1 3 1.5"/>
          <geom name="left" type="box" pos="0 3 1.5" size="3 .1 1.5"/>
          <geom name="right" type="box" pos="0 -3 1.5" size="3 .1 1.5"/>
          <body name="mimo_location"><body name="head" pos="0 0 1">
            <geom name="head" type="sphere" size=".08"/>
            <geom name="geom:left_eye1" type="sphere" size=".01" pos=".07 .03 .03"/>
            <geom name="geom:right_eye1" type="sphere" size=".01" pos=".07 -.03 .03"/>
          </body></body>
          <body name="blocker" pos=".85 2 1.06"><freejoint/>
            <geom name="blocker_geom" type="box" size=".1 .4 .4" mass="1"/>
          </body>
        </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MujocoAcousticScene(model, data)


def test_actual_sdk_shared_flight_time_and_binaural_latency(acoustic_room) -> None:
    """Both real DSP branches shift by D0; HRIR timing remains additional."""

    clips = ClipLibrary(48_000)
    clips.register("impulse", np.concatenate((np.zeros(240), np.ones(1))), reference_peak_pressure_pa_at_1m=1.)
    renderer = SteamAudioRenderer(AuditionConfig(backend="steam_audio"), clips=clips)
    try:
        position, rotation = acoustic_room.listener_pose()
        source = StaticAttachment(tuple(position + np.asarray((1.7, 0., 0.))))
        for name in ("reference", "delayed"):
            renderer.register_source(AudioSourceSpec(name, "impulse", source))
            renderer.play(name)
        renderer.render_substep(acoustic_room)
        reference, delayed = renderer._streams.values()
        assert reference.outputs.reflections.ir and delayed.outputs.reflections.ir
        assert reference.outputs.reflections.ir != delayed.outputs.reflections.ir
        # Each opaque IR publication belongs to one effect. Two coincident
        # sources produce matching stationary IRs, and a truly silent block
        # primes each source's own effect before testing propagation timing.
        listener = _coordinate_space(position, rotation)
        direct, reflected = [[], []], [[], []]
        finished = [False, False]
        for block in range(410):
            renderer._sample_cursor = (block + 1) * 240
            mono = np.zeros(240)
            if block == 0:
                mono[0] = 1
            for branch, stream, distance in ((0, reference, 0.), (1, delayed, 1.7)):
                _, finished[branch] = renderer._process(
                    stream, mono, distance, np.asarray((0., 0., -1.)), listener, 241)
                direct[branch].append(renderer._direct_out.samples.copy())
                reflected[branch].append(renderer._reflected_out.samples.copy())
        dry_reference, dry_delayed = (np.concatenate(values, axis=1) for values in direct)
        wet_reference, wet_delayed = (np.concatenate(values, axis=1) for values in reflected)
        assert renderer.binaural_latency_samples == 60
        onset = np.flatnonzero(np.max(np.abs(dry_reference), axis=0) > 1e-5)[0]
        assert onset == 60
        assert np.max(np.abs(wet_reference)) > 1e-5
        for original, shifted in ((dry_reference, dry_delayed), (wet_reference, wet_delayed)):
            np.testing.assert_allclose(shifted[:, :240], 0, atol=2e-6)
            np.testing.assert_allclose(shifted[:, 240:], original[:, :-240], atol=3e-6, rtol=3e-4)
        assert all(finished)
        assert np.count_nonzero(wet_delayed[:, -240:]) == 0
    finally:
        renderer.close()


def test_continuous_resume_advances_the_reflection_history(acoustic_room) -> None:
    """A silent gap and resume match uninterrupted zero-valued dry blocks."""

    config = AuditionConfig(backend="steam_audio", continuous=ContinuousSoundConfig(
        max_continuous_voices=2, unexpected_disappearance_fade_seconds=0))
    renderer = SteamAudioRenderer(config)
    try:
        position, rotation = acoustic_room.listener_pose()
        attachment = StaticAttachment(tuple(position + (.85, 0., 0.)))
        silent = np.zeros(240, dtype=np.float32)
        renderer.render_substep(acoustic_room, tuple(
            ContinuousEmission(name, "fixture", attachment, silent)
            for name in ("reference", "resumed")))
        reference = renderer._streams[('continuous', 'reference')]
        resumed = renderer._streams[('continuous', 'resumed')]
        listener = _coordinate_space(position, rotation)
        outputs = [[], []]
        for block in range(12):
            renderer._sample_cursor = (block + 1) * 240
            mono = np.zeros(240)
            if block in (0, 5):
                mono[0] = 1
            for branch, stream in enumerate((reference, resumed)):
                dry_end = 241 if branch == 1 and 1 <= block < 5 else None
                output, _ = renderer._process(stream, mono, .85,
                    np.asarray((0., 0., -1.)), listener, dry_end, can_resume=True)
                outputs[branch].append(output)
        np.testing.assert_allclose(np.concatenate(outputs[0], axis=1),
                                   np.concatenate(outputs[1], axis=1), atol=3e-6, rtol=3e-4)
    finally:
        renderer.close()


def test_dynamic_occluder_uses_each_block_pose_and_omits_head(acoustic_room) -> None:
    """A rigid body moved between blocks changes actual SDK ray occlusion."""

    clips = ClipLibrary(48_000)
    clips.register("tone", np.ones(240), reference_peak_pressure_pa_at_1m=1.)
    config = AuditionConfig(backend="steam_audio", enable_first_order_reflections=False,
                            enable_late_reverb=False)
    renderer = SteamAudioRenderer(config, clips=clips)
    try:
        position, _ = acoustic_room.listener_pose()
        renderer.register_source(AudioSourceSpec("source", "tone",
            StaticAttachment(tuple(position + (1.7, 0., 0.))), loop=True))
        renderer.play("source")
        renderer.render_substep(acoustic_room)
        stream = next(iter(renderer._streams.values()))
        assert stream.outputs.direct.occlusion == 1
        assert len(renderer._geometry.static_meshes) == 6
        assert len(renderer._geometry.instances) == 1
        acoustic_room.data.qpos[:3] = position + (.85, 0., 0.)
        mujoco.mj_forward(acoustic_room.model, acoustic_room.data)
        renderer.render_substep(acoustic_room)
        assert stream.outputs.direct.occlusion == 0
        acoustic_room.data.qpos[1] = 2
        mujoco.mj_forward(acoustic_room.model, acoustic_room.data)
        renderer.render_substep(acoustic_room)
        assert stream.outputs.direct.occlusion == 1
    finally:
        renderer.close()

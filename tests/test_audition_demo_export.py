"""Focused contract tests for the deterministic audiovisual exporter."""

from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

import numpy as np
import pytest

from blight_world import AuditionConfig, BlightWorldEnv, ContinuousSoundConfig
from tools.export_audition_demo import (
    AUDIO_BLOCKS,
    AUDIO_SAMPLES_PER_EAR,
    AUDIO_SAMPLE_RATE_HZ,
    BLOCK_DRAG_START_SECONDS,
    BLOCK_DRAG_STOP_SECONDS,
    CAMERA_AZIMUTH_DEGREES,
    CAMERA_DISTANCE,
    CAMERA_ELEVATION_DEGREES,
    CAMERA_LOOKAT,
    CYLINDER_DRAG_START_SECONDS,
    DURATION_SECONDS,
    PHYSICS_SUBSTEPS,
    PHYSICS_TIMESTEP_SECONDS,
    SAMPLES_PER_FRAME,
    VIDEO_FRAME_RATE_HZ,
    VIDEO_FRAMES,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    EXPECTED_BALL_IMPACTS,
    _apply_substep_fixture,
    _free_joint_addresses,
    _ild_db,
    _prepare_demo_state,
    export_demo,
    pose_values,
    storyboard_phase,
)


def test_demo_timing_camera_and_storyboard_are_frozen() -> None:
    """Video/audio clocks and all phase boundaries remain exact."""

    assert PHYSICS_SUBSTEPS * PHYSICS_TIMESTEP_SECONDS == DURATION_SECONDS
    assert VIDEO_FRAMES / VIDEO_FRAME_RATE_HZ == DURATION_SECONDS
    assert AUDIO_BLOCKS * 480 == AUDIO_SAMPLES_PER_EAR
    assert AUDIO_SAMPLES_PER_EAR / AUDIO_SAMPLE_RATE_HZ == DURATION_SECONDS
    assert SAMPLES_PER_FRAME == AUDIO_SAMPLE_RATE_HZ // VIDEO_FRAME_RATE_HZ
    assert (VIDEO_WIDTH, VIDEO_HEIGHT) == (640, 360)
    assert CAMERA_LOOKAT == (0.55, 0.0, 0.55)
    assert CAMERA_DISTANCE == 2.2
    assert CAMERA_AZIMUTH_DEGREES == 135.0
    assert CAMERA_ELEVATION_DEGREES == -25.0

    assert storyboard_phase(0.0) == "establish_and_hold_ball"
    assert storyboard_phase(1.0) == "ball_release_and_impact"
    assert storyboard_phase(BLOCK_DRAG_START_SECONDS) == "block_drag"
    assert storyboard_phase(BLOCK_DRAG_STOP_SECONDS) == "block_release"
    assert storyboard_phase(CYLINDER_DRAG_START_SECONDS) == "cylinder_drag"
    assert storyboard_phase(5.6) == "cylinder_release_and_room_tail"
    assert storyboard_phase(7.995) == "cylinder_release_and_room_tail"


def test_scripted_pose_uses_the_exact_ramp_and_oscillators() -> None:
    """The neutral start and post-ramp formulas remain explicit."""

    assert pose_values(0.0) == {
        "head_swivel": 0.0,
        "right_shoulder_horizontal": 0.0,
        "right_shoulder_ad_ab": 0.0,
        "right_elbow": -0.0,
    }
    time_seconds = 0.8
    ramp = 0.5 - 0.5 * cos(pi * min(time_seconds / 0.8, 1.0))
    pose = pose_values(time_seconds)
    assert ramp == 1.0
    assert pose["head_swivel"] == pytest.approx(
        0.35 * sin(2.0 * pi * time_seconds / 4.0)
    )
    assert pose["right_shoulder_horizontal"] == 0.30
    assert pose["right_shoulder_ad_ab"] == pytest.approx(
        0.42 + 0.12 * sin(2.0 * pi * time_seconds / 1.6)
    )
    assert pose["right_elbow"] == pytest.approx(
        -0.70 + 0.12 * sin(2.0 * pi * time_seconds / 1.2)
    )


def test_demo_fixture_clamps_root_and_drag_velocities() -> None:
    """The manual substep fixture applies only the scripted object motion."""

    environment = BlightWorldEnv(
        width=64,
        height=48,
        audition=AuditionConfig.quality_48k(
            continuous=ContinuousSoundConfig()
        ),
    )
    try:
        addresses = _prepare_demo_state(environment)
        unrelated_joint = environment.model.joint(
            "robot:right_shoulder_rotation"
        )
        unrelated_dof = int(np.asarray(unrelated_joint.dofadr).item())
        environment.data.qvel[unrelated_dof] = 0.123
        block_qpos, block_dof = _free_joint_addresses(environment, "block")
        environment.data.qpos[block_qpos] = 0.9
        _apply_substep_fixture(environment, addresses, BLOCK_DRAG_START_SECONDS)
        cylinder_qpos, cylinder_dof = _free_joint_addresses(
            environment,
            "cylinder",
        )
        mimo_qpos, mimo_dof = _free_joint_addresses(
            environment,
            "mimo_location",
        )
        np.testing.assert_array_equal(
            environment.data.qpos[block_qpos : block_qpos + 3],
            np.asarray((0.9, 0.38, 0.699)),
        )
        assert environment.data.qvel[block_dof] == 0.30
        assert environment.data.qvel[cylinder_dof] == 0.0
        np.testing.assert_array_equal(
            environment.data.qpos[mimo_qpos : mimo_qpos + 7],
            np.asarray((0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0)),
        )
        assert not np.any(environment.data.qvel[mimo_dof : mimo_dof + 6])
        for joint_name in (
            "robot:head_swivel",
            "robot:right_shoulder_horizontal",
            "robot:right_shoulder_ad_ab",
            "robot:right_elbow",
        ):
            dof = int(np.asarray(environment.model.joint(joint_name).dofadr).item())
            assert environment.data.qvel[dof] == 0.0
        assert environment.data.qvel[unrelated_dof] == 0.123

        environment.data.qpos[block_qpos] = 0.9
        _apply_substep_fixture(environment, addresses, BLOCK_DRAG_STOP_SECONDS)
        assert environment.data.qpos[block_qpos] == 0.9
        assert environment.data.qvel[block_dof] == 0.0

        environment.data.qpos[cylinder_qpos] = 0.8
        _apply_substep_fixture(environment, addresses, CYLINDER_DRAG_START_SECONDS)
        assert environment.data.qpos[cylinder_qpos] == 0.8
        assert environment.data.qvel[cylinder_dof] == 0.30
    finally:
        environment.close()


def test_ild_metric_preserves_pressure_ratio_and_sign() -> None:
    pressure = np.vstack(
        (
            np.full(48_000, 0.2, dtype=np.float32),
            np.full(48_000, 0.1, dtype=np.float32),
        )
    )
    assert _ild_db(pressure, 0.0, 1.0) == pytest.approx(
        20.0 * np.log10(2.0)
    )


def test_full_demo_timeline_replays_public_audio_and_scrape_windows(
    tmp_path: Path,
) -> None:
    """The complete exporter retains its public pressure and contact identity."""

    manifest = export_demo(tmp_path / "audition_demo")

    assert manifest["raw_sources"][
        "audio_pressure_pa_channels_first_float32_sha256"
    ] == "20daab84d6e896b4ba4477dce4499744af9001198f0863631aab300fd24e026e"
    assert round(
        manifest["media_pressure_mapping"]["peak_pressure_pa"],
        4,
    ) == 0.7738
    assert manifest["configuration"]["behavior_label"] == (
        "kinematically_constrained_external_camera_audition_diagnostic"
    )
    assert manifest["configuration"]["contact_physics"] == (
        "current production scene contact pairs"
    )
    ball_events = [
        event
        for event in manifest["storyboard_evidence"]["impact_events"]
        if event["clip_id"].startswith("impact_wood_rubber_")
    ]
    assert len(ball_events) == 4
    assert all(event["accepted"] for event in ball_events)
    for event, (expected_time, expected_clip, expected_energy) in zip(
        ball_events,
        EXPECTED_BALL_IMPACTS,
        strict=True,
    ):
        assert event["clip_id"] == expected_clip
        assert event["simulation_time_s"] == pytest.approx(expected_time, abs=1e-12)
        assert event["impact_energy_j"] == pytest.approx(
            expected_energy,
            abs=5e-13,
        )
    windows = manifest["storyboard_evidence"]["scrape_windows"]
    assert windows["scrape:6:11"] == {
        "first_active_time_s": pytest.approx(2.215),
        "last_active_time_s": pytest.approx(3.405),
        "active_substeps": 239,
    }
    assert windows["scrape:6:13"] == {
        "first_active_time_s": pytest.approx(4.415),
        "last_active_time_s": pytest.approx(5.605),
        "active_substeps": 239,
    }
    assert manifest["media_validation"]["lossless_master_matches_raw_sources"]
    for artifact in manifest["artifacts"].values():
        assert (tmp_path / "audition_demo" / artifact["path"]).is_file()

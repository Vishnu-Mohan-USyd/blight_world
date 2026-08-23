"""Focused contracts for the public binocular physics demo exporter."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from blight_world import BlightWorldEnv
from tools.export_binocular_physics_demo import (
    AUDIO_BLOCKS,
    AUDIO_SAMPLES_PER_EAR,
    BLOCK_FORCE_START_SECONDS,
    BLOCK_FORCE_STOP_SECONDS,
    DEMO_SEED,
    DURATION_SECONDS,
    ENVIRONMENT_STEP_SECONDS,
    EYE_HEIGHT,
    EYE_WIDTH,
    PHYSICS_SUBSTEPS,
    PHYSICS_TIMESTEP_SECONDS,
    PUBLIC_STEPS,
    VIDEO_FRAME_RATE_HZ,
    VIDEO_FRAMES,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    _apply_step_inputs,
    _stage_demo_state,
    _validate_model,
    export_demo,
    force_commands,
    make_demo_environment,
)


def test_timing_and_force_windows_are_exact() -> None:
    """Public-step, video, and finite-force clocks remain synchronized."""

    assert PUBLIC_STEPS * ENVIRONMENT_STEP_SECONDS == DURATION_SECONDS
    assert PHYSICS_SUBSTEPS * PHYSICS_TIMESTEP_SECONDS == DURATION_SECONDS
    assert VIDEO_FRAMES / VIDEO_FRAME_RATE_HZ == DURATION_SECONDS
    assert AUDIO_BLOCKS == PUBLIC_STEPS
    assert AUDIO_SAMPLES_PER_EAR == 240_000
    assert (EYE_WIDTH, EYE_HEIGHT) == (320, 240)
    assert (VIDEO_WIDTH, VIDEO_HEIGHT) == (640, 240)
    commands = [force_commands(step * ENVIRONMENT_STEP_SECONDS) for step in range(PUBLIC_STEPS)]
    assert sum(block != 0.0 for block, _ in commands) == 20
    assert sum(cylinder != 0.0 for _, cylinder in commands) == 15
    assert force_commands(BLOCK_FORCE_START_SECONDS) == (1.0, 0.0)
    assert force_commands(BLOCK_FORCE_STOP_SECONDS) == (0.0, 0.0)


def test_contact_pairs_and_demo_weld_are_narrowly_scoped() -> None:
    """Contact pairs are universal, while torso support exists only in the demo."""

    baseline = BlightWorldEnv(width=16, height=12)
    demo = make_demo_environment()
    try:
        assert (
            mujoco.mj_name2id(
                baseline.model,
                mujoco.mjtObj.mjOBJ_EQUALITY,
                "demo_mimo_root_support",
            )
            == -1
        )
        model = _validate_model(demo)
        assert set(model["contact_pairs"]) == {
            "ball_table",
            "block_table",
            "cylinder_table",
        }
        assert model["demo_world_weld"] == {
            "name": "demo_mimo_root_support",
            "equality_id": model["demo_world_weld"]["equality_id"],
            "active": True,
            "solref": [0.01, 1.0],
            "solimp": [0.99, 0.999, 0.001, 0.5, 2.0],
        }
    finally:
        baseline.close()
        demo.close()


def test_one_time_stage_and_step_inputs_do_not_rewrite_object_state() -> None:
    """After staging, the controller changes only applied forces and actions."""

    environment = make_demo_environment()
    try:
        addresses = _stage_demo_state(environment, seed=DEMO_SEED)
        qpos_before = environment.data.qpos.copy()
        qvel_before = environment.data.qvel.copy()
        expected_hip = np.clip(
            np.sum(environment.data.qfrc_bias[list(addresses.hip_dof_addresses)])
            / 10.58,
            environment.action_space.low[addresses.hip_action_index],
            environment.action_space.high[addresses.hip_action_index],
        )
        expected_head = np.clip(
            environment.data.qfrc_bias[addresses.head_dof_address] / 3.3,
            environment.action_space.low[addresses.head_action_index],
            environment.action_space.high[addresses.head_action_index],
        )
        action, forces = _apply_step_inputs(
            environment,
            addresses,
            BLOCK_FORCE_START_SECONDS,
        )
        np.testing.assert_array_equal(environment.data.qpos, qpos_before)
        np.testing.assert_array_equal(environment.data.qvel, qvel_before)
        assert forces == (1.0, 0.0)
        allowed_indices = {addresses.hip_action_index, addresses.head_action_index}
        assert not np.any(
            [
                value != 0.0
                for index, value in enumerate(action)
                if index not in allowed_indices
            ]
        )
        assert action[addresses.hip_action_index] == np.float32(expected_hip)
        assert action[addresses.head_action_index] == np.float32(expected_head)
        assert environment.action_space.contains(action)
        for name, (qpos_address, dof_address) in addresses.object_qpos_dof.items():
            expected_position = {
                "ball": (1.15, 0.0, 1.20),
                "block": (0.65, 0.30, 0.699),
                "cylinder": (0.65, -0.30, 0.699),
            }[name]
            np.testing.assert_array_equal(
                environment.data.qpos[qpos_address : qpos_address + 3],
                expected_position,
            )
            np.testing.assert_array_equal(
                environment.data.qvel[dof_address : dof_address + 6],
                0.0,
            )
    finally:
        environment.close()


def test_full_export_uses_public_binocular_and_auditory_streams(tmp_path: Path) -> None:
    """The encoded lossless master is byte-identical to public observations."""

    manifest = export_demo(tmp_path / "binocular_physics_demo")

    assert manifest["configuration"]["public_environment_steps"] == 500
    assert manifest["configuration"]["physics_diagnostic_interval_seconds"] == 0.005
    assert manifest["configuration"]["physics_diagnostic_rows_per_object"] == 1_000
    assert manifest["configuration"]["post_stage_object_qpos_qvel_writes"] == 0
    assert manifest["configuration"]["countergravity_after_staging"] is False
    public = manifest["public_source_validation"]
    assert public["every_observation_in_space"] is True
    assert public["panel_byte_equal_frame_count"] == 250
    assert public["left_right_nonidentical_frame_count"] == 250
    assert public["left_temporally_changed_transitions"] == 249
    assert public["right_temporally_changed_transitions"] == 249
    visibility = public["visibility"]
    assert visibility["all_scheduled_targets_visible_in_both_eyes"] is True
    assert visibility["table_in_both_eye_views_every_frame"] is True
    expected_frames = {"ball": 32, "block": 25, "cylinder": 22}
    minimum_masks = {"ball": 192, "block": 768, "cylinder": 768}
    for target, frame_count in expected_frames.items():
        for eye_name in ("eye_left", "eye_right"):
            summary = visibility["target_summary"][target][eye_name]
            assert summary["frame_count"] == frame_count
            assert summary["minimum_mask_pixels"] >= minimum_masks[target]
            assert summary["minimum_depth_m"] > 0.0
            assert summary["maximum_absolute_horizontal_degrees"] <= 31.0
            assert summary["maximum_absolute_vertical_degrees"] <= 23.0
    for eye_name in ("eye_left", "eye_right"):
        table = visibility["table_summary"][eye_name]
        assert table["frame_count"] == 250
        assert table["minimum_depth_m"] > 0.0
        assert table["maximum_absolute_vertical_degrees"] <= 14.0
    assert manifest["media_validation"]["lossless_master_matches_public_sources"]
    support = manifest["physics_validation"]["support"]
    assert support["maximum_root_position_error_m"] <= 0.002
    assert support["external_torso_contact_count"] == 0
    audition = manifest["audition_validation"]
    assert audition["accepted_impact_count"] == 6
    assert audition["rejected_impact_count"] == 0
    assert audition["source_projection_count"] == 0
    assert audition["continuous_dropped_stream_ids"] == []
    assert manifest["physics_validation"]["objects"]["cylinder"][
        "state_at_5s"
    ] == "near_rest_still_dissipating"
    ball = manifest["physics_validation"]["objects"]["ball"]
    assert [
        apex["time_s"] for apex in ball["rebound_apexes"]
    ] == [0.4600000000000003, 0.6200000000000004, 0.6550000000000005]
    np.testing.assert_allclose(
        [apex["clearance_above_rest_m"] for apex in ball["rebound_apexes"]],
        (0.075554689, 0.001735113, 0.000062188),
        rtol=0,
        atol=1e-9,
    )
    flight_substeps = [
        apex["flight_substep_count"] for apex in ball["rebound_apexes"]
    ]
    assert flight_substeps == [49, 8, 1]
    np.testing.assert_allclose(
        [apex["flight_duration_s"] for apex in ball["rebound_apexes"]],
        np.asarray(flight_substeps) * PHYSICS_TIMESTEP_SECONDS,
        rtol=0,
        atol=np.finfo(np.float64).eps,
    )
    assert ball["settle_time_s"] == 0.6650000000000005
    assert ball["max_mechanical_after_first_contact_j"] < ball[
        "initial_mechanical_energy_j"
    ]
    for artifact in manifest["artifacts"].values():
        assert (tmp_path / "binocular_physics_demo" / artifact["path"]).is_file()

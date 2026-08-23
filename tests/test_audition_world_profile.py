"""Focused live-world regressions for the reproducible audition profile."""

from __future__ import annotations

from tools.profile_audition_world import (
    profile_block_occlusion,
    profile_direct_path_distance,
    profile_directional_drops,
    profile_material_matrix,
    profile_mirrored_feet,
    profile_moving_ball_crossing,
)


EXPECTED_MATERIAL_FAMILIES = {
    "hard_plastic",
    "hard_rubber",
    "hard_soft",
    "hard_wood",
    "plastic_soft",
    "rubber_plastic",
    "rubber_soft",
    "wood_plastic",
    "wood_rubber",
    "wood_soft",
    "wood_wood",
}


def test_physical_direction_distance_and_mirrored_feet() -> None:
    """Live impacts retain distance law and reverse lateral cue signs."""

    directional = profile_directional_drops()
    direct = profile_direct_path_distance()
    feet = profile_mirrored_feet()

    assert directional["equal_energy"] is True
    assert directional["left_right_ild_sign_reversal"] is True
    assert directional["repeat_exact"] is True
    assert -20.2 < direct["fit_db_per_decade"] < -19.8
    assert direct["stable_ild"] is True
    assert direct["stable_itd"] is True
    assert feet["cue_sign_reversal"] is True
    assert feet["repeat_exact"] is True


def test_physical_material_matrix_is_complete_and_resting_silent() -> None:
    """Physical contacts cover all material sounds without resting retriggers."""

    result = profile_material_matrix()

    assert set(result["covered_material_families"]) == EXPECTED_MATERIAL_FAMILIES
    assert len(result["recipes"]) == 13
    assert result["repeat_exact"] is True
    assert result["resting_control"] == {
        "event_count": 0,
        "waveform_peak_pa": 0.0,
        "peak_reserved_voices": 0,
    }


def test_moving_body_source_and_real_block_occlusion() -> None:
    """Actual MuJoCo motion updates cues and geometry changes direct sound."""

    moving = profile_moving_ball_crossing()
    occlusion = profile_block_occlusion()

    assert moving["start"]["azimuth_deg"] < 0 < moving["end"]["azimuth_deg"]
    assert (
        moving["mean_endpoint_ild_db"]["start"]
        * moving["mean_endpoint_ild_db"]["end"]
        < 0
    )
    assert moving["impact_event_count"] == 0
    assert occlusion["measurements"]["blocked"]["ray_occluded"] is True
    assert occlusion["measurements"]["clear"]["ray_occluded"] is False
    assert 0 < occlusion["blocked_over_clear_rms"] < 0.5
    assert (
        0
        < occlusion["blocked_over_clear_high_frequency"]
        < occlusion["blocked_over_clear_rms"]
    )

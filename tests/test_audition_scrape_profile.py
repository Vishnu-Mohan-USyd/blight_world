"""Focused regressions for the reproducible live-world scrape profile."""

from __future__ import annotations

import numpy as np
import pytest

from tools.profile_audition_scrape import (
    _geom_pair,
    _new_environment,
    _run_supported_cell,
    profile_mirrored_scrapes,
    profile_scrape_matrix,
    profile_scrape_replay_and_capacity,
    profile_scrape_transitions,
)


def _block_floor_arguments() -> dict[str, object]:
    return {
        "fixture_name": "block_floor",
        "body_name": "block",
        "support_geom": "floor",
        "position_m": (1.8, 0.0, 0.099),
        "material_pair": ("hard", "wood"),
        "speed_m_s": 0.4,
        "extra_downward_force_n": 0.0,
    }


def test_live_scrape_matrix_profile_covers_transport_and_admission() -> None:
    """All 24 live cells prove contact, admission, and public transport."""

    result = profile_scrape_matrix()

    assert result["cell_count"] == 24
    assert result["all_contacts_continuous"]
    assert result["all_target_streams_admitted"]
    assert result["public_observation_transport_valid"]
    assert set(result["material_source_spectra"]) == {
        "hard_plastic",
        "hard_wood",
        "wood_plastic",
        "wood_wood",
    }
    for cell in result["cells"]:
        assert cell["received_transport"] == "gym_observation_float32"
        assert cell["post_render_observation_reads"] == 19
        assert cell["steady_observation_blocks"] == 10
        assert cell["public_observation_conformant"]
        assert cell["observation_reads_idempotent"]
        assert cell["solved_measurement_present_every_analyzed_substep"]
        assert cell["target_source_block_present_every_analyzed_substep"]
        assert cell["target_material_verified"]
        assert cell["target_stream_admitted_every_analyzed_substep"]
        assert cell["diagnostic_matches_solved_contact_controls"]


def test_supported_profile_reads_public_observation_idempotently() -> None:
    """The reported receive trace is exactly ten steady Gym audio blocks."""

    environment = _new_environment()
    assert environment.audition is not None
    renderer = environment.audition.renderer
    original_get_obs = environment._get_obs
    original_render = renderer.render_substep
    render_count = 0
    captured: list[np.ndarray] = []
    read_invariants: list[bool] = []

    def counting_render(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal render_count
        output = original_render(*args, **kwargs)  # type: ignore[arg-type]
        render_count += 1
        return output

    def recording_get_obs() -> dict[str, np.ndarray]:
        time_before = float(environment.data.time)
        cursor_before = renderer.sample_cursor
        render_count_before = render_count
        observation = original_get_obs()
        if cursor_before > 0 and not environment.audition._pending_substeps:
            audition = observation["audition"]
            captured.append(audition.copy(order="C"))
            read_invariants.append(
                float(environment.data.time) == time_before
                and renderer.sample_cursor == cursor_before
                and render_count == render_count_before
                and environment.observation_space.contains(observation)
                and audition.shape == (2, 480)
                and audition.dtype == np.float32
                and audition.flags.c_contiguous
            )
        return observation

    renderer.render_substep = counting_render  # type: ignore[method-assign]
    environment._get_obs = recording_get_obs  # type: ignore[method-assign]
    try:
        _, _, received, _ = _run_supported_cell(
            environment,
            **_block_floor_arguments(),  # type: ignore[arg-type]
        )
    finally:
        environment._get_obs = original_get_obs  # type: ignore[method-assign]
        renderer.render_substep = original_render  # type: ignore[method-assign]
        environment.close()

    assert render_count == 38
    assert len(captured) == 19
    assert all(read_invariants)
    np.testing.assert_array_equal(received, np.concatenate(captured[9:], axis=1))


def test_supported_profile_fails_on_solved_measurement_dropout() -> None:
    """A retained gate diagnostic cannot conceal a missing solved contact."""

    environment = _new_environment()
    assert environment.audition is not None
    tracker = environment.audition._scrapes
    assert tracker is not None
    pair = _geom_pair(environment, "floor", "block")
    original_measurements = tracker._measurements
    target_time_calls = 0

    def dropping_measurement() -> dict[tuple[int, int], object]:
        nonlocal target_time_calls
        measurements = original_measurements()
        if np.isclose(environment.data.time, 0.095, rtol=0, atol=1e-12):
            target_time_calls += 1
            if target_time_calls == 2:
                measurements.pop(pair, None)
        return measurements  # type: ignore[return-value]

    tracker._measurements = dropping_measurement  # type: ignore[method-assign]
    try:
        with pytest.raises(AssertionError, match="measurement disappeared"):
            _run_supported_cell(
                environment,
                **_block_floor_arguments(),  # type: ignore[arg-type]
            )
    finally:
        tracker._measurements = original_measurements  # type: ignore[method-assign]
        environment.close()
    assert target_time_calls == 2


def test_supported_profile_fails_on_expected_material_mismatch() -> None:
    """The fixture's expected material identity is independently checked."""

    environment = _new_environment()
    arguments = _block_floor_arguments()
    arguments["material_pair"] = ("hard", "plastic")
    try:
        with pytest.raises(AssertionError, match="material identity changed"):
            _run_supported_cell(
                environment,
                **arguments,  # type: ignore[arg-type]
            )
    finally:
        environment.close()


def test_supported_profile_fails_when_target_stream_is_not_admitted() -> None:
    """A generated dry block is insufficient without renderer admission."""

    environment = _new_environment()
    assert environment.audition is not None
    renderer = environment.audition.renderer
    target_stream = "scrape:0:11"
    original_render = renderer.render_substep

    def hiding_admission(*args: object, **kwargs: object) -> np.ndarray:
        output = original_render(*args, **kwargs)  # type: ignore[arg-type]
        renderer._continuous_active_stream_ids = tuple(
            stream_id
            for stream_id in renderer.continuous_active_stream_ids
            if stream_id != target_stream
        )
        return output

    renderer.render_substep = hiding_admission  # type: ignore[method-assign]
    try:
        with pytest.raises(AssertionError, match="stream was not admitted"):
            _run_supported_cell(
                environment,
                **_block_floor_arguments(),  # type: ignore[arg-type]
            )
    finally:
        renderer.render_substep = original_render  # type: ignore[method-assign]
        environment.close()


def test_live_scrape_transition_profile() -> None:
    """Reversal, release, lift, and drop-to-slide retain their contracts."""

    transitions = profile_scrape_transitions()

    assert transitions["reversal"]["direction_cosine"] == -1.0
    assert transitions["reversal"]["speed_norm_ratio"] == 1.0
    assert transitions["stop"]["phase_by_substep"] == [
        "release",
        "release",
        "release",
        "release",
        "release",
        "release",
        None,
        None,
    ]
    assert transitions["lift"]["phase_by_substep"] == [
        "active",
        "release",
        "release",
        "release",
        "release",
        "release",
        "release",
        None,
    ]
    assert transitions["drop"]["impact_count"] == 1
    assert transitions["drop"]["impact"]["clip_id"].startswith(
        "impact_hard_wood_"
    )


def test_live_scrape_mirror_profile() -> None:
    """Mirrored sources reverse cues under the proven physics tolerances."""

    result = profile_mirrored_scrapes()
    fixtures = result["fixtures"]

    assert set(fixtures) == {
        "block_floor",
        "block_table",
        "cylinder_floor",
        "cylinder_table",
    }
    assert all(fixture["ild_sign_reversal"] for fixture in fixtures.values())
    for name in ("block_floor", "block_table", "cylinder_floor"):
        assert fixtures[name]["control_mirror_within_float64_roundoff"]
        assert fixtures[name]["control_mirror_max_ulp_allowed"] == 16
    assert not fixtures["cylinder_table"][
        "control_mirror_within_float64_roundoff"
    ]
    assert fixtures["cylinder_table"]["control_mirror_max_ulp_allowed"] is None


def test_live_scrape_replay_and_capacity_profile() -> None:
    """Same-side replay is exact while clip and scrape pools stay independent."""

    result = profile_scrape_replay_and_capacity()

    assert result["same_side_seeded_replay_byte_exact"]
    assert result["source_waveform_sha256"] == (
        "82ca449de87f639033794ef2dd5f8383d4bcf3ec1efb91d9185cea375206c642"
    )
    assert result["received_waveform_sha256"] == (
        "e6e0187e4e7809f365b11f34942086f4aac65995f79fb29ea318ae1abf4726c0"
    )
    assert result["clip_active_voices"] == 8
    assert result["continuous_active_voices"] == 4
    assert result["continuous_retained_voices"] == 4
    assert result["combined_output_finite"]
    assert result["combined_output_rms_pa"] > 0.0

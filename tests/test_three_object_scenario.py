"""Focused contracts for the constraint-assisted three-object episode."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from blight_world.constraint_assisted_three_object import (
    ConstraintAssistedControllerState,
    ConstraintAssistedThreeObjectController,
    constraint_assisted_three_object_controller_factory,
)
from blight_world.three_object_scenario import (
    ControllerHandoffPendingError,
    PICKUP_MECHANISM,
    ROOT_MOTION_MECHANISM,
    SCENARIO_BODY_NAMES,
    ScenarioEvent,
    ScenarioEventKind,
    ScenarioObject,
    ScenarioPhase,
    ScenarioSchedule,
    ThreeObjectScenarioEnv,
    dispatch_controller,
    next_object,
)
from mimo_audition.materials import AcousticMaterialResolver


TOOLS_DIRECTORY = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIRECTORY))
try:
    from export_three_object_walk_pick_throw import (
        AUDIO_SAMPLES_PER_EAR,
        DEFAULT_OUTPUT_DIRECTORY,
        EYE_HEIGHT,
        EYE_WIDTH,
        EXPECTED_ACTION_SHA256,
        EXPECTED_OBJECT_XPOS_SHA256,
        EXPECTED_QPOS_SHA256,
        SEGMENTATION_VISIBILITY_GATES,
        VIDEO_FRAMES,
        build_plan_manifest,
        capture_segmentation_visibility_gate,
        export_scenario,
        planned_gaze_records,
        planned_phase_records,
        validate_event_coverage,
    )
finally:
    sys.path.pop(0)


def test_schedule_has_three_contiguous_calibrated_cycles() -> None:
    schedule = ScenarioSchedule.default()

    assert len(schedule.windows) == 18
    assert schedule.total_steps == 4_500
    assert schedule.duration_seconds == 45.0
    assert VIDEO_FRAMES == 2_250
    assert AUDIO_SAMPLES_PER_EAR == 2_160_000
    assert tuple(window.start_step for window in schedule.windows[1:]) == tuple(
        window.stop_step for window in schedule.windows[:-1]
    )
    expected_boundaries = {
        0: (0, ScenarioPhase.GO_TO_TARGET, ScenarioObject.BALL),
        299: (0, ScenarioPhase.GO_TO_TARGET, ScenarioObject.BALL),
        300: (0, ScenarioPhase.PICK_TARGET, ScenarioObject.BALL),
        700: (0, ScenarioPhase.INSPECT_HELD_TARGET, ScenarioObject.BALL),
        900: (0, ScenarioPhase.GLANCE_AT_NEXT_OBJECT, ScenarioObject.BALL),
        1_100: (0, ScenarioPhase.THROW_TO_FLOOR, ScenarioObject.BALL),
        1_300: (0, ScenarioPhase.SETTLE, ScenarioObject.BALL),
        1_500: (1, ScenarioPhase.GO_TO_TARGET, ScenarioObject.BLOCK),
        3_000: (2, ScenarioPhase.GO_TO_TARGET, ScenarioObject.CYLINDER),
        4_499: (2, ScenarioPhase.SETTLE, ScenarioObject.CYLINDER),
    }
    for step_index, expected in expected_boundaries.items():
        context = schedule.context_at(step_index)
        assert (context.cycle_index, context.phase, context.target) == expected
    assert schedule.context_at(700).look_target is ScenarioObject.BALL
    assert schedule.context_at(900).look_target is ScenarioObject.BLOCK
    assert next_object(ScenarioObject.CYLINDER) is ScenarioObject.BALL
    with pytest.raises(IndexError):
        schedule.context_at(4_500)


def test_plan_manifest_labels_both_assistance_mechanisms() -> None:
    plan = build_plan_manifest()
    records = planned_phase_records()
    gaze_records = planned_gaze_records()

    assert plan["status"] == "calibrated_constraint_assisted_controller_available"
    assert plan["output_directory"] == str(DEFAULT_OUTPUT_DIRECTORY)
    assert plan["clock"]["duration_seconds"] == 45.0
    assert plan["clock"]["public_steps"] == 4_500
    assert plan["clock"]["video_frames"] == 2_250
    assert plan["clock"]["audio_samples_per_ear"] == 2_160_000
    assert len(records) == 18
    assert len(gaze_records) == 12
    assert records[0]["start_time_seconds"] == 0.0
    assert records[-1]["stop_time_seconds"] == 45.0
    assert [
        (
            record["label"],
            record["subject"],
            record["start_public_step"],
            record["stop_public_step_exclusive"],
        )
        for record in gaze_records[:4]
    ] == [
        ("current_target_acquire_and_lift", "ball", 0, 700),
        ("first_other_object", "block", 700, 850),
        ("second_other_object", "cylinder", 850, 1_000),
        ("current_target_release_and_follow", "ball", 1_000, 1_500),
    ]
    physics = plan["physics_contract"]
    assert physics["root_motion_mechanism"] == ROOT_MOTION_MECHANISM
    assert physics["pickup_mechanism"] == PICKUP_MECHANISM
    assert physics["grip_active_local_time_seconds"] == [6.0, 12.0]
    assert physics["runtime_object_qpos_qvel_writes"] is False
    assert physics["runtime_external_forces"] is False
    assert physics["post_release_objects_unconstrained"] is True
    assert plan["media_sources"]["video"] == (
        "public_eye_left_plus_public_eye_right"
    )
    assert plan["media_sources"]["audio"] == "public_audition_pressure_only"


def test_scene_has_exact_free_objects_mocap_anchors_and_weld_defaults() -> None:
    environment = ThreeObjectScenarioEnv(width=32, height=24)
    scene_path = environment.scene_path
    try:
        observation, info = environment.reset(seed=1_401)
        assert info == {}
        assert environment.observation_space.contains(observation)
        assert observation["eye_left"].shape == (24, 32, 3)
        assert observation["eye_right"].shape == (24, 32, 3)
        assert not np.array_equal(observation["eye_left"], observation["eye_right"])

        model = environment.model
        table_id = int(model.body("demo_table").id)
        assert int(model.body_jntnum[table_id]) == 0
        rgba: list[np.ndarray] = []
        expected_types = {
            ScenarioObject.BALL: mujoco.mjtGeom.mjGEOM_SPHERE,
            ScenarioObject.BLOCK: mujoco.mjtGeom.mjGEOM_BOX,
            ScenarioObject.CYLINDER: mujoco.mjtGeom.mjGEOM_CYLINDER,
        }
        for target in ScenarioObject:
            body_name = SCENARIO_BODY_NAMES[target]
            body_id = int(model.body(body_name).id)
            geom_id = int(model.geom(f"{body_name}_geom").id)
            assert int(model.body_jntnum[body_id]) == 1
            joint_id = int(model.body_jntadr[body_id])
            assert int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
            assert int(model.geom_type[geom_id]) == int(expected_types[target])
            assert int(model.geom_contype[geom_id]) != 0
            assert int(model.geom_conaffinity[geom_id]) != 0
            rgba.append(model.geom_rgba[geom_id].copy())
        materials = AcousticMaterialResolver(model)
        assert materials.material(int(model.geom("demo_table_top").id)) == "wood"
        assert materials.material(int(model.geom("demo_ball_geom").id)) == "rubber"
        assert materials.material(int(model.geom("demo_block_geom").id)) == "wood"
        assert (
            materials.material(int(model.geom("demo_cylinder_geom").id))
            == "plastic"
        )
        assert all(
            not np.array_equal(first, second)
            for index, first in enumerate(rgba)
            for second in rgba[index + 1 :]
        )

        assert model.nmocap == 2
        root_anchor = int(model.body("demo_root_anchor").id)
        grip_anchor = int(model.body("demo_grip_anchor").id)
        assert int(model.body_mocapid[root_anchor]) >= 0
        assert int(model.body_mocapid[grip_anchor]) >= 0
        expected_active = {
            "demo_root_support": 1,
            "demo_grip_ball": 0,
            "demo_grip_block": 0,
            "demo_grip_cylinder": 0,
        }
        for name, active in expected_active.items():
            equality_id = int(model.equality(name).id)
            assert int(model.eq_type[equality_id]) == int(mujoco.mjtEq.mjEQ_WELD)
            assert int(model.eq_active0[equality_id]) == active
        assert "demo_root_anchor" in scene_path.read_text(encoding="utf-8")
    finally:
        environment.close()
    assert not scene_path.exists()


def _event(
    kind: ScenarioEventKind,
    cycle_index: int,
    target: ScenarioObject,
    step_index: int,
    subject: ScenarioObject | None = None,
) -> ScenarioEvent:
    return ScenarioEvent(
        kind=kind,
        step_index=step_index,
        time_seconds=step_index * 0.01,
        cycle_index=cycle_index,
        target=target,
        subject=subject,
    )


def _complete_event_set() -> tuple[ScenarioEvent, ...]:
    events: list[ScenarioEvent] = []
    for cycle_index, target in enumerate(ScenarioObject):
        base = cycle_index * 1_500
        events.extend(
            (
                _event(ScenarioEventKind.STATION_REACHED, cycle_index, target, base + 300),
                _event(
                    ScenarioEventKind.GRIP_CONSTRAINT_ACTIVATED,
                    cycle_index,
                    target,
                    base + 600,
                ),
                _event(
                    ScenarioEventKind.HELD_OBJECT_INSPECTED,
                    cycle_index,
                    target,
                    base + 699,
                    target,
                ),
                _event(
                    ScenarioEventKind.NEXT_OBJECT_GLANCE_MEASURED,
                    cycle_index,
                    target,
                    base + 849,
                    next_object(target),
                ),
                _event(
                    ScenarioEventKind.SECOND_OTHER_OBJECT_GLANCE_MEASURED,
                    cycle_index,
                    target,
                    base + 999,
                    next_object(next_object(target)),
                ),
                _event(
                    ScenarioEventKind.GRIP_CONSTRAINT_RELEASED,
                    cycle_index,
                    target,
                    base + 1_200,
                ),
                _event(
                    ScenarioEventKind.FLOOR_IMPACT,
                    cycle_index,
                    target,
                    base + 1_238,
                ),
            )
        )
    return tuple(events)


def test_event_coverage_requires_disclosed_grip_release_and_floor_impact() -> None:
    events = _complete_event_set()
    coverage = validate_event_coverage(events)
    assert set(coverage) == {"0", "1", "2"}
    assert all(row["event_count"] == 7 for row in coverage.values())

    missing_grip = tuple(
        event
        for event in events
        if not (
            event.cycle_index == 1
            and event.kind is ScenarioEventKind.GRIP_CONSTRAINT_ACTIVATED
        )
    )
    with pytest.raises(AssertionError, match="lacks physical event coverage"):
        validate_event_coverage(missing_grip)


def test_export_explicit_none_controller_fails_before_output(tmp_path: Path) -> None:
    output = tmp_path / "three_object_walk_pick_throw"
    with pytest.raises(ControllerHandoffPendingError, match="controller factory"):
        export_scenario(output, controller_factory=None)
    assert not output.exists()


def test_constraint_assisted_controller_matches_frozen_physics_hashes() -> None:
    environment = ThreeObjectScenarioEnv(width=16, height=12)
    segmentation_renderer: mujoco.Renderer | None = None
    try:
        environment.reset(seed=1_401)
        schedule = ScenarioSchedule.default()
        controller = constraint_assisted_three_object_controller_factory(
            environment, schedule
        )
        state = controller.reset(environment, schedule, 1_401)
        environment.model.vis.global_.offwidth = EYE_WIDTH
        environment.model.vis.global_.offheight = EYE_HEIGHT
        segmentation_renderer = mujoco.Renderer(
            environment.model, height=EYE_HEIGHT, width=EYE_WIDTH
        )
        segmentation_renderer.enable_segmentation_rendering()
        action_digest = sha256()
        qpos_digest = sha256()
        object_digest = sha256()
        events: list[ScenarioEvent] = []
        maximum_root_error = 0.0
        maximum_active_grips = 0
        visibility_records: list[dict[str, object]] = []

        for step_index in range(schedule.total_steps):
            context = schedule.context_at(step_index)
            qpos_before = environment.data.qpos.copy()
            qvel_before = environment.data.qvel.copy()
            xfrc_before = environment.data.xfrc_applied.copy()
            result = dispatch_controller(controller, environment, state, context)
            assert np.array_equal(environment.data.qpos, qpos_before)
            assert np.array_equal(environment.data.qvel, qvel_before)
            assert np.array_equal(environment.data.xfrc_applied, xfrc_before)
            state = result.state
            events.extend(result.events)
            action = np.ascontiguousarray(result.action, dtype=np.float32)
            assert environment.action_space.contains(action)
            action_digest.update(action.tobytes(order="C"))

            grip_active = sum(
                int(environment.data.eq_active[equality_id])
                for equality_id in controller.grip_equality_ids.values()
            )
            expected_grip = int(600 <= step_index % 1_500 < 1_200)
            assert grip_active == expected_grip
            maximum_active_grips = max(maximum_active_grips, grip_active)
            assert int(
                environment.data.eq_active[controller.root_support_equality_id]
            ) == 1

            observation, _, terminated, truncated, _ = environment.step(action)
            assert not terminated and not truncated
            assert environment.observation_space.contains(observation)
            assert not np.any(environment.data.xfrc_applied)
            qpos_digest.update(
                np.ascontiguousarray(
                    environment.data.qpos, dtype=np.float64
                ).tobytes(order="C")
            )
            positions = np.concatenate(
                [
                    np.asarray(
                        environment.data.body(SCENARIO_BODY_NAMES[target]).xpos
                    )
                    for target in ScenarioObject
                ]
            )
            object_digest.update(
                np.ascontiguousarray(positions, dtype=np.float64).tobytes(order="C")
            )
            local_step = step_index % 1_500
            for (
                label,
                gate_local_step,
                subject_offset,
                minimum_pixels,
                require_in_frame,
            ) in SEGMENTATION_VISIBILITY_GATES:
                if local_step == gate_local_step:
                    visibility_records.append(
                        capture_segmentation_visibility_gate(
                            segmentation_renderer,
                            environment,
                            context,
                            label=label,
                            local_step=gate_local_step,
                            subject_offset=subject_offset,
                            minimum_pixels_per_eye=minimum_pixels,
                            require_in_frame=require_in_frame,
                        )
                    )
            diagnostics = dict(result.diagnostics)
            target_position = np.asarray(
                (
                    diagnostics["root_target_x_m"],
                    diagnostics["root_target_y_m"],
                    diagnostics["root_target_z_m"],
                )
            )
            maximum_root_error = max(
                maximum_root_error,
                float(
                    np.linalg.norm(
                        environment.data.body("mimo_location").xpos
                        - target_position
                    )
                ),
            )

        assert isinstance(state, ConstraintAssistedControllerState)
        assert state.step_index == 4_500
        assert state.saturation_rows == 0
        assert action_digest.hexdigest() == EXPECTED_ACTION_SHA256
        assert qpos_digest.hexdigest() == EXPECTED_QPOS_SHA256
        assert object_digest.hexdigest() == EXPECTED_OBJECT_XPOS_SHA256
        assert maximum_active_grips == 1
        assert maximum_root_error <= 0.0041, maximum_root_error
        assert len(visibility_records) == 21
        assert {record["label"] for record in visibility_records} == {
            gate[0] for gate in SEGMENTATION_VISIBILITY_GATES
        }
        for record in visibility_records:
            assert record["passed"] is True
            eyes = record["eyes"]
            assert isinstance(eyes, dict)
            assert set(eyes) == {"eye_left", "eye_right"}
            for metrics in eyes.values():
                assert isinstance(metrics, dict)
                assert metrics["geom_pixels"] >= record["minimum_pixels_per_eye"]
                if record["require_in_frame"]:
                    assert metrics["in_frame"] is True

        coverage = validate_event_coverage(events)
        assert all(row["event_count"] == 7 for row in coverage.values())
        impacts = [
            event
            for event in events
            if event.kind is ScenarioEventKind.FLOOR_IMPACT
        ]
        assert [event.time_seconds for event in impacts] == pytest.approx(
            [12.38, 27.36, 42.37], abs=5e-11
        )
        assert all(dict(event.metrics)["grip_active"] == 0.0 for event in impacts)
    finally:
        if segmentation_renderer is not None:
            segmentation_renderer.close()
        environment.close()

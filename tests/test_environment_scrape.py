"""Live MuJoCo checks for continuous scrape sensing and synthesis."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError

import mujoco
import numpy as np
import pytest

from blight_world import (
    AuditionConfig,
    AudioSourceSpec,
    BlightWorldEnv,
    ContinuousSoundConfig,
    StaticAttachment,
)
from mimo_audition.scrape import ScrapeContact, ScrapeDiagnostic, ScrapeTracker


@pytest.fixture(scope="module")
def scrape_environment() -> Iterator[BlightWorldEnv]:
    """Create one explicitly scrape-enabled live MIMo world."""

    environment = BlightWorldEnv(
        width=16,
        height=12,
        audition=AuditionConfig.quality_48k(
            continuous=ContinuousSoundConfig()
        ),
    )
    try:
        yield environment
    finally:
        environment.close()


def _free_joint_addresses(
    environment: BlightWorldEnv,
    body_name: str,
) -> tuple[int, int]:
    joint = environment.model.joint(f"{body_name}_joint")
    return (
        int(np.asarray(joint.qposadr).item()),
        int(np.asarray(joint.dofadr).item()),
    )


def _place_object_fixture(
    environment: BlightWorldEnv,
    *,
    active_body: str,
    position_m: tuple[float, float, float],
    velocity_m_s: tuple[float, float, float],
    seed: int,
) -> tuple[int, int]:
    """Park unused objects and seed support contacts before acoustic tracking."""

    environment.reset(seed=seed)
    qpos = environment.data.qpos.copy()
    qvel = np.zeros_like(environment.data.qvel)
    parked = {
        "block": (-1.8, 1.5, 0.099),
        "ball": (-1.8, 0.0, 0.099),
        "cylinder": (-1.8, -1.5, 0.099),
    }
    for body_name, parked_position in parked.items():
        qpos_address, dof_address = _free_joint_addresses(
            environment,
            body_name,
        )
        qpos[qpos_address : qpos_address + 3] = (
            position_m if body_name == active_body else parked_position
        )
        qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
        qvel[dof_address : dof_address + 6] = 0.0
    qpos_address, dof_address = _free_joint_addresses(
        environment,
        active_body,
    )
    qvel[dof_address : dof_address + 3] = velocity_m_s
    environment.data.xfrc_applied.fill(0)
    environment.set_state(qpos, qvel)
    mujoco.mj_forward(environment.model, environment.data)
    assert environment.audition is not None
    environment.audition.reset(seed)
    return qpos_address, dof_address


def _step_supported_object(
    environment: BlightWorldEnv,
    *,
    body_name: str,
    qpos_address: int,
    dof_address: int,
    contact_height_m: float,
    velocity_m_s: tuple[float, float, float],
    downward_force_n: float = 0.0,
) -> tuple[np.ndarray, tuple[ScrapeDiagnostic, ...]]:
    """Advance one action while preserving a continuous solved support contact."""

    environment.data.qpos[qpos_address + 2] = contact_height_m
    environment.data.qvel[dof_address : dof_address + 6] = 0.0
    environment.data.qvel[dof_address : dof_address + 3] = velocity_m_s
    environment.data.xfrc_applied.fill(0)
    body_id = environment.model.body(body_name).id
    environment.data.xfrc_applied[body_id, 2] = -downward_force_n
    mujoco.mj_forward(environment.model, environment.data)
    observation, *_ = environment.step(
        np.zeros(environment.action_space.shape, dtype=np.float32)
    )
    assert environment.audition is not None
    return observation["audition"], environment.audition.scrape_diagnostics


def _target_diagnostic(
    diagnostics: tuple[ScrapeDiagnostic, ...],
    material_pair: tuple[str, str],
) -> ScrapeDiagnostic:
    matches = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.material_pair == material_pair
    ]
    assert len(matches) == 1
    return matches[0]


def _supported_trial(
    environment: BlightWorldEnv,
    *,
    body_name: str,
    position_m: tuple[float, float, float],
    velocity_m_s: tuple[float, float, float],
    material_pair: tuple[str, str],
    downward_force_n: float = 0.0,
) -> tuple[ScrapeDiagnostic, np.ndarray]:
    qpos_address, dof_address = _place_object_fixture(
        environment,
        active_body=body_name,
        position_m=position_m,
        velocity_m_s=velocity_m_s,
        seed=211,
    )
    pressure = np.zeros((2, 480), dtype=np.float32)
    diagnostics: tuple[ScrapeDiagnostic, ...] = ()
    for _ in range(3):
        pressure, diagnostics = _step_supported_object(
            environment,
            body_name=body_name,
            qpos_address=qpos_address,
            dof_address=dof_address,
            contact_height_m=position_m[2],
            velocity_m_s=velocity_m_s,
            downward_force_n=downward_force_n,
        )
    return _target_diagnostic(diagnostics, material_pair), pressure


def _tracker_cell(
    environment: BlightWorldEnv,
    *,
    body_name: str,
    position_m: tuple[float, float, float],
    speed_m_s: float,
    extra_downward_force_n: float,
    material_pair: tuple[str, str],
) -> tuple[ScrapeContact, ...]:
    """Run the source-verified 8-settle/30-drive substep fixture."""

    qpos_address, dof_address = _place_object_fixture(
        environment,
        active_body=body_name,
        position_m=position_m,
        velocity_m_s=(0.0, 0.0, 0.0),
        seed=311,
    )
    assert environment.audition is not None
    tracker = environment.audition._scrapes
    assert isinstance(tracker, ScrapeTracker)
    support_geom = "floor" if position_m[2] == 0.099 else "table_top"
    target_pair = tuple(
        sorted(
            (
                environment.model.geom(support_geom).id,
                environment.model.geom(f"{body_name}_geom").id,
            )
        )
    )
    body_id = environment.model.body(body_name).id
    driven: list[ScrapeContact] = []
    for substep in range(38):
        speed = 0.0 if substep < 8 else speed_m_s
        environment.data.qpos[qpos_address + 2] = position_m[2]
        environment.data.qvel[dof_address : dof_address + 6] = 0.0
        environment.data.qvel[dof_address] = speed
        environment.data.xfrc_applied.fill(0)
        environment.data.xfrc_applied[body_id, 2] = -extra_downward_force_n
        mujoco.mj_forward(environment.model, environment.data)
        tracker.before_substep()
        mujoco.mj_step(environment.model, environment.data)
        mujoco.mj_rnePostConstraint(environment.model, environment.data)
        measurements = tracker._measurements()
        assert target_pair in measurements
        controls = tracker._gate.update(measurements)
        matches = [control for control in controls if control.geom_pair == target_pair]
        if substep >= 18:
            assert len(matches) == 1
            control = matches[0]
            assert control.material_pair == material_pair
            assert control.stream_id == f"scrape:{target_pair[0]}:{target_pair[1]}"
            assert control.tangential_speed_m_s == pytest.approx(
                speed_m_s,
                abs=1e-12,
            )
            driven.append(control)
    assert len(driven) == 20
    return tuple(driven)


def test_block_floor_scrape_reverses_and_releases_without_contact_gaps(
    scrape_environment: BlightWorldEnv,
) -> None:
    """A supported block tracks reversal, stop, lift, and finite release."""

    qpos_address, dof_address = _place_object_fixture(
        scrape_environment,
        active_body="block",
        position_m=(1.8, 0.0, 0.099),
        velocity_m_s=(0.0, 0.4, 0.0),
        seed=223,
    )
    assert scrape_environment.audition is not None
    tracker = scrape_environment.audition._scrapes
    assert isinstance(tracker, ScrapeTracker)
    target_pair = tuple(
        sorted(
            (
                scrape_environment.model.geom("floor").id,
                scrape_environment.model.geom("block_geom").id,
            )
        )
    )
    original_measurements = tracker._measurements
    contact_history: list[bool] = []

    def recording_measurements() -> dict[tuple[int, int], object]:
        measurements = original_measurements()
        contact_history.append(target_pair in measurements)
        return measurements

    tracker._measurements = recording_measurements  # type: ignore[method-assign]
    try:
        pressure_blocks = []
        forward: ScrapeDiagnostic | None = None
        for _ in range(3):
            pressure, diagnostics = _step_supported_object(
                scrape_environment,
                body_name="block",
                qpos_address=qpos_address,
                dof_address=dof_address,
                contact_height_m=0.099,
                velocity_m_s=(0.0, 0.4, 0.0),
            )
            pressure_blocks.append(pressure)
            forward = _target_diagnostic(diagnostics, ("hard", "wood"))
        assert forward is not None

        reverse: ScrapeDiagnostic | None = None
        for _ in range(3):
            pressure, diagnostics = _step_supported_object(
                scrape_environment,
                body_name="block",
                qpos_address=qpos_address,
                dof_address=dof_address,
                contact_height_m=0.099,
                velocity_m_s=(0.0, -0.4, 0.0),
            )
            pressure_blocks.append(pressure)
            reverse = _target_diagnostic(diagnostics, ("hard", "wood"))
        assert reverse is not None
        forward_velocity = np.asarray(forward.tangential_velocity_world_m_s)
        reverse_velocity = np.asarray(reverse.tangential_velocity_world_m_s)
        assert np.linalg.norm(reverse_velocity) == pytest.approx(
            np.linalg.norm(forward_velocity),
            rel=0.03,
        )
        cosine = float(
            np.dot(forward_velocity, reverse_velocity)
            / (
                np.linalg.norm(forward_velocity)
                * np.linalg.norm(reverse_velocity)
            )
        )
        assert cosine == pytest.approx(-1.0, abs=1e-6)

        _, stopped = _step_supported_object(
            scrape_environment,
            body_name="block",
            qpos_address=qpos_address,
            dof_address=dof_address,
            contact_height_m=0.099,
            velocity_m_s=(0.0, 0.0, 0.0),
        )
        assert _target_diagnostic(stopped, ("hard", "wood")).phase == "release"

        for _ in range(2):
            _step_supported_object(
                scrape_environment,
                body_name="block",
                qpos_address=qpos_address,
                dof_address=dof_address,
                contact_height_m=0.099,
                velocity_m_s=(0.0, 0.4, 0.0),
            )
        scrape_environment.data.qpos[qpos_address + 2] = 0.30
        scrape_environment.data.qvel[dof_address : dof_address + 6] = 0.0
        scrape_environment.data.xfrc_applied.fill(0)
        mujoco.mj_forward(scrape_environment.model, scrape_environment.data)
        scrape_environment.step(
            np.zeros(scrape_environment.action_space.shape, dtype=np.float32)
        )
        lifted = scrape_environment.audition.scrape_diagnostics
        assert _target_diagnostic(lifted, ("hard", "wood")).phase == "release"
        for _ in range(4):
            scrape_environment.data.qpos[qpos_address + 2] = 0.30
            scrape_environment.data.qvel[dof_address : dof_address + 6] = 0.0
            mujoco.mj_forward(scrape_environment.model, scrape_environment.data)
            scrape_environment.step(
                np.zeros(scrape_environment.action_space.shape, dtype=np.float32)
            )
        assert all(
            diagnostic.geom_pair != target_pair
            for diagnostic in scrape_environment.audition.scrape_diagnostics
        )
    finally:
        tracker._measurements = original_measurements  # type: ignore[method-assign]

    assert contact_history
    assert all(contact_history[:12])
    pressure_trace = np.concatenate(pressure_blocks, axis=1)
    assert np.all(np.isfinite(pressure_trace))
    assert np.max(np.abs(pressure_trace)) > 0
    assert not np.array_equal(pressure_trace[0], pressure_trace[1])
    with pytest.raises(FrozenInstanceError):
        forward.normal_load_n = 0.0  # type: ignore[misc]


def test_speed_load_and_material_change_live_scrape_controls(
    scrape_environment: BlightWorldEnv,
) -> None:
    """All four object/support pairs respond monotonically to speed and load."""

    fixtures = (
        ("block", (1.8, 0.0, 0.099), ("hard", "wood")),
        ("block", (0.8, 0.0, 0.699), ("wood", "wood")),
        ("cylinder", (1.8, 0.0, 0.099), ("hard", "plastic")),
        ("cylinder", (0.8, 0.0, 0.699), ("wood", "plastic")),
    )
    for body_name, position_m, material_pair in fixtures:
        body_id = scrape_environment.model.body(body_name).id
        body_weight_n = float(
            scrape_environment.model.body_mass[body_id] * 9.81
        )
        cells: dict[tuple[float, float], tuple[float, float]] = {}
        for extra_load in (0.0, body_weight_n):
            for speed in (0.2, 0.4, 0.8):
                controls = _tracker_cell(
                    scrape_environment,
                    body_name=body_name,
                    position_m=position_m,
                    speed_m_s=speed,
                    extra_downward_force_n=extra_load,
                    material_pair=material_pair,
                )
                loads = np.asarray(
                    [control.normal_load_n for control in controls]
                )
                powers = np.asarray(
                    [control.friction_power_w for control in controls]
                )
                assert np.all(np.isfinite(loads)) and np.all(loads > 0)
                assert np.all(np.isfinite(powers)) and np.all(powers > 0)
                cells[(extra_load, speed)] = (
                    float(np.median(loads)),
                    float(np.median(powers)),
                )
        for extra_load in (0.0, body_weight_n):
            powers = [cells[(extra_load, speed)][1] for speed in (0.2, 0.4, 0.8)]
            assert powers[0] < powers[1] < powers[2]
        for speed in (0.2, 0.4, 0.8):
            unloaded = cells[(0.0, speed)]
            loaded = cells[(body_weight_n, speed)]
            assert loaded[0] > unloaded[0]
            assert loaded[1] > unloaded[1]

    cylinder, cylinder_pressure = _supported_trial(
        scrape_environment,
        body_name="cylinder",
        position_m=(1.0, 0.0, 0.699),
        velocity_m_s=(0.0, 0.6, 0.0),
        material_pair=("wood", "plastic"),
    )
    cylinder_replay, replay_pressure = _supported_trial(
        scrape_environment,
        body_name="cylinder",
        position_m=(1.0, 0.0, 0.699),
        velocity_m_s=(0.0, 0.6, 0.0),
        material_pair=("wood", "plastic"),
    )

    assert cylinder.phase == "active"
    assert cylinder.target_rms_pressure_pa_at_1m > 0
    assert np.all(np.isfinite(cylinder_pressure))
    assert np.max(np.abs(cylinder_pressure)) > 0
    assert cylinder_replay == cylinder
    np.testing.assert_array_equal(replay_pressure, cylinder_pressure)


def test_drop_emits_one_impact_then_persistent_scrape(
    scrape_environment: BlightWorldEnv,
) -> None:
    """Scrape synthesis does not retrigger a supported contact's impact onset."""

    qpos_address, dof_address = _place_object_fixture(
        scrape_environment,
        active_body="block",
        position_m=(1.8, 0.0, 0.13),
        velocity_m_s=(0.0, 0.4, -2.0),
        seed=131,
    )
    assert scrape_environment.audition is not None
    renderer = scrape_environment.audition.renderer
    original_play_transient = renderer.play_transient
    clip_ids: list[str] = []

    def recording_play_transient(
        clip_id: str,
        *args: object,
        **kwargs: object,
    ) -> int | None:
        clip_ids.append(clip_id)
        return original_play_transient(clip_id, *args, **kwargs)  # type: ignore[arg-type]

    renderer.play_transient = recording_play_transient  # type: ignore[method-assign]
    pressure_blocks = []
    try:
        for _ in range(8):
            if clip_ids:
                scrape_environment.data.qpos[qpos_address + 2] = 0.099
                scrape_environment.data.qvel[dof_address : dof_address + 6] = 0.0
                scrape_environment.data.qvel[dof_address + 1] = 0.4
                mujoco.mj_forward(
                    scrape_environment.model,
                    scrape_environment.data,
                )
            observation, *_ = scrape_environment.step(
                np.zeros(scrape_environment.action_space.shape, dtype=np.float32)
            )
            pressure_blocks.append(observation["audition"])
    finally:
        renderer.play_transient = original_play_transient  # type: ignore[method-assign]

    assert len(clip_ids) == 1
    assert clip_ids[0].startswith("impact_hard_wood_")
    diagnostic = _target_diagnostic(
        scrape_environment.audition.scrape_diagnostics,
        ("hard", "wood"),
    )
    assert diagnostic.phase == "active"
    pressure = np.concatenate(pressure_blocks, axis=1)
    assert np.max(np.abs(pressure)) > 0


def test_eight_clip_voices_coexist_with_live_scrape_pool(
    scrape_environment: BlightWorldEnv,
) -> None:
    """A live scrape uses continuous capacity without reserving clip voices."""

    qpos_address, dof_address = _place_object_fixture(
        scrape_environment,
        active_body="block",
        position_m=(1.8, 0.0, 0.099),
        velocity_m_s=(0.0, 0.4, 0.0),
        seed=271,
    )
    assert scrape_environment.audition is not None
    renderer = scrape_environment.audition.renderer
    if "scrape_capacity_fixture" not in renderer.source_ids:
        scrape_environment.audition.register_source(
            AudioSourceSpec(
                "scrape_capacity_fixture",
                "chime",
                StaticAttachment((1.0, 0.0, 0.8)),
                loop=True,
            )
        )
    for _ in range(8):
        scrape_environment.audition.play("scrape_capacity_fixture")
    _step_supported_object(
        scrape_environment,
        body_name="block",
        qpos_address=qpos_address,
        dof_address=dof_address,
        contact_height_m=0.099,
        velocity_m_s=(0.0, 0.4, 0.0),
    )

    assert renderer.active_voice_count == 8
    assert renderer.continuous_active_voice_count == 1
    assert renderer.continuous_retained_voice_count >= 1

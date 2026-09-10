"""Checks for the approved endpoint snapshot and sensorimotor timing contract."""

from __future__ import annotations

import numpy as np
import pytest

from blight_world.biology import CorrectedMuscleModel
from blight_world.environment import BlightWorldEnv
from mimo_audition import AuditionConfig, AudioSourceSpec, StaticAttachment


class ClockVision:
    """Expose capture time in pixels without needing an external renderer."""

    def __init__(self, context, parameters):
        self.context = context
        self.parameters = parameters
        self.capture_times = []
        self.closed = False

    def get_vision_obs(self):
        time_ms = round(self.context.data.time * 1000)
        self.capture_times.append(time_ms)
        return {
            eye: np.full((p["height"], p["width"], 3), time_ms, dtype=np.uint8)
            for eye, p in self.parameters.items()
        }

    def close(self):
        self.closed = True


def test_snapshot_preserves_authoritative_trajectory_and_refreshes_geometry():
    with (
        BlightWorldEnv(width=16, height=12) as baseline,
        BlightWorldEnv(
            width=16, height=12, vision_provider_factory=ClockVision
        ) as observed,
    ):
        baseline.reset(seed=71)
        observed.reset(seed=71)
        authoritative = observed.data
        provider = observed.vision._provider
        assert observed.snapshot_sensors
        assert not observed.mujoco_renderer._viewers
        assert not observed.vision._renderers
        for step in range(4):
            action = np.zeros(observed.action_space.shape, dtype=np.float32)
            baseline.step(action)
            observation, *_ = observed.step(action)
            assert observed.data is authoritative
            assert observed.sensor_context.data is not authoritative
            assert observed.sensor_context.data.time == observed.data.time
            for field in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
                np.testing.assert_array_equal(
                    getattr(observed.data, field), getattr(baseline.data, field)
                )
            np.testing.assert_array_equal(
                observed.model.actuator_gear, baseline.model.actuator_gear
            )
            ball_qpos = int(observed.model.joint("ball_joint").qposadr[0])
            np.testing.assert_array_equal(
                observed.sensor_context.data.body("ball").xpos,
                observed.data.qpos[ball_qpos:ball_qpos + 3],
            )
            assert observation["eye_left"][0, 0, 0] == (step + 1) * 10
            captures = len(provider.capture_times)
            before = observed.data.qacc_warmstart.copy()
            for _ in range(3):
                observed._get_obs()
            assert len(provider.capture_times) == captures
            np.testing.assert_array_equal(before, observed.data.qacc_warmstart)
        assert not np.array_equal(
            observed.data.body("ball").xpos,
            observed.sensor_context.data.body("ball").xpos,
        )
    assert provider.closed


@pytest.mark.parametrize("delay", [1, 2])
def test_exact_motor_and_sensory_delays_copy_inputs_and_reset(delay):
    with BlightWorldEnv(
        width=8, height=6, vision_provider_factory=ClockVision,
        motor_delay=delay, sensory_delay=delay,
    ) as env:
        reset, _ = env.reset(seed=72)
        commands = []
        for index in range(delay + 3):
            action = np.full(env.action_space.shape, (index + 1) * .01, dtype=np.float32)
            commands.append(action.copy())
            observation, *_ = env.step(action)
            expected_action = (
                np.zeros_like(action) if index < delay else commands[index - delay]
            )
            np.testing.assert_array_equal(env.actuation_model.control_input, expected_action)
            expected_time = max(index + 1 - delay, 0) * 10
            assert observation["eye_left"][0, 0, 0] == expected_time
            action.fill(-1)
            observation["eye_left"].fill(255)
            for _ in range(3):
                assert env._get_obs()["eye_left"][0, 0, 0] == expected_time
        again, _ = env.reset(seed=72)
        for key in reset:
            np.testing.assert_array_equal(again[key], reset[key])
        np.testing.assert_array_equal(env.actuation_model.control_input, 0)
        for _ in range(delay):
            observation, *_ = env.step(np.ones(env.action_space.shape, dtype=np.float32))
            np.testing.assert_array_equal(env.actuation_model.control_input, 0)
            np.testing.assert_array_equal(observation["observation"], reset["observation"])
            assert observation["eye_left"][0, 0, 0] == 0


def test_sensory_delay_includes_audio_and_substeps_use_start_pose_copies(monkeypatch):
    with BlightWorldEnv(
        width=8, height=6, vision_provider_factory=ClockVision, sensory_delay=2,
        audition=AuditionConfig(
            enable_occlusion=False, enable_first_order_reflections=False,
            enable_late_reverb=False,
        ),
    ) as env:
        initial, _ = env.reset(seed=73)
        env.audition.register_source(AudioSourceSpec(
            source_id="chime", clip_id="chime", attachment=StaticAttachment((1., 0., 1.))
        ))
        env.audition.play("chime")
        original_capture = env.audition.capture_substep
        interval_times = []

        def capture(*args, pose_data=None, **kwargs):
            interval_times.append((pose_data.time, env.data.time))
            assert env.audition._impacts.data is env.data
            return original_capture(*args, pose_data=pose_data, **kwargs)

        monkeypatch.setattr(env.audition, "capture_substep", capture)
        pressure = [initial["audition"].copy()]
        for step in range(1, 6):
            observation, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            pressure.append(env.audition.observation)
            np.testing.assert_array_equal(observation["audition"], pressure[max(step - 2, 0)])
            assert observation["eye_left"][0, 0, 0] == max(step - 2, 0) * 10
        assert any(np.any(block) for block in pressure[1:])
        np.testing.assert_allclose(
            interval_times, [(i * .005, (i + 1) * .005) for i in range(10)], atol=1e-14
        )
        assert env.audition.renderer.sample_cursor == 2400
        reset, _ = env.reset(seed=73)
        assert not np.any(reset["audition"])


def test_muscle_activation_has_one_update_per_substep_and_snapshot_is_read_only(monkeypatch):
    with BlightWorldEnv(
        width=8, height=6, vision_provider_factory=ClockVision,
        actuation_model=CorrectedMuscleModel,
        proprio_params={"components": ["actuation"]},
    ) as env:
        muscle = env.actuation_model
        update_activity = muscle._update_activity
        updates = []

        def update():
            updates.append(float(env.data.time))
            update_activity()

        monkeypatch.setattr(muscle, "_update_activity", update)
        env.reset(seed=74)
        assert not updates
        muscle.action(np.ones(env.action_space.shape, dtype=np.float32))
        np.testing.assert_array_equal(muscle.activity, 0)
        observation, *_ = env.step(np.ones(env.action_space.shape, dtype=np.float32))
        assert len(updates) == 2
        np.testing.assert_array_equal(muscle.activity, .75)
        assert np.all(env.data.ctrl[muscle.actuators] == 1)
        assert np.all(np.isfinite(observation["observation"]))
        before = {
            "activity": muscle.activity.copy(),
            "forces": muscle.muscle_forces.copy(),
            "gear": env.model.actuator_gear.copy(),
            "ctrl": env.data.ctrl.copy(),
            "warmstart": env.data.qacc_warmstart.copy(),
        }
        env._refresh_sensor_context()
        env._get_obs()
        evaluator = env.sensor_context.actuation_model
        assert evaluator is not muscle
        assert not np.shares_memory(evaluator.activity, muscle.activity)
        np.testing.assert_array_equal(muscle.activity, before["activity"])
        np.testing.assert_array_equal(muscle.muscle_forces, before["forces"])
        np.testing.assert_array_equal(env.model.actuator_gear, before["gear"])
        np.testing.assert_array_equal(env.data.ctrl, before["ctrl"])
        np.testing.assert_array_equal(env.data.qacc_warmstart, before["warmstart"])
        assert len(updates) == 2
        env.reset(seed=74)
        assert len(updates) == 2
        np.testing.assert_array_equal(muscle.activity, 0)
        np.testing.assert_array_equal(muscle.target_activity, 0)


def test_full_profile_exposes_existing_upstream_modalities():
    with BlightWorldEnv(
        width=8, height=6, vision_provider_factory=ClockVision, biology_profile="full"
    ) as env:
        observation, _ = env.reset(seed=75)
        assert set(observation) == {
            "observation", "touch", "vestibular", "eye_left", "eye_right"
        }
        assert observation["observation"].shape == (621,)
        assert observation["vestibular"].shape == (6,)
        assert observation["touch"].size > 0
        assert observation["touch"].size % 3 == 0
        assert len(env.touch.sensor_scales) == 55
        assert env.observation_space.contains(observation)
        observation, *_ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert all(np.isfinite(value).all() for value in observation.values())
        assert env.touch.env.data is env.sensor_context.data
        assert env.vestibular.env.data is env.sensor_context.data


def test_exact_acuity_age_uses_filter_age_without_changing_body_age():
    class PatternVision(ClockVision):
        def get_vision_obs(self):
            images = super().get_vision_obs()
            for image in images.values():
                pattern = np.indices(image.shape[:2]).sum(axis=0) % 2 * 255
                image[:] = pattern[:, :, None]
            return images

    with BlightWorldEnv(
        width=16, height=12, vision_provider_factory=PatternVision,
        vision_params={"eye_left": {"acuity": 1.0}},
    ) as env:
        observation, _ = env.reset(seed=76)
        assert env.age is None
        assert env.model.geom("head").size[0] == .0735
        assert observation["eye_left"].std() < observation["eye_right"].std()


def test_native_vision_uses_the_same_acuity_correction():
    with BlightWorldEnv(
        width=16, height=12, vision_params={"eye_left": {"acuity": 1.0}}
    ) as env:
        observation, _ = env.reset(seed=77)
        assert not env.snapshot_sensors
        assert env.vision.env is env
        assert "rgb_array" in env.mujoco_renderer._viewers
        assert env.age is None
        assert observation["eye_left"].shape == (12, 16, 3)

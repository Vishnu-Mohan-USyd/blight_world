"""Focused contracts for the short assisted ball-throw controller."""

from __future__ import annotations

import numpy as np

from blight_world.assisted_ball_throw import (
    BALL_THROW_IMPACT_SOURCE_CLIP_ID,
    FINGER_OPEN_START_SWING_STEP,
    FINGER_OPEN_STOP_SWING_STEP,
    LAUNCH_ARM_REFERENCE_RAD,
    MAXIMUM_HAND_RELATIVE_ERROR_M,
    PRE_SWING_PUBLIC_STEPS,
    RECORDED_PUBLIC_STEPS,
    RELEASE_RECORDED_STEP,
    RETURN_STOP_RECORDED_STEP,
    SWING_PUBLIC_STEPS,
    THROW_ARM_TARGET_RAD,
    THROW_PULSE_ACTIONS,
    THROW_PULSE_RAMP_STEPS,
    WINDUP_ARM_REFERENCE_RAD,
    WINDUP_STOP_RECORDED_STEP,
    derive_single_hit_rubber_clip,
    onset_cluster_starts,
)
from blight_world.motor_skills import RIGHT_ARM_ACTUATOR_NAMES
from blight_world.supported_pick_place import LIFT_ARM_SEED_RAD
from mimo_audition.clips import ClipLibrary


def test_throw_clock_and_actuator_contract_are_explicit() -> None:
    assert RECORDED_PUBLIC_STEPS == 500
    assert MAXIMUM_HAND_RELATIVE_ERROR_M == 0.008
    assert WINDUP_STOP_RECORDED_STEP == 30
    assert RETURN_STOP_RECORDED_STEP == 70
    assert PRE_SWING_PUBLIC_STEPS == 92
    assert SWING_PUBLIC_STEPS == 7
    assert RELEASE_RECORDED_STEP == 99
    assert FINGER_OPEN_START_SWING_STEP < SWING_PUBLIC_STEPS
    assert FINGER_OPEN_STOP_SWING_STEP == SWING_PUBLIC_STEPS
    assert THROW_PULSE_RAMP_STEPS == 5
    assert THROW_PULSE_ACTIONS == (
        ("act:right_shoulder_horizontal", 1.0),
        ("act:right_elbow", 0.30),
        ("act:right_shoulder_abduction", 0.152),
    )
    assert len(THROW_ARM_TARGET_RAD) == len(RIGHT_ARM_ACTUATOR_NAMES) == 7
    assert np.all(np.isfinite(THROW_ARM_TARGET_RAD))
    assert len(WINDUP_ARM_REFERENCE_RAD) == len(RIGHT_ARM_ACTUATOR_NAMES)
    np.testing.assert_array_equal(THROW_ARM_TARGET_RAD, LAUNCH_ARM_REFERENCE_RAD)
    assert WINDUP_ARM_REFERENCE_RAD[0] < LIFT_ARM_SEED_RAD[0]
    assert WINDUP_ARM_REFERENCE_RAD[3] < LIFT_ARM_SEED_RAD[3]


def test_demo_rubber_clip_is_single_onset_and_does_not_mutate_source() -> None:
    library = ClipLibrary(48_000)
    source = library.get(BALL_THROW_IMPACT_SOURCE_CLIP_ID)
    source_before = source.samples.copy()

    derived = derive_single_hit_rubber_clip(source)

    np.testing.assert_array_equal(source.samples, source_before)
    assert derived.shape == source.samples.shape
    assert derived.dtype == np.float64
    assert derived.flags.c_contiguous
    assert np.all(np.isfinite(derived))
    assert np.max(np.abs(derived)) > 0.0
    assert len(onset_cluster_starts(derived, source.sample_rate_hz)) == 1

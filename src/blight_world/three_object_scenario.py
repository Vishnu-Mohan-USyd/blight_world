"""Constraint-assisted three-object demonstration scene and timeline.

The scene contains three free task objects plus two explicitly disclosed mocap
anchors. Root travel is carried by an always-active six-DOF mocap weld, and
pickup uses at most one six-DOF object weld from local time 6 through 12 s.
Those are engineering demonstration constraints: they are not free walking or
a physical hand grasp. After release every task object is unconstrained and
falls through ordinary MuJoCo dynamics to a measured floor impact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from blight_world.environment import BlightWorldEnv


PUBLIC_STEP_SECONDS = 0.010
PHYSICS_STEP_SECONDS = 0.005
SCENARIO_SEED = 1_401
SCENARIO_CYCLE_STEPS = 1_500
SCENARIO_TOTAL_STEPS = 4_500
SCENARIO_DURATION_SECONDS = 45.0
ROOT_MOTION_MECHANISM = "constraint_carried_stepping"
PICKUP_MECHANISM = "mocap_grip_assisted"

CONSTRAINT_ASSISTED_SCENE_EXTENSION_XML = """<worldbody>
  <body name="demo_table" pos="-.4 0 .169">
    <geom name="demo_table_top" type="box" size=".235 .50 .169"
          rgba=".45 .27 .12 1" friction=".8 .05 .02" condim="6"/>
  </body>
  <body name="demo_ball" pos="-.18 -.30 .360">
    <freejoint name="demo_ball_joint"/>
    <geom name="demo_ball_geom" type="sphere" size=".022" mass=".030"
          rgba="1 .15 .10 1"/>
  </body>
  <body name="demo_block" pos="-.18 0 .360">
    <freejoint name="demo_block_joint"/>
    <geom name="demo_block_geom" type="box" size=".022 .022 .022"
          mass=".035" rgba=".10 .25 1 1"/>
  </body>
  <body name="demo_cylinder" pos="-.18 .30 .363">
    <freejoint name="demo_cylinder_joint"/>
    <geom name="demo_cylinder_geom" type="cylinder" size=".022 .025"
          mass=".032" rgba=".10 .90 .20 1"/>
  </body>
  <body name="demo_root_anchor" mocap="true" pos="0 0 .4"
        quat="0 0 0 1"/>
  <body name="demo_grip_anchor" mocap="true" pos="-.18 -.30 .36"/>
</worldbody>
<equality>
  <weld name="demo_root_support" body1="demo_root_anchor"
        body2="mimo_location" relpose="0 0 0 1 0 0 0" active="true"
        solref=".01 1" solimp=".99 .999 .001 .5 2"/>
  <weld name="demo_grip_ball" body1="demo_grip_anchor"
        body2="demo_ball" relpose="0 0 0 1 0 0 0" active="false"
        solref=".005 1" solimp=".99 .999 .001 .5 2"/>
  <weld name="demo_grip_block" body1="demo_grip_anchor"
        body2="demo_block" relpose="0 0 0 1 0 0 0" active="false"
        solref=".005 1" solimp=".99 .999 .001 .5 2"/>
  <weld name="demo_grip_cylinder" body1="demo_grip_anchor"
        body2="demo_cylinder" relpose="0 0 0 1 0 0 0" active="false"
        solref=".005 1" solimp=".99 .999 .001 .5 2"/>
</equality>"""


class ScenarioObject(str, Enum):
    """Three task objects in the frozen cycle order."""

    BALL = "ball"
    BLOCK = "block"
    CYLINDER = "cylinder"


SCENARIO_BODY_NAMES = {
    ScenarioObject.BALL: "demo_ball",
    ScenarioObject.BLOCK: "demo_block",
    ScenarioObject.CYLINDER: "demo_cylinder",
}


def next_object(target: ScenarioObject) -> ScenarioObject:
    """Return the next cyclic task object used by the measured glance."""

    if not isinstance(target, ScenarioObject):
        raise TypeError("target must be a ScenarioObject")
    ordered = tuple(ScenarioObject)
    return ordered[(ordered.index(target) + 1) % len(ordered)]


class ScenarioPhase(str, Enum):
    """Frozen behavior/control windows for each 15-second cycle."""

    GO_TO_TARGET = "go_to_target"
    PICK_TARGET = "mocap_grip_assisted_pick"
    INSPECT_HELD_TARGET = "inspect_held_target"
    GLANCE_AT_NEXT_OBJECT = "glance_at_next_object"
    THROW_TO_FLOOR = "release_throw_to_floor"
    SETTLE = "unconstrained_floor_settle"


class ScenarioEventKind(str, Enum):
    """Honest milestones emitted by the calibrated controller."""

    STATION_REACHED = "station_reached"
    GRIP_CONSTRAINT_ACTIVATED = "grip_constraint_activated"
    HELD_OBJECT_INSPECTED = "held_object_inspected"
    NEXT_OBJECT_GLANCE_MEASURED = "next_object_glance_measured"
    SECOND_OTHER_OBJECT_GLANCE_MEASURED = (
        "second_other_object_glance_measured"
    )
    GRIP_CONSTRAINT_RELEASED = "grip_constraint_released"
    FLOOR_IMPACT = "floor_impact"
    PHASE_FAILED = "phase_failed"


_PHASE_STEP_COUNTS = (
    (ScenarioPhase.GO_TO_TARGET, 300),
    (ScenarioPhase.PICK_TARGET, 400),
    (ScenarioPhase.INSPECT_HELD_TARGET, 200),
    (ScenarioPhase.GLANCE_AT_NEXT_OBJECT, 200),
    (ScenarioPhase.THROW_TO_FLOOR, 200),
    (ScenarioPhase.SETTLE, 200),
)


@dataclass(frozen=True, slots=True)
class PhaseWindow:
    """One half-open deterministic control window measured in public steps."""

    cycle_index: int
    phase: ScenarioPhase
    target: ScenarioObject
    look_target: ScenarioObject | None
    start_step: int
    stop_step: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.cycle_index, bool)
            or not isinstance(self.cycle_index, int)
            or self.cycle_index < 0
        ):
            raise ValueError("cycle_index must be a non-negative integer")
        if (
            isinstance(self.start_step, bool)
            or not isinstance(self.start_step, int)
            or isinstance(self.stop_step, bool)
            or not isinstance(self.stop_step, int)
            or self.start_step < 0
            or self.stop_step <= self.start_step
        ):
            raise ValueError("phase window must have non-negative increasing steps")
        if self.phase is ScenarioPhase.INSPECT_HELD_TARGET:
            if self.look_target is not self.target:
                raise ValueError("held inspection must look at the held target")
        elif self.phase is ScenarioPhase.GLANCE_AT_NEXT_OBJECT:
            if self.look_target is not next_object(self.target):
                raise ValueError("next-object glance must use cyclic next target")
        elif self.look_target is not None:
            raise ValueError("only gaze phases may define look_target")

    @property
    def duration_steps(self) -> int:
        return self.stop_step - self.start_step


@dataclass(frozen=True, slots=True)
class ScenarioContext:
    """Immutable phase context presented to exactly one controller hook."""

    step_index: int
    nominal_time_seconds: float
    cycle_index: int
    phase: ScenarioPhase
    target: ScenarioObject
    look_target: ScenarioObject | None
    phase_start_step: int
    phase_stop_step: int
    phase_progress: float


@dataclass(frozen=True, slots=True)
class ScenarioSchedule:
    """Validated contiguous schedule for three 15-second object cycles."""

    windows: tuple[PhaseWindow, ...]
    public_step_seconds: float = PUBLIC_STEP_SECONDS

    def __post_init__(self) -> None:
        if not isfinite(self.public_step_seconds) or self.public_step_seconds <= 0.0:
            raise ValueError("public_step_seconds must be finite and positive")
        if not self.windows or self.windows[0].start_step != 0:
            raise ValueError("schedule must contain windows beginning at step zero")
        for previous, current in zip(self.windows, self.windows[1:]):
            if previous.stop_step != current.start_step:
                raise ValueError("schedule windows must be contiguous")
        expected_phases = tuple(phase for phase, _ in _PHASE_STEP_COUNTS)
        for cycle_index, target in enumerate(ScenarioObject):
            cycle = tuple(
                window
                for window in self.windows
                if window.cycle_index == cycle_index
            )
            if tuple(window.phase for window in cycle) != expected_phases:
                raise ValueError(f"cycle {cycle_index} has an invalid phase order")
            if any(window.target is not target for window in cycle):
                raise ValueError(f"cycle {cycle_index} targets the wrong object")
            if cycle[2].look_target is not target:
                raise ValueError(f"cycle {cycle_index} held gaze target changed")
            if cycle[3].look_target is not next_object(target):
                raise ValueError(f"cycle {cycle_index} next glance target changed")
        if {window.cycle_index for window in self.windows} != {0, 1, 2}:
            raise ValueError("schedule must contain exactly cycles 0, 1, and 2")

    @classmethod
    def default(cls) -> ScenarioSchedule:
        """Build the frozen 45-second constraint-assisted storyboard."""

        windows: list[PhaseWindow] = []
        cursor = 0
        for cycle_index, target in enumerate(ScenarioObject):
            for phase, duration_steps in _PHASE_STEP_COUNTS:
                look_target = (
                    target
                    if phase is ScenarioPhase.INSPECT_HELD_TARGET
                    else next_object(target)
                    if phase is ScenarioPhase.GLANCE_AT_NEXT_OBJECT
                    else None
                )
                windows.append(
                    PhaseWindow(
                        cycle_index=cycle_index,
                        phase=phase,
                        target=target,
                        look_target=look_target,
                        start_step=cursor,
                        stop_step=cursor + duration_steps,
                    )
                )
                cursor += duration_steps
        schedule = cls(windows=tuple(windows))
        if schedule.total_steps != SCENARIO_TOTAL_STEPS:
            raise AssertionError("default scenario clock changed")
        return schedule

    @property
    def total_steps(self) -> int:
        return self.windows[-1].stop_step

    @property
    def duration_seconds(self) -> float:
        return self.total_steps * self.public_step_seconds

    def context_at(self, step_index: int) -> ScenarioContext:
        """Resolve one public step to its unique phase context."""

        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or not 0 <= step_index < self.total_steps
        ):
            raise IndexError(f"step_index outside schedule: {step_index!r}")
        window = next(
            candidate
            for candidate in self.windows
            if candidate.start_step <= step_index < candidate.stop_step
        )
        return ScenarioContext(
            step_index=step_index,
            nominal_time_seconds=step_index * self.public_step_seconds,
            cycle_index=window.cycle_index,
            phase=window.phase,
            target=window.target,
            look_target=window.look_target,
            phase_start_step=window.start_step,
            phase_stop_step=window.stop_step,
            phase_progress=(step_index - window.start_step) / window.duration_steps,
        )


@dataclass(frozen=True, slots=True)
class ScenarioEvent:
    """Controller-reported measured or constraint-transition milestone."""

    kind: ScenarioEventKind
    step_index: int
    time_seconds: float
    cycle_index: int
    target: ScenarioObject
    subject: ScenarioObject | None = None
    metrics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or self.step_index < 0
        ):
            raise ValueError("event step_index must be a non-negative integer")
        if not isfinite(self.time_seconds) or self.time_seconds < 0.0:
            raise ValueError("event time_seconds must be finite and non-negative")
        if (
            isinstance(self.cycle_index, bool)
            or not isinstance(self.cycle_index, int)
            or self.cycle_index < 0
        ):
            raise ValueError("event cycle_index must be a non-negative integer")
        metric_names = [name for name, _ in self.metrics]
        if any(not name for name in metric_names) or len(set(metric_names)) != len(
            metric_names
        ):
            raise ValueError("event metric names must be unique and non-empty")
        if not all(isfinite(value) for _, value in self.metrics):
            raise ValueError("event metrics must be finite")


@dataclass(frozen=True, slots=True)
class ControllerStep:
    """Immutable controller action, state, events, and scalar diagnostics."""

    action: NDArray[np.float32]
    state: object
    events: tuple[ScenarioEvent, ...] = ()
    diagnostics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(self.action)
        if values.dtype != np.float32 or values.ndim != 1:
            raise ValueError("controller action must be a one-dimensional float32 array")
        if not np.all(np.isfinite(values)):
            raise ValueError("controller action must contain only finite values")
        immutable = np.frombuffer(
            np.ascontiguousarray(values).tobytes(order="C"), dtype=np.float32
        )
        object.__setattr__(self, "action", immutable)
        names = [name for name, _ in self.diagnostics]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("diagnostic names must be unique and non-empty")
        if not all(isfinite(value) for _, value in self.diagnostics):
            raise ValueError("diagnostics must be finite")


class ScenarioControllerHooks(Protocol):
    """Controller boundary allowing only disclosed mocap/equality mechanisms."""

    @property
    def controller_id(self) -> str: ...

    def reset(
        self,
        environment: ThreeObjectScenarioEnv,
        schedule: ScenarioSchedule,
        seed: int,
    ) -> object: ...

    def compose_locomotion(
        self,
        environment: ThreeObjectScenarioEnv,
        state: object,
        context: ScenarioContext,
    ) -> ControllerStep: ...

    def compose_grasp(
        self,
        environment: ThreeObjectScenarioEnv,
        state: object,
        context: ScenarioContext,
    ) -> ControllerStep: ...

    def compose_gaze(
        self,
        environment: ThreeObjectScenarioEnv,
        state: object,
        context: ScenarioContext,
    ) -> ControllerStep: ...

    def compose_release(
        self,
        environment: ThreeObjectScenarioEnv,
        state: object,
        context: ScenarioContext,
    ) -> ControllerStep: ...

    def compose_settle(
        self,
        environment: ThreeObjectScenarioEnv,
        state: object,
        context: ScenarioContext,
    ) -> ControllerStep: ...


class ControllerHandoffPendingError(RuntimeError):
    """Raised before simulation when active hooks are unavailable."""


def require_controller(
    hooks: ScenarioControllerHooks | None,
) -> ScenarioControllerHooks:
    if hooks is None:
        raise ControllerHandoffPendingError(
            "constraint-assisted locomotion/grip/gaze/release hooks are required"
        )
    controller_id = hooks.controller_id
    if not isinstance(controller_id, str) or not controller_id.strip():
        raise ValueError("controller_id must be a non-empty string")
    return hooks


def dispatch_controller(
    hooks: ScenarioControllerHooks,
    environment: ThreeObjectScenarioEnv,
    state: object,
    context: ScenarioContext,
) -> ControllerStep:
    """Route a context to exactly one calibrated behavior hook."""

    if context.phase is ScenarioPhase.GO_TO_TARGET:
        step = hooks.compose_locomotion(environment, state, context)
    elif context.phase is ScenarioPhase.PICK_TARGET:
        step = hooks.compose_grasp(environment, state, context)
    elif context.phase in {
        ScenarioPhase.INSPECT_HELD_TARGET,
        ScenarioPhase.GLANCE_AT_NEXT_OBJECT,
    }:
        step = hooks.compose_gaze(environment, state, context)
    elif context.phase is ScenarioPhase.THROW_TO_FLOOR:
        step = hooks.compose_release(environment, state, context)
    else:
        step = hooks.compose_settle(environment, state, context)
    if not isinstance(step, ControllerStep):
        raise TypeError("controller hooks must return ControllerStep")
    return step


class ThreeObjectScenarioEnv(BlightWorldEnv):
    """Basic room plus the exact calibrated table, free objects, and anchors."""

    _scene_extension_xml = CONSTRAINT_ASSISTED_SCENE_EXTENSION_XML
    object_names = tuple(SCENARIO_BODY_NAMES.values())

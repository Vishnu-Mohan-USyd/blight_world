"""Local adapters for MIMo's existing actuation and endpoint senses."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from types import SimpleNamespace
from typing import Any, Protocol, TypeAlias

import mujoco
import numpy as np
from mimoActuation.actuation import ActuationModel
from mimoActuation.muscle import MuscleModel
from mimoVision.vision import SimpleVision
from numpy.typing import NDArray


class CorrectedMuscleModel(MuscleModel):
    """Advance MIMo activation once per physics substep, using its own curves."""

    def action(self, action: NDArray[Any]) -> None:
        self.control_input = np.clip(
            action, self.action_space.low, self.action_space.high
        ).copy()
        self.target_activity = self.control_input.copy()

    def _update_muscle_state(self) -> None:
        self._update_virtual_lengths()
        self._update_virtual_velocities()
        self._update_torque()

    def substep_update(self) -> None:
        self._update_activity()
        self._update_muscle_state()
        self._apply_torque()
        self.env.data.ctrl[self.actuators] = 1.0

    def reset(self) -> None:
        self.control_input = np.zeros(self.action_space.shape, dtype=np.float32)
        self._set_initial_muscle_state()
        self._apply_torque()
        self.env.data.ctrl[self.actuators] = 1.0


class SensorContext:
    """Sensor-owned state; the environment's authoritative data is never swapped."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        actuation_model: ActuationModel,
        age: float | None,
    ) -> None:
        self.model = model
        self.age = age
        self.refresh(data, actuation_model)

    def refresh(
        self, data: mujoco.MjData, actuation_model: ActuationModel
    ) -> None:
        endpoint = copy(data)
        mujoco.mj_forward(self.model, endpoint)
        self.data = endpoint
        evaluator = copy(actuation_model)
        evaluator.env = self
        evaluator.control_input = np.asarray(actuation_model.control_input).copy()
        if isinstance(evaluator, MuscleModel):
            evaluator.activity = actuation_model.activity.copy()
            evaluator.target_activity = actuation_model.target_activity.copy()
            evaluator._update_virtual_lengths()
            evaluator._update_virtual_velocities()
            evaluator._update_torque()
        self.actuation_model = evaluator


VisionParameters: TypeAlias = dict[str, dict[str, Any]]


class VisionProvider(Protocol):
    def get_vision_obs(self) -> dict[str, NDArray[np.uint8]]: ...

    def close(self) -> None: ...


VisionProviderFactory: TypeAlias = Callable[
    [SensorContext, VisionParameters], VisionProvider
]


class CorrectedSimpleVision(SimpleVision):
    """Keep native vision behavior while using the requested acuity age."""

    def _get_acuity_function(
        self, width: int, height: int, acuity_age: float, fovy: float
    ) -> NDArray[Any]:
        # The pinned implementation indexes exact age knots through env.age.
        # Give that calculation its requested filter age without changing body age.
        calculation = copy(self)
        calculation.env = SimpleNamespace(age=acuity_age)
        return SimpleVision._get_acuity_function(
            calculation, width, height, acuity_age, fovy
        )


class BiologicalVision(CorrectedSimpleVision):
    """Apply existing MIMo vision filters to native or externally supplied RGB."""

    def __init__(
        self,
        context: SensorContext,
        camera_parameters: VisionParameters,
        provider: VisionProvider | None = None,
    ) -> None:
        super().__init__(context, camera_parameters)
        self._provider = provider
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def get_vision_obs(self) -> dict[str, NDArray[np.uint8]]:
        if self._provider is not None:
            images = self._provider.get_vision_obs()
        else:
            images = {}
            for camera, parameters in self.camera_parameters.items():
                size = (parameters["height"], parameters["width"])
                if size not in self._renderers:
                    self._renderers[size] = mujoco.Renderer(
                        self.env.model, height=size[0], width=size[1]
                    )
                renderer = self._renderers[size]
                renderer.update_scene(self.env.data, camera=camera)
                images[camera] = renderer.render().copy()
        outputs = {}
        for camera, parameters in self.camera_parameters.items():
            image = np.asarray(images[camera])
            expected = (parameters["height"], parameters["width"], 3)
            if image.shape != expected or image.dtype != np.uint8:
                raise ValueError(f"{camera} must provide uint8 RGB with shape {expected}")
            if self._acuity_functions[camera] is not None:
                image = self._apply_acuity(image, camera)
            if self._foveation[camera]:
                image = self._apply_foveation(image, self._foveation[camera])
            outputs[camera] = np.ascontiguousarray(image).copy()
        self.sensor_outputs = outputs
        return outputs

    def close(self) -> None:
        try:
            if self._provider is not None:
                self._provider.close()
        finally:
            for renderer in self._renderers.values():
                renderer.close()
            self._renderers.clear()

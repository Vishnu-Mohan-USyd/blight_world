"""MIMo worlds with configurable body senses, vision and numerical audition."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from html import escape
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mimoEnv
import mujoco
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv
from mimoActuation.actuation import ActuationModel, SpringDamperModel
from mimoEnv.envs.mimo_env import (
    DEFAULT_PROPRIOCEPTION_PARAMS,
    DEFAULT_TOUCH_PARAMS_V2,
    DEFAULT_VESTIBULAR_PARAMS,
    EMOTES,
    MIMoEnv,
)
from mimoProprioception.proprio import SimpleProprioception
from mimoTouch.touch import TrimeshTouch
from mimoVestibular.vestibular import SimpleVestibular
from mimo_audition import AuditionConfig, MIMoAudition
from numpy.typing import NDArray

from blight_world.biology import (
    BiologicalVision,
    CorrectedSimpleVision,
    SensorContext,
    VisionParameters,
    VisionProviderFactory,
)

_TEXTURE_PLACEHOLDER = "{{MIMO_TEXTURE_DIR}}"
_META_PLACEHOLDER = "{{MIMO_META_XML}}"
_MODEL_PLACEHOLDER = "{{MIMO_MODEL_XML}}"
_SCENE_EXTENSION_PLACEHOLDER = "{{SCENE_EXTENSION_XML}}"


def _xml_path(path: Path) -> str:
    """Return an absolute POSIX path escaped for use in an XML attribute."""

    return escape(path.resolve().as_posix(), quote=True)


class BlightWorldEnv(MIMoEnv):
    """MIMo in a static room with binocular vision and three table objects.

    Parameters
    ----------
    width:
        Width in pixels of each eye image. Must be a positive integer.
    height:
        Height in pixels of each eye image. Must be a positive integer.
    proprio_params:
        Optional MIMo proprioception configuration. ``None`` preserves this
        environment's minimal joint-position-only default.
    touch_params, vestibular_params:
        Optional upstream MIMo modality configurations. ``None`` keeps each
        modality disabled as in the original environment.
    actuation_model:
        Upstream MIMo actuation-model class. The default remains the pinned
        spring-damper torque model.
    motor_delay:
        Number of complete public steps between a command and its application.
    sensory_delay:
        Number of complete public steps delaying the entire observation dictionary.
    snapshot_sensors:
        Evaluate sensors on a refreshed independent copy of the integrated state.
    vision_provider_factory:
        Optional factory receiving the sensor context and camera parameters.
        Supplying it enables snapshots and replaces native RGB capture.
    biology_profile:
        ``full`` enables snapshots and the existing full MIMo proprioception,
        touch and vestibular defaults. ``legacy`` preserves the minimal defaults.
    scene_template:
        Optional MJCF template path containing the existing MIMo and scene-extension
        placeholders. ``None`` uses the packaged basic room.

    Notes
    -----
    The MuJoCo scene is materialized in an instance-owned temporary directory.
    Installed MIMo assets are referenced read-only and are never rewritten.
    """

    _scene_extension_xml = ""

    def __init__(
        self,
        width: int = 96,
        height: int = 96,
        audition: AuditionConfig | None = None,
        proprio_params: dict[str, Any] | None = None,
        touch_params: dict[str, Any] | None = None,
        vestibular_params: dict[str, Any] | None = None,
        actuation_model: type[ActuationModel] = SpringDamperModel,
        motor_delay: int = 0,
        *,
        snapshot_sensors: bool = False,
        sensory_delay: int = 0,
        vision_params: VisionParameters | None = None,
        vision_provider_factory: VisionProviderFactory | None = None,
        age: float | None = None,
        biology_profile: str = "legacy",
        scene_template: str | Path | None = None,
    ) -> None:
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError("height must be a positive integer")
        if audition is not None and not isinstance(audition, AuditionConfig):
            raise TypeError("audition must be an AuditionConfig or None")
        if biology_profile not in ("legacy", "full"):
            raise ValueError("biology_profile must be 'legacy' or 'full'")
        if not isinstance(snapshot_sensors, bool):
            raise TypeError("snapshot_sensors must be a bool")
        if vision_provider_factory is not None and not callable(vision_provider_factory):
            raise TypeError("vision_provider_factory must be callable or None")
        if vision_params is not None and not isinstance(vision_params, dict):
            raise TypeError("vision_params must be a dict or None")
        for name, parameters in (
            ("proprio_params", proprio_params),
            ("touch_params", touch_params),
            ("vestibular_params", vestibular_params),
        ):
            if parameters is not None and not isinstance(parameters, dict):
                raise TypeError(f"{name} must be a dict or None")
        if (
            not isinstance(actuation_model, type)
            or not issubclass(actuation_model, ActuationModel)
        ):
            raise TypeError("actuation_model must be an ActuationModel class")
        for name, delay in (("motor_delay", motor_delay), ("sensory_delay", sensory_delay)):
            if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        self._audition_config = audition
        self.audition: MIMoAudition | None = None
        self._closed = False
        self.snapshot_sensors = (
            snapshot_sensors or biology_profile == "full"
            or vision_provider_factory is not None
            or (audition is not None and audition.backend == "steam_audio")
        )
        self.biology_profile = biology_profile
        self._vision_provider_factory = vision_provider_factory
        self.sensor_context: SensorContext | None = None
        self._observation_cache: dict[str, NDArray[Any]] | None = None
        self._latest_raw_observation: dict[str, NDArray[Any]] | None = None
        self._delivered_observation: dict[str, NDArray[Any]] | None = None
        self._motor_history: deque[NDArray[Any]] = deque()
        self._sensory_history: deque[dict[str, NDArray[Any]]] = deque()
        self._during_step = False
        if biology_profile == "full":
            if proprio_params is None:
                proprio_params = deepcopy(DEFAULT_PROPRIOCEPTION_PARAMS)
            if touch_params is None:
                touch_params = deepcopy(DEFAULT_TOUCH_PARAMS_V2)
            if vestibular_params is None:
                vestibular_params = deepcopy(DEFAULT_VESTIBULAR_PARAMS)

        upstream_assets = Path(mimoEnv.__file__).resolve().parent / "assets"
        meta_xml = upstream_assets / "mimo" / "MIMo_metav2.xml"
        model_xml = upstream_assets / "mimo" / "MIMo_modelv2.xml"
        if not upstream_assets.is_dir():
            raise FileNotFoundError(f"MIMo asset directory not found: {upstream_assets}")
        for required_file in (meta_xml, model_xml):
            if not required_file.is_file():
                raise FileNotFoundError(f"Required MIMo asset not found: {required_file}")

        if scene_template is None:
            template = (
                resources.files("blight_world")
                .joinpath("assets", "basic_room.xml.template")
                .read_text(encoding="utf-8")
            )
        else:
            scene_template = Path(scene_template).expanduser().resolve()
            template = scene_template.read_text(encoding="utf-8")
        replacements = {
            _TEXTURE_PLACEHOLDER: _xml_path(upstream_assets),
            _META_PLACEHOLDER: _xml_path(meta_xml),
            _MODEL_PLACEHOLDER: _xml_path(model_xml),
            _SCENE_EXTENSION_PLACEHOLDER: self._scene_extension_xml,
        }
        scene_xml = template
        for placeholder, value in replacements.items():
            if placeholder not in scene_xml:
                raise ValueError(f"Scene template is missing placeholder {placeholder}")
            scene_xml = scene_xml.replace(placeholder, value)

        self._scene_directory: TemporaryDirectory[str] | None = TemporaryDirectory(
            prefix="blight-world-"
        )
        self._scene_path = Path(self._scene_directory.name) / "basic_room.xml"
        self._scene_path.write_text(scene_xml, encoding="utf-8")

        default_vision: VisionParameters = {
            eye: {
                "width": width,
                "height": height,
                "fovy": 60,
                "acuity": False,
                "foveation": False,
            }
            for eye in ("eye_left", "eye_right")
        }
        if vision_params is not None:
            for eye, parameters in vision_params.items():
                default_vision[eye].update(deepcopy(parameters))
        vision_params = default_vision

        try:
            super().__init__(
                model_path=str(self._scene_path),
                width=width,
                height=height,
                age=age,
                proprio_params=(
                    {"components": []}
                    if proprio_params is None
                    else proprio_params
                ),
                touch_params=touch_params,
                vision_params=vision_params,
                vestibular_params=vestibular_params,
                actuation_model=actuation_model,
                motor_delay=motor_delay,
                sensory_delay=sensory_delay,
                goals_in_observation=False,
                done_active=False,
                render_mode=None,
            )
        except BaseException:
            self._scene_directory.cleanup()
            self._scene_directory = None
            raise
        utils.EzPickle.__init__(
            self,
            width=width,
            height=height,
            audition=audition,
            proprio_params=proprio_params,
            touch_params=touch_params,
            vestibular_params=vestibular_params,
            actuation_model=actuation_model,
            motor_delay=motor_delay,
            snapshot_sensors=snapshot_sensors,
            sensory_delay=sensory_delay,
            vision_params=vision_params,
            vision_provider_factory=vision_provider_factory,
            age=age,
            biology_profile=biology_profile,
            scene_template=scene_template,
        )

    @property
    def scene_path(self) -> Path:
        """Path to this instance's materialized MuJoCo scene."""

        return self._scene_path

    def _initialize_simulation(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Initialize MuJoCo while preserving MIMo's post-load setup."""

        model, data = MujocoEnv._initialize_simulation(self)
        self.model = model
        self.data = data

        fps = int(np.round(1 / self.dt))
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": fps,
        }
        self._get_joints()
        self._get_actuators()
        self._get_facial_expressions(EMOTES)
        self._set_initial_position(self._initial_qpos)
        self.actuation_model = self.actuation_model(self, self.mimo_actuators)
        return model, data

    def _env_setup(self) -> None:
        """Initialize upstream senses, then the optional auditory sensor."""

        if self.snapshot_sensors:
            self.sensor_context = SensorContext(
                self.model, self.data, self.actuation_model, self.age
            )
        super()._env_setup()
        if self._audition_config is not None:
            self.audition = MIMoAudition(self.model, self.data, self._audition_config)

    def proprio_setup(self, proprio_params: dict[str, Any]) -> None:
        context = self.sensor_context if self.snapshot_sensors else self
        self.proprioception = SimpleProprioception(context, proprio_params)

    def touch_setup(self, touch_params: dict[str, Any]) -> None:
        context = self.sensor_context if self.snapshot_sensors else self
        self.touch = TrimeshTouch(context, touch_params)

    def vestibular_setup(self, vestibular_params: dict[str, Any]) -> None:
        context = self.sensor_context if self.snapshot_sensors else self
        self.vestibular = SimpleVestibular(context, vestibular_params)

    def vision_setup(self, vision_params: VisionParameters) -> None:
        for parameters in vision_params.values():
            if parameters["acuity"] is True:
                parameters["acuity"] = self.age
            if parameters["foveation"] is True:
                parameters["foveation"] = 10.0
        if not self.snapshot_sensors:
            self.vision = CorrectedSimpleVision(self, vision_params)
            return
        assert self.sensor_context is not None
        provider = (
            self._vision_provider_factory(self.sensor_context, vision_params)
            if self._vision_provider_factory is not None else None
        )
        self.vision = BiologicalVision(self.sensor_context, vision_params, provider)

    def _refresh_sensor_context(self) -> None:
        if self.sensor_context is not None:
            self.sensor_context.refresh(self.data, self.actuation_model)
            self._observation_cache = None

    def _set_observation_space(self) -> None:
        """Extend MIMo's hard-coded modality space with binaural pressure."""

        super()._set_observation_space()
        if self._audition_config is not None:
            modality_spaces = dict(self.observation_space.spaces)
            modality_spaces["audition"] = spaces.Box(
                -np.inf,
                np.inf,
                shape=self._audition_config.spec.observation_shape,
                dtype=np.float32,
            )
            self.observation_space = spaces.Dict(modality_spaces)
        assert self._latest_raw_observation is not None
        self._prime_sensory_history(self._latest_raw_observation)

    def _set_action_space(self) -> None:
        super()._set_action_space()
        self._prime_motor_history()

    def _prime_motor_history(self) -> None:
        self._motor_history.clear()
        for _ in range(self.motor_delay):
            self._motor_history.append(
                np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
            )

    @staticmethod
    def _copy_observation(
        observation: dict[str, NDArray[Any]],
    ) -> dict[str, NDArray[Any]]:
        return {key: value.copy() for key, value in observation.items()}

    def _prime_sensory_history(self, observation: dict[str, NDArray[Any]]) -> None:
        self._sensory_history.clear()
        for _ in range(self.sensory_delay):
            self._sensory_history.append(self._copy_observation(observation))
        self._delivered_observation = self._copy_observation(observation)

    def _delayed_action(self, action: NDArray[Any]) -> NDArray[Any]:
        self._motor_history.append(np.asarray(action).copy())
        return self._motor_history.popleft()

    def _delayed_observation(
        self, observation: dict[str, NDArray[Any]],
    ) -> dict[str, NDArray[Any]]:
        # Delay only after audio and all other modalities have been assembled.
        return observation

    def step(self, action: NDArray[Any]) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._during_step = True
        try:
            observation, reward, terminated, truncated, info = super().step(action)
        finally:
            self._during_step = False
        if self.sensory_delay:
            self._sensory_history.append(self._copy_observation(observation))
            observation = self._sensory_history.popleft()
        self._delivered_observation = self._copy_observation(observation)
        return observation, reward, terminated, truncated, info

    def _get_obs(self) -> dict[str, NDArray[Any]]:
        """Return upstream observations plus an idempotent audio cache read."""

        if (
            self.sensory_delay and not self._during_step
            and self._delivered_observation is not None
        ):
            return self._copy_observation(self._delivered_observation)
        if self.snapshot_sensors and self._observation_cache is not None:
            return self._copy_observation(self._observation_cache)
        observation = super()._get_obs()
        if self.audition is not None:
            observation["audition"] = self.audition.observation
        if self.snapshot_sensors:
            self._observation_cache = self._copy_observation(observation)
        self._latest_raw_observation = self._copy_observation(observation)
        return observation

    def _single_mujoco_step(self) -> None:
        """Capture pre-step contact kinematics, then advance MuJoCo once."""

        if self.audition is not None:
            if self.sensor_context is None:
                self.audition.before_substep()
            else:
                self.audition.before_substep(kinematic_data=self.sensor_context.data)
        super()._single_mujoco_step()

    def _substep_callback(self) -> None:
        """Render one acoustic block after every 5 ms physics substep."""

        super()._substep_callback()
        if self.audition is not None:
            if self.sensor_context is None:
                self.audition.capture_substep()
            else:
                self.audition.capture_substep(pose_data=self.sensor_context.data)
        self._refresh_sensor_context()

    def _step_callback(self) -> None:
        """Publish the two rendered substeps before observations are read."""

        super()._step_callback()
        if self.audition is not None:
            self.audition.finish_step()

    def render(self) -> NDArray[Any] | None:
        """Render from the camera selected through MIMo's legacy interface."""

        if self.camera_id is not None and self.camera_name is not None:
            raise ValueError("camera_id and camera_name cannot both be set")
        camera_id = self.camera_id
        if camera_id is None:
            camera_name = self.camera_name if self.camera_name is not None else "track"
            camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )
        self.mujoco_renderer.camera_id = camera_id
        return self.mujoco_renderer.render(self.render_mode)

    def reset_model(self) -> dict[str, NDArray[Any]]:
        """Restore the complete initial MuJoCo state and return observations."""

        self._observation_cache = None
        self._delivered_observation = None
        self.set_state(self.init_qpos, self.init_qvel)
        self.actuation_model.reset()
        mujoco.mj_forward(self.model, self.data)
        if self.audition is not None:
            self.audition.reset(self.np_random_seed)
        self._refresh_sensor_context()
        observation = self._get_obs()
        self._prime_motor_history()
        self._prime_sensory_history(observation)
        return observation

    def sample_goal(self) -> NDArray[np.float64]:
        """Return the empty goal required by the goal-environment interface."""

        return np.empty((0,), dtype=np.float64)

    def get_achieved_goal(self) -> NDArray[np.float64]:
        """Return the empty achieved goal for this non-task environment."""

        return np.empty((0,), dtype=np.float64)

    def is_success(
        self,
        achieved_goal: NDArray[Any],
        desired_goal: NDArray[Any],
    ) -> bool:
        """Report no task success because this world defines no task."""

        del achieved_goal, desired_goal
        return False

    def is_failure(
        self,
        achieved_goal: NDArray[Any],
        desired_goal: NDArray[Any],
    ) -> bool:
        """Report no task failure because this world defines no task."""

        del achieved_goal, desired_goal
        return False

    def is_truncated(self) -> bool:
        """Keep the neutral world untruncated."""

        return False

    def compute_reward(
        self,
        achieved_goal: NDArray[Any],
        desired_goal: NDArray[Any],
        info: dict[str, Any],
    ) -> float:
        """Return the neutral scalar required by the Gymnasium step API."""

        del achieved_goal, desired_goal, info
        return 0.0

    def close(self) -> None:
        """Close MIMo renderers and remove the instance-owned scene directory."""

        try:
            if not self._closed:
                try:
                    vision = getattr(self, "vision", None)
                    if isinstance(vision, BiologicalVision):
                        vision.close()
                    close_audio = getattr(self.audition, "close", None)
                    if close_audio is not None:
                        close_audio()
                    renderer = getattr(self, "mujoco_renderer", None)
                    if renderer is not None:
                        for viewer in tuple(renderer._viewers.values()):
                            viewer.make_context_current()
                            viewer.con.free()
                finally:
                    try:
                        super().close()
                    finally:
                        self._closed = True
        finally:
            scene_directory = self._scene_directory
            if scene_directory is not None:
                scene_directory.cleanup()
                self._scene_directory = None

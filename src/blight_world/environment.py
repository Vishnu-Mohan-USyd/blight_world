"""A minimal vision-only room environment for MIMo v2."""

from __future__ import annotations

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
from mimoEnv.envs.mimo_env import EMOTES, MIMoEnv
from mimo_audition import AuditionConfig, MIMoAudition
from numpy.typing import NDArray


_TEXTURE_PLACEHOLDER = "{{MIMO_TEXTURE_DIR}}"
_META_PLACEHOLDER = "{{MIMO_META_XML}}"
_MODEL_PLACEHOLDER = "{{MIMO_MODEL_XML}}"


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

    Notes
    -----
    The MuJoCo scene is materialized in an instance-owned temporary directory.
    Installed MIMo assets are referenced read-only and are never rewritten.
    """

    def __init__(
        self,
        width: int = 96,
        height: int = 96,
        audition: AuditionConfig | None = None,
    ) -> None:
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError("height must be a positive integer")
        if audition is not None and not isinstance(audition, AuditionConfig):
            raise TypeError("audition must be an AuditionConfig or None")

        self._audition_config = audition
        self.audition: MIMoAudition | None = None
        self._closed = False

        upstream_assets = Path(mimoEnv.__file__).resolve().parent / "assets"
        meta_xml = upstream_assets / "mimo" / "MIMo_metav2.xml"
        model_xml = upstream_assets / "mimo" / "MIMo_modelv2.xml"
        if not upstream_assets.is_dir():
            raise FileNotFoundError(f"MIMo asset directory not found: {upstream_assets}")
        for required_file in (meta_xml, model_xml):
            if not required_file.is_file():
                raise FileNotFoundError(f"Required MIMo asset not found: {required_file}")

        template = (
            resources.files("blight_world")
            .joinpath("assets", "basic_room.xml.template")
            .read_text(encoding="utf-8")
        )
        replacements = {
            _TEXTURE_PLACEHOLDER: _xml_path(upstream_assets),
            _META_PLACEHOLDER: _xml_path(meta_xml),
            _MODEL_PLACEHOLDER: _xml_path(model_xml),
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

        vision_params: dict[str, dict[str, int | float | bool]] = {
            eye: {
                "width": width,
                "height": height,
                "fovy": 60,
                "acuity": False,
                "foveation": False,
            }
            for eye in ("eye_left", "eye_right")
        }

        try:
            super().__init__(
                model_path=str(self._scene_path),
                width=width,
                height=height,
                age=None,
                proprio_params={"components": []},
                touch_params=None,
                vision_params=vision_params,
                vestibular_params=None,
                goals_in_observation=False,
                done_active=False,
                render_mode=None,
            )
        except BaseException:
            self._scene_directory.cleanup()
            self._scene_directory = None
            raise
        utils.EzPickle.__init__(self, width=width, height=height, audition=audition)

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

        super()._env_setup()
        if self._audition_config is not None:
            self.audition = MIMoAudition(self.model, self.data, self._audition_config)

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

    def _get_obs(self) -> dict[str, NDArray[Any]]:
        """Return upstream observations plus an idempotent audio cache read."""

        observation = super()._get_obs()
        if self.audition is not None:
            observation["audition"] = self.audition.observation
        return observation

    def _single_mujoco_step(self) -> None:
        """Capture pre-step contact kinematics, then advance MuJoCo once."""

        if self.audition is not None:
            self.audition.before_substep()
        super()._single_mujoco_step()

    def _substep_callback(self) -> None:
        """Render one acoustic block after every 5 ms physics substep."""

        super()._substep_callback()
        if self.audition is not None:
            self.audition.capture_substep()

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

        self.set_state(self.init_qpos, self.init_qvel)
        mujoco.mj_forward(self.model, self.data)
        if self.audition is not None:
            self.audition.reset(self.np_random_seed)
        return self._get_obs()

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

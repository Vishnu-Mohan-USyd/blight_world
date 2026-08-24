"""Visible free-ball fixture for the scripted unsupported belly crawl.

The fixture preserves the validated open-loop actuator sequence and adds one
small, free red sphere in the left-hand crawl path.  The table is moved to a
physics-neutral room corner and the unused block/cylinder are non-colliding.
After reset, neither this environment nor its contact recorder writes object
state, applied forces, model parameters, equality state, or mocap state.

This is a scripted belly/army-crawl interaction fixture.  It is not a turn,
pick, lift, grasp, learned policy, or autonomous object-manipulation skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from importlib import resources
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mimoEnv
import mujoco
import numpy as np
from gymnasium import utils
from mimoActuation.actuation import SpringDamperModel
from mimoEnv.envs.mimo_env import MIMoEnv
from mimo_audition import AuditionConfig
from numpy.typing import NDArray

from blight_world.locomotion_skills import UnsupportedStaticSupportEnv


CRAWL_BALL_INITIAL_POSITION_M = (0.540, 0.200, 0.025)
CRAWL_BALL_RADIUS_M = 0.025
CRAWL_BALL_MASS_KG = 0.150
CRAWL_BALL_FRICTION = (1.1, 0.2, 0.10)
CRAWL_BALL_TABLE_POSITION_M = (-1.8, 1.8, 0.0)
CRAWL_BALL_CONTACT_GEOM_NAMES = frozenset(
    {"geom:left_ffdistal1", "geom:left_mfdistal1"}
)

_TEXTURE_PLACEHOLDER = "{{MIMO_TEXTURE_DIR}}"
_META_PLACEHOLDER = "{{MIMO_META_XML}}"
_MODEL_PLACEHOLDER = "{{MIMO_MODEL_XML}}"
_SCENE_EXTENSION_PLACEHOLDER = "{{SCENE_EXTENSION_XML}}"


def _xml_path(path: Path) -> str:
    """Return an absolute XML-safe POSIX path."""

    return escape(path.resolve().as_posix(), quote=True)


@dataclass(frozen=True, slots=True)
class CrawlBallContact:
    """One solved MIMo-to-ball contact sampled after a 5 ms substep."""

    public_step_index: int
    physics_substep_index: int
    time_seconds: float
    mimo_geom_name: str
    ball_geom_name: str
    position_m: tuple[float, float, float]
    normal_force_n: float


class CrawlBallContactRecorder:
    """Read-only deterministic recorder for solved MIMo/ball contacts."""

    __slots__ = (
        "_ball_geom_id",
        "_environment",
        "_mimo_root_body_id",
        "_records",
        "_substep_index",
    )

    def __init__(self, environment: CrawlBallFixtureEnv) -> None:
        self._environment = environment
        self._ball_geom_id = int(environment.model.geom("ball_geom").id)
        self._mimo_root_body_id = int(environment.model.body("mimo_location").id)
        self._records: list[CrawlBallContact] = []
        self._substep_index = 0

    @property
    def records(self) -> tuple[CrawlBallContact, ...]:
        """Return an immutable snapshot in MuJoCo contact-array order."""

        return tuple(self._records)

    def reset(self) -> None:
        """Clear records and restart the deterministic substep clock."""

        self._records.clear()
        self._substep_index = 0

    def capture_substep(self) -> None:
        """Read the current solved contacts without modifying simulation state."""

        model = self._environment.model
        data = self._environment.data
        public_step_index = self._substep_index // 2
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if int(contact.efc_address) < 0:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self._ball_geom_id not in (geom1, geom2):
                continue
            opposing_geom = geom2 if geom1 == self._ball_geom_id else geom1
            opposing_body = int(model.geom_bodyid[opposing_geom])
            if int(model.body_rootid[opposing_body]) != self._mimo_root_body_id:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, force)
            mimo_geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, opposing_geom
            )
            ball_geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, self._ball_geom_id
            )
            if mimo_geom_name is None or ball_geom_name is None:
                raise ValueError("crawl-ball contact geom must have a name")
            self._records.append(
                CrawlBallContact(
                    public_step_index=public_step_index,
                    physics_substep_index=self._substep_index,
                    time_seconds=float(data.time),
                    mimo_geom_name=mimo_geom_name,
                    ball_geom_name=ball_geom_name,
                    position_m=tuple(float(value) for value in contact.pos),
                    normal_force_n=max(float(force[0]), 0.0),
                )
            )
        self._substep_index += 1


class CrawlBallFixtureEnv(UnsupportedStaticSupportEnv):
    """Unsupported crawl scene with one visible, free, contact-driven sphere.

    The class inherits :class:`BlightWorldEnv`'s camera-name-aware render path,
    audition lifecycle, Gym API, and safe renderer cleanup.  Its constructor
    differs only by materializing the dedicated packaged scene template.
    """

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        audition: AuditionConfig | None = None,
    ) -> None:
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError("height must be a positive integer")
        if audition is not None and not isinstance(audition, AuditionConfig):
            raise TypeError("audition must be an AuditionConfig or None")

        self._audition_config = audition
        self.audition = None
        self._closed = False
        self._contact_recorder: CrawlBallContactRecorder | None = None

        upstream_assets = Path(mimoEnv.__file__).resolve().parent / "assets"
        meta_xml = upstream_assets / "mimo" / "MIMo_metav2.xml"
        model_xml = upstream_assets / "mimo" / "MIMo_modelv2.xml"
        if not upstream_assets.is_dir():
            raise FileNotFoundError(f"MIMo asset directory not found: {upstream_assets}")
        for required_file in (meta_xml, model_xml):
            if not required_file.is_file():
                raise FileNotFoundError(
                    f"Required MIMo asset not found: {required_file}"
                )

        template = (
            resources.files("blight_world")
            .joinpath("assets", "crawl_ball_room.xml.template")
            .read_text(encoding="utf-8")
        )
        replacements = {
            _TEXTURE_PLACEHOLDER: _xml_path(upstream_assets),
            _META_PLACEHOLDER: _xml_path(meta_xml),
            _MODEL_PLACEHOLDER: _xml_path(model_xml),
            _SCENE_EXTENSION_PLACEHOLDER: "",
        }
        scene_xml = template
        for placeholder, value in replacements.items():
            if placeholder not in scene_xml:
                raise ValueError(f"Scene template is missing placeholder {placeholder}")
            scene_xml = scene_xml.replace(placeholder, value)

        self._scene_directory: TemporaryDirectory[str] | None = TemporaryDirectory(
            prefix="blight-world-crawl-ball-"
        )
        self._scene_path = Path(self._scene_directory.name) / "crawl_ball_room.xml"
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
            MIMoEnv.__init__(
                self,
                model_path=str(self._scene_path),
                width=width,
                height=height,
                age=None,
                proprio_params={"components": []},
                touch_params=None,
                vision_params=vision_params,
                vestibular_params=None,
                actuation_model=SpringDamperModel,
                motor_delay=0,
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
        )

    def start_contact_recording(self) -> CrawlBallContactRecorder:
        """Install and return a fresh read-only 5 ms contact recorder."""

        recorder = CrawlBallContactRecorder(self)
        self._contact_recorder = recorder
        return recorder

    def stop_contact_recording(self) -> CrawlBallContactRecorder | None:
        """Detach and return the current recorder, if one is installed."""

        recorder = self._contact_recorder
        self._contact_recorder = None
        return recorder

    def _substep_callback(self) -> None:
        super()._substep_callback()
        if self._contact_recorder is not None:
            self._contact_recorder.capture_substep()

    def reset_model(self) -> dict[str, NDArray[Any]]:
        observation = super().reset_model()
        if self._contact_recorder is not None:
            self._contact_recorder.reset()
        return observation

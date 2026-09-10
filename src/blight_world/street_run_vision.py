"""The street scene's binocular renderer and one full-body following camera."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from blight_world.unreal_vision import UnrealVisionProvider, _position, _REFLECT


_ROOT = Path(__file__).resolve().parents[2]
_SPECTATOR = {"width": 960, "height": 540, "fovy": 64.0}


def make_street_run_vision_provider(context, vision_params):
    return StreetRunVisionProvider(context, vision_params)


class StreetRunVisionProvider(UnrealVisionProvider):
    """Capture the two biological eyes and a spectator from one copied endpoint."""

    def __init__(self, context, vision_params, *, config_file=None, asset_manifest=None):
        self._spectator_actor = None
        self._spectator_capture = None
        self._spectator_rgb = None
        self._captured_snapshot = None
        self._mimo_root = mujoco.mj_name2id(context.model, mujoco.mjtObj.mjOBJ_BODY, "mimo_location")
        if self._mimo_root < 0:
            raise ValueError("The street model requires the original mimo_location body")
        asset_manifest = asset_manifest or os.environ.get("BLIGHT_UNREAL_ASSET_MANIFEST", _ROOT / "artifacts/unreal/street_assets/manifest.json")
        super().__init__(context, vision_params, config_file=config_file, asset_manifest=asset_manifest)
        try:
            with self._instance.begin_frame():
                # Reuse the verified Blueprint capture setup, keeping this view
                # outside the biological camera dictionary and its observation.
                self._spawn_eye("spectator", _SPECTATOR)
                self._spectator_actor = self._camera_actors.pop("spectator")
                self._spectator_capture = self._captures.pop("spectator")
                self._apply_snapshot()
            with self._instance.end_frame(single_step=True):
                pass
            self._game.async_loading_service.wait_for_engine_idle()
        except BaseException:
            self._instance.close(force=True)
            self._closed = True
            raise

    def _spawn_lights(self) -> None:
        model, service = self.context.model, self._game.unreal_service
        atmosphere = service.spawn_actor(uclass="ASkyAtmosphere", spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"})
        self._actors.append(atmosphere)
        atmosphere_component = service.get_component_by_class(actor=atmosphere, uclass="USkyAtmosphereComponent")
        atmosphere_component.SetSkyLuminanceFactor(NewValue={"R": 3.0, "G": 4.0, "B": 6.0, "A": 1.0})
        for index in range(model.nlight):
            directional = bool(model.light_directional[index])
            direction = _REFLECT @ model.light_dir[index]
            direction /= np.linalg.norm(direction)
            up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(direction, up)) > 0.99:
                up = np.array([0.0, 1.0, 0.0])
            right = np.cross(up, direction)
            right /= np.linalg.norm(right)
            rotation = np.column_stack((direction, right, np.cross(direction, right)))
            actor = service.spawn_actor(
                uclass="ADirectionalLight" if directional else "APointLight",
                location=self._vector(_position(model.light_pos[index])),
                rotation=self._rotator(rotation),
                spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
            )
            self._actors.append(actor)
            light = service.get_component_by_class(actor=actor, uclass="UDirectionalLightComponent" if directional else "UPointLightComponent")
            light.SetMobility(NewMobility="Movable")
            diffuse = np.asarray(model.light_diffuse[index], dtype=float)
            strength = float(np.max(diffuse))
            color = diffuse / max(strength, 1e-12)
            light.SetLightColor(NewLightColor={"R": float(color[0]), "G": float(color[1]), "B": float(color[2]), "A": 1.0}, bSRGB=False)
            light.SetCastShadows(bNewValue=bool(model.light_castshadow[index]))
            if directional:
                light.SetIntensity(NewIntensity=20.0 if model.light(index).name == "dusk_sun" else 18.0)
                light.SetAtmosphereSunLight(bNewValue=model.light(index).name == "dusk_sun")
                light.SetAtmosphereSunLightIndex(NewValue=0)
            else:
                light.SetIntensityUnits(NewIntensityUnits="Lumens")
                light.SetIntensity(NewIntensity=2500.0 * strength)
                light.SetAttenuationRadius(NewRadius=450.0)
                light.SetSourceRadius(bNewValue=8.0)

    def _spawn_eye(self, eye: str, parameters: dict) -> None:
        super()._spawn_eye(eye, parameters)
        self._captures[eye].PostProcessSettings.AutoExposureBias = -3.0

    def _apply_snapshot(self) -> None:
        super()._apply_snapshot()
        if self._spectator_actor is None:
            return
        origin = self.context.data.xpos[self._mimo_root]
        target = origin + np.array([0.10, 0.0, 0.55])
        location = origin + np.array([-1.0, -1.5, 0.50])
        forward = _REFLECT @ (target - location)
        forward /= np.linalg.norm(forward)
        right = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        right /= np.linalg.norm(right)
        rotation = np.column_stack((forward, right, np.cross(forward, right)))
        self._spectator_actor.send_async.K2_SetActorLocationAndRotation(
            NewLocation=self._vector(_position(location)),
            NewRotation=self._rotator(rotation),
            bSweep=False,
            bTeleport=True,
        )

    @staticmethod
    def _read_rgb(capture, parameters) -> NDArray[np.uint8]:
        bgra = capture.read_pixels()["arrays"]["data"]
        expected = (int(parameters["height"]), int(parameters["width"]), 4)
        if bgra.shape != expected or bgra.dtype != np.uint8:
            raise ValueError(f"Street camera returned {bgra.dtype} {bgra.shape}; expected uint8 {expected}")
        return bgra[:, :, [2, 1, 0]].copy(order="C")

    def get_vision_obs(self) -> dict[str, NDArray[np.uint8]]:
        if self._closed:
            raise RuntimeError("Street vision provider is closed")
        snapshot = self.context.data
        with self._instance.begin_frame():
            self._apply_snapshot()
        with self._instance.end_frame(single_step=True):
            images = {eye: self._read_rgb(capture, self.vision_params[eye]) for eye, capture in self._captures.items()}
            self._spectator_rgb = self._read_rgb(self._spectator_capture, _SPECTATOR)
        self._captured_snapshot = snapshot
        return images

    def get_spectator_obs(self) -> NDArray[np.uint8]:
        """Return owned 960×540 RGB for the current snapshot, without MJ stepping."""
        if self._closed:
            raise RuntimeError("Street vision provider is closed")
        if self._captured_snapshot is not self.context.data or self._spectator_rgb is None:
            self.get_vision_obs()
        return self._spectator_rgb.copy()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._spectator_capture is not None:
                with self._instance.begin_frame():
                    pass
                with self._instance.end_frame(single_step=True):
                    self._spectator_capture.terminate_sp_funcs()
                    self._spectator_capture.Terminate()
        finally:
            super().close()

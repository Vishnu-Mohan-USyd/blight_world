"""SPEAR binocular rendering of a copied MuJoCo endpoint; Unreal owns no physics."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray


_ROOT = Path(__file__).resolve().parents[2]
_REFLECT = np.diag([1.0, -1.0, 1.0])


def _position(position: NDArray[Any]) -> NDArray[np.float64]:
    return 100.0 * (_REFLECT @ position)


def _body_rotation(rotation: NDArray[Any]) -> NDArray[np.float64]:
    return _REFLECT @ rotation.reshape(3, 3) @ _REFLECT


def _camera_rotation(rotation: NDArray[Any]) -> NDArray[np.float64]:
    reflected = _REFLECT @ rotation.reshape(3, 3)
    return np.column_stack((-reflected[:, 2], reflected[:, 0], reflected[:, 1]))


def make_unreal_vision_provider(context, vision_params):
    """Factory accepted by ``BlightWorldEnv(vision_provider_factory=...)``."""
    return UnrealVisionProvider(context, vision_params)


class UnrealVisionProvider:
    """Render both eye cameras in one paused Unreal frame through shared memory.

    Geom size and geom-to-body offsets are baked into the imported body meshes.
    Each capture applies the snapshot's body poses and actual MuJoCo camera
    extrinsics before either image is read. Returned arrays own their storage.
    """

    def __init__(
        self,
        context,
        vision_params,
        *,
        config_file: str | Path | None = None,
        asset_manifest: str | Path | None = None,
    ) -> None:
        import spear

        self._spear = spear
        self.context = context
        self.vision_params = vision_params
        self._instance = None
        self._actors = []
        self._body_actors = {}
        self._camera_actors = {}
        self._captures = {}
        self._id_camera_actors = {}
        self._id_captures = {}
        self._closed = False
        if set(vision_params) != {"eye_left", "eye_right"}:
            raise ValueError("Unreal vision requires eye_left and eye_right camera parameters")
        self._camera_ids = {
            eye: mujoco.mj_name2id(context.model, mujoco.mjtObj.mjOBJ_CAMERA, eye)
            for eye in vision_params
        }
        if any(index < 0 for index in self._camera_ids.values()):
            raise ValueError("The compiled model must contain both named eye cameras")
        config_file = Path(config_file or os.environ.get("BLIGHT_UNREAL_CONFIG", _ROOT / "config/spear_unreal.yaml"))
        asset_manifest = Path(asset_manifest or os.environ.get("BLIGHT_UNREAL_ASSET_MANIFEST", _ROOT / "artifacts/unreal/assets/manifest.json"))
        if not config_file.is_file():
            raise FileNotFoundError(f"Unreal SPEAR config does not exist: {config_file}")
        if not asset_manifest.is_file():
            raise FileNotFoundError(f"Prepare and import the Unreal assets first: {asset_manifest}")
        self._manifest = json.loads(asset_manifest.read_text())
        self._check_asset_geometry()
        config = spear.get_config(user_config_files=[str(config_file.resolve())])
        spear.configure_system(config=config)
        try:
            self._instance = spear.Instance(config=config)
            self._game = self._instance.get_game()
            with self._instance.begin_frame():
                self._game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(bPaused=True)
                self._configure_rendering()
                self._spawn_bodies()
                self._spawn_lights()
                for eye, parameters in vision_params.items():
                    self._spawn_eye(eye, parameters)
                self._apply_snapshot()
            with self._instance.end_frame(single_step=True):
                pass
            # Imported materials/textures and render targets must be ready before
            # MIMo requests its first observation during environment construction.
            self._game.async_loading_service.wait_for_engine_idle()
            self._instance.step(num_frames=2, single_step=True)
        except BaseException:
            if self._instance is not None:
                self._instance.close(force=True)
            self._closed = True
            raise

    def _check_asset_geometry(self) -> None:
        model = self.context.model
        if self._manifest["ngeom"] != model.ngeom or self._manifest["nbody"] != model.nbody:
            raise ValueError("Unreal assets do not match this compiled MuJoCo model")
        for body in self._manifest["bodies"]:
            body_id = body["body_id"]
            if (model.body(body_id).name or "world") != body["body_name"]:
                raise ValueError(f"Unreal body mapping differs at body {body_id}")
            for geom in body["geoms"]:
                index = geom["geom_id"]
                if int(model.geom_bodyid[index]) != body_id or int(model.geom_type[index]) != geom["type"]:
                    raise ValueError(f"Unreal geometry mapping differs at geom {index}")
                for actual, exported in ((model.geom_size[index], geom["size_m"]), (model.geom_pos[index], geom["local_position_m"]), (model.geom_quat[index], geom["local_quaternion_wxyz"])):
                    if not np.allclose(actual, exported, rtol=0, atol=1e-12):
                        raise ValueError(f"Unreal asset has different size/local pose at geom {index}")

    def _configure_rendering(self) -> None:
        for name, value in {
            "r.AntiAliasingMethod": 1,  # FXAA: no history from a different pose.
            "r.MotionBlurQuality": 0,
            "r.DepthOfFieldQuality": 0,
            "r.EyeAdaptationQuality": 0,
            "r.Streaming.FramesForFullUpdate": 0,
            "r.Streaming.FullyLoadUsedTextures": 1,
        }.items():
            self._game.console_service.set_cvar(name=name, value=value)

    def _spawn_bodies(self) -> None:
        service = self._game.unreal_service
        for body in self._manifest["bodies"]:
            mesh = service.load_object(uclass="UStaticMesh", name=body["mesh_asset"])
            actor = service.spawn_actor(uclass="AStaticMeshActor", spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"})
            self._actors.append(actor)
            service.set_stable_name_for_actor(actor=actor, stable_name=f"BlightWorld/{body['body_name']}")
            component = service.get_component_by_class(actor=actor, uclass="UStaticMeshComponent")
            component.SetMobility(NewMobility="Movable")
            component.SetStaticMesh(NewMesh=mesh)
            component.SetSimulatePhysics(bSimulate=False)
            component.SetCollisionEnabled(NewType="NoCollision")
            component.SetGenerateOverlapEvents(bInGenerateOverlapEvents=False)
            self._body_actors[body["body_id"]] = actor

    def _spawn_lights(self) -> None:
        # Rendering approximation: finite interior area lights replace MuJoCo's
        # directional sources so the room's closed ceiling does not block them.
        # Their locations, directions, relative strengths and colors follow MJCF.
        model, service = self.context.model, self._game.unreal_service
        for index in range(model.nlight):
            direction = _REFLECT @ model.light_dir[index]
            direction /= np.linalg.norm(direction)
            up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(direction, up)) > 0.99:
                up = np.array([0.0, 1.0, 0.0])
            right = np.cross(up, direction)
            right /= np.linalg.norm(right)
            rotation = np.column_stack((direction, right, np.cross(direction, right)))
            actor = service.spawn_actor(uclass="ARectLight", location=self._vector(_position(model.light_pos[index])), rotation=self._rotator(rotation), spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"})
            self._actors.append(actor)
            light = service.get_component_by_class(actor=actor, uclass="URectLightComponent")
            light.SetMobility(NewMobility="Movable")
            light.SetIntensityUnits(NewIntensityUnits="Lumens")
            diffuse = np.asarray(model.light_diffuse[index], dtype=float)
            strength = float(np.max(diffuse))
            color = diffuse / max(strength, 1e-12)
            light.SetLightColor(NewLightColor={"R": float(color[0]), "G": float(color[1]), "B": float(color[2]), "A": 1.0}, bSRGB=False)
            light.SetIntensity(NewIntensity=2500.0 * strength)
            light.SetAttenuationRadius(NewRadius=700.0)
            light.SetSourceWidth(NewValue=100.0)
            light.SetSourceHeight(NewValue=100.0)

    def _spawn_eye(self, eye: str, parameters: dict) -> None:
        service = self._game.unreal_service
        camera_class = service.load_class(uclass="AActor", name="/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C")
        actor = service.spawn_actor(uclass=camera_class, spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"})
        self._actors.append(actor)
        capture = service.get_component_by_name(actor=actor, component_name="DefaultSceneRoot.final_tone_curve_hdr_", uclass="USpSceneCaptureComponent2D")
        width, height = int(parameters["width"]), int(parameters["height"])
        fovy = float(parameters["fovy"])
        capture.Width = width
        capture.Height = height
        capture.NumChannelsPerPixel = 4
        capture.ChannelDataType = "UInt8"
        capture.BufferingMode = "SingleBuffered"
        capture.bUseSharedMemory = True
        capture.CaptureSource = "SCS_FinalToneCurveHDR"
        capture.bOverrideTextureRenderTargetFormat = True
        capture.TextureRenderTargetFormat = "RTF_RGBA8_SRGB"
        capture.ProjectionType = "Perspective"
        capture.FOVAngle = math.degrees(2 * math.atan(width / height * math.tan(math.radians(fovy) / 2)))
        capture.bOverride_CustomNearClippingPlane = True
        capture.CustomNearClippingPlane = 0.5
        capture.bCaptureEveryFrame = True
        capture.bCaptureOnMovement = False
        capture.PostProcessBlendWeight = 1.0
        capture.PostProcessSettings.bOverride_AutoExposureMethod = True
        capture.PostProcessSettings.AutoExposureMethod = "AEM_Manual"
        capture.PostProcessSettings.bOverride_AutoExposureApplyPhysicalCameraExposure = True
        capture.PostProcessSettings.AutoExposureApplyPhysicalCameraExposure = False
        capture.PostProcessSettings.bOverride_AutoExposureBias = True
        capture.PostProcessSettings.AutoExposureBias = -4.5
        capture.PostProcessSettings.bOverride_MotionBlurAmount = True
        capture.PostProcessSettings.MotionBlurAmount = 0.0
        capture.PostProcessSettings.bOverride_BloomIntensity = True
        capture.PostProcessSettings.BloomIntensity = 0.0
        capture.Initialize()
        capture.initialize_sp_funcs()
        self._camera_actors[eye] = actor
        self._captures[eye] = capture

    def _vector(self, value):
        return self._spear.math.to_spear_vector_from_numpy_array(numpy_array=np.asarray(value))

    def _rotator(self, value):
        return self._spear.math.to_spear_rotator_from_numpy_matrix(numpy_matrix=np.asarray(value))

    def _apply_snapshot(self) -> None:
        data = self.context.data
        for body_id, actor in self._body_actors.items():
            actor.send_async.K2_SetActorLocationAndRotation(NewLocation=self._vector(_position(data.xpos[body_id])), NewRotation=self._rotator(_body_rotation(data.xmat[body_id])), bSweep=False, bTeleport=True)
        for eye, actor in self._camera_actors.items():
            camera_id = self._camera_ids[eye]
            actor.send_async.K2_SetActorLocationAndRotation(NewLocation=self._vector(_position(data.cam_xpos[camera_id])), NewRotation=self._rotator(_camera_rotation(data.cam_xmat[camera_id])), bSweep=False, bTeleport=True)
            if eye in self._id_camera_actors:
                self._id_camera_actors[eye].send_async.K2_SetActorLocationAndRotation(NewLocation=self._vector(_position(data.cam_xpos[camera_id])), NewRotation=self._rotator(_camera_rotation(data.cam_xmat[camera_id])), bSweep=False, bTeleport=True)

    def get_vision_obs(self) -> dict[str, NDArray[np.uint8]]:
        if self._closed:
            raise RuntimeError("Unreal vision provider is closed")
        images = {}
        with self._instance.begin_frame():
            self._apply_snapshot()
        with self._instance.end_frame(single_step=True):
            for eye, capture in self._captures.items():
                bgra = capture.read_pixels()["arrays"]["data"]
                expected = (int(self.vision_params[eye]["height"]), int(self.vision_params[eye]["width"]), 4)
                if bgra.shape != expected or bgra.dtype != np.uint8:
                    raise ValueError(f"SPEAR {eye} returned {bgra.dtype} {bgra.shape}, expected uint8 {expected}")
                images[eye] = bgra[:, :, [2, 1, 0]].copy(order="C")
        return images

    def get_body_id_obs(self) -> dict[str, NDArray[np.int32]]:
        """Capture Unreal visibility at a hold: MuJoCo body IDs, background -1."""
        if self._closed:
            raise RuntimeError("Unreal vision provider is closed")
        service = self._game.unreal_service
        segmentation = self._game.segmentation_service
        if not self._id_captures:
            with self._instance.begin_frame():
                segmentation.initialize()
                segmentation.set_allowed_actors(allowed_actors=list(self._body_actors.values()))
                # RGB captures were initialized before this optional proxy actor.
                for capture in self._captures.values():
                    capture.HiddenActors = [segmentation.proxy_component_manager]
                camera_class = service.load_class(uclass="AActor", name="/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C")
                for eye, parameters in self.vision_params.items():
                    actor = service.spawn_actor(uclass=camera_class, spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"})
                    self._actors.append(actor)
                    capture = service.get_component_by_name(actor=actor, component_name="DefaultSceneRoot.sp_object_ids_uint8_", uclass="USpSceneCaptureComponent2D")
                    capture.Width = int(parameters["width"])
                    capture.Height = int(parameters["height"])
                    capture.FOVAngle = math.degrees(2 * math.atan(parameters["width"] / parameters["height"] * math.tan(math.radians(parameters["fovy"]) / 2)))
                    capture.bOverride_CustomNearClippingPlane = True
                    capture.CustomNearClippingPlane = 0.5
                    capture.BufferingMode = "SingleBuffered"
                    capture.bUseSharedMemory = True
                    capture.Initialize()
                    capture.initialize_sp_funcs()
                    self._id_camera_actors[eye] = actor
                    self._id_captures[eye] = capture
                self._apply_snapshot()
            with self._instance.end_frame(single_step=True):
                pass
            self._game.async_loading_service.wait_for_engine_idle()
        images = {}
        actor_bodies = {self._spear.to_handle(obj=actor): body for body, actor in self._body_actors.items()}
        with self._instance.begin_frame():
            self._apply_snapshot()
            for capture in self._id_captures.values():
                capture.SetVisibility(bNewVisibility=True)
        with self._instance.end_frame(single_step=True):
            descriptors = segmentation.get_mesh_proxy_geometry_descs(as_raw=True)
            for eye, capture in self._id_captures.items():
                bgra = capture.read_pixels()["arrays"]["data"]
                raw_ids = self._spear.rendering.get_object_ids_bgra_uint8_as_uint32(object_ids_bgra_uint8=bgra)
                body_ids = np.full(raw_ids.shape, -1, dtype=np.int32)
                for desc in descriptors:
                    body_id = actor_bodies.get(desc["actor"])
                    if body_id is not None:
                        body_ids[raw_ids == desc["rawId"]] = body_id
                images[eye] = body_ids
                capture.SetVisibility(bNewVisibility=False)
        return images

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._instance is None:
            return
        try:
            with self._instance.begin_frame():
                pass
            with self._instance.end_frame(single_step=True):
                for capture in (*self._captures.values(), *self._id_captures.values()):
                    capture.terminate_sp_funcs()
                    capture.Terminate()
                if self._id_captures:
                    self._game.segmentation_service.terminate()
                for actor in reversed(self._actors):
                    self._game.unreal_service.destroy_actor(actor=actor)
        finally:
            self._instance.close()

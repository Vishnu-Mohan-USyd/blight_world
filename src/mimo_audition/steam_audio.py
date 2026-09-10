"""Standalone Steam Audio spatial rendering with MuJoCo scene geometry.

The dry clips retain their one-metre source calibration. Output is a pressure
proxy using Steam's reference-head HRTF, not measured absolute ear pressure.
Poses and propagation delay are held over each 240-sample interval. Steam's
first-order Ambisonic reflections are a ray/energy approximation with 10 ms
relative-delay bins; both binaural stages add 60 samples of OLA latency.
"""

from __future__ import annotations

import ctypes as C
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import ceil

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from mimo_audition import _steam_audio_sdk as sdk
from mimo_audition.clips import ClipLibrary
from mimo_audition.config import AuditionConfig
from mimo_audition.continuous import ContinuousEmission
from mimo_audition.materials import AcousticMaterialResolver
from mimo_audition.renderer import AcousticScene, BinauralRenderer, FloatArray
from mimo_audition.sources import Attachment


SPEED_OF_SOUND_M_S = 340.0
BINAURAL_LATENCY_SAMPLES = 60
AMBISONICS_ORDER = 1
AMBISONICS_CHANNELS = 4
MUJOCO_TO_STEAM = np.asarray(((0, -1, 0), (0, 0, 1), (-1, 0, 0)), dtype=np.float64)


def _vector(values: FloatArray) -> sdk.Vector3:
    return sdk.Vector3(*map(float, values))


def _coordinate_space(position: FloatArray, rotation: FloatArray) -> sdk.CoordinateSpace3:
    return sdk.CoordinateSpace3(
        _vector(MUJOCO_TO_STEAM @ -rotation[:, 1]),
        _vector(MUJOCO_TO_STEAM @ rotation[:, 2]),
        _vector(MUJOCO_TO_STEAM @ rotation[:, 0]),
        _vector(MUJOCO_TO_STEAM @ position),
    )


def _transform(position: FloatArray, rotation: FloatArray) -> sdk.Matrix4x4:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = MUJOCO_TO_STEAM @ rotation @ MUJOCO_TO_STEAM.T
    transform[:3, 3] = MUJOCO_TO_STEAM @ position
    return sdk.Matrix4x4(((sdk.F * 4) * 4)(*(tuple(row) for row in transform)))


def _primitive_mesh(model: mujoco.MjModel, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate visible MuJoCo primitives; curved bodies are coarse proxies."""

    kind = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        return model.mesh_vert[va:va + vn].copy(), model.mesh_face[fa:fa + fn].copy()
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        # MuJoCo planes are infinite for contact but have editable visual extents.
        # The acoustic room uses those actual finite extents, without a shoebox.
        if np.any(size[:2] <= 0):
            raise ValueError("Steam Audio planes require positive finite x/y extents")
        vertices = np.asarray([(-size[0], -size[1], 0), (size[0], -size[1], 0),
                               (size[0], size[1], 0), (-size[0], size[1], 0)])
        return vertices, np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = np.asarray(tuple(product((-1, 1), repeat=3))) * size
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        angles = np.arange(12) * (2 * np.pi / 12)
        vertices = np.asarray([(size[0] * np.cos(a), size[0] * np.sin(a), z)
                               for z in (-size[1], size[1]) for a in angles])
    elif kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                  mujoco.mjtGeom.mjGEOM_CAPSULE):
        angles = np.arange(12) * (2 * np.pi / 12)
        latitudes = np.linspace(-np.pi / 2, np.pi / 2, 7)
        vertices = np.asarray([(np.cos(lat) * np.cos(a), np.cos(lat) * np.sin(a), np.sin(lat))
                               for lat in latitudes for a in angles])
        vertices *= size if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID else size[0]
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            vertices = np.concatenate((vertices + (0, 0, size[1]), vertices - (0, 0, size[1])))
    else:
        raise ValueError(f"Steam Audio cannot triangulate MuJoCo geom type {kind}")
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    reverse = np.sum(normals * hull.equations[:, :3], axis=1) < 0
    faces[reverse] = faces[reverse][:, (0, 2, 1)]
    return vertices, faces


class _Geometry:
    """Static room triangles and independently transformed dynamic geoms."""

    def __init__(self, api: sdk.SDK, context: sdk.H, parent: sdk.H, model: mujoco.MjModel,
                 data: mujoco.MjData) -> None:
        self.api, self.parent, self.model = api, parent, model
        self.static_meshes: list[sdk.H] = []
        self.instances: list[tuple[int, sdk.H, sdk.H, sdk.H]] = []
        materials = AcousticMaterialResolver(model)
        head = model.body("head").id
        # Values are engineering proxies: SDK concrete/wood/carpet presets and
        # generic rubber/plastic. They are not measurements of these objects.
        coefficients = {
            "hard": ((.05, .07, .08), .05, (.015, .002, .001)),
            "wood": ((.11, .07, .06), .05, (.070, .014, .005)),
            "rubber": ((.10, .20, .30), .05, (.02, .01, .005)),
            "plastic": ((.10, .20, .30), .05, (.10, .05, .03)),
            "soft": ((.24, .69, .73), .05, (.020, .005, .003)),
        }
        for geom_id in range(model.ngeom):
            body = int(model.geom_bodyid[geom_id])
            ancestor = body
            while ancestor and ancestor != head:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == head:
                continue
            material_id = int(model.geom_matid[geom_id])
            alpha = model.mat_rgba[material_id, 3] if material_id >= 0 else model.geom_rgba[geom_id, 3]
            if alpha <= 0 or np.max(model.geom_size[geom_id]) < 1e-5:
                continue
            vertices, faces = _primitive_mesh(model, geom_id)
            absorption, scattering, transmission = coefficients[materials.material(geom_id)]
            material = sdk.Material(sdk.F3(*absorption), scattering, sdk.F3(*transmission))
            if int(model.body_weldid[body]) == 0:
                vertices = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
                self.static_meshes.append(self._mesh(parent, vertices, faces, material))
            else:
                subscene = sdk.H()
                api.check(api.SceneCreate(context, C.byref(sdk.SceneSettings()), C.byref(subscene)), "SceneCreate")
                mesh = self._mesh(subscene, vertices, faces, material)
                api.SceneCommit(subscene)
                instance = sdk.H()
                settings = sdk.InstancedMeshSettings(subscene, _transform(data.geom_xpos[geom_id], data.geom_xmat[geom_id].reshape(3, 3)))
                api.check(api.InstancedMeshCreate(parent, C.byref(settings), C.byref(instance)), "InstancedMeshCreate")
                api.InstancedMeshAdd(instance, parent)
                self.instances.append((geom_id, instance, mesh, subscene))

    def _mesh(self, scene: sdk.H, vertices: np.ndarray, faces: np.ndarray, material: sdk.Material) -> sdk.H:
        vertices = np.ascontiguousarray(vertices @ MUJOCO_TO_STEAM.T, dtype=np.float32)
        faces = np.ascontiguousarray(faces, dtype=np.int32)
        indices = np.zeros(len(faces), dtype=np.int32)
        settings = sdk.StaticMeshSettings(len(vertices), len(faces), 1,
            vertices.ctypes.data_as(sdk.P(sdk.Vector3)), faces.ctypes.data_as(sdk.P(sdk.Triangle)),
            indices.ctypes.data_as(sdk.P(sdk.I)), C.pointer(material))
        mesh = sdk.H()
        self.api.check(self.api.StaticMeshCreate(scene, C.byref(settings), C.byref(mesh)), "StaticMeshCreate")
        self.api.StaticMeshAdd(mesh, scene)
        return mesh

    def update(self, data: mujoco.MjData) -> None:
        for geom_id, instance, _, _ in self.instances:
            self.api.InstancedMeshUpdateTransform(instance, self.parent,
                _transform(data.geom_xpos[geom_id], data.geom_xmat[geom_id].reshape(3, 3)))

    def close(self) -> None:
        for _, instance, mesh, scene in self.instances:
            self.api.InstancedMeshRemove(instance, self.parent)
            self.api.InstancedMeshRelease(C.byref(instance))
            self.api.StaticMeshRelease(C.byref(mesh))
            self.api.SceneRelease(C.byref(scene))
        for mesh in self.static_meshes:
            self.api.StaticMeshRemove(mesh, self.parent)
            self.api.StaticMeshRelease(C.byref(mesh))
        self.instances.clear()
        self.static_meshes.clear()


class _PropagationDelay:
    """Causal linear fractional interpolation, including distances below 16 samples."""

    def __init__(self) -> None:
        self.history = np.zeros(0, dtype=np.float64)
        self.max_delay_samples = 0.0

    def process(self, samples: FloatArray, delay_samples: float) -> FloatArray:
        self.max_delay_samples = max(self.max_delay_samples, delay_samples)
        combined = np.concatenate((self.history, samples))
        indices = len(self.history) + np.arange(len(samples), dtype=np.float64) - delay_samples
        output = np.interp(indices, np.arange(len(combined)), combined, left=0.0, right=0.0)
        retain = ceil(self.max_delay_samples) + len(samples) + 1
        self.history = combined[-retain:].copy()
        return output


@dataclass(slots=True)
class _Stream:
    source: sdk.H
    direct: sdk.H
    binaural: sdk.H
    reflection: sdk.H
    decoder: sdk.H
    delay: _PropagationDelay
    outputs: sdk.SimulationOutputs
    direct_done: bool = False
    reflection_done: bool = False
    decoder_done: bool = False
    resumable_tail_samples: int | None = None


class SteamAudioRenderer(BinauralRenderer):
    """Keep the established dry scheduler and replace its spatial DSP only."""

    binaural_latency_samples = BINAURAL_LATENCY_SAMPLES
    speed_of_sound_m_s = SPEED_OF_SOUND_M_S
    output_units = "source-calibrated acoustic-pressure proxy"

    def __init__(self, config: AuditionConfig, *, clips: ClipLibrary | None = None) -> None:
        if config.backend != "steam_audio":
            raise ValueError("SteamAudioRenderer requires backend='steam_audio'")
        self.config, self.spec = config, config.spec
        self.clips = ClipLibrary(48_000) if clips is None else clips
        if self.clips.sample_rate_hz != 48_000:
            raise ValueError("Steam Audio requires a 48000 Hz clip library")
        self._sources = {}
        self._voices = []
        self._continuous_voices = {}
        self._continuous_active_stream_ids = ()
        self._continuous_dropped_stream_ids = ()
        self._next_voice_id, self._sample_cursor = 1, 0
        self._streams: dict[tuple[str, int | str], _Stream] = {}
        self._geometry: _Geometry | None = None
        self._closed = False
        self._api = sdk.SDK(config.steam_audio_library_path)
        self._context, self._scene, self._simulator, self._hrtf = (sdk.H() for _ in range(4))
        self._audio_settings = sdk.AudioSettings(48_000, 240)
        self._reflections = config.enable_first_order_reflections or config.enable_late_reverb
        self._flags = sdk.DIRECT | (sdk.REFLECTIONS if self._reflections else 0)
        self._ir_size = round(config.steam_audio_ir_duration_s * 48_000)
        self._mono, self._filtered = sdk.Buffer(1), sdk.Buffer(1)
        self._direct_out, self._reflected_out = sdk.Buffer(2), sdk.Buffer(2)
        self._ambi = sdk.Buffer(AMBISONICS_CHANNELS)
        api = self._api
        api.check(api.ContextCreate(C.byref(sdk.ContextSettings(version=sdk.VERSION, simdLevel=3)), C.byref(self._context)), "ContextCreate")
        api.check(api.SceneCreate(self._context, C.byref(sdk.SceneSettings()), C.byref(self._scene)), "SceneCreate")
        api.SceneCommit(self._scene)
        api.check(api.HRTFCreate(self._context, C.byref(self._audio_settings),
            C.byref(sdk.HRTFSettings(type=0, volume=1.0, normType=0)), C.byref(self._hrtf)), "HRTFCreate")
        capacity = config.max_scheduled_events + config.max_active_voices
        if config.continuous is not None:
            capacity += config.continuous.max_continuous_voices
        settings = sdk.SimulationSettings(flags=self._flags, sceneType=0, reflectionType=0,
            maxNumOcclusionSamples=1, maxNumRays=config.steam_audio_num_rays,
            numDiffuseSamples=32, maxDuration=config.steam_audio_ir_duration_s,
            maxOrder=AMBISONICS_ORDER, maxNumSources=capacity,
            numThreads=config.steam_audio_num_threads, rayBatchSize=1,
            samplingRate=48_000, frameSize=240)
        api.check(api.SimulatorCreate(self._context, C.byref(settings), C.byref(self._simulator)), "SimulatorCreate")
        api.SimulatorSetScene(self._simulator, self._scene)
        api.SimulatorCommit(self._simulator)

    def _new_paths(self) -> list:
        # Base scheduler voice state owns no native HRIR, image path, or FDN.
        return []

    def _new_stream(self) -> _Stream:
        api = self._api
        source, direct, binaural, reflection, decoder = (sdk.H() for _ in range(5))
        api.check(api.SourceCreate(self._simulator, C.byref(sdk.SourceSettings(self._flags)), C.byref(source)), "SourceCreate")
        for name, settings, handle in (
            ("DirectEffect", sdk.DirectEffectSettings(1), direct),
            ("BinauralEffect", sdk.BinauralEffectSettings(self._hrtf), binaural),
        ):
            api.check(getattr(api, name + "Create")(self._context, C.byref(self._audio_settings), C.byref(settings), C.byref(handle)), name + "Create")
        if self._reflections:
            api.check(api.ReflectionEffectCreate(self._context, C.byref(self._audio_settings),
                C.byref(sdk.ReflectionEffectSettings(0, self._ir_size, 4)), C.byref(reflection)), "ReflectionEffectCreate")
            api.check(api.AmbisonicsDecodeEffectCreate(self._context, C.byref(self._audio_settings),
                C.byref(sdk.AmbisonicsDecodeEffectSettings(sdk.SpeakerLayout(type=1), self._hrtf, 1)),
                C.byref(decoder)), "AmbisonicsDecodeEffectCreate")
        api.SourceAdd(source, self._simulator)
        return _Stream(source, direct, binaural, reflection, decoder, _PropagationDelay(), sdk.SimulationOutputs(),
                       reflection_done=not self._reflections, decoder_done=not self._reflections)

    def _release_stream(self, stream: _Stream) -> None:
        self._api.SourceRemove(stream.source, self._simulator)
        for name, handle in (("DirectEffect", stream.direct), ("BinauralEffect", stream.binaural),
                             ("ReflectionEffect", stream.reflection), ("AmbisonicsDecodeEffect", stream.decoder),
                             ("Source", stream.source)):
            if handle:
                getattr(self._api, name + "Release")(C.byref(handle))

    def reset(self) -> None:
        for stream in self._streams.values():
            self._release_stream(stream)
        self._streams.clear()
        self._api.SimulatorCommit(self._simulator)
        self._voices.clear()
        self._continuous_voices.clear()
        self._continuous_active_stream_ids = ()
        self._continuous_dropped_stream_ids = ()
        self._next_voice_id, self._sample_cursor = 1, 0

    def close(self) -> None:
        if self._closed:
            return
        self.reset()
        if self._geometry is not None:
            self._geometry.close()
        for name, handle in (("Simulator", self._simulator), ("Scene", self._scene),
                             ("HRTF", self._hrtf), ("Context", self._context)):
            getattr(self._api, name + "Release")(C.byref(handle))
        self._closed = True

    def _dry_blocks(self, emissions: Sequence[ContinuousEmission]) -> list[tuple[tuple[str, int | str], Attachment, FloatArray, int | None]]:
        blocks = [(('clip', voice.voice_id), voice.source.attachment, self._voice_block(voice, 240), self._dry_end_sample(voice))
                  for voice in self._voices if voice.start_sample < self._sample_cursor + 240]
        admitted = self._select_continuous_emissions(emissions)
        for stream_id, emission in admitted.items():
            voice = self._continuous_voices.get(stream_id)
            if voice is None:
                voice = self._new_continuous_voice(emission)
                self._continuous_voices[stream_id] = voice
            elif voice.kind != emission.kind or not self._same_continuous_attachment_anchor(voice.attachment, emission.attachment):
                raise ValueError(f"continuous stream identity changed for stream_id: {stream_id}")
            voice.attachment = emission.attachment
        for stream_id in sorted(self._continuous_voices):
            voice = self._continuous_voices[stream_id]
            emission = admitted.get(stream_id)
            if emission is None:
                mono = self._continuous_release_block(voice, 240)
            else:
                mono = np.asarray(emission.samples_pa_at_1m, dtype=np.float64)
                voice.last_dry_sample = float(mono[-1])
                voice.release_total_samples = voice.release_elapsed_samples = 0
                voice.dry_end_sample = None
            blocks.append((('continuous', stream_id), voice.attachment, mono, voice.dry_end_sample))
        return blocks

    def _process(self, stream: _Stream, mono: FloatArray, distance: float, local_direction: FloatArray,
                 listener: sdk.CoordinateSpace3, dry_end: int | None,
                 *, can_resume: bool = False) -> tuple[FloatArray, bool]:
        api = self._api
        delayed = stream.delay.process(mono, distance * 48_000 / SPEED_OF_SOUND_M_S)
        self._mono.samples[0] = delayed
        self._direct_out.samples.fill(0)
        self._reflected_out.samples.fill(0)
        input_done = dry_end is not None and self._sample_cursor >= dry_end + ceil(stream.delay.max_delay_samples) + 1
        if not input_done:
            stream.resumable_tail_samples = None
        elif can_resume and stream.resumable_tail_samples is None:
            stream.resumable_tail_samples = max(
                api.BinauralEffectGetTailSize(stream.binaural),
                (api.ReflectionEffectGetTailSize(stream.reflection)
                 + api.AmbisonicsDecodeEffectGetTailSize(stream.decoder)) if self._reflections else 0,
            )
        decoder_params = sdk.AmbisonicsDecodeEffectParams(1, self._hrtf, listener, 1)
        if not input_done or can_resume:
            # SDK GetTail reads retained convolution blocks without advancing
            # their FIFO. A resumable stream must advance it with silent Apply
            # blocks, or resuming would replay an earlier part of its tail.
            direct = stream.outputs.direct
            direct.flags = 1 | 2 | ((8 | 16) if self.config.enable_occlusion else 0)
            direct.transmissionType = 1
            api.DirectEffectApply(stream.direct, C.byref(direct), C.byref(self._mono.c), C.byref(self._filtered.c))
            params = sdk.BinauralEffectParams(_vector(local_direction), 1, 1.0, self._hrtf, None)
            api.BinauralEffectApply(stream.binaural, C.byref(params), C.byref(self._filtered.c), C.byref(self._direct_out.c))
            stream.direct_done = False
            if self._reflections:
                reflections = stream.outputs.reflections
                reflections.type, reflections.numChannels, reflections.irSize = 0, 4, self._ir_size
                # Convolution ignores reflections.delay. The shared input above
                # already restores direct flight time to this relative IR.
                api.ReflectionEffectApply(stream.reflection, C.byref(reflections), C.byref(self._mono.c), C.byref(self._ambi.c), None)
                api.AmbisonicsDecodeEffectApply(stream.decoder, C.byref(decoder_params), C.byref(self._ambi.c), C.byref(self._reflected_out.c))
                stream.reflection_done = stream.decoder_done = False
        else:
            if not stream.direct_done:
                stream.direct_done = api.BinauralEffectGetTail(stream.binaural, C.byref(self._direct_out.c)) == sdk.TAIL_COMPLETE
            if not stream.reflection_done:
                stream.reflection_done = api.ReflectionEffectGetTail(stream.reflection, C.byref(self._ambi.c), None) == sdk.TAIL_COMPLETE
                api.AmbisonicsDecodeEffectApply(stream.decoder, C.byref(decoder_params), C.byref(self._ambi.c), C.byref(self._reflected_out.c))
            elif not stream.decoder_done:
                stream.decoder_done = api.AmbisonicsDecodeEffectGetTail(stream.decoder, C.byref(self._reflected_out.c)) == sdk.TAIL_COMPLETE
        finished = input_done and stream.direct_done and stream.reflection_done and stream.decoder_done
        if can_resume and input_done:
            stream.resumable_tail_samples -= 240
            finished = stream.resumable_tail_samples <= 0
        return self._direct_out.samples.astype(np.float64) + self._reflected_out.samples, finished

    def render_substep(self, scene: AcousticScene, continuous_emissions: Sequence[ContinuousEmission] = ()) -> FloatArray:
        """Commit start poses, simulate synchronously, then emit [t,t+5ms)."""

        if self._closed:
            raise RuntimeError("Steam Audio renderer is closed")
        api = self._api
        if hasattr(scene, "model") and hasattr(scene, "data"):
            if self._geometry is None:
                self._geometry = _Geometry(api, self._context, self._scene, scene.model, scene.data)
            elif self._geometry.model is not scene.model:
                raise ValueError("cannot change the MuJoCo model of a Steam Audio renderer")
            self._geometry.update(scene.data)
        api.SceneCommit(self._scene)
        position, rotation = scene.listener_pose()
        position, rotation = np.asarray(position), np.asarray(rotation).reshape(3, 3)
        listener = _coordinate_space(position, rotation)
        shared = sdk.SimulationSharedInputs(listener=listener, numRays=self.config.steam_audio_num_rays,
            numBounces=self.config.steam_audio_num_bounces, duration=self.config.steam_audio_ir_duration_s,
            order=1, irradianceMinDistance=self.config.distance_floor_m)
        api.SimulatorSetSharedInputs(self._simulator, self._flags, C.byref(shared))
        blocks = self._dry_blocks(continuous_emissions)
        live_keys = {key for key, _, _, _ in blocks}
        for key in tuple(self._streams):
            if key not in live_keys:
                self._release_stream(self._streams.pop(key))
        prepared = []
        for key, attachment, mono, dry_end in blocks:
            stream = self._streams.get(key)
            if stream is None:
                stream = self._new_stream()
                self._streams[key] = stream
            source_position, _ = scene.attachment_pose(attachment)
            source_position = np.asarray(source_position, dtype=np.float64)
            offset = source_position - position
            distance = float(np.linalg.norm(offset))
            direction = MUJOCO_TO_STEAM @ (rotation.T @ offset / distance) if distance > 0 else np.asarray((0., 0., -1.))
            inputs = sdk.SimulationInputs(flags=self._flags, directFlags=1 | 2 | ((8 | 16) if self.config.enable_occlusion else 0),
                source=_coordinate_space(source_position, np.eye(3)),
                distanceAttenuationModel=sdk.DistanceAttenuationModel(type=1, minDistance=self.config.distance_floor_m),
                directivity=sdk.Directivity(dipolePower=1), numOcclusionSamples=1,
                reverbScale=sdk.F3(1, 1, 1), numTransmissionRays=4)
            api.SourceSetInputs(stream.source, self._flags, C.byref(inputs))
            prepared.append((key, stream, mono, distance, direction, dry_end))
        api.SimulatorCommit(self._simulator)
        output = np.zeros((2, 240), dtype=np.float64)
        if prepared:
            api.SimulatorRunDirect(self._simulator)
            if self._reflections:
                api.SimulatorRunReflections(self._simulator)
            completed = set()
            for key, stream, mono, distance, direction, dry_end in prepared:
                api.SourceGetOutputs(stream.source, self._flags, C.byref(stream.outputs))
                spatial, finished = self._process(stream, mono, distance, direction, listener, dry_end,
                                                  can_resume=key[0] == "continuous")
                output += spatial
                if finished:
                    completed.add(key)
            for key in completed:
                self._release_stream(self._streams.pop(key))
            self._voices = [voice for voice in self._voices if ('clip', voice.voice_id) not in completed]
            self._continuous_voices = {key: voice for key, voice in self._continuous_voices.items()
                                       if ('continuous', key) not in completed}
            if completed:
                api.SimulatorCommit(self._simulator)
        self._sample_cursor += 240
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("Steam Audio produced non-finite pressure proxy samples")
        return output

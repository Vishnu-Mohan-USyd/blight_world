"""The standalone Steam Audio 4.8.1 C ABI used by the auditory backend.

Declarations match the official SDK's include/phonon.h. No engine plugin or
audio device is involved; buffers are caller-owned planar float32 arrays.
"""

from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np


VERSION = 0x040801
DIRECT = 1
REFLECTIONS = 2
TAIL_COMPLETE = 1
F = C.c_float
I = C.c_int32
H = C.c_void_p
P = C.POINTER
F3 = F * 3


class Vector3(C.Structure):
    _fields_ = [("x", F), ("y", F), ("z", F)]


class Matrix4x4(C.Structure):
    _fields_ = [("elements", (F * 4) * 4)]


class CoordinateSpace3(C.Structure):
    _fields_ = [("right", Vector3), ("up", Vector3), ("ahead", Vector3), ("origin", Vector3)]


class ContextSettings(C.Structure):
    _fields_ = [("version", C.c_uint32), ("logCallback", H), ("allocateCallback", H),
                ("freeCallback", H), ("simdLevel", I), ("flags", I)]


class SceneSettings(C.Structure):
    _fields_ = [("type", I), ("closestHitCallback", H), ("anyHitCallback", H),
                ("batchedClosestHitCallback", H), ("batchedAnyHitCallback", H),
                ("userData", H), ("embreeDevice", H), ("radeonRaysDevice", H)]


class Material(C.Structure):
    _fields_ = [("absorption", F3), ("scattering", F), ("transmission", F3)]


class Triangle(C.Structure):
    _fields_ = [("indices", I * 3)]


class StaticMeshSettings(C.Structure):
    _fields_ = [("numVertices", I), ("numTriangles", I), ("numMaterials", I),
                ("vertices", P(Vector3)), ("triangles", P(Triangle)),
                ("materialIndices", P(I)), ("materials", P(Material))]


class InstancedMeshSettings(C.Structure):
    _fields_ = [("subScene", H), ("transform", Matrix4x4)]


class AudioSettings(C.Structure):
    _fields_ = [("samplingRate", I), ("frameSize", I)]


class AudioBuffer(C.Structure):
    _fields_ = [("numChannels", I), ("numSamples", I), ("data", P(P(F)))]


class HRTFSettings(C.Structure):
    _fields_ = [("type", I), ("sofaFileName", C.c_char_p), ("sofaData", H),
                ("sofaDataSize", I), ("volume", F), ("normType", I)]


class BinauralEffectSettings(C.Structure):
    _fields_ = [("hrtf", H)]


class BinauralEffectParams(C.Structure):
    _fields_ = [("direction", Vector3), ("interpolation", I), ("spatialBlend", F),
                ("hrtf", H), ("peakDelays", P(F))]


class DirectEffectSettings(C.Structure):
    _fields_ = [("numChannels", I)]


class DirectEffectParams(C.Structure):
    _fields_ = [("flags", I), ("transmissionType", I), ("distanceAttenuation", F),
                ("airAbsorption", F3), ("directivity", F), ("occlusion", F),
                ("transmission", F3)]


class ReflectionEffectSettings(C.Structure):
    _fields_ = [("type", I), ("irSize", I), ("numChannels", I)]


class ReflectionEffectParams(C.Structure):
    _fields_ = [("type", I), ("ir", H), ("reverbTimes", F3), ("eq", F3),
                ("delay", I), ("numChannels", I), ("irSize", I),
                ("tanDevice", H), ("tanSlot", I)]


class SpeakerLayout(C.Structure):
    _fields_ = [("type", I), ("numSpeakers", I), ("speakers", P(Vector3))]


class AmbisonicsDecodeEffectSettings(C.Structure):
    _fields_ = [("speakerLayout", SpeakerLayout), ("hrtf", H), ("maxOrder", I)]


class AmbisonicsDecodeEffectParams(C.Structure):
    _fields_ = [("order", I), ("hrtf", H), ("orientation", CoordinateSpace3), ("binaural", I)]


class DistanceAttenuationModel(C.Structure):
    _fields_ = [("type", I), ("minDistance", F), ("callback", H), ("userData", H), ("dirty", I)]


class AirAbsorptionModel(C.Structure):
    _fields_ = [("type", I), ("coefficients", F3), ("callback", H), ("userData", H), ("dirty", I)]


class Directivity(C.Structure):
    _fields_ = [("dipoleWeight", F), ("dipolePower", F), ("callback", H), ("userData", H)]


class Sphere(C.Structure):
    _fields_ = [("center", Vector3), ("radius", F)]


class BakedDataIdentifier(C.Structure):
    _fields_ = [("type", I), ("variation", I), ("endpointInfluence", Sphere)]


class SimulationSettings(C.Structure):
    _fields_ = [("flags", I), ("sceneType", I), ("reflectionType", I),
                ("maxNumOcclusionSamples", I), ("maxNumRays", I), ("numDiffuseSamples", I),
                ("maxDuration", F), ("maxOrder", I), ("maxNumSources", I),
                ("numThreads", I), ("rayBatchSize", I), ("numVisSamples", I),
                ("samplingRate", I), ("frameSize", I), ("openCLDevice", H),
                ("radeonRaysDevice", H), ("tanDevice", H)]


class SourceSettings(C.Structure):
    _fields_ = [("flags", I)]


class SimulationInputs(C.Structure):
    _fields_ = [("flags", I), ("directFlags", I), ("source", CoordinateSpace3),
                ("distanceAttenuationModel", DistanceAttenuationModel),
                ("airAbsorptionModel", AirAbsorptionModel), ("directivity", Directivity),
                ("occlusionType", I), ("occlusionRadius", F), ("numOcclusionSamples", I),
                ("reverbScale", F3), ("hybridReverbTransitionTime", F),
                ("hybridReverbOverlapPercent", F), ("baked", I),
                ("bakedDataIdentifier", BakedDataIdentifier), ("pathingProbes", H),
                ("visRadius", F), ("visThreshold", F), ("visRange", F), ("pathingOrder", I),
                ("enableValidation", I), ("findAlternatePaths", I),
                ("numTransmissionRays", I), ("deviationModel", H)]


class SimulationSharedInputs(C.Structure):
    _fields_ = [("listener", CoordinateSpace3), ("numRays", I), ("numBounces", I),
                ("duration", F), ("order", I), ("irradianceMinDistance", F),
                ("pathingVisCallback", H), ("pathingUserData", H)]


class PathEffectParams(C.Structure):
    _fields_ = [("eqCoeffs", F3), ("shCoeffs", P(F)), ("order", I), ("binaural", I),
                ("hrtf", H), ("listener", CoordinateSpace3), ("normalizeEQ", I)]


class SimulationOutputs(C.Structure):
    _fields_ = [("direct", DirectEffectParams), ("reflections", ReflectionEffectParams),
                ("pathing", PathEffectParams)]


class Buffer:
    """Own both NumPy storage and C channel pointers for one fixed block."""

    def __init__(self, channels: int, samples: int = 240) -> None:
        self.samples = np.zeros((channels, samples), dtype=np.float32)
        self._pointers = (P(F) * channels)(
            *(row.ctypes.data_as(P(F)) for row in self.samples)
        )
        self.c = AudioBuffer(channels, samples, self._pointers)


class SDK:
    """Typed functions from the official Linux standalone SDK library."""

    def __init__(self, library_path: str | None = None) -> None:
        path = library_path or os.environ.get("STEAM_AUDIO_LIBRARY")
        if path is None:
            path = str(Path(__file__).resolve().parents[2] / "third_party/steam_audio/lib/linux-x64/libphonon.so")
        if not Path(path).is_file():
            raise FileNotFoundError(
                "Steam Audio 4.8.1 libphonon.so is required; set "
                "steam_audio_library_path or STEAM_AUDIO_LIBRARY"
            )
        self.library_path = str(Path(path).resolve())
        self.lib = C.CDLL(self.library_path)
        definitions = {
            "ContextCreate": (I, [P(ContextSettings), P(H)]),
            "SceneCreate": (I, [H, P(SceneSettings), P(H)]),
            "StaticMeshCreate": (I, [H, P(StaticMeshSettings), P(H)]),
            "InstancedMeshCreate": (I, [H, P(InstancedMeshSettings), P(H)]),
            "InstancedMeshUpdateTransform": (None, [H, H, Matrix4x4]),
            "SceneCommit": (None, [H]),
            "HRTFCreate": (I, [H, P(AudioSettings), P(HRTFSettings), P(H)]),
            "SimulatorCreate": (I, [H, P(SimulationSettings), P(H)]),
            "SimulatorSetScene": (None, [H, H]),
            "SimulatorCommit": (None, [H]),
            "SimulatorSetSharedInputs": (None, [H, I, P(SimulationSharedInputs)]),
            "SimulatorRunDirect": (None, [H]),
            "SimulatorRunReflections": (None, [H]),
            "SourceCreate": (I, [H, P(SourceSettings), P(H)]),
            "SourceSetInputs": (None, [H, I, P(SimulationInputs)]),
            "SourceGetOutputs": (None, [H, I, P(SimulationOutputs)]),
        }
        effects = {
            "DirectEffect": (DirectEffectSettings, DirectEffectParams),
            "BinauralEffect": (BinauralEffectSettings, BinauralEffectParams),
            "ReflectionEffect": (ReflectionEffectSettings, ReflectionEffectParams),
            "AmbisonicsDecodeEffect": (AmbisonicsDecodeEffectSettings, AmbisonicsDecodeEffectParams),
        }
        for name, (settings, params) in effects.items():
            definitions[name + "Create"] = (I, [H, P(AudioSettings), P(settings), P(H)])
            extra = [H] if name == "ReflectionEffect" else []
            definitions[name + "Apply"] = (I, [H, P(params), P(AudioBuffer), P(AudioBuffer)] + extra)
            definitions[name + "GetTail"] = (I, [H, P(AudioBuffer)] + extra)
            definitions[name + "GetTailSize"] = (I, [H])
        for name in ("Source", "StaticMesh", "InstancedMesh"):
            definitions[name + "Add"] = (None, [H, H])
            definitions[name + "Remove"] = (None, [H, H])
        for name in ("Context", "Scene", "StaticMesh", "InstancedMesh", "HRTF", "Simulator", "Source", *effects):
            definitions[name + "Release"] = (None, [P(H)])
        for name, (result, arguments) in definitions.items():
            function = getattr(self.lib, "ipl" + name)
            function.restype = result
            function.argtypes = arguments
            setattr(self, name, function)

    @staticmethod
    def check(status: int, operation: str) -> None:
        if status != 0:
            raise RuntimeError(f"Steam Audio {operation} failed with IPLerror {status}")

"""Generate the documented dry impact/effect bank reproducibly.

The waveforms are engineering assets synthesized from damped modes and
filtered noise; they are not field recordings or infant-specific calibration.
They are stored as ordinary PCM WAV files so runtime loading needs only SciPy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import wave

import numpy as np
from scipy.signal import firwin, lfilter


SAMPLE_RATE_HZ = 48_000
MATERIAL_PAIRS = (
    ("hard", "wood"),
    ("hard", "rubber"),
    ("hard", "plastic"),
    ("hard", "soft"),
    ("wood", "wood"),
    ("wood", "rubber"),
    ("wood", "plastic"),
    ("wood", "soft"),
    ("rubber", "plastic"),
    ("rubber", "soft"),
    ("plastic", "soft"),
)
MATERIAL_CHARACTER = {
    "hard": (3_200.0, 0.045, 0.55),
    "wood": (1_250.0, 0.095, 0.70),
    "rubber": (240.0, 0.070, 0.30),
    "plastic": (2_100.0, 0.060, 0.62),
    "soft": (130.0, 0.045, 0.18),
}


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def _normalize(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal)))
    if peak <= 0:
        raise ValueError("generated audio must not be silent")
    return np.asarray(signal * (0.98 / peak), dtype=np.float64)


def _impact(material_a: str, material_b: str, variant: int) -> np.ndarray:
    label = f"impact_{material_a}_{material_b}_{variant}"
    rng = np.random.default_rng(_seed(label))
    frequency_a, decay_a, brightness_a = MATERIAL_CHARACTER[material_a]
    frequency_b, decay_b, brightness_b = MATERIAL_CHARACTER[material_b]
    base_frequency = np.sqrt(frequency_a * frequency_b) * (0.92 + 0.08 * variant)
    decay_s = 0.5 * (decay_a + decay_b) * (0.9 + 0.1 * variant)
    brightness = 0.5 * (brightness_a + brightness_b)
    duration_s = max(0.16, 5.5 * decay_s)
    time_s = np.arange(round(duration_s * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
    attack = 1.0 - np.exp(-time_s / 0.00035)
    envelope = attack * np.exp(-time_s / decay_s)
    modes = np.zeros_like(time_s)
    ratios = np.asarray((1.0, 1.47, 2.11, 2.83, 3.76))
    phases = rng.uniform(-np.pi, np.pi, ratios.size)
    weights = np.power(ratios, -1.1 + 0.55 * brightness)
    for ratio, phase, weight in zip(ratios, phases, weights, strict=True):
        modes += weight * np.sin(2 * np.pi * base_frequency * ratio * time_s + phase)
    noise = rng.normal(size=time_s.size)
    cutoff_hz = min(10_000.0, 700.0 + brightness * 9_000.0)
    coefficient = np.exp(-2 * np.pi * cutoff_hz / SAMPLE_RATE_HZ)
    noise = lfilter((1.0 - coefficient,), (1.0, -coefficient), noise)
    transient = np.zeros_like(time_s)
    transient[:12] = np.hanning(24)[:12]
    signal = envelope * (0.72 * modes + 0.22 * noise) + brightness * transient
    return _normalize(signal)


def _effect(name: str) -> np.ndarray:
    rng = np.random.default_rng(_seed(name))
    duration_s = {"rattle": 0.36, "squeak": 0.42, "chime": 0.72, "click": 0.10}[name]
    time_s = np.arange(round(duration_s * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
    if name == "rattle":
        carrier = rng.normal(size=time_s.size)
        gate = np.square(np.maximum(0.0, np.sin(2 * np.pi * 28 * time_s)))
        signal = carrier * gate * np.exp(-time_s / 0.24)
    elif name == "squeak":
        phase = 2 * np.pi * (650 * time_s + 1_200 * time_s**2)
        signal = np.sin(phase) * np.sin(np.pi * time_s / duration_s) ** 2
    elif name == "chime":
        signal = sum(
            weight * np.sin(2 * np.pi * frequency * time_s)
            for frequency, weight in ((880, 1.0), (1_321, 0.55), (2_137, 0.28))
        ) * np.exp(-time_s / 0.24)
    else:
        signal = np.zeros_like(time_s)
        signal[:20] = np.hanning(40)[:20]
        signal += 0.08 * rng.normal(size=time_s.size) * np.exp(-time_s / 0.008)
    return _normalize(np.asarray(signal))


def _downsample(signal: np.ndarray) -> np.ndarray:
    kernel = firwin(
        127,
        cutoff=10_800.0,
        window=("kaiser", 8.6),
        fs=48_000.0,
    )
    group_delay = (kernel.size - 1) // 2
    filtered = np.convolve(signal, kernel, mode="full")
    return filtered[group_delay : group_delay + signal.size : 2]


def _write_wav(path: Path, signal: np.ndarray, sample_rate_hz: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.asarray(np.round(np.clip(signal, -1, 1) * 32_767), dtype="<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        output.writeframes(pcm.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output_root: Path) -> dict[str, object]:
    clips: dict[str, object] = {}
    waveforms: list[tuple[str, np.ndarray, str, tuple[str, str] | None]] = []
    for material_a, material_b in MATERIAL_PAIRS:
        for variant in range(3):
            clip_id = f"impact_{material_a}_{material_b}_{variant}"
            waveforms.append(
                (clip_id, _impact(material_a, material_b, variant), "impact", (material_a, material_b))
            )
    for effect_name in ("rattle", "squeak", "chime", "click"):
        waveforms.append((effect_name, _effect(effect_name), "effect", None))

    for clip_id, master, kind, materials in waveforms:
        derived = _downsample(master)
        path_48 = output_root / "48k" / f"{clip_id}.wav"
        path_24 = output_root / "24k" / f"{clip_id}.wav"
        sha_48 = _write_wav(path_48, master, 48_000)
        sha_24 = _write_wav(path_24, derived, 24_000)
        # 0.2 Pa peak at one metre is about 80 dB re 20 uPa. Impact energy,
        # source gain, distance, HRTF, and room paths scale this reference.
        clips[clip_id] = {
            "kind": kind,
            "materials": materials,
            "reference_peak_pressure_pa_at_1m": 0.2,
            "files": {
                "48000": {"path": f"48k/{clip_id}.wav", "sha256": sha_48},
                "24000": {"path": f"24k/{clip_id}.wav", "sha256": sha_24},
            },
        }

    manifest = {
        "version": 1,
        "description": "Deterministic generated dry impact/effect waveforms",
        "license": "CC0-1.0 (project-generated)",
        "master_sample_rate_hz": 48_000,
        "derivation_24k": "127-tap Kaiser beta=8.6 LP at 10.8 kHz, delay-compensated, decimate 2",
        "clips": clips,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "clip_count": len(clips),
        "wav_count": 2 * len(clips),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_root), indent=2, sort_keys=True))

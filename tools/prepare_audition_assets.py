"""Reproducibly prepare bundled audition data from documented source assets.

This developer utility requires ``h5py`` to convert the official SADIE II H10
SOFA file.  Runtime users load only the generated NumPy archive and therefore
do not need an HDF5 dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_SOFA_SHA256 = "b48cf93b1b3919fa61a787242c75763fad2a4b0ee3464e53e165ba12a9720fde"
SADIE_LICENSE = "Apache-2.0"


def _is_zero_data_delay(data_delay: np.ndarray) -> bool:
    """Return whether SOFA delay metadata is the expected paired zero row."""

    return data_delay.shape == (1, 2) and bool(np.all(data_delay == 0.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directions(source_position: np.ndarray) -> np.ndarray:
    azimuth = np.deg2rad(source_position[:, 0])
    elevation = np.deg2rad(source_position[:, 1])
    cos_elevation = np.cos(elevation)
    return np.column_stack(
        (
            cos_elevation * np.cos(azimuth),
            cos_elevation * np.sin(azimuth),
            np.sin(elevation),
        )
    )


def prepare_hrtf(sofa_path: Path, output_path: Path) -> dict[str, object]:
    """Validate and convert the official 48 kHz H10 SOFA measurement."""

    import h5py

    source_sha256 = _sha256(sofa_path)
    if source_sha256 != EXPECTED_SOFA_SHA256:
        raise ValueError(
            f"unexpected SOFA SHA256 {source_sha256}; expected {EXPECTED_SOFA_SHA256}"
        )
    with h5py.File(sofa_path, "r") as sofa:
        impulse_responses = np.asarray(sofa["Data.IR"], dtype=np.float64)
        source_position = np.asarray(sofa["SourcePosition"], dtype=np.float64)
        data_delay = np.asarray(sofa["Data.Delay"], dtype=np.float64)
        sample_rate_hz = int(np.asarray(sofa["Data.SamplingRate"]).item())
        if impulse_responses.shape != (2_114, 2, 256):
            raise ValueError(f"unexpected H10 Data.IR shape {impulse_responses.shape}")
        if source_position.shape != (2_114, 3):
            raise ValueError(
                f"unexpected H10 SourcePosition shape {source_position.shape}"
            )
        if sample_rate_hz != 48_000:
            raise ValueError(f"unexpected H10 sample rate {sample_rate_hz}")
        if not _is_zero_data_delay(data_delay):
            raise ValueError(
                "this converter expects H10 v2.2's zero Data.Delay; non-zero "
                "SOFA delays must be explicitly materialized before use"
            )

    # The paired linear-phase FIRs contain a large common delay.  Retain the
    # filters and their ITD intact, but publish one global delay so the runtime
    # can compensate it once when adding geometric propagation.
    peak_indices = np.argmax(np.abs(impulse_responses), axis=2)
    common_delay_samples = float(np.median(np.mean(peak_indices, axis=1)))

    # One global scalar establishes approximately unity frontal gain over the
    # 500--2,000 Hz band without altering any interaural or directional cues.
    front_index = int(
        np.argmin(np.square(source_position[:, 0]) + np.square(source_position[:, 1]))
    )
    frequency_hz = np.fft.rfftfreq(8_192, d=1 / sample_rate_hz)
    front_response = np.abs(
        np.fft.rfft(impulse_responses[front_index], n=8_192, axis=1)
    )
    calibration_band = (frequency_hz >= 500) & (frequency_hz <= 2_000)
    front_band_gain = float(np.mean(front_response[:, calibration_band]))
    calibration_scalar = 1.0 / front_band_gain
    impulse_responses *= calibration_scalar

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        directions_xyz=_directions(source_position).astype(np.float32),
        hrir=impulse_responses.astype(np.float32),
        sample_rate_hz=np.asarray(sample_rate_hz, dtype=np.int64),
        common_delay_samples=np.asarray(common_delay_samples, dtype=np.float64),
        subject=np.asarray("SADIE II H10"),
        license_name=np.asarray(SADIE_LICENSE),
        source_sha256=np.asarray(source_sha256),
        source_position_count=np.asarray(source_position.shape[0], dtype=np.int64),
        data_delay_samples=data_delay,
        front_band_calibration_scalar=np.asarray(calibration_scalar),
    )
    return {
        "output": output_path.as_posix(),
        "output_sha256": _sha256(output_path),
        "source_sha256": source_sha256,
        "directions": source_position.shape[0],
        "sample_rate_hz": sample_rate_hz,
        "common_delay_samples": common_delay_samples,
        "front_band_calibration_scalar": calibration_scalar,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sofa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = prepare_hrtf(arguments.sofa, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Focused regressions for the continuous embodied demo exporter."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from tools.export_continuous_embodied_demo import (
    AUDIO_SAMPLE_RATE_HZ,
    AUDIO_SAMPLES_PER_EAR,
    CONTACT_VIDEO_FRAME,
    DURATION_SECONDS,
    EXPECTED_ACTION_SHA256,
    EXPECTED_AUDIO_TRANSPORT_SHA256,
    EXPECTED_BALL_TRAJECTORY_SHA256,
    EXPECTED_QPOS_SHA256,
    MEDIA_MONITOR_GAIN,
    PUBLIC_STEPS,
    VIDEO_FRAME_RATE_HZ,
    VIDEO_FRAMES,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    export_demo,
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_fixed_export_clock_and_monitor_gain_are_predeclared() -> None:
    assert DURATION_SECONDS == 41.8
    assert PUBLIC_STEPS == 4_180
    assert VIDEO_FRAMES == 2_090
    assert VIDEO_FRAME_RATE_HZ == 50
    assert (VIDEO_WIDTH, VIDEO_HEIGHT) == (640, 240)
    assert AUDIO_SAMPLE_RATE_HZ == 48_000
    assert AUDIO_SAMPLES_PER_EAR == 2_006_400
    assert CONTACT_VIDEO_FRAME == 1_425
    assert MEDIA_MONITOR_GAIN == 0.04
    assert 12.987203598022461 * MEDIA_MONITOR_GAIN < 1.0


def test_full_export_preserves_public_sources_and_codec_contract(
    tmp_path: Path,
) -> None:
    output = tmp_path / "continuous_embodied_demo"
    manifest = export_demo(output)

    configuration = manifest["configuration"]
    assert configuration["reset_count"] == 1
    assert configuration["public_environment_steps"] == 4_180
    assert configuration["video_frames"] == 2_090
    assert configuration["audio_samples_per_ear"] == 2_006_400
    assert configuration["camera_source"] == (
        "public_eye_left_plus_public_eye_right"
    )
    assert configuration["audio_source"] == "public_audition_pressure_only"
    assert configuration["cuts_or_overlays"] is False
    assert configuration["external_camera"] is False
    assert configuration["external_forces"] is False
    assert configuration["runtime_state_or_model_writes"] is False
    assert configuration["synthetic_sound_injection"] is False

    public = manifest["public_source_validation"]
    assert public["action_float32_step_major_sha256"] == EXPECTED_ACTION_SHA256
    assert public["qpos_float64_post_step_sha256"] == EXPECTED_QPOS_SHA256
    assert public["ball_xpos_float64_post_step_sha256"] == (
        EXPECTED_BALL_TRAJECTORY_SHA256
    )
    assert public["audition_block_transport_sha256"] == (
        EXPECTED_AUDIO_TRANSPORT_SHA256
    )
    assert public["audition_nonzero_blocks"] == 3_665
    assert set(public["source_frame_hashes"]) == {"0", "1425", "2089"}
    contact_frame = public["contact_frame"]
    assert contact_frame["eye_left_red_pixels"] == 2_270
    assert contact_frame["eye_right_red_pixels"] == 2_163
    assert contact_frame["binocular_images_equal"] is False

    physics = manifest["physics_validation"]
    assert physics["contacts"]["count"] == 5
    assert physics["contacts"]["first_public_step"] == 2_851
    assert physics["ball_planar_displacement_m"] == pytest.approx(
        0.3279796105546315, abs=1e-12
    )
    assert physics["runtime_qpos_qvel_writes"] == 0
    assert physics["runtime_model_equality_mocap_writes"] == 0
    assert physics["runtime_xfrc_nonzero_steps"] == 0

    mapping = manifest["media_pressure_mapping"]
    assert mapping["playback_fixed_monitor_gain"] == 0.04
    assert mapping["automatic_gain_control"] is False
    assert mapping["normalization"] is False
    assert mapping["fixed_gain_clips"] is False
    assert mapping["fixed_gain_peak_full_scale_before_aac"] < 1.0

    media = manifest["media_validation"]
    assert media["duration_seconds"] == pytest.approx(41.8, abs=0.02)
    assert media["video_frames"] == 2_090
    assert media["decoded_audio_clipped"] is False
    assert media["decoded_audio_peak_full_scale"] < 1.0
    assert media["full_decode_exit_code"] == 0

    playback = output / manifest["artifacts"]["playback_mp4"]["path"]
    pressure_path = output / manifest["artifacts"]["pressure_wav"]["path"]
    manifest_path = output / "continuous_embodied_demo_manifest.json"
    assert playback.is_file()
    assert pressure_path.is_file()
    assert manifest_path.is_file()
    assert _sha256_file(playback) == manifest["artifacts"]["playback_mp4"][
        "sha256"
    ]
    assert _sha256_file(pressure_path) == manifest["artifacts"]["pressure_wav"][
        "sha256"
    ]
    wav_rate, pressure = wavfile.read(pressure_path)
    assert wav_rate == 48_000
    assert pressure.shape == (2_006_400, 2)
    assert pressure.dtype == np.float32
    assert np.all(np.isfinite(pressure))
    assert float(np.max(np.abs(pressure))) == pytest.approx(
        12.987203598022461, abs=1e-12
    )
    assert not any(path.name.startswith("continuous-embodied-demo-") for path in output.iterdir())

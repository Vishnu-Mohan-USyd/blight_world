#!/usr/bin/env python3
"""Record assisted running in Unreal with contact-driven Steam Audio footsteps."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import mujoco
import numpy as np
from scipy.io import wavfile

from blight_world.street_run import StreetRunController, StreetRunEnv
from blight_world.street_run_vision import make_street_run_vision_provider
from mimo_audition import AuditionConfig


ROOT = Path(__file__).resolve().parents[1]


def export(output: Path, seconds: float, warmup: float) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write the video")
    os.environ["BLIGHT_UNREAL_ASSET_MANIFEST"] = str(
        ROOT / "artifacts/unreal/street_assets/manifest.json"
    )
    os.environ.setdefault("BLIGHT_UNREAL_CONFIG", str(ROOT / "config/spear_unreal.yaml"))
    steps = round(seconds * 100)
    if steps <= 0 or steps % 2 or abs(steps / 100 - seconds) > 1e-9:
        raise ValueError("duration must be a positive multiple of 20 ms")
    env = StreetRunEnv(
        width=320,
        height=240,
        audition=AuditionConfig(backend="steam_audio"),
        vision_provider_factory=make_street_run_vision_provider,
        scene_template=ROOT / "config/street_run.xml.template",
    )
    encoder = None
    try:
        env.reset(seed=1401)
        controller = StreetRunController(env)
        provider = env.vision._provider
        assert env.audition is not None
        footstep_clips = []
        for index in range(4):
            clip_id = f"footstep_concrete_{index}"
            sample_rate, pcm = wavfile.read(
                ROOT / f"src/mimo_audition/assets/audio/48k/{clip_id}.wav"
            )
            if sample_rate != 48000 or pcm.ndim != 1:
                raise RuntimeError(f"Invalid footstep recording: {clip_id}")
            env.audition.clips.register(clip_id, pcm.astype(np.float64) / 32768.0)
            footstep_clips.append(clip_id)
        footstep_index = {"left": 0, "right": 2}
        impact_events = []
        detect = env.audition._impacts.detect
        armed = {"left": False, "right": False}
        last_contact = {"left": float(env.data.time), "right": float(env.data.time)}
        foot_sides = {f"{side}_{part}": side for side in armed
                      for part in ("foot", "toes", "big_toe")}
        foot_bodies = {env.model.body(name).id: side for name, side in foot_sides.items()}
        floor_geom = env.model.geom("floor").id
        foot_geoms = [g for g in range(env.model.ngeom)
                      if int(env.model.geom_bodyid[g]) in foot_bodies]
        foot_geom_columns = {g: i for i, g in enumerate(foot_geoms)}
        body_names = ["hip", "upper_body", "head", "right_foot", "left_foot"]
        body_ids = [env.model.body(name).id for name in body_names]
        motion = {key: [] for key in ("time", "qpos", "qvel", "ctrl", "pelvis_wrench",
                                      "contact_normal", "pose_time", "body_pos", "body_rot",
                                      "foot_geom_pos", "foot_geom_rot", "camera_rot")}

        def record_pose():
            data = env.sensor_context.data
            motion["pose_time"].append(float(data.time))
            motion["body_pos"].append(data.xpos[body_ids].copy())
            motion["body_rot"].append(data.xmat[body_ids].reshape(-1, 3, 3).copy())
            motion["foot_geom_pos"].append(data.geom_xpos[foot_geoms].copy())
            motion["foot_geom_rot"].append(data.geom_xmat[foot_geoms].reshape(-1, 3, 3).copy())
            camera_ids = [env.model.camera(name).id for name in ("eye_left", "eye_right")]
            motion["camera_rot"].append(data.cam_xmat[camera_ids].reshape(-1, 3, 3).copy())

        def record_impacts():
            # A whole-foot air gap separates landings from toe-contact chatter.
            now = float(env.data.time)
            grounded = {"left": False, "right": False}
            normal_load = np.zeros(len(foot_geoms))
            force = np.zeros(6)
            for contact_index, contact in enumerate(env.data.contact):
                geom_a, geom_b = map(int, contact.geom)
                other = geom_b if geom_a == floor_geom else geom_a if geom_b == floor_geom else -1
                if other >= 0 and contact.dist <= 0:
                    side = foot_bodies.get(int(env.model.geom_bodyid[other]))
                    if side is not None:
                        grounded[side] = True
                        mujoco.mj_contactForce(env.model, env.data, contact_index, force)
                        normal_load[foot_geom_columns[other]] += max(0.0, float(force[0]))
            motion["time"].append(now)
            motion["qpos"].append(env.data.qpos.copy())
            motion["qvel"].append(env.data.qvel.copy())
            motion["ctrl"].append(env.data.ctrl.copy())
            motion["pelvis_wrench"].append(env.data.xfrc_applied[body_ids[0]].copy())
            motion["contact_normal"].append(normal_load)
            for side in armed:
                if now - last_contact[side] >= 0.040 - 1e-9:
                    armed[side] = True
                if grounded[side]:
                    last_contact[side] = now
            strongest = {}
            for event in detect():
                side = foot_sides.get(event.attachment.body_name)
                if side is not None and (side not in strongest or
                                         event.energy_j > strongest[side].energy_j):
                    strongest[side] = event
            events = []
            for side, event in strongest.items():
                if not armed[side] or not grounded[side]:
                    continue
                armed[side] = False
                event = replace(event, clip_id=footstep_clips[footstep_index[side] % 4])
                footstep_index[side] += 1
                events.append(event)
                impact_events.append({
                    "simulation_time_s": now,
                    "body": event.attachment.body_name,
                    "position_m": event.world_position_m.tolist(),
                    "energy_j": float(event.energy_j),
                    "clip": event.clip_id,
                })
            return tuple(events)

        env.audition._impacts.detect = record_impacts
        for _ in range(round(warmup * 100)):
            env.step(controller.action())
            record_pose()
        start_time = float(env.data.time)
        start_sample = env.audition.renderer.sample_cursor
        start_position = env.data.qpos[:3].copy()
        print(f"Recording from simulation time {start_time:.3f} s", flush=True)
        silent = output / "running_video_only.mp4"
        encoder = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", "960x816",
             "-framerate", "50", "-i", "-", "-an", "-c:v", "libx264",
             "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             str(silent)],
            stdin=subprocess.PIPE,
        )
        assert encoder.stdin is not None
        audio_blocks = []
        frame_count = 0
        obs = env._get_obs()
        for frame_index in range(steps // 2):
            # Frame PTS is this state's time; sound covers the following 20 ms.
            canvas = np.zeros((816, 960, 3), dtype=np.uint8)
            canvas[:540] = provider.get_spectator_obs()
            canvas[576:, 160:480] = obs["eye_left"]
            canvas[576:, 480:800] = obs["eye_right"]
            cv2.rectangle(canvas, (0, 0), (960, 49), (13, 17, 24), -1)
            cv2.putText(canvas, "RUNNING UNDER STREETLIGHTS", (14, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Scripted motion with balance and pacing assistance", (14, 41),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (190, 195, 205), 1, cv2.LINE_AA)
            cv2.putText(canvas, "LEFT EYE", (170, 562),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 225, 235), 1, cv2.LINE_AA)
            cv2.putText(canvas, "RIGHT EYE", (490, 562),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 225, 235), 1, cv2.LINE_AA)
            encoder.stdin.write(canvas.tobytes())
            if frame_count in (0, steps // 4, steps // 2 - 1):
                cv2.imwrite(str(output / f"frame_{frame_count:04d}.png"),
                            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Recorded {frame_count}/{steps // 2} frames", flush=True)
            for subframe in range(2):
                obs, _, terminated, truncated, _ = env.step(controller.action())
                if terminated or truncated:
                    raise RuntimeError(f"Episode ended during recording at frame {frame_index}")
                sound = obs["audition"]
                if sound.shape != (2, 480) or not np.isfinite(sound).all():
                    raise RuntimeError("Invalid stereo audio observation")
                audio_blocks.append(sound.copy())
                record_pose()
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
        encoder = None
        pressure = np.concatenate(audio_blocks, axis=1)
        peak = float(np.max(np.abs(pressure)))
        if peak == 0:
            raise RuntimeError("The running sequence produced no audible pressure")
        # One constant listening gain; the scientific sensor signal stays untouched.
        monitor_gain = 0.75 / peak
        wav_path = output / "footsteps.wav"
        wavfile.write(wav_path, 48000,
                      np.round(pressure.T * monitor_gain * 32767).astype(np.int16))
        final_path = output / "street_run.mp4"
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent),
             "-i", str(wav_path), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final_path)],
            check=True,
        )
        events = [dict(event, video_time_s=event["simulation_time_s"] - start_time)
                  for event in impact_events
                  if start_time <= event["simulation_time_s"] < start_time + seconds]
        summary = {
            "duration_s": seconds,
            "frames": frame_count,
            "start_simulation_time_s": start_time,
            "start_audio_sample": start_sample,
            "audio_samples_per_ear": pressure.shape[1],
            "audio_peak_before_monitor_gain": peak,
            "monitor_gain": monitor_gain,
            "start_root_position_m": start_position.tolist(),
            "end_root_position_m": env.data.qpos[:3].tolist(),
            "impact_events": events,
        }
        (output / "footfalls.json").write_text(json.dumps(summary, indent=2) + "\n")
        metadata = {
            "recording_start_time_s": start_time,
            "bodies": body_names,
            "foot_geoms": [env.model.geom(g).name for g in foot_geoms],
            "foot_geom_size": env.model.geom_size[foot_geoms].tolist(),
            "joints": [env.model.joint(j).name for j in range(env.model.njnt)],
            "qpos_adr": env.model.jnt_qposadr.tolist(),
            "dof_adr": env.model.jnt_dofadr.tolist(),
            "actuators": [env.model.actuator(a).name for a in range(env.model.nu)],
        }
        np.savez_compressed(output / "motion.npz", metadata=json.dumps(metadata),
                            **{key: np.asarray(values) for key, values in motion.items()})
        silent.unlink()
        print(f"Saved {final_path}: {len(events)} contact sounds, "
              f"{pressure.shape[1]} samples per ear", flush=True)
        return final_path
    finally:
        try:
            if encoder is not None and encoder.poll() is None:
                if encoder.stdin is not None and not encoder.stdin.closed:
                    try:
                        encoder.stdin.close()
                    except BrokenPipeError:
                        pass
                encoder.terminate()
                try:
                    encoder.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait()
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/unreal_street_run")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    args = parser.parse_args()
    export(args.output, args.seconds, args.warmup)

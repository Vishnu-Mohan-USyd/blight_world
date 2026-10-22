"""Render continuous stereo shape videos with a fixed camera or a smooth camera arc."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import subprocess
import time

import numpy as np
from numba import njit
from PIL import Image, ImageDraw
import trimesh


SIZE, SS, FPS, FRAMES = 128, 2, 12, 48
N = SIZE * SS
FOCAL = N / (2 * np.tan(np.deg2rad(60) / 2))
EYES = (-0.0245, 0.0245)
LIGHT = np.array([-0.4, 0.65, 1.0])
LIGHT /= np.linalg.norm(LIGHT)
ORANGE = np.array([230., 119., 39.])
CLASSES = ["box", "sphere", "cylinder", "cone", "pyramid", "torus",
           "capsule", "ellipsoid", "triangular_prism", "hexagonal_prism",
           "tetrahedron", "truncated_cone"]
SPLITS = {"train": (4800, 19701), "val": (1200, 19702), "test": (1200, 19703)}
DEFAULT_OUT = Path("artifacts/simple_shapes/data")


def room(eye, camera_angle=0.):
    yy, xx = np.mgrid[:N, :N]
    cosine, sine = np.cos(camera_angle), np.sin(camera_angle)
    right = np.array([cosine, 0., -sine])
    backward = np.array([sine, 0., cosine])
    rays = (((xx + .5 - N / 2) / FOCAL)[..., None] * right
            + (-(yy + .5 - N / 2) / FOCAL)[..., None] * np.array([0., 1., 0.])
            - backward)
    origin = np.array([0., 0., .20]) + 1.05 * backward + eye * right
    depth = np.full((N, N), np.inf)
    rgb = np.full((N, N, 3), 235., dtype=float)
    walls = [(0, -.5, [204, 216, 227]), (0, .5, [221, 224, 216]),
             (1, -.5, [191, 201, 207]), (1, .5, [227, 229, 225]),
             (2, -.5, [224, 228, 231])]
    for axis, coordinate, colour in walls:
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (coordinate - origin[axis]) / rays[..., axis]
            hit = origin + distance[..., None] * rays
        others = [a for a in range(3) if a != axis]
        visible = (distance > 0) & (distance < depth)
        for a in others:
            visible &= (hit[..., a] >= -.500001) & (hit[..., a] <= .500001)
        grid = np.zeros((N, N), dtype=bool)
        for a in others:
            with np.errstate(invalid="ignore"):
                grid |= np.abs((hit[..., a] + .125) % .25 - .125) < .010
        wall_rgb = np.broadcast_to(np.array(colour, dtype=float), rgb.shape).copy()
        wall_rgb[grid] = np.array(colour) * .44
        rgb[visible] = wall_rgb[visible]
        depth[visible] = distance[visible]
    return rgb, depth


def make_shape(name, rng):
    parameters = {}
    if name == "box":
        parameters["extents"] = rng.uniform(.85, 1.15, 3).tolist()
        mesh = trimesh.creation.box(extents=parameters["extents"])
    elif name == "sphere":
        parameters["axis_scales"] = rng.uniform(.95, 1.05, 3).tolist()
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.vertices *= parameters["axis_scales"]
    elif name in ("cylinder", "cone", "pyramid"):
        parameters.update(radius=float(rng.uniform(.45, .55)),
                          height=float(rng.uniform(.9, 1.1)))
        if name == "cylinder":
            mesh = trimesh.creation.cylinder(**parameters, sections=32)
        else:
            mesh = trimesh.creation.cone(**parameters,
                                        sections=4 if name == "pyramid" else 32)
    elif name == "capsule":
        parameters.update(radius=float(rng.uniform(.30, .35)),
                          height=float(rng.uniform(.8, 1.0)), count=[16, 32])
        mesh = trimesh.creation.capsule(**parameters)
    elif name == "ellipsoid":
        parameters["axis_scales"] = [float(rng.uniform(.65, .80)),
                                     float(rng.uniform(.95, 1.05)),
                                     float(rng.uniform(1.50, 1.70))]
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.vertices *= parameters["axis_scales"]
    elif name in ("triangular_prism", "hexagonal_prism"):
        parameters.update(radius=float(rng.uniform(.45, .55)),
                          height=float(rng.uniform(.9, 1.1)),
                          sections=3 if name == "triangular_prism" else 6)
        mesh = trimesh.creation.cylinder(**parameters)
    elif name == "tetrahedron":
        parameters["radius"] = float(rng.uniform(.45, .55))
        parameters.update(height=parameters["radius"] * float(rng.uniform(1.25, 1.55)),
                          sections=3)
        mesh = trimesh.creation.cone(**parameters)
    elif name == "truncated_cone":
        parameters.update(radius=float(rng.uniform(.45, .55)),
                          height=float(rng.uniform(.9, 1.1)),
                          top_radius_ratio=float(rng.uniform(.45, .65)), sections=32)
        radius, height = parameters["radius"], parameters["height"]
        profile = [[0., -height / 2], [radius, -height / 2],
                   [radius * parameters["top_radius_ratio"], height / 2], [0., height / 2]]
        mesh = trimesh.creation.revolve(profile, sections=parameters["sections"])
    else:
        parameters.update(major_radius=float(rng.uniform(.60, .70)),
                          minor_radius=float(rng.uniform(.21, .26)))
        mesh = trimesh.creation.torus(**parameters, major_sections=32, minor_sections=12)
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    mesh.apply_transform(trimesh.transformations.quaternion_matrix(quaternion))
    mesh.vertices -= mesh.bounds.mean(axis=0)
    size = float(rng.uniform(.32, .40))
    mesh.vertices *= size / np.ptp(mesh.vertices, axis=0).max()
    parameters["maximum_rotated_extent_m"] = size
    return mesh, quaternion, parameters


def trajectory(rng, interval_seconds=1., speed_scale=1.):
    dt = 1 / 120
    # Keep the original world-space trajectory when changing render resolution.
    px_per_m = (64 / (2 * np.tan(np.deg2rad(60) / 2))) / (1.25 - .20)
    times = (np.arange(480) + .5) * dt
    segment_count = int(4 / interval_seconds)
    segment = np.minimum((times / interval_seconds).astype(int), segment_count - 1)
    fraction = np.clip((times - interval_seconds * segment - (interval_seconds - .25)) / .25, 0, 1)
    fraction = fraction * fraction * (3 - 2 * fraction)
    for _ in range(10000):
        angles = rng.uniform(0, 2 * np.pi, segment_count)
        targets = (rng.permutation([.5, 1., 1.5, 1.]) if segment_count == 4
                   else rng.choice([.5, 1., 1.5, 1.], size=segment_count))
        speeds = targets * speed_scale * FPS / px_per_m
        velocities = np.column_stack((np.cos(angles), np.sin(angles))) * speeds[:, None]
        blended = ((1 - fraction[:, None]) * velocities[segment] +
                   fraction[:, None] * velocities[np.minimum(segment + 1, segment_count - 1)])
        start = rng.uniform(-.05, .05, 2)
        points = np.vstack((start, start + np.cumsum(blended * dt, axis=0)))
        if np.abs(points).max() <= .22:
            chosen = points[np.arange(FRAMES) * 10]
            return np.column_stack((chosen, np.full(FRAMES, .20)))
    raise RuntimeError("No bounded trajectory found")


@njit(cache=True)
def rasterize(vertices, faces, vertex_normals, face_normals, position,
              eye, background_rgb, background_depth, smooth, camera_angle=0.):
    rgb = background_rgb.copy()
    depth_buffer = background_depth.copy()
    screen = np.empty((vertices.shape[0], 2))
    inverse_depth = np.empty(vertices.shape[0])
    cosine, sine = np.cos(camera_angle), np.sin(camera_angle)
    for i in range(vertices.shape[0]):
        world_x = vertices[i, 0] + position[0]
        relative_z = vertices[i, 2] + position[2] - .20
        inverse_depth[i] = 1 / (1.05 - sine * world_x - cosine * relative_z)
        screen[i, 0] = N / 2 + FOCAL * (cosine * world_x - sine * relative_z - eye) * inverse_depth[i]
        screen[i, 1] = N / 2 - FOCAL * (vertices[i, 1] + position[1]) * inverse_depth[i]
    for f in range(faces.shape[0]):
        i0, i1, i2 = faces[f]
        x0, y0 = screen[i0]
        x1, y1 = screen[i1]
        x2, y2 = screen[i2]
        lo_x = max(int(np.floor(min(x0, x1, x2))), 0)
        hi_x = min(int(np.ceil(max(x0, x1, x2))), N - 1)
        lo_y = max(int(np.floor(min(y0, y1, y2))), 0)
        hi_y = min(int(np.ceil(max(y0, y1, y2))), N - 1)
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < 1e-10:
            continue
        flat_diffuse = max(face_normals[f, 0] * LIGHT[0] +
                           face_normals[f, 1] * LIGHT[1] +
                           face_normals[f, 2] * LIGHT[2], 0.)
        for y in range(lo_y, hi_y + 1):
            for x in range(lo_x, hi_x + 1):
                a = ((y1 - y2) * (x + .5 - x2) + (x2 - x1) * (y + .5 - y2)) / denominator
                b = ((y2 - y0) * (x + .5 - x2) + (x0 - x2) * (y + .5 - y2)) / denominator
                c = 1 - a - b
                if a < -1e-7 or b < -1e-7 or c < -1e-7:
                    continue
                wa, wb, wc = a * inverse_depth[i0], b * inverse_depth[i1], c * inverse_depth[i2]
                depth = 1 / (wa + wb + wc)
                if depth >= depth_buffer[y, x]:
                    continue
                diffuse = flat_diffuse
                if smooth:
                    nx = wa * vertex_normals[i0, 0] + wb * vertex_normals[i1, 0] + wc * vertex_normals[i2, 0]
                    ny = wa * vertex_normals[i0, 1] + wb * vertex_normals[i1, 1] + wc * vertex_normals[i2, 1]
                    nz = wa * vertex_normals[i0, 2] + wb * vertex_normals[i1, 2] + wc * vertex_normals[i2, 2]
                    norm = max(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-10)
                    diffuse = max((nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]) / norm, 0.)
                shade = .35 + .65 * diffuse
                for channel in range(3):
                    rgb[y, x, channel] = ORANGE[channel] * shade
                depth_buffer[y, x] = depth
    return rgb.astype(np.uint8)


BACKGROUNDS = None
OUTPUTS = None
CAMERA_ANGLES = None
MOTION_INTERVAL = 1.
MOTION_SPEED_SCALE = 1.


def initialize_worker(out, camera_sweep_degrees, motion_interval_seconds, motion_speed_scale):
    global BACKGROUNDS, OUTPUTS, CAMERA_ANGLES, MOTION_INTERVAL, MOTION_SPEED_SCALE
    MOTION_INTERVAL, MOTION_SPEED_SCALE = motion_interval_seconds, motion_speed_scale
    CAMERA_ANGLES = np.deg2rad(np.linspace(-camera_sweep_degrees / 2,
                                         camera_sweep_degrees / 2, FRAMES))
    BACKGROUNDS = [[room(eye, angle) for eye in EYES] for angle in CAMERA_ANGLES]
    OUTPUTS = {split: np.load(Path(out) / f"{split}.npy", mmap_mode="r+")
               for split in SPLITS}


def render_recording(job):
    split, index, label, seed = job
    pose_seed, motion_seed = np.random.SeedSequence(seed).spawn(2)
    mesh, quaternion, parameters = make_shape(CLASSES[label], np.random.default_rng(pose_seed))
    positions = trajectory(np.random.default_rng(motion_seed), MOTION_INTERVAL, MOTION_SPEED_SCALE)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    vertex_normals = np.asarray(mesh.vertex_normals)
    face_normals = np.asarray(mesh.face_normals)
    smooth = CLASSES[label] in ("sphere", "torus", "capsule", "ellipsoid")
    camera_direction = int(np.random.default_rng(np.random.SeedSequence([seed, 1903])).choice([-1, 1]))
    camera_order = np.arange(FRAMES) if camera_direction == 1 else np.arange(FRAMES)[::-1]
    recording = np.empty((FRAMES, 2, 3, SIZE, SIZE), dtype=np.uint8)
    for t, position in enumerate(positions):
        camera_index = camera_order[t]
        for eye_index, eye in enumerate(EYES):
            rgb = rasterize(vertices, faces, vertex_normals, face_normals, position,
                            eye, *BACKGROUNDS[camera_index][eye_index], smooth,
                            CAMERA_ANGLES[camera_index])
            frame = Image.fromarray(rgb).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
            recording[t, eye_index] = np.asarray(frame).transpose(2, 0, 1)
    OUTPUTS[split][index] = recording
    return {"id": f"{split}_{index:04d}", "asset_id": f"{split}_{index:04d}", "array_index": index,
            "label": label, "class": CLASSES[label], "category": CLASSES[label], "seed": seed,
            "orientation_quaternion_wxyz": quaternion.tolist(),
            "shape_parameters": parameters, "positions_m": positions.tolist(),
            "camera_angles_degrees": np.rad2deg(CAMERA_ANGLES[camera_order]).tolist()}


def save_inspection(out, index):
    data = np.load(out / "train.npy", mmap_mode="r")
    examples = [next(r for r in index["splits"]["train"] if r["label"] == c)
                for c in range(len(CLASSES))]
    rows = (len(CLASSES) + 2) // 3
    sheet = Image.new("RGB", (3 * 256, rows * 282), "#20262d")
    draw = ImageDraw.Draw(sheet)
    for c, record in enumerate(examples):
        frames = data[record["array_index"]]
        x, y = (c % 3) * 256, (c // 3) * 282
        draw.text((x + 8, y + 6), CLASSES[c], fill="white")
        first = Image.fromarray(frames[0, 0].transpose(1, 2, 0))
        sheet.paste(first.resize((256, 256), Image.Resampling.NEAREST), (x, y + 26))
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                   "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", "256x128", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
                   "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   str(out / f"sample_{CLASSES[c]}.mp4")]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        for frame in frames:
            stereo = np.concatenate([eye.transpose(1, 2, 0) for eye in frame], axis=1)
            process.stdin.write(stereo.tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Video encoding failed")
    sheet.save(out / "sample_shapes.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--camera-sweep-degrees", type=float, default=0.,
                        help="Horizontal camera arc around fixed point (0,0,0.20); 0 keeps camera fixed")
    parser.add_argument("--motion-interval-seconds", type=int, choices=(1, 2), default=1)
    parser.add_argument("--motion-speed-scale", type=float, choices=(.5, 1.), default=1.)
    args = parser.parse_args()
    if not 0 <= args.camera_sweep_degrees <= 60:
        parser.error("--camera-sweep-degrees must be between 0 and 60 for the open-front room")
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = []
    index = {"classes": CLASSES, "fps": FPS, "frame_indices": list(range(FRAMES)),
             "dtype": "uint8", "array_axes": ["recording", "frame", "eye", "rgb", "y", "x"],
             "per_eye_resolution": [SIZE, SIZE], "camera_positions_m": [[e, 0, 1.25] for e in EYES],
             "camera_looks_along": "negative z; y is up", "camera_vertical_fov_degrees": 60,
             "object_colour_rgb": ORANGE.astype(int).tolist(), "wall_grid_line_width_m": .020, "wall_grid_spacing_m": .25,
             "split_metadata": {}, "splits": {}}
    index["object_motion"] = {"interval_seconds": args.motion_interval_seconds,
        "transition_seconds": .25, "speed_scale": args.motion_speed_scale,
        "reference_speeds_pixels_per_frame": [s * args.motion_speed_scale for s in (.5, 1., 1.5)],
        "reference_camera": "original fixed camera at (0,0,1.25), object plane z=0.20",
        "center_bound_m": .22, "selection": "reject complete trajectories outside the bound"}
    if args.camera_sweep_degrees:
        del index["camera_positions_m"], index["camera_looks_along"]
        index["camera_motion"] = {"type": "constant-speed horizontal arc",
            "sweep_degrees": args.camera_sweep_degrees, "radius_m": 1.05,
            "fixed_look_at_m": [0., 0., .20], "world_up": [0., 1., 0.],
            "stereo_baseline_m": .049, "stereo_axes": "parallel",
            "direction": "independently seeded left-to-right or right-to-left per recording"}
    for split, (count, split_seed) in SPLITS.items():
        array = np.lib.format.open_memmap(args.out / f"{split}.npy", mode="w+", dtype=np.uint8,
                                         shape=(count, FRAMES, 2, 3, SIZE, SIZE))
        del array
        rng = np.random.default_rng(split_seed)
        labels = np.repeat(np.arange(len(CLASSES)), count // len(CLASSES))
        rng.shuffle(labels)
        for i, label in enumerate(labels):
            jobs.append((split, i, int(label), int(rng.integers(0, 2**63))))
        index["split_metadata"][split] = {"file": f"{split}.npy", "count": count, "seed": split_seed}
        index["splits"][split] = []
    background = room(EYES[0])
    mesh, _, _ = make_shape("box", np.random.default_rng(0))
    rasterize(np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.vertex_normals),
              np.asarray(mesh.face_normals), np.array([0., 0., .2]), EYES[0], *background, False)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("fork"),
                             initializer=initialize_worker,
                             initargs=(str(args.out), args.camera_sweep_degrees,
                                       args.motion_interval_seconds, args.motion_speed_scale)) as pool:
        for n, (job, record) in enumerate(zip(jobs, pool.map(render_recording, jobs, chunksize=4)), 1):
            index["splits"][job[0]].append(record)
            if n % 60 == 0 or n == len(jobs):
                elapsed = time.monotonic() - started
                print(f"Rendered {n}/{len(jobs)} recordings; {elapsed:.1f}s; "
                      f"estimated remaining {elapsed / n * (len(jobs) - n):.1f}s", flush=True)
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    save_inspection(args.out, index)
    print(f"Complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()

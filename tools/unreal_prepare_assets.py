"""Export this room's compiled MIMo primitive geometry for Unreal's OBJ importer."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import mimoEnv
import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONTENT = "/Game/BlightWorld"
_CUBE_PADDING = 2


def _name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", value)


def _revolve(profile: list[tuple[float, float, float, float, float]], slices=48):
    vertices, normals, directions, rings, faces = [], [], [], [], []
    for radius, z, nr, nz, tz in profile:
        ring = []
        for angle in np.linspace(0, 2 * np.pi, slices, endpoint=False) if radius > 1e-14 else [0]:
            c, s = math.cos(angle), math.sin(angle)
            ring.append(len(vertices))
            vertices.append((radius * c, radius * s, z))
            normals.append((nr * c, nr * s, nz))
            directions.append((radius * c, radius * s, tz))
        rings.append(ring)
    for lower, upper in zip(rings[:-1], rings[1:]):
        for j in range(slices):
            k = (j + 1) % slices
            if len(lower) == 1:
                faces.append((lower[0], upper[k], upper[j]))
            elif len(upper) == 1:
                faces.append((lower[j], lower[k], upper[0]))
            else:
                faces.extend(((lower[j], lower[k], upper[k]), (lower[j], upper[k], upper[j])))
    return tuple(np.asarray(v) for v in (vertices, normals, directions, faces))


def _primitive(kind: int, size: np.ndarray):
    """Return metre vertices, outward normals, cube directions, and triangles."""
    if kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE):
        r = float(size[0])
        h = float(size[1]) if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
        profile = []
        for sign in (-1, 1):
            angles = np.linspace(-np.pi / 2, 0, 13) if sign < 0 else np.linspace(0, np.pi / 2, 13)
            if h == 0 and sign > 0:
                angles = angles[1:]
            for angle in angles:
                nr, nz = math.cos(angle), math.sin(angle)
                # MuJoCo's capsule hemisphere display lists use z=sign+sin(angle)
                # before their model-view translation; retain that cube mapping.
                tz = h * (sign + nz) if h else r * nz
                profile.append((r * nr, sign * h + r * nz, nr, nz, tz))
        return _revolve(profile)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = (float(v) for v in size[:2])
        vertices, normals, directions, faces = _revolve([(r, -h, 1, 0, -h), (r, h, 1, 0, h)])
        vertices, normals, directions, faces = map(list, (vertices, normals, directions, faces))
        for sign in (-1, 1):
            center = len(vertices)
            vertices.append((0, 0, sign * h))
            normals.append((0, 0, sign))
            directions.append((0, 0, sign * h))
            ring = []
            for angle in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                p = (r * math.cos(angle), r * math.sin(angle), sign * h)
                ring.append(len(vertices))
                vertices.append(p)
                normals.append((0, 0, sign))
                directions.append(p)
            for j in range(48):
                tri = (center, ring[j], ring[(j + 1) % 48])
                faces.append(tri if sign > 0 else tri[::-1])
        return tuple(np.asarray(v) for v in (vertices, normals, directions, faces))
    if kind in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_PLANE):
        vertices, normals, faces = [], [], []
        axes = [(2, 1)] if kind == mujoco.mjtGeom.mjGEOM_PLANE else [(a, s) for a in range(3) for s in (-1, 1)]
        for axis, sign in axes:
            normal = np.eye(3)[axis] * sign
            a, b = [i for i in range(3) if i != axis]
            quad = []
            for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[a], p[b] = x * size[a], y * size[b]
                if kind != mujoco.mjtGeom.mjGEOM_PLANE:
                    p[axis] = sign * size[axis]
                quad.append(p)
            if np.dot(np.cross(quad[1] - quad[0], quad[2] - quad[0]), normal) < 0:
                quad.reverse()
            offset = len(vertices)
            vertices.extend(quad)
            normals.extend([normal] * 4)
            faces.extend(((offset, offset + 1, offset + 2), (offset, offset + 2, offset + 3)))
        return np.asarray(vertices), np.asarray(normals), np.asarray(vertices), np.asarray(faces)
    raise ValueError(f"Room geometry type is not supported: {kind}")


def _clip_cube_triangle(triangle: np.ndarray, face: int):
    """Split at cube-map seams so UV interpolation never crosses atlas tiles."""
    axis, sign = face // 2, (1 if face % 2 == 0 else -1)
    polygon = list(triangle)
    for other in (i for i in range(3) if i != axis):
        for other_sign in (-1, 1):
            old, polygon = polygon, []
            for a, b in zip(old, old[1:] + old[:1]):
                da = sign * a[6 + axis] - other_sign * a[6 + other]
                db = sign * b[6 + axis] - other_sign * b[6 + other]
                if da >= -1e-14:
                    polygon.append(a)
                if (da < -1e-14) != (db < -1e-14):
                    polygon.append(a + (b - a) * da / (da - db))
            if not polygon:
                return
    for j in range(1, len(polygon) - 1):
        result = np.asarray((polygon[0], polygon[j], polygon[j + 1]))
        if np.linalg.norm(np.cross(result[1, :3] - result[0, :3], result[2, :3] - result[0, :3])) > 1e-16:
            yield result


def _cube_uv(directions: np.ndarray, face: int, width: int) -> np.ndarray:
    x, y, z = directions.T
    sc, tc, ma = ((-z, -y, x), (z, -y, -x), (x, z, y), (x, -z, -y), (x, -y, z), (-x, -y, -z))[face]
    uv = np.column_stack((sc, tc)) / ma[:, None] * 0.5 + 0.5
    uv = np.clip(uv, 0.5 / width, 1 - 0.5 / width)
    tile = width + 2 * _CUBE_PADDING
    uv = (uv * width + _CUBE_PADDING + np.array([face % 3, face // 3]) * tile) / np.array([3 * tile, 2 * tile])
    uv[:, 1] = 1 - uv[:, 1]  # Unreal's legacy importer flips OBJ V.
    return uv


def _export_texture(model, texture_id: int, directory: Path, content_root: str) -> dict:
    width, height, channels = (int(a[texture_id]) for a in (model.tex_width, model.tex_height, model.tex_nchannel))
    start = int(model.tex_adr[texture_id])
    data = model.tex_data[start : start + width * height * channels].reshape(height, width, channels)
    cube = int(model.tex_type[texture_id]) == mujoco.mjtTexture.mjTEXTURE_CUBE
    if cube:
        tile = width + 2 * _CUBE_PADDING
        pixels = np.empty((2 * tile, 3 * tile, channels), dtype=np.uint8)
        for face in range(6):
            image = data if height == width else data[face * width : (face + 1) * width]
            pixels[(face // 3) * tile : (face // 3 + 1) * tile, (face % 3) * tile : (face % 3 + 1) * tile] = np.pad(image, ((_CUBE_PADDING, _CUBE_PADDING), (_CUBE_PADDING, _CUBE_PADDING), (0, 0)), mode="edge")
    else:
        pixels = data
    name = f"T_{texture_id:02d}_{_name(model.tex(texture_id).name)}"
    path = directory / f"{name}.png"
    Image.fromarray(pixels).save(path)
    return {"name": name, "file": str(path), "asset": f"{content_root}/Textures/{name}.{name}", "cube": cube, "face_width": width}


def prepare_assets(output: Path, model_path: Path | None = None, scene_template: Path | None = None, content_root: str = CONTENT) -> Path:
    if model_path is not None and scene_template is not None:
        raise ValueError("Choose either a compiled-scene XML path or a scene template")
    content_root = content_root.rstrip("/")
    if not content_root.startswith("/Game/"):
        raise ValueError("content_root must be an Unreal /Game/... asset directory")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if model_path is None:
        upstream = Path(mimoEnv.__file__).resolve().parent / "assets"
        template = scene_template or ROOT / "src/blight_world/assets/basic_room.xml.template"
        xml = template.expanduser().resolve().read_text()
        for key, value in {
            "{{MIMO_TEXTURE_DIR}}": str(upstream),
            "{{MIMO_META_XML}}": str(upstream / "mimo/MIMo_metav2.xml"),
            "{{MIMO_MODEL_XML}}": str(upstream / "mimo/MIMo_modelv2.xml"),
            "{{SCENE_EXTENSION_XML}}": "",
        }.items():
            xml = xml.replace(key, value)
        model = mujoco.MjModel.from_xml_string(xml)
    else:
        model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))

    textures, materials, bodies = {}, {}, []
    geometry = defaultdict(list)
    for geom_id in range(model.ngeom):
        geometry[int(model.geom_bodyid[geom_id])].append(geom_id)
    for body_id, geom_ids in geometry.items():
        body_name = model.body(body_id).name or "world"
        mesh_name = f"SM_{body_id:03d}_{_name(body_name)}"
        obj_path = output / f"{mesh_name}.obj"
        body = {"body_id": body_id, "body_name": body_name, "mesh_name": mesh_name, "mesh_file": str(obj_path), "mesh_asset": f"{content_root}/Meshes/{mesh_name}.{mesh_name}", "materials": [], "geoms": []}
        with obj_path.open("w") as stream:
            stream.write(f"# MuJoCo right-handed centimetres; UE legacy importer reflects Y.\nmtllib materials.mtl\no {mesh_name}\n")
            vertex_index = 1
            for geom_id in geom_ids:
                mat_id = int(model.geom_matid[geom_id])
                rgba = model.mat_rgba[mat_id] if mat_id >= 0 else model.geom_rgba[geom_id]
                material_name = f"M_{mat_id:02d}_{_name(model.mat(mat_id).name)}" if mat_id >= 0 else f"M_geom_{geom_id}"
                texture_id = int(model.mat_texid[mat_id, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]) if mat_id >= 0 else -1
                if texture_id >= 0 and texture_id not in textures:
                    textures[texture_id] = _export_texture(model, texture_id, output, content_root)
                texture = textures.get(texture_id)
                if material_name not in materials:
                    shine = float(model.mat_shininess[mat_id]) if mat_id >= 0 else 0.5
                    materials[material_name] = {"name": material_name, "asset": f"{content_root}/Materials/{material_name}.{material_name}", "rgba": rgba.tolist(), "specular": float(model.mat_specular[mat_id]) if mat_id >= 0 else 0.5, "roughness": math.sqrt(2 / (128 * shine + 2)), "emission": float(model.mat_emission[mat_id]) if mat_id >= 0 else 0.0, "texture": texture}
                if material_name not in body["materials"]:
                    body["materials"].append(material_name)
                size = model.geom_size[geom_id]
                kind = int(model.geom_type[geom_id])
                vertices, normals, directions, faces = _primitive(kind, size)
                if texture and not bool(model.mat_texuniform[mat_id]):
                    visual_size = size if kind == mujoco.mjtGeom.mjGEOM_BOX else np.array([size[0], size[0], size[1] if kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER) else size[0]])
                    directions = directions / visual_size
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
                rotation = rotation.reshape(3, 3)
                local = vertices.copy()
                vertices = (vertices @ rotation.T + model.geom_pos[geom_id]) * 100
                normals = normals @ rotation.T
                attributes = np.column_stack((vertices, normals, directions))
                stream.write(f"g geom_{geom_id}_{_name(model.geom(geom_id).name)}\nusemtl {material_name}\ns 1\n")
                for indices in faces:
                    source = attributes[indices]
                    if texture and texture["cube"]:
                        triangles = ((tri, _cube_uv(tri[:, 6:9], face, texture["face_width"])) for face in range(6) for tri in _clip_cube_triangle(source, face))
                    else:
                        uv = np.zeros((3, 2))
                        if texture:
                            scale = model.mat_texrepeat[mat_id] / (np.ones(2) if model.mat_texuniform[mat_id] else size[:2])
                            uv = local[indices, :2] * np.array([0.5, -0.5]) * scale - 0.5
                            uv[:, 1] = 1 - uv[:, 1]
                        triangles = [(source, uv)]
                    for triangle, uv in triangles:
                        for vertex, texcoord in zip(triangle, uv):
                            normal = vertex[3:6] / np.linalg.norm(vertex[3:6])
                            stream.write("v {:.10g} {:.10g} {:.10g}\n".format(*vertex[:3]))
                            stream.write("vt {:.10g} {:.10g}\n".format(*texcoord))
                            stream.write("vn {:.10g} {:.10g} {:.10g}\n".format(*normal))
                        stream.write("f " + " ".join(f"{j}/{j}/{j}" for j in range(vertex_index, vertex_index + 3)) + "\n")
                        vertex_index += 3
                body["geoms"].append({"geom_id": geom_id, "name": model.geom(geom_id).name, "type": kind, "size_m": size.tolist(), "local_position_m": model.geom_pos[geom_id].tolist(), "local_quaternion_wxyz": model.geom_quat[geom_id].tolist(), "material": material_name})
        bodies.append(body)
    with (output / "materials.mtl").open("w") as stream:
        for material in materials.values():
            stream.write(f"newmtl {material['name']}\nKd " + " ".join(str(v) for v in material["rgba"][:3]) + "\nd 1\n\n")
    manifest = {"content_root": content_root, "ngeom": model.ngeom, "nbody": model.nbody, "bodies": bodies, "materials": list(materials.values()), "textures": list(textures.values()), "mesh_coordinates": "MuJoCo right-handed centimetres. Unreal legacy OBJ import ConvertPos reflects Y; convert_scene and convert_scene_unit must be false."}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {model.ngeom} geoms in {len(bodies)} body meshes with {len(textures)} textures: {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/unreal/assets")
    parser.add_argument("--content-root", default=CONTENT)
    scene = parser.add_mutually_exclusive_group()
    scene.add_argument("--model", type=Path)
    scene.add_argument("--scene-template", type=Path)
    arguments = parser.parse_args()
    prepare_assets(arguments.output, arguments.model, arguments.scene_template, arguments.content_root)

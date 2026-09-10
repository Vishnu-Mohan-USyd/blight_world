"""Populate WorldEditor or export its primitive actors as a MuJoCo room template."""

from __future__ import annotations

import argparse
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import unreal


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/blight_world/assets/basic_room.xml.template"
WORLD_MAP = "/Game/Maps/WorldEditor"


def _source():
    document = ET.fromstring(SOURCE.read_text(encoding="utf-8"))
    return document, document.find("worldbody")


def _numbers(element, attribute, default):
    return [float(value) for value in element.get(attribute, default).split()]


def _vector(values):
    return unreal.Vector(*values)


def _xyz(vector):
    return [float(vector.x), float(vector.y), float(vector.z)]


def _format(values):
    return " ".join(format(float(value), ".17g") for value in values)


def _actors():
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    return subsystem, [actor for actor in subsystem.get_all_level_actors() if isinstance(actor, unreal.BlightWorldPrimitive)]


def _open_world(create=False):
    levels = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    if world is not None and world.get_path_name().split(".")[0] == WORLD_MAP:
        return levels
    if unreal.EditorAssetLibrary.does_asset_exist(WORLD_MAP):
        if not levels.load_level(WORLD_MAP):
            raise RuntimeError(f"Could not open {WORLD_MAP}")
    elif create:
        unreal.EditorAssetLibrary.make_directory("/Game/Maps")
        if not levels.new_level(WORLD_MAP):
            raise RuntimeError(f"Could not create {WORLD_MAP}")
    else:
        raise RuntimeError(f"Populate {WORLD_MAP} before exporting it")
    return levels


def populate():
    """Make the fixed basic-room world geometry individually editable."""
    levels = _open_world(create=True)
    subsystem, existing = _actors()
    if existing:
        raise RuntimeError("WorldEditor already contains primitive actors; export its current edits instead of repopulating it")
    document, worldbody = _source()
    default_geom = document.find("default/geom")
    default_friction = default_geom.get("friction", "1 0.005 0.0001")
    materials = {}
    for index, material in enumerate(document.findall("asset/material")):
        name = material.get("name")
        asset_name = f"M_{index:02d}_{re.sub(r'[^a-zA-Z0-9_]', '_', name)}"
        asset_path = f"/Game/BlightWorld/Materials/{asset_name}.{asset_name}"
        materials[name] = unreal.load_asset(asset_path)
        if materials[name] is None:
            raise RuntimeError(f"Import the prepared source material before populating WorldEditor: {asset_path}")
    shapes = {
        "box": unreal.BlightWorldPrimitiveShape.BOX,
        "sphere": unreal.BlightWorldPrimitiveShape.SPHERE,
        "cylinder": unreal.BlightWorldPrimitiveShape.CYLINDER,
        "plane": unreal.BlightWorldPrimitiveShape.PLANE,
    }
    entries = [(geom, None) for geom in worldbody.findall("geom")]
    for body in worldbody.findall("body"):
        if body.get("name") != "mimo_location":
            entries.extend((geom, body) for geom in body.findall("geom"))
    for geom, body in entries:
        # The source basic room has identity world/body/geom rotations; only its
        # table and three free objects contribute an additional local position.
        position = _numbers(geom, "pos", "0 0 0")
        if body is not None:
            position = [a + b for a, b in zip(position, _numbers(body, "pos", "0 0 0"))]
        location = _vector([100 * position[0], -100 * position[1], 100 * position[2]])
        actor = subsystem.spawn_actor_from_class(unreal.BlightWorldPrimitive, location, unreal.Rotator())
        if actor is None:
            raise RuntimeError(f"Could not spawn the source primitive {geom.get('name')}")
        dynamic = body is not None and body.find("freejoint") is not None
        name = body.get("name") if dynamic else geom.get("name")
        shape = geom.get("type", "sphere")
        size = _numbers(geom, "size", "0.1")
        friction = _numbers(geom, "friction", default_friction)
        actor.set_actor_label(name)
        actor.set_actor_scale3d(unreal.Vector(1, 1, 1))
        actor.set_editor_property("physics_name", name)
        actor.set_editor_property("shape", shapes[shape])
        if shape in ("box", "plane"):
            actor.set_editor_property("half_size_m", _vector(size))
        else:
            actor.set_editor_property("radius_m", size[0])
            if shape == "cylinder":
                actor.set_editor_property("half_height_m", size[1])
        actor.set_editor_property("mass_kg", float(geom.get("mass", "0")) if dynamic else 0.0)
        for property_name, value in zip(("sliding_friction", "torsional_friction", "rolling_friction"), friction):
            actor.set_editor_property(property_name, value)
        material_name = geom.get("material")
        actor.set_editor_property("material_name", material_name)
        actor.get_editor_property("preview_mesh").set_material(0, materials[material_name])
    # Match the runtime renderer's two finite interior area lights so the closed
    # ceiling does not block the source room's directional illumination.
    for source in worldbody.findall("light"):
        position = _numbers(source, "pos", "0 0 0")
        direction = _numbers(source, "dir", "0 0 -1")
        direction[1] = -direction[1]
        rotation = unreal.Rotator(
            pitch=math.degrees(math.atan2(direction[2], math.hypot(direction[0], direction[1]))),
            yaw=math.degrees(math.atan2(direction[1], direction[0])),
            roll=0.0,
        )
        location = _vector([100 * position[0], -100 * position[1], 100 * position[2]])
        actor = subsystem.spawn_actor_from_class(unreal.RectLight, location, rotation)
        if actor is None:
            raise RuntimeError(f"Could not spawn the source light {source.get('name')}")
        actor.set_actor_label(source.get("name"))
        light = actor.get_editor_property("rect_light_component")
        light.set_mobility(unreal.ComponentMobility.MOVABLE)
        light.set_intensity_units(unreal.LightUnits.LUMENS)
        diffuse = _numbers(source, "diffuse", "1 1 1")
        strength = max(diffuse)
        light.set_light_color(unreal.LinearColor(*(value / max(strength, 1e-12) for value in diffuse), 1.0), False)
        light.set_intensity(2500.0 * strength)
        light.set_attenuation_radius(700.0)
        light.set_source_width(100.0)
        light.set_source_height(100.0)
    if not levels.save_current_level():
        raise RuntimeError(f"Could not save {WORLD_MAP}")
    unreal.log(f"Populated {WORLD_MAP} with {len(entries)} editable world primitives; MIMo remains in the MuJoCo template.")


def _primitive_attributes(actor, material_names):
    name = str(actor.get_editor_property("physics_name"))
    if name in ("", "None"):
        raise ValueError(f"Set Physics Name on {actor.get_actor_label()}")
    if any(abs(value - 1.0) > 1e-9 for value in _xyz(actor.get_actor_scale3d())):
        raise ValueError(f"{name}: actor scale must stay 1; edit the SI size fields")
    mass = float(actor.get_editor_property("mass_kg"))
    if not math.isfinite(mass) or mass < 0:
        raise ValueError(f"{name}: Mass Kg must be finite and non-negative")
    shape = actor.get_editor_property("shape")
    if shape == unreal.BlightWorldPrimitiveShape.BOX:
        kind, size = "box", _xyz(actor.get_editor_property("half_size_m"))
    elif shape == unreal.BlightWorldPrimitiveShape.PLANE:
        kind, size = "plane", _xyz(actor.get_editor_property("half_size_m"))
        if mass > 0:
            raise ValueError(f"{name}: a plane must have zero mass")
    elif shape == unreal.BlightWorldPrimitiveShape.SPHERE:
        kind, size = "sphere", [float(actor.get_editor_property("radius_m"))]
    elif shape == unreal.BlightWorldPrimitiveShape.CYLINDER:
        kind, size = "cylinder", [float(actor.get_editor_property("radius_m")), float(actor.get_editor_property("half_height_m"))]
    else:
        raise ValueError(f"{name}: unsupported primitive shape {shape}")
    if any(not math.isfinite(value) or value <= 0 for value in (size[:2] if kind == "plane" else size)):
        raise ValueError(f"{name}: primitive dimensions must be finite and positive")
    if kind == "plane" and (not math.isfinite(size[2]) or size[2] < 0):
        raise ValueError(f"{name}: plane half-size Z must be finite and non-negative")
    friction = [float(actor.get_editor_property(key)) for key in ("sliding_friction", "torsional_friction", "rolling_friction")]
    if any(not math.isfinite(value) or value < 0 for value in friction):
        raise ValueError(f"{name}: friction values must be finite and non-negative")
    geom_name = f"{name}_geom" if mass > 0 else name
    attributes = {"name": geom_name, "type": kind, "size": _format(size), "friction": _format(friction)}
    material = str(actor.get_editor_property("material_name"))
    if material not in ("", "None"):
        if material not in material_names:
            raise ValueError(f"{name}: Material Name must name a material in the source room: {material}")
        attributes["material"] = material
    location = actor.get_actor_location()
    rotation = actor.get_actor_rotation().quaternion()
    # The reflection S R S maps UE quaternion (x,y,z,w) to MJ (w,-x,y,-z).
    pose = {"pos": _format([location.x / 100, -location.y / 100, location.z / 100]), "quat": _format([rotation.w, -rotation.x, rotation.y, -rotation.z])}
    return name, mass, attributes, pose


def export_world(output):
    """Replace only world primitives while retaining MIMo and source assets."""
    _open_world()
    _, actors = _actors()
    document, worldbody = _source()
    material_names = {material.get("name") for material in document.findall("asset/material")}
    descriptions = [_primitive_attributes(actor, material_names) for actor in actors]
    physics_names = [desc[0] for desc in descriptions]
    geom_names = [desc[2]["name"] for desc in descriptions]
    if len(physics_names) != len(set(physics_names)) or len(geom_names) != len(set(geom_names)):
        raise ValueError("Each primitive needs a unique Physics Name")
    for child in list(worldbody):
        if child.tag == "geom" or (child.tag == "body" and child.get("name") != "mimo_location"):
            worldbody.remove(child)
    mimo = worldbody.find("body[@name='mimo_location']")
    insertion = list(worldbody).index(mimo)
    for name, mass, attributes, pose in sorted(descriptions, key=lambda desc: desc[0]):
        if mass > 0:
            element = ET.Element("body", {"name": name, **pose})
            ET.SubElement(element, "freejoint", {"name": f"{name}_joint"})
            ET.SubElement(element, "geom", {**attributes, "mass": _format([mass])})
        else:
            element = ET.Element("geom", {**attributes, **pose})
        worldbody.insert(insertion, element)
        insertion += 1
    retained_names = set(geom_names)
    contact = document.find("contact")
    if contact is not None:
        for pair in list(contact):
            if pair.tag == "pair" and (pair.get("geom1") not in retained_names or pair.get("geom2") not in retained_names):
                contact.remove(pair)
    ET.indent(document, space="  ")
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ET.tostring(document, encoding="unicode") + "\n", encoding="utf-8")
    unreal.log(f"Exported {len(descriptions)} world primitives to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("populate", "export"), default=os.environ.get("BLIGHT_UNREAL_WORLD_MODE"))
    parser.add_argument("--output", type=Path, default=ROOT / "config/edited_room.xml.template")
    arguments = parser.parse_args()
    if arguments.mode is None:
        parser.error("Choose populate or export (or set BLIGHT_UNREAL_WORLD_MODE)")
    if arguments.mode == "populate":
        populate()
    else:
        export_world(arguments.output)

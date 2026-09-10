"""Run in the Unreal Editor to import the prepared MIMo meshes and room map."""

import json
import os
from pathlib import Path

import unreal


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(os.environ.get("BLIGHT_UNREAL_ASSET_MANIFEST", ROOT / "artifacts/unreal/assets/manifest.json"))


def _constant(material, value, property_name):
    expression = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionConstant)
    expression.set_editor_property("r", value)
    unreal.MaterialEditingLibrary.connect_material_property(expression, "", property_name)


def import_assets():
    manifest = json.loads(MANIFEST.read_text())
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    library = unreal.EditorAssetLibrary
    content = manifest["content_root"]
    for folder in (f"{content}/Meshes", f"{content}/Materials", f"{content}/Textures", "/Game/Maps"):
        library.make_directory(folder)

    textures = {}
    for desc in manifest["textures"]:
        task = unreal.AssetImportTask()
        task.set_editor_property("automated", True)
        task.set_editor_property("filename", desc["file"])
        task.set_editor_property("destination_path", f"{content}/Textures")
        task.set_editor_property("destination_name", desc["name"])
        task.set_editor_property("replace_existing", True)
        task.set_editor_property("save", True)
        asset_tools.import_asset_tasks([task])
        texture = unreal.load_asset(desc["asset"])
        if texture is None:
            raise RuntimeError(f"Texture import did not produce {desc['asset']}")
        texture.set_editor_property("srgb", True)
        texture.set_editor_property("never_stream", True)
        textures[desc["name"]] = texture
        library.save_loaded_asset(texture)

    materials = {}
    for desc in manifest["materials"]:
        material = unreal.load_asset(desc["asset"])
        if material is None:
            material = asset_tools.create_asset(desc["name"], f"{content}/Materials", unreal.Material, unreal.MaterialFactoryNew())
        unreal.MaterialEditingLibrary.delete_all_material_expressions(material)
        if desc["texture"]:
            expression = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionTextureSample)
            expression.set_editor_property("texture", textures[desc["texture"]["name"]])
            output = "RGB"
        else:
            expression = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionConstant3Vector)
            expression.set_editor_property("constant", unreal.LinearColor(*desc["rgba"]))
            output = ""
        unreal.MaterialEditingLibrary.connect_material_property(expression, output, unreal.MaterialProperty.MP_BASE_COLOR)
        if desc.get("emission", 0.0) > 0:
            glow = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionMultiply)
            glow.set_editor_property("const_b", desc["emission"])
            unreal.MaterialEditingLibrary.connect_material_expressions(expression, output, glow, "A")
            unreal.MaterialEditingLibrary.connect_material_property(glow, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        _constant(material, desc["roughness"], unreal.MaterialProperty.MP_ROUGHNESS)
        _constant(material, desc["specular"], unreal.MaterialProperty.MP_SPECULAR)
        _constant(material, 0.0, unreal.MaterialProperty.MP_METALLIC)
        # The finite floor plane is viewed from inside the room; walls are closed.
        material.set_editor_property("two_sided", True)
        unreal.MaterialEditingLibrary.recompile_material(material)
        library.save_loaded_asset(material)
        materials[desc["name"]] = material

    for desc in manifest["bodies"]:
        options = unreal.FbxImportUI()
        options.set_editor_property("import_mesh", True)
        options.set_editor_property("import_as_skeletal", False)
        options.set_editor_property("import_materials", False)
        options.set_editor_property("import_textures", False)
        options.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_STATIC_MESH)
        data = options.get_editor_property("static_mesh_import_data")
        for property_name, value in {
            "combine_meshes": True,
            "convert_scene": False,
            "convert_scene_unit": False,
            "force_front_x_axis": False,
            "import_uniform_scale": 1.0,
            "generate_lightmap_u_vs": False,
            "auto_generate_collision": False,
            "remove_degenerates": False,
            "normal_import_method": unreal.FBXNormalImportMethod.FBXNIM_IMPORT_NORMALS,
        }.items():
            data.set_editor_property(property_name, value)
        task = unreal.AssetImportTask()
        for property_name, value in {
            "automated": True,
            "filename": desc["mesh_file"],
            "destination_path": f"{content}/Meshes",
            "destination_name": desc["mesh_name"],
            "replace_existing": True,
            "replace_existing_settings": True,
            "save": True,
            "factory": unreal.FbxFactory(),
            "options": options,
        }.items():
            task.set_editor_property(property_name, value)
        asset_tools.import_asset_tasks([task])
        mesh = unreal.load_asset(desc["mesh_asset"])
        if mesh is None:
            raise RuntimeError(f"Mesh import did not produce {desc['mesh_asset']}")
        for index, slot in enumerate(mesh.get_editor_property("static_materials")):
            name = str(slot.get_editor_property("imported_material_slot_name"))
            if name not in materials:
                name = str(slot.get_editor_property("material_slot_name"))
            if name not in materials:
                raise RuntimeError(f"Unexpected material slot {name!r} in {desc['mesh_name']}")
            mesh.set_material(index, materials[name])
        library.save_loaded_asset(mesh)

    # Runtime owns the movable body actors, lights, and the two eye cameras.
    if not library.does_asset_exist("/Game/Maps/Room"):
        subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        if not subsystem.new_level("/Game/Maps/Room"):
            raise RuntimeError("Could not create /Game/Maps/Room")
    unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
    unreal.log(f"Imported {manifest['ngeom']} MuJoCo geoms in {len(manifest['bodies'])} body meshes.")


if __name__ == "__main__":
    import_assets()

"""Exact acoustic material classification shared by contact sound models."""

from __future__ import annotations

from typing import Literal

import mujoco


AcousticMaterial = Literal["hard", "wood", "rubber", "plastic", "soft"]
MATERIAL_ORDER: tuple[AcousticMaterial, ...] = (
    "hard",
    "wood",
    "rubber",
    "plastic",
    "soft",
)
IMPACT_PAIRS = frozenset(
    {
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
    }
)


class AcousticMaterialResolver:
    """Map loaded MuJoCo geoms to the frozen impact material classes."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._mimo_root_id = model.body("mimo_location").id

    def material(self, geom_id: int) -> AcousticMaterial:
        """Return the exact acoustic class historically used for one geom."""

        name = (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            or ""
        )
        body_id = int(self.model.geom_bodyid[geom_id])
        if int(self.model.body_rootid[body_id]) == self._mimo_root_id:
            return "soft"
        if name in {"block_geom", "demo_block_geom"} or name.startswith(
            ("table_", "demo_table")
        ):
            return "wood"
        if name in {"ball_geom", "demo_ball_geom"}:
            return "rubber"
        if name in {"cylinder_geom", "demo_cylinder_geom"}:
            return "plastic"
        return "hard"


def canonical_material_pair(
    first: AcousticMaterial,
    second: AcousticMaterial,
) -> tuple[AcousticMaterial, AcousticMaterial]:
    """Return two material classes in the frozen canonical order."""

    indices = {material: index for index, material in enumerate(MATERIAL_ORDER)}
    return (first, second) if indices[first] <= indices[second] else (second, first)

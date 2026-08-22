"""MuJoCo implementation of the renderer's dynamic acoustic-scene protocol."""

from __future__ import annotations

import mujoco
import numpy as np
from numpy.typing import NDArray

from mimo_audition.sources import BodyAttachment, SiteAttachment, StaticAttachment


FloatArray = NDArray[np.float64]
LEFT_EAR_LOCAL_M = np.asarray((0.01, 0.0625, 0.0575), dtype=np.float64)
RIGHT_EAR_LOCAL_M = np.asarray((0.01, -0.0625, 0.0575), dtype=np.float64)


class MujocoAcousticScene:
    """Resolve virtual ears, attachments, and direct occlusion from MuJoCo."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self._head_body_id = model.body("head").id
        self._ignored_geom_ids = frozenset(
            model.geom(name).id
            for name in ("head", "geom:left_eye1", "geom:right_eye1")
        )

    def ear_positions(self) -> tuple[FloatArray, FloatArray]:
        """Return left/right virtual ear positions in the world frame."""

        head = self.data.body(self._head_body_id)
        rotation = head.xmat.reshape(3, 3)
        return (
            head.xpos + rotation @ LEFT_EAR_LOCAL_M,
            head.xpos + rotation @ RIGHT_EAR_LOCAL_M,
        )

    def listener_pose(self) -> tuple[FloatArray, FloatArray]:
        """Return ear midpoint and current world-from-head rotation."""

        left, right = self.ear_positions()
        rotation = self.data.body(self._head_body_id).xmat.reshape(3, 3).copy()
        return (left + right) / 2, rotation

    def attachment_pose(
        self,
        attachment: StaticAttachment | BodyAttachment | SiteAttachment,
    ) -> tuple[FloatArray, int]:
        """Resolve a typed attachment through MuJoCo's named accessors."""

        if isinstance(attachment, StaticAttachment):
            return np.asarray(attachment.world_position_m, dtype=np.float64), -1
        if isinstance(attachment, BodyAttachment):
            body = self.data.body(attachment.body_name)
            position = body.xpos + body.xmat.reshape(3, 3) @ attachment.local_position_m
            return np.asarray(position, dtype=np.float64), body.id
        if isinstance(attachment, SiteAttachment):
            site = self.data.site(attachment.site_name)
            return np.asarray(site.xpos, dtype=np.float64), int(
                self.model.site_bodyid[site.id]
            )
        raise TypeError("unsupported source attachment")

    def _geom_visible(self, geom_id: int) -> bool:
        material_id = int(self.model.geom_matid[geom_id])
        alpha = (
            self.model.mat_rgba[material_id, 3]
            if material_id >= 0
            else self.model.geom_rgba[geom_id, 3]
        )
        return bool(alpha > 0)

    def direct_path_occluded(
        self,
        listener_position_m: FloatArray,
        source_position_m: FloatArray,
        source_body_id: int,
    ) -> bool:
        """Ray-test each eligible geom without conflating child body IDs."""

        offset = source_position_m - listener_position_m
        source_distance = float(np.linalg.norm(offset))
        if source_distance <= 2e-4:
            return False
        direction = offset / source_distance
        epsilon_m = 1e-4
        ray_origin = listener_position_m + epsilon_m * direction
        for geom_id in range(self.model.ngeom):
            if geom_id in self._ignored_geom_ids:
                continue
            if int(self.model.geom_bodyid[geom_id]) == source_body_id:
                continue
            if not self._geom_visible(geom_id):
                continue
            hit_distance = mujoco.mju_rayGeom(
                self.data.geom_xpos[geom_id],
                self.data.geom_xmat[geom_id],
                self.model.geom_size[geom_id],
                ray_origin,
                direction,
                int(self.model.geom_type[geom_id]),
            )
            if epsilon_m < hit_distance < source_distance - epsilon_m:
                return True
        return False

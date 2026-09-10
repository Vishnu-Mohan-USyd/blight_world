"""Contact-onset detection and material-dependent impact specifications."""

from __future__ import annotations

from dataclasses import dataclass
from math import log10

import mujoco
import numpy as np
from numpy.typing import NDArray

from mimo_audition.materials import (
    IMPACT_PAIRS,
    AcousticMaterialResolver,
    canonical_material_pair,
)
from mimo_audition.sources import BodyAttachment


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ImpactEvent:
    """A clustered physical contact onset ready for acoustic scheduling."""

    clip_id: str
    attachment: BodyAttachment
    gain_db: float
    energy_j: float
    world_position_m: FloatArray


def impact_amplitude(energy_j: float) -> float:
    """Map estimated impact energy to the frozen bounded pressure factor."""

    if not np.isfinite(energy_j) or energy_j < 0:
        raise ValueError("energy_j must be finite and non-negative")
    return float(np.clip(np.sqrt(energy_j / 0.01), 0.25, 4.0))


class ImpactDetector:
    """Detect new energetic contacts from pre/post MuJoCo substep state."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
    ) -> None:
        self.model = model
        self.data = data
        self.rng = rng
        self._pre_velocity = np.zeros((model.nbody, 6), dtype=np.float64)
        self._pre_body_position = np.zeros((model.nbody, 3), dtype=np.float64)
        self._previous_pairs: set[tuple[int, int]] = set()
        self._last_event_time: dict[tuple[int, int], float] = {}
        self._mimo_root_id = model.body("mimo_location").id
        self._materials = AcousticMaterialResolver(model)
        self.reset(rng)

    def reset(self, rng: np.random.Generator) -> None:
        """Seed resting contacts and clear episode-specific event history."""

        self.rng = rng
        self._previous_pairs = {
            tuple(sorted(map(int, self.data.contact[index].geom)))
            for index in range(self.data.ncon)
            if np.all(self.data.contact[index].geom >= 0)
        }
        self._last_event_time.clear()
        self.before_substep()

    def before_substep(self, kinematic_data: mujoco.MjData | None = None) -> None:
        """Capture body velocities immediately before MuJoCo advances."""

        data = self.data if kinematic_data is None else kinematic_data
        for body_id in range(self.model.nbody):
            mujoco.mj_objectVelocity(
                self.model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                self._pre_velocity[body_id],
                0,
            )
        self._pre_body_position[:] = data.xpos

    def _point_velocity(self, body_id: int, world_position_m: FloatArray) -> FloatArray:
        if body_id == 0 or int(self.model.body_weldid[body_id]) == 0:
            return np.zeros(3, dtype=np.float64)
        angular = self._pre_velocity[body_id, :3]
        linear = self._pre_velocity[body_id, 3:]
        return linear + np.cross(
            angular,
            world_position_m - self._pre_body_position[body_id],
        )

    def _eligible_bodies(self, body_a: int, body_b: int) -> bool:
        static_a = int(self.model.body_weldid[body_a]) == 0
        static_b = int(self.model.body_weldid[body_b]) == 0
        if static_a and static_b:
            return False
        root_a = int(self.model.body_rootid[body_a])
        root_b = int(self.model.body_rootid[body_b])
        return not (root_a == root_b == self._mimo_root_id)

    def detect(self) -> tuple[ImpactEvent, ...]:
        """Return clustered onsets satisfying impulse and approach thresholds."""

        current_pairs: set[tuple[int, int]] = set()
        candidates: list[ImpactEvent] = []
        current_time = float(self.data.time)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom_a, geom_b = map(int, contact.geom)
            if geom_a < 0 or geom_b < 0:
                continue
            pair = tuple(sorted((geom_a, geom_b)))
            current_pairs.add(pair)
            if pair in self._previous_pairs:
                continue
            if current_time - self._last_event_time.get(pair, -np.inf) < 0.010:
                continue
            body_a = int(self.model.geom_bodyid[geom_a])
            body_b = int(self.model.geom_bodyid[geom_b])
            if not self._eligible_bodies(body_a, body_b):
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            impulse_ns = max(float(force[0]), 0.0) * float(self.model.opt.timestep)
            if impulse_ns < 5e-4:
                continue
            position = np.asarray(contact.pos, dtype=np.float64).copy()
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            velocity_a = self._point_velocity(body_a, position)
            velocity_b = self._point_velocity(body_b, position)
            approach_m_s = float(np.dot(velocity_a - velocity_b, normal))
            if approach_m_s < 0.05:
                continue
            material_pair = canonical_material_pair(
                self._materials.material(geom_a),
                self._materials.material(geom_b),
            )
            if material_pair not in IMPACT_PAIRS:
                continue
            static_a = int(self.model.body_weldid[body_a]) == 0
            static_b = int(self.model.body_weldid[body_b]) == 0
            if static_a:
                radiating_body = body_b
            elif static_b:
                radiating_body = body_a
            else:
                radiating_body = (
                    body_a
                    if np.linalg.norm(velocity_a) >= np.linalg.norm(velocity_b)
                    else body_b
                )
            body_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                radiating_body,
            )
            if body_name is None:
                continue
            body = self.data.body(radiating_body)
            local_position = body.xmat.reshape(3, 3).T @ (position - body.xpos)
            energy_j = 0.5 * impulse_ns * approach_m_s
            amplitude = impact_amplitude(energy_j)
            variant = int(self.rng.integers(0, 3))
            clip_id = f"impact_{material_pair[0]}_{material_pair[1]}_{variant}"
            candidates.append(
                ImpactEvent(
                    clip_id=clip_id,
                    attachment=BodyAttachment(
                        body_name,
                        tuple(map(float, local_position)),
                    ),
                    gain_db=20 * log10(amplitude),
                    energy_j=energy_j,
                    world_position_m=position,
                )
            )
            self._last_event_time[pair] = current_time

        self._previous_pairs = current_pairs
        candidates.sort(key=lambda event: event.energy_j, reverse=True)
        clustered: list[ImpactEvent] = []
        for candidate in candidates:
            if all(
                np.linalg.norm(candidate.world_position_m - event.world_position_m) >= 0.02
                for event in clustered
            ):
                clustered.append(candidate)
        return tuple(clustered)

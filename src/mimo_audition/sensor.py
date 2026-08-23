"""Lifecycle controller joining MuJoCo, impacts, and the binaural renderer."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
from numpy.typing import NDArray

from mimo_audition.clips import ClipLibrary
from mimo_audition.config import AuditionConfig
from mimo_audition.continuous import ContinuousEmission
from mimo_audition.contacts import ImpactDetector
from mimo_audition.mujoco_scene import MujocoAcousticScene
from mimo_audition.renderer import BinauralRenderer
from mimo_audition.sources import AudioSourceSpec


class MIMoAudition:
    """Optional deterministic auditory sensor for a loaded MIMo model."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: AuditionConfig,
    ) -> None:
        self.model = model
        self.data = data
        self.config = config
        self.scene = MujocoAcousticScene(model, data)
        self.renderer = BinauralRenderer(config)
        self._rng = np.random.default_rng(np.random.SeedSequence(0).spawn(1)[0])
        self._impacts = ImpactDetector(model, data, self._rng)
        self._pending_substeps: list[NDArray[np.float64]] = []
        self._cache = np.zeros(config.spec.observation_shape, dtype=np.float32)

    @property
    def observation(self) -> NDArray[np.float32]:
        """Return an idempotent C-contiguous copy of the latest pressure block."""

        return self._cache.copy(order="C")

    @property
    def clips(self) -> ClipLibrary:
        """Clip library used by registered and physical sources."""

        return self.renderer.clips

    def register_source(self, source: AudioSourceSpec) -> None:
        """Validate a MuJoCo attachment and register a playable source."""

        self.scene.attachment_pose(source.attachment)
        self.renderer.register_source(source)

    def play(self, source_id: str, *, delay_s: float = 0.0) -> int:
        """Schedule a registered positioned source."""

        return self.renderer.play(source_id, delay_s=delay_s)

    def stop(self, source_id: str, *, delay_s: float = 0.0) -> None:
        """Stop all voices of a registered source."""

        self.renderer.stop(source_id, delay_s=delay_s)

    def reset(self, seed: int) -> None:
        """Reset audio state from an RNG stream independent of Gym sampling."""

        child_seed = np.random.SeedSequence(int(seed)).spawn(1)[0]
        self._rng = np.random.default_rng(child_seed)
        self.renderer.reset()
        self._impacts.reset(self._rng)
        self._pending_substeps.clear()
        self._cache.fill(0)

    def before_substep(self) -> None:
        """Capture pre-integration velocities for impact approach estimates."""

        self._impacts.before_substep()

    def capture_substep(
        self,
        continuous_emissions: Sequence[ContinuousEmission] = (),
    ) -> None:
        """Schedule new impacts and render one 5 ms acoustic interval."""

        for event in self._impacts.detect():
            self.renderer.play_transient(
                event.clip_id,
                event.attachment,
                gain_db=event.gain_db,
                delay_s=0.010,
                impact_energy_j=event.energy_j,
            )
        emissions = (
            continuous_emissions if self.config.continuous is not None else ()
        )
        self._pending_substeps.append(
            self.renderer.render_substep(self.scene, emissions)
        )

    def finish_step(self) -> NDArray[np.float32]:
        """Publish exactly two substeps as one Gymnasium observation block."""

        if len(self._pending_substeps) != self.config.spec.frame_skip:
            raise RuntimeError(
                "audition expected exactly two acoustic substeps per environment step"
            )
        block = np.concatenate(self._pending_substeps, axis=1)
        self._cache = np.ascontiguousarray(block, dtype=np.float32)
        self._pending_substeps.clear()
        return self.observation

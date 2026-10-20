"""Room-relative interval object velocities for the simple-shape video cache."""

from pathlib import Path

import numpy as np


def build_motion_targets(cache: Path, index: dict) -> dict[str, np.ndarray]:
    """Return [video, time, two components] velocities in cm/s for each split.

    Shape recordings use fixed room X/Y axes and their saved positions.
    The first frame has no observed interval; its zero placeholder must be
    masked by the caller. ``cache`` is retained for the training loader interface.
    """
    frame_indices = np.asarray(index["frame_indices"], dtype=np.int64)
    targets = {}
    for split, rows in index["splits"].items():
        values = np.zeros((len(rows), len(frame_indices), 2), dtype=np.float32)
        positions = np.asarray([row["positions_m"] for row in rows], dtype=np.float64)
        dt = np.diff(frame_indices) / index["fps"]
        values[:, 1:] = np.diff(positions[..., :2], axis=1) * 100 / dt[None, :, None]
        targets[split] = values
    return targets

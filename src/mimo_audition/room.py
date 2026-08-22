"""Shoebox first-order image sources and an eight-line late-reverb FDN."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
ROOM_MIN_M = np.asarray((-2.5, -2.5, 0.0), dtype=np.float64)
ROOM_MAX_M = np.asarray((2.5, 2.5, 2.5), dtype=np.float64)
SOURCE_BOUNDARY_TOLERANCE_M = 0.010
SURFACE_ABSORPTION = {
    "wall_x_min": 0.20,
    "wall_x_max": 0.20,
    "wall_y_min": 0.20,
    "wall_y_max": 0.20,
    "floor": 0.10,
    "ceiling": 0.30,
}
FDN_DELAYS = {
    48_000: np.asarray((1_499, 1_601, 1_747, 1_861, 1_999, 2_131, 2_269, 2_381)),
    24_000: np.asarray((743, 797, 877, 929, 997, 1_063, 1_129, 1_193)),
}


@dataclass(frozen=True, slots=True)
class ImageSource:
    """One first-order image-source path in world coordinates."""

    surface: str
    world_position_m: FloatArray
    pressure_reflection: float


def project_source_position(
    source_position_m: NDArray[np.floating],
) -> FloatArray:
    """Project contact-scale boundary offsets onto the acoustic shoebox.

    MuJoCo contact positions are surface midpoints and a source attached to a
    moving contact can therefore resolve a few millimetres beyond a room
    plane.  Offsets up to 10 mm are treated as boundary positions; larger
    excursions fail closed because they indicate an invalid acoustic source.
    """

    source = np.asarray(source_position_m, dtype=np.float64)
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise ValueError("source_position_m must be a finite three-vector")
    if np.any(source < ROOM_MIN_M - SOURCE_BOUNDARY_TOLERANCE_M) or np.any(
        source > ROOM_MAX_M + SOURCE_BOUNDARY_TOLERANCE_M
    ):
        raise ValueError("source lies more than 10 mm outside the audition shoebox")
    return np.clip(source, ROOM_MIN_M, ROOM_MAX_M)


def first_order_image_sources(
    source_position_m: NDArray[np.floating],
) -> tuple[ImageSource, ...]:
    """Return the six shoebox image positions and pressure coefficients."""

    source = np.asarray(source_position_m, dtype=np.float64)
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise ValueError("source_position_m must be a finite three-vector")
    if np.any(source < ROOM_MIN_M) or np.any(source > ROOM_MAX_M):
        raise ValueError("source must lie inside the audition shoebox")
    paths: list[ImageSource] = []
    surface_names = (
        "wall_x_min",
        "wall_x_max",
        "wall_y_min",
        "wall_y_max",
        "floor",
        "ceiling",
    )
    axes_and_planes = (
        (0, ROOM_MIN_M[0]),
        (0, ROOM_MAX_M[0]),
        (1, ROOM_MIN_M[1]),
        (1, ROOM_MAX_M[1]),
        (2, ROOM_MIN_M[2]),
        (2, ROOM_MAX_M[2]),
    )
    for surface, (axis, plane) in zip(surface_names, axes_and_planes, strict=True):
        image_position = source.copy()
        image_position[axis] = 2 * plane - source[axis]
        image_position.setflags(write=False)
        paths.append(
            ImageSource(
                surface=surface,
                world_position_m=image_position,
                pressure_reflection=sqrt(1.0 - SURFACE_ABSORPTION[surface]),
            )
        )
    return tuple(paths)


def sabine_rt60_s() -> float:
    """Return the frozen shoebox's low-frequency Sabine decay time."""

    dimensions = ROOM_MAX_M - ROOM_MIN_M
    volume = float(np.prod(dimensions))
    area_xy = dimensions[0] * dimensions[1]
    area_xz = dimensions[0] * dimensions[2]
    area_yz = dimensions[1] * dimensions[2]
    absorption_area = (
        area_xy * SURFACE_ABSORPTION["floor"]
        + area_xy * SURFACE_ABSORPTION["ceiling"]
        + 2 * area_xz * SURFACE_ABSORPTION["wall_y_min"]
        + 2 * area_yz * SURFACE_ABSORPTION["wall_x_min"]
    )
    return float(np.clip(0.161 * volume / absorption_area, 0.2, 1.2))


def normalized_sylvester_8() -> FloatArray:
    """Return the orthonormal order-eight Sylvester-Hadamard matrix."""

    matrix = np.asarray(((1.0,),), dtype=np.float64)
    while matrix.shape[0] < 8:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    matrix /= sqrt(8.0)
    matrix.setflags(write=False)
    return matrix


class FeedbackDelayNetwork:
    """Eight-line stereo FDN with frequency-dependent RT60 damping."""

    def __init__(self, sample_rate_hz: int) -> None:
        if sample_rate_hz not in FDN_DELAYS:
            raise ValueError("sample_rate_hz must be 24000 or 48000")
        self.sample_rate_hz = sample_rate_hz
        self.delays = FDN_DELAYS[sample_rate_hz].copy()
        self.predelay_samples = round(0.010 * sample_rate_hz)
        self.rt60_low_s = sabine_rt60_s()
        self.rt60_high_s = max(0.15, 0.55 * self.rt60_low_s)
        self.feedback_matrix = normalized_sylvester_8()
        self.input_vector = np.ones(8, dtype=np.float64) / sqrt(8.0)
        self.output_left = np.asarray((1, -1, 1, -1, 1, -1, 1, -1), dtype=np.float64) / sqrt(8.0)
        self.output_right = np.asarray((1, 1, -1, -1, 1, 1, -1, -1), dtype=np.float64) / sqrt(8.0)

        gain_dc = np.power(
            10.0,
            -3.0 * self.delays / (sample_rate_hz * self.rt60_low_s),
        )
        gain_nyquist = np.power(
            10.0,
            -3.0 * self.delays / (sample_rate_hz * self.rt60_high_s),
        )
        self._damping_pole = (gain_dc - gain_nyquist) / (gain_dc + gain_nyquist)
        self._damping_feed = 2 * gain_dc * gain_nyquist / (gain_dc + gain_nyquist)
        self._lines = [np.zeros(int(delay), dtype=np.float64) for delay in self.delays]
        self._indices = np.zeros(8, dtype=np.int64)
        self._damping_state = np.zeros(8, dtype=np.float64)
        self._predelay = np.zeros(self.predelay_samples, dtype=np.float64)
        self._predelay_index = 0

    def reset(self) -> None:
        """Clear predelay, delay lines, damping filters, and cursors."""

        for line in self._lines:
            line.fill(0)
        self._indices.fill(0)
        self._damping_state.fill(0)
        self._predelay.fill(0)
        self._predelay_index = 0

    def process(self, mono: NDArray[np.floating]) -> FloatArray:
        """Render one mono excitation block to a decorrelated stereo tail."""

        values = np.asarray(mono, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("mono input must be a finite one-dimensional array")
        output = np.empty((2, values.size), dtype=np.float64)
        for sample_index, sample in enumerate(values):
            delayed_input = self._predelay[self._predelay_index]
            self._predelay[self._predelay_index] = sample
            self._predelay_index = (self._predelay_index + 1) % self.predelay_samples

            line_output = np.asarray(
                [line[index] for line, index in zip(self._lines, self._indices, strict=True)]
            )
            output[0, sample_index] = self.output_left @ line_output
            output[1, sample_index] = self.output_right @ line_output
            mixed = self.feedback_matrix @ line_output
            self._damping_state = (
                self._damping_feed * mixed
                + self._damping_pole * self._damping_state
            )
            line_input = self.input_vector * delayed_input + self._damping_state
            for line_index, line in enumerate(self._lines):
                line[self._indices[line_index]] = line_input[line_index]
                self._indices[line_index] = (self._indices[line_index] + 1) % line.size
        output[np.abs(output) < 1e-20] = 0.0
        return output

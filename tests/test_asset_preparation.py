"""Regression tests for deterministic asset-conversion guards."""

from __future__ import annotations

import numpy as np
import pytest

from tools.prepare_audition_assets import _is_zero_data_delay


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (np.asarray([[0.0, 0.0]], dtype=np.float64), True),
        (np.asarray([[0.0, 1.0]], dtype=np.float64), False),
        (np.asarray([[0.0, np.nan]], dtype=np.float64), False),
        (np.asarray([0.0, 0.0], dtype=np.float64), False),
    ),
)
def test_sofa_data_delay_guard(value: np.ndarray, expected: bool) -> None:
    """Only the exact finite paired-zero metadata shape is accepted."""

    assert _is_zero_data_delay(value) is expected

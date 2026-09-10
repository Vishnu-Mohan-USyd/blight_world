"""Focused clock and schedule contract for the self-examination exporter."""

from __future__ import annotations

from pathlib import Path
import sys


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from export_self_examination_demo import (  # noqa: E402
    DURATION_SECONDS,
    EVIDENCE_SEGMENTS,
    SEGMENTS,
    TOTAL_PUBLIC_STEPS,
    VIDEO_FRAMES,
    movement_envelope,
    segment_at,
)


def test_self_examination_schedule_is_contiguous_and_midpoints_are_visible_frames() -> None:
    assert TOTAL_PUBLIC_STEPS == 1_000
    assert DURATION_SECONDS == 10.0
    assert VIDEO_FRAMES == 500
    assert SEGMENTS[0].start_step == 0
    assert SEGMENTS[-1].stop_step == TOTAL_PUBLIC_STEPS
    assert all(
        left.stop_step == right.start_step
        for left, right in zip(SEGMENTS, SEGMENTS[1:])
    )
    assert [segment.name for segment in EVIDENCE_SEGMENTS] == [
        "right_hand",
        "left_hand",
        "right_lower_leg_foot",
        "left_lower_leg_foot",
    ]
    for segment in SEGMENTS:
        assert segment_at(segment.start_step) == segment
        assert segment_at(segment.stop_step - 1) == segment
    for segment in EVIDENCE_SEGMENTS:
        assert segment.midpoint_step % 2 == 1
        assert movement_envelope(segment.midpoint_step, segment) == 1.0

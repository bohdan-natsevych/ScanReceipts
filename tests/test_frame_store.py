from __future__ import annotations

import numpy as np

from scan_receipts.frame_store import FullFrameStore
from scan_receipts.models import FramePacket


def frame_with_value(value: int) -> np.ndarray:
    return np.full((120, 160, 3), value, np.uint8)


def test_recent_frame_returned_raw_and_exact() -> None:
    store = FullFrameStore(raw_seconds=3.0, compressed_seconds=20.0)
    for index in range(5):
        store.put(FramePacket(float(index), frame_with_value(40 + index), index=index))

    packet = store.get(2)

    assert packet is not None
    assert packet.index == 2
    assert int(packet.frame[0, 0, 0]) == 42


def test_old_frame_survives_in_compressed_tier() -> None:
    store = FullFrameStore(raw_seconds=1.0, compressed_seconds=20.0)
    store.put(FramePacket(0.0, frame_with_value(50), index=0))
    store.put(FramePacket(5.0, frame_with_value(90), index=1))

    old = store.get(0)

    assert old is not None
    assert old.index == 0
    # CURSOR: JPEG round-trip tolerance on a flat frame.
    assert abs(int(old.frame[10, 10, 0]) - 50) <= 3


def test_frames_beyond_compressed_window_are_dropped() -> None:
    store = FullFrameStore(raw_seconds=1.0, compressed_seconds=4.0)
    store.put(FramePacket(0.0, frame_with_value(50), index=0))
    store.put(FramePacket(10.0, frame_with_value(90), index=1))

    assert store.get(0) is None
    assert store.get(1) is not None


def test_missing_index_returns_none_and_clear_empties() -> None:
    store = FullFrameStore()
    store.put(FramePacket(0.0, frame_with_value(10), index=7))
    assert store.get(99) is None
    store.clear()
    assert store.get(7) is None

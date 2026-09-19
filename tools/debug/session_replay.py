"""Shared helpers for replaying recorded capture sessions off-line.

Recordings live under %LOCALAPPDATA%\ScanReceipts\Videos\<date>\<session-uuid>
and hold segment_*.avi plus frame_index.jsonl (one JSON object per frame with the
capture timestamp). Every debug script in this folder resolves sessions through
`find_session` so a short UUID prefix is enough on the command line.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def videos_root() -> Path:
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(local) / "ScanReceipts" / "Videos"


def find_session(spec: str) -> Path:
    """Resolve a full path, a `<date>/<uuid>` pair, or a bare UUID prefix."""
    direct = Path(spec)
    if direct.is_dir():
        return direct
    root = videos_root()
    nested = root / spec
    if nested.is_dir():
        return nested
    matches = sorted(
        path
        for path in root.glob("*/*")
        if path.is_dir() and path.name.startswith(spec)
    )
    if not matches:
        raise SystemExit(f"no session under {root} matches {spec!r}")
    if len(matches) > 1:
        joined = "\n  ".join(str(path) for path in matches)
        raise SystemExit(f"{spec!r} is ambiguous:\n  {joined}")
    return matches[0]


def list_sessions() -> list[Path]:
    return sorted(
        path
        for path in videos_root().glob("*/*")
        if (path / "frame_index.jsonl").exists()
    )


def frame_timestamps(root: Path) -> list[float]:
    lines = (root / "frame_index.jsonl").read_text(encoding="utf-8").splitlines()
    return [float(json.loads(line)["timestamp"]) for line in lines if line.strip()]


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp: float
    offset: float
    image: np.ndarray


def iter_frames(root: Path, stride: int = 1) -> Iterator[Frame]:
    """Yield decoded frames in capture order, paired with their real timestamps."""
    timestamps = frame_timestamps(root)
    started_at = timestamps[0]
    index = 0
    for filename in sorted(root.glob("segment_*.avi")):
        capture = cv2.VideoCapture(str(filename))
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if index < len(timestamps) and index % stride == 0:
                    timestamp = timestamps[index]
                    yield Frame(index, timestamp, timestamp - started_at, image)
                index += 1
        finally:
            capture.release()


def build_detector(profile: str = "realtime"):
    from scan_receipts.config import default_settings
    from scan_receipts.detection import ReceiptDetector

    return ReceiptDetector(default_settings().detection, profile=profile)


def replay(root: Path, stride: int = 1, detector=None, on_frame=None):
    """Feed a whole recording through the detector; return (started_at, candidates).

    `on_frame(frame, metrics, candidate, detector)` is called per fed frame so a
    caller can trace detector state without duplicating the decode loop.
    """
    from scan_receipts.models import FramePacket

    detector = detector or build_detector()
    started_at = frame_timestamps(root)[0]
    candidates = []
    for frame in iter_frames(root, stride):
        metrics, candidate = detector.feed(
            FramePacket(frame.timestamp, frame.image, index=frame.index)
        )
        if on_frame is not None:
            on_frame(frame, metrics, candidate, detector)
        if candidate is not None:
            candidates.append(candidate)
    candidates.extend(detector.flush_all())
    return started_at, candidates

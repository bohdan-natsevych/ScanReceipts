from __future__ import annotations

import threading
from collections import deque

import cv2
import numpy as np

from .models import FramePacket


class FullFrameStore:
    """Index-addressed full-resolution frames: recent raw, older JPEG-compressed."""

    def __init__(
        self,
        raw_seconds: float = 3.0,
        compressed_seconds: float = 20.0,
        jpeg_quality: int = 92,
    ) -> None:
        self.raw_seconds = raw_seconds
        self.compressed_seconds = compressed_seconds
        self.jpeg_quality = jpeg_quality
        self._raw: deque[FramePacket] = deque()
        self._compressed: deque[tuple[int, float, np.ndarray]] = deque()
        self._lock = threading.Lock()

    def put(self, packet: FramePacket) -> None:
        with self._lock:
            self._raw.append(
                FramePacket(packet.timestamp, packet.frame.copy(), index=packet.index)
            )
            now = packet.timestamp
            while self._raw and self._raw[0].timestamp < now - self.raw_seconds:
                old = self._raw.popleft()
                ok, encoded = cv2.imencode(
                    ".jpg", old.frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if ok:
                    self._compressed.append((old.index, old.timestamp, encoded))
            while (
                self._compressed
                and self._compressed[0][1] < now - self.compressed_seconds
            ):
                self._compressed.popleft()

    def get(self, index: int) -> FramePacket | None:
        with self._lock:
            for packet in reversed(self._raw):
                if packet.index == index:
                    return FramePacket(
                        packet.timestamp, packet.frame.copy(), index=packet.index
                    )
            for stored_index, timestamp, encoded in reversed(self._compressed):
                if stored_index == index:
                    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if frame is None:
                        return None
                    return FramePacket(timestamp, frame, index=stored_index)
        return None

    def clear(self) -> None:
        with self._lock:
            self._raw.clear()
            self._compressed.clear()

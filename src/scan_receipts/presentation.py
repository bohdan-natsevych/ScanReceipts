from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def majority_hash(hashes: Sequence[int], bits: int = 1024) -> int:
    """Bitwise majority vote suppresses per-frame hash jitter."""
    if not hashes:
        return 0
    half = len(hashes) / 2.0
    result = 0
    for bit in range(bits):
        mask = 1 << bit
        if sum(1 for value in hashes if value & mask) > half:
            result |= mask
    return result


@dataclass(frozen=True, slots=True)
class PresentationDecision:
    is_new_presentation: bool
    changed_bits: int | None


class PresentationTracker:
    """Decide whether the settled document is a new receipt presentation.

    A presentation ends by document absence, by a content change across a
    transition, or by a long transition (a full flip) even when the new page
    looks identical - identical pages are captured and left to the duplicate
    pipeline, never silently dropped.
    """

    def __init__(
        self, change_threshold: int, long_transition_seconds: float = 0.5
    ) -> None:
        self.change_threshold = change_threshold
        self.long_transition_seconds = long_transition_seconds
        self._captured = False
        self._captured_fingerprint: int | None = None
        self._transition_seen = False
        self._long_transition_seen = False

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def long_transition_seen(self) -> bool:
        """A full flip happened, so the next page is new even if it looks alike."""
        return self._long_transition_seen

    def note_transition(self, duration_seconds: float) -> None:
        self._transition_seen = True
        if duration_seconds >= self.long_transition_seconds:
            self._long_transition_seen = True

    def note_document_lost(self) -> None:
        self._captured = False
        self._captured_fingerprint = None
        self._transition_seen = False
        self._long_transition_seen = False

    def evaluate(self, fingerprint: int) -> PresentationDecision:
        if not self._captured or self._captured_fingerprint is None:
            return PresentationDecision(True, None)
        changed_bits = (fingerprint ^ self._captured_fingerprint).bit_count()
        if self._long_transition_seen:
            return PresentationDecision(True, changed_bits)
        if self._transition_seen and changed_bits > self.change_threshold:
            return PresentationDecision(True, changed_bits)
        return PresentationDecision(False, changed_bits)

    def confirm_capture(self, fingerprint: int) -> None:
        self._captured = True
        self._captured_fingerprint = fingerprint
        self._transition_seen = False
        self._long_transition_seen = False

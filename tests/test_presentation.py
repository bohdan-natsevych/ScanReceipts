from __future__ import annotations

from scan_receipts.presentation import PresentationTracker, majority_hash


def test_majority_hash_votes_per_bit() -> None:
    assert majority_hash([0b1100, 0b1010, 0b1001], bits=4) == 0b1000
    assert majority_hash([], bits=4) == 0
    assert majority_hash([0b0110], bits=4) == 0b0110


def test_first_stable_view_is_a_new_presentation() -> None:
    tracker = PresentationTracker(change_threshold=128)
    decision = tracker.evaluate(0)
    assert decision.is_new_presentation


def test_same_content_after_short_motion_is_suppressed() -> None:
    tracker = PresentationTracker(change_threshold=8, long_transition_seconds=0.5)
    tracker.confirm_capture(0b1111)
    tracker.note_transition(0.2)
    decision = tracker.evaluate(0b1110)
    assert not decision.is_new_presentation
    assert decision.changed_bits == 1


def test_changed_content_after_motion_is_new() -> None:
    tracker = PresentationTracker(change_threshold=8, long_transition_seconds=0.5)
    tracker.confirm_capture(0)
    tracker.note_transition(0.2)
    assert tracker.evaluate((1 << 32) - 1).is_new_presentation


def test_changed_content_without_any_transition_is_suppressed() -> None:
    # CURSOR: hash jitter alone must never recapture a still receipt.
    tracker = PresentationTracker(change_threshold=8)
    tracker.confirm_capture(0)
    assert not tracker.evaluate((1 << 32) - 1).is_new_presentation


def test_long_transition_with_identical_content_is_new() -> None:
    tracker = PresentationTracker(change_threshold=8, long_transition_seconds=0.5)
    tracker.confirm_capture(0b1111)
    tracker.note_transition(0.7)
    assert tracker.evaluate(0b1111).is_new_presentation


def test_document_lost_resets_capture_state() -> None:
    tracker = PresentationTracker(change_threshold=8)
    tracker.confirm_capture(0b1111)
    assert tracker.captured
    tracker.note_document_lost()
    assert not tracker.captured
    assert tracker.evaluate(0b1111).is_new_presentation

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from scan_receipts.camera import VideoFileSource
from scan_receipts.config import default_settings
from scan_receipts.database import Repository
from scan_receipts.detection import (
    ReceiptDetector,
    difference_hash,
    hamming_distance,
    order_corners,
    perceptual_hash,
    perspective_crop,
    receipt_content_evidence,
)
from scan_receipts.models import (
    CaptureCandidate,
    DetectionSettings,
    FramePacket,
    ManualCaptureRequest,
)
from scan_receipts.processing import (
    ReceiptProcessor,
    compose_grid,
    encode_output,
    grid_shape,
    process_image,
)
from scan_receipts.recovery import matches_existing_capture


def receipt_frame(label: str = "ONE") -> np.ndarray:
    frame = np.full((600, 800, 3), 35, np.uint8)
    corners = np.array([[190, 45], [610, 60], [635, 555], [170, 540]], np.int32)
    cv2.fillConvexPoly(frame, corners, (245, 245, 238))
    cv2.polylines(frame, [corners], True, (255, 255, 255), 4)
    for y in range(140, 500, 45):
        cv2.line(frame, (230, y), (570, y), (70, 70, 70), 3)
    cv2.putText(
        frame, label, (260, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 10, 10), 3
    )
    return frame


def patterned_receipt(label: str, pattern: str) -> np.ndarray:
    frame = receipt_frame(label)
    if pattern == "vertical":
        for x in range(260, 560, 35):
            cv2.line(frame, (x, 180), (x, 470), (15, 15, 15), 4)
    elif pattern == "blocks":
        for y in range(160, 480, 70):
            cv2.rectangle(frame, (250, y), (560, y + 30), (25, 25, 25), -1)
    return frame


def phone_link_frame(label: str) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), np.uint8)
    active = np.full((720, 520, 3), 35, np.uint8)
    corners = np.array([[165, 120], [365, 130], [380, 610], [150, 600]], np.int32)
    cv2.fillConvexPoly(active, corners, (235, 235, 225))
    cv2.polylines(active, [corners], True, (255, 255, 250), 3)
    cv2.putText(
        active, label, (190, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3
    )
    for y in range(280, 560, 45):
        cv2.line(active, (185, y), (350, y), (80, 80, 80), 2)
    frame[:, 384:904] = active
    return frame


def faint_receipt_with_strong_header() -> np.ndarray:
    """A faint paper edge with an internal box that is stronger than the edge."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    active = np.full((720, 520, 3), 175, np.uint8)
    cv2.rectangle(active, (150, 80), (380, 650), (205, 205, 205), -1)
    cv2.rectangle(active, (165, 100), (365, 210), (80, 80, 80), 4)
    for y in range(260, 620, 45):
        cv2.line(active, (170, y), (360, y), (170, 170, 170), 2)
    frame[:, 384:904] = active
    return frame


def phone_link_empty_desk() -> np.ndarray:
    """A bright, lightly textured desk that can look like a paper contour."""
    random = np.random.default_rng(42)
    frame = np.zeros((720, 1280, 3), np.uint8)
    texture = random.normal(0, 3, (720, 520, 1))
    desk = np.clip(188 + texture, 0, 255).astype(np.uint8)
    frame[:, 384:904] = np.repeat(desk, 3, axis=2)
    return frame


def lamp_glare_on_dark_table(with_receipt: bool = False) -> np.ndarray:
    """One lamp's glare pool on a dark speckled table, seen through a phone link.

    The pool is bright, large and roughly oval - the shape a paper detector
    looks for - but it has no boundary: it fades into the table, and the coarse
    grain of the stone puts Canny edges everywhere.
    """
    random = np.random.default_rng(7)
    height, width = 720, 520
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    pool = np.exp(
        -(((x - 250) ** 2) / (2 * 150.0**2) + ((y - 190) ** 2) / (2 * 130.0**2))
    )
    stone = 18 + 250.0 * pool + random.normal(0, 14, pool.shape)
    active = np.repeat(np.clip(stone, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    if with_receipt:
        corners = np.array([[150, 300], [360, 305], [370, 660], [140, 655]], np.int32)
        cv2.fillConvexPoly(active, corners, (238, 236, 230))
        for row in range(340, 640, 34):
            cv2.line(active, (170, row), (345, row), (60, 60, 60), 2)
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[:, 384:904] = active
    return frame


def test_candidate_frame_is_exactly_the_scored_frame() -> None:
    settings = DetectionSettings(
        analysis_width=800,
        target_analysis_fps=30,
        stable_seconds=0.10,
        motion_threshold=0.02,
        min_document_area=0.1,
    )
    detector = ReceiptDetector(settings)
    candidate = None
    for index in range(15):
        frame = receipt_frame("EXACT")
        # CURSOR: encode the frame number into one pixel far from the receipt.
        frame[0, 0] = (index, index, index)
        _, found = detector.feed(FramePacket(index * 0.04, frame, index=index))
        if found:
            candidate = found
    assert candidate is not None
    marker = int(candidate.frame[0, 0, 0])
    stored = detector.frames.get(marker)
    assert stored is not None
    assert stored.timestamp == candidate.timestamp


def test_hash_distinguishes_receipt_content() -> None:
    left = difference_hash(receipt_frame("ONE"))
    right = difference_hash(receipt_frame("TWO"))
    assert hamming_distance(left, left) == 0
    assert hamming_distance(left, right) > 0


def test_visual_fingerprint_survives_small_camera_and_lighting_changes() -> None:
    original = receipt_frame("SAME")
    transform = np.float32([[1, 0, 5], [0, 1, -4]])
    changed = cv2.warpAffine(
        original,
        transform,
        (original.shape[1], original.shape[0]),
        borderValue=(35, 35, 35),
    )
    changed = cv2.convertScaleAbs(changed, alpha=1.03, beta=5)

    distance = hamming_distance(perceptual_hash(original), perceptual_hash(changed))

    assert distance <= 10


def test_continuous_presentation_captured_once_then_reappears() -> None:
    settings = DetectionSettings(
        analysis_width=800,
        target_analysis_fps=30,
        stable_seconds=0.10,
        motion_threshold=0.02,
        min_document_area=0.1,
    )
    detector = ReceiptDetector(settings)
    frame = receipt_frame()
    captures = []
    for index in range(15):
        _, candidate = detector.feed(FramePacket(index * 0.04, frame.copy()))
        if candidate:
            captures.append(candidate)
    assert len(captures) == 1

    blank = np.full_like(frame, 35)
    for index in range(15, 40):
        detector.feed(FramePacket(index * 0.04, blank.copy()))
    for index in range(40, 55):
        _, candidate = detector.feed(FramePacket(index * 0.04, frame.copy()))
        if candidate:
            captures.append(candidate)
    assert len(captures) == 2


def test_changed_content_after_motion_captures_without_receipt_leaving_frame() -> None:
    detector = ReceiptDetector(
        DetectionSettings(
            analysis_width=800,
            target_analysis_fps=30,
            stable_seconds=0.10,
            motion_threshold=0.02,
            min_document_area=0.1,
        )
    )
    captures = []
    timestamp = 0.0
    for label in ("ONE", "TWO"):
        if label == "TWO":
            moving = receipt_frame("ONE")
            cv2.rectangle(moving, (180, 180), (640, 380), (15, 15, 15), -1)
            for _ in range(24):
                detector.feed(FramePacket(timestamp, moving.copy()))
                timestamp += 0.04
        frame = receipt_frame(label)
        if label == "TWO":
            for x in range(260, 560, 35):
                cv2.line(frame, (x, 180), (x, 470), (15, 15, 15), 4)
        for _ in range(12):
            _, candidate = detector.feed(FramePacket(timestamp, frame.copy()))
            timestamp += 0.04
            if candidate:
                captures.append(candidate)
    assert len(captures) == 2


def test_large_content_change_captures_even_below_motion_threshold() -> None:
    detector = ReceiptDetector(
        DetectionSettings(
            analysis_width=800,
            target_analysis_fps=30,
            stable_seconds=0.10,
            motion_threshold=1.0,
            min_document_area=0.1,
        )
    )
    first = receipt_frame("ONE")
    second = receipt_frame("TWO")
    for x in range(260, 560, 35):
        cv2.line(second, (x, 180), (x, 470), (15, 15, 15), 4)

    captures = []
    timestamp = 0.0
    for page_number, frame in enumerate((first, second)):
        if page_number > 0:
            # CURSOR: occlusion is a presentation boundary even when motion
            # stays below the disabled motion threshold.
            for _ in range(10):
                moving = first.copy()
                cv2.rectangle(moving, (170, 50), (640, 540), (15, 15, 15), -1)
                detector.feed(FramePacket(timestamp, moving))
                timestamp += 0.04
        for _ in range(12):
            _, candidate = detector.feed(FramePacket(timestamp, frame.copy()))
            timestamp += 0.04
            if candidate:
                captures.append(candidate)

    assert len(captures) == 2


def test_stapled_flip_without_empty_table_captures_each_receipt() -> None:
    detector = ReceiptDetector(
        DetectionSettings(
            analysis_width=800,
            target_analysis_fps=30,
            stable_seconds=0.10,
            minimum_capture_interval=0.2,
            motion_threshold=0.02,
            min_document_area=0.1,
        )
    )
    pages = [
        patterned_receipt("ONE", "plain"),
        patterned_receipt("TWO", "vertical"),
        patterned_receipt("THREE", "blocks"),
    ]
    captures = []
    timestamp = 0.0
    for page_number, page in enumerate(pages):
        if page_number > 0:
            # CURSOR: a hand flips the page; paper never leaves the frame.
            for step in range(8):
                moving = pages[page_number - 1].copy()
                x = 150 + step * 55
                cv2.rectangle(moving, (x, 90), (x + 260, 520), (15, 15, 15), -1)
                detector.feed(FramePacket(timestamp, moving))
                timestamp += 0.04
        for _ in range(14):
            _, candidate = detector.feed(FramePacket(timestamp, page.copy()))
            timestamp += 0.04
            if candidate:
                captures.append(candidate)
    assert len(captures) == 3


def test_phone_link_pillarbox_and_fast_presentations_are_detected() -> None:
    detector = ReceiptDetector(DetectionSettings())
    captures = []
    timestamp = 0.0
    for label in ("FIRST", "SECOND"):
        frame = phone_link_frame(label)
        for _ in range(18):
            _, candidate = detector.feed(FramePacket(timestamp, frame.copy()))
            timestamp += 1 / 30
            if candidate:
                captures.append(candidate)
        moving = np.zeros_like(frame)
        moving[:, 384:904] = 35
        for _ in range(30):
            detector.feed(FramePacket(timestamp, moving.copy()))
            timestamp += 1 / 30

    assert len(captures) == 2
    assert all(candidate.corners[:, 0].min() >= 384 for candidate in captures)


def test_receipt_content_validator_rejects_desk_and_accepts_faint_receipt() -> None:
    detector = ReceiptDetector(DetectionSettings(), profile="offline")
    receipt = faint_receipt_with_strong_header()
    corners, _confidence = detector.detect_document(receipt)

    assert corners is not None
    assert receipt_content_evidence(perspective_crop(receipt, corners)).is_receipt
    assert not receipt_content_evidence(phone_link_empty_desk()[:, 384:904]).is_receipt


def test_manual_redetection_prefers_faint_full_paper_over_strong_header() -> None:
    detector = ReceiptDetector(DetectionSettings(), profile="offline")

    corners, confidence = detector.detect_document(faint_receipt_with_strong_header())

    assert corners is not None
    assert confidence >= 0.7
    assert corners[:, 1].max() - corners[:, 1].min() > 500


def test_weak_automatic_header_never_crops_away_the_receipt_body() -> None:
    settings = default_settings().output
    settings.minimum_long_edge = 0
    settings.max_width = 0
    settings.enhancement = False
    header = np.array([[548, 100], [750, 100], [750, 212], [548, 212]], np.float32)

    processed = process_image(
        faint_receipt_with_strong_header(), header, 0.51, settings
    )

    # Outer paper is kept; the full Phone Link viewport is not substituted.
    assert processed.shape[0] >= 500
    assert processed.shape[1] < 450


def test_clipped_paper_is_cropped_with_margin_not_full_viewport() -> None:
    settings = default_settings().output
    settings.minimum_long_edge = 0
    settings.max_width = 0
    settings.enhancement = False
    frame = phone_link_frame("CLIPPED")
    # Paper touches the top of the active viewport, which used to zero crop
    # confidence and dump the entire camera frame.
    corners = np.array([[549, 0], [749, 8], [764, 610], [534, 600]], np.float32)

    processed = process_image(frame, corners, 0.40, settings)

    assert processed.shape[1] < 400
    assert processed.shape[0] > processed.shape[1]


def test_fast_successive_receipts_are_captured_without_waiting_two_seconds() -> None:
    detector = ReceiptDetector(
        DetectionSettings(
            analysis_width=800,
            target_analysis_fps=30,
            stable_seconds=0.10,
            minimum_capture_interval=0.35,
            motion_threshold=0.02,
            min_document_area=0.1,
        )
    )
    captures = []
    timestamp = 0.0
    for label in ("ONE", "TWO"):
        frame = receipt_frame(label)
        if label == "TWO":
            for x in range(260, 560, 35):
                cv2.line(frame, (x, 180), (x, 470), (15, 15, 15), 4)
        for _ in range(10):
            _, candidate = detector.feed(FramePacket(timestamp, frame.copy()))
            timestamp += 0.04
            if candidate:
                captures.append(candidate)
    assert len(captures) == 2
    assert captures[1].timestamp - captures[0].timestamp < 2.0


def test_flip_of_a_similar_form_is_captured_as_a_new_receipt() -> None:
    detector = ReceiptDetector(
        DetectionSettings(
            analysis_width=800,
            target_analysis_fps=30,
            stable_seconds=0.10,
            minimum_capture_interval=0.20,
            motion_threshold=0.02,
            min_document_area=0.1,
        )
    )
    frame = receipt_frame("FORM")
    captures = []
    timestamp = 0.0
    for _ in range(12):
        _, candidate = detector.feed(FramePacket(timestamp, frame.copy()))
        timestamp += 0.04
        if candidate:
            captures.append(candidate)
    # CURSOR: a covering flip hides the form so the tracker records a lost
    # document; identical content after that is a new presentation (ACC-6).
    moving = frame.copy()
    cv2.rectangle(moving, (170, 50), (640, 540), (15, 15, 15), -1)
    for _ in range(10):
        detector.feed(FramePacket(timestamp, moving.copy()))
        timestamp += 0.04
    for _ in range(12):
        _, candidate = detector.feed(FramePacket(timestamp, frame.copy()))
        timestamp += 0.04
        if candidate:
            captures.append(candidate)
    assert len(captures) == 2


def test_handheld_phone_receipts_are_captured_once_per_presentation() -> None:
    detector = ReceiptDetector(DetectionSettings())
    captures = []
    timestamp = 0.0
    offsets = [(-7, -3), (6, 3), (-5, 2), (7, -2)]

    for receipt_number in range(10):
        active = phone_link_frame(f"RECEIPT {receipt_number}")[:, 384:904]
        for frame_number in range(18):
            x, y = offsets[frame_number % len(offsets)]
            transform = np.float32([[1, 0, x], [0, 1, y]])
            frame = np.zeros((720, 1280, 3), np.uint8)
            frame[:, 384:904] = cv2.warpAffine(
                active,
                transform,
                (520, 720),
                borderValue=(35, 35, 35),
            )
            _, candidate = detector.feed(FramePacket(timestamp, frame))
            timestamp += 1 / 30
            if candidate:
                captures.append(candidate)

        blank = np.zeros((720, 1280, 3), np.uint8)
        blank[:, 384:904] = 35
        for _ in range(30):
            detector.feed(FramePacket(timestamp, blank.copy()))
            timestamp += 1 / 30

    assert len(captures) == 10


def test_recovery_matches_an_existing_capture_by_recording_timestamp() -> None:
    candidate = CaptureCandidate(
        timestamp=42.4,
        frame=np.zeros((10, 10, 3), np.uint8),
        corners=None,
        boundary_confidence=0.9,
        score=0.8,
        content_hash=0,
    )

    unrelated_hash = (1 << 1024) - 1
    assert matches_existing_capture(candidate, [unrelated_hash], [42.0])
    assert not matches_existing_capture(candidate, [unrelated_hash], [40.0])


def test_processor_preserves_original_never_overwrites_and_flags_near_duplicate(
    tmp_path: Path,
) -> None:
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Replay", settings)
    frame = receipt_frame()
    corners = np.array([[190, 45], [610, 60], [635, 555], [170, 540]], np.float32)
    content_hash = difference_hash(frame)
    candidate = CaptureCandidate(1.0, frame, corners, 0.95, 0.9, content_hash)
    processor = ReceiptProcessor(repository)

    first = processor.save_candidate(session.id, candidate, settings.output)
    second = processor.save_candidate(session.id, candidate, settings.output)

    assert first.processed_path != second.processed_path
    assert Path(first.processed_path).stat().st_size >= settings.output.minimum_bytes
    assert Path(first.original_path).exists()
    records = repository.list_receipts(session.id)
    assert len(records) == 2
    assert all(item.duplicate_group for item in records)


def test_visual_duplicate_is_flagged_when_structural_hash_drifted(
    tmp_path: Path,
) -> None:
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Replay", settings)
    frame = receipt_frame("SAME RECEIPT")
    corners = np.array([[190, 45], [610, 60], [635, 555], [170, 540]], np.float32)
    processor = ReceiptProcessor(repository)

    processor.save_candidate(
        session.id,
        CaptureCandidate(1.0, frame, corners, 0.95, 0.9, 0),
        settings.output,
    )
    processor.save_candidate(
        session.id,
        CaptureCandidate(3.0, frame, corners, 0.95, 0.9, (1 << 1024) - 1),
        settings.output,
    )

    records = repository.list_receipts(session.id)
    assert records[0].duplicate_group == records[1].duplicate_group
    assert records[0].duplicate_group is not None


def test_small_receipt_is_upscaled_only_when_asked_for() -> None:
    """Interpolation is opt-in, because it destroys the detail it pretends to add."""
    settings = default_settings().output
    small = cv2.resize(receipt_frame(), (240, 420), interpolation=cv2.INTER_AREA)

    assert settings.minimum_long_edge == 0
    assert max(process_image(small, None, 0.0, settings).shape[:2]) == 420

    settings.minimum_long_edge = 1600
    processed = process_image(small, None, 0.0, settings)

    assert max(processed.shape[:2]) == settings.minimum_long_edge


def test_quickbooks_size_floor_is_enforced() -> None:
    settings = default_settings().output
    tiny = np.full((30, 30, 3), 255, np.uint8)
    data, extension = encode_output(tiny, settings)
    assert extension == "jpg"
    assert settings.minimum_bytes <= len(data) <= settings.maximum_bytes


def test_pdf_output_is_one_quickbooks_sized_document() -> None:
    settings = default_settings().output
    settings.format = "PDF"
    data, extension = encode_output(receipt_frame(), settings)
    assert extension == "pdf"
    assert data.startswith(b"%PDF")
    assert data.count(b"%%EOF") == 1
    assert settings.minimum_bytes <= len(data) <= settings.maximum_bytes


def test_recorded_video_uses_the_same_detector_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "replay.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (800, 600)
    )
    assert writer.isOpened()
    for _ in range(20):
        writer.write(receipt_frame())
    writer.release()

    source = VideoFileSource(path)
    source.open()
    detector = ReceiptDetector(
        DetectionSettings(
            analysis_width=800,
            target_analysis_fps=30,
            stable_seconds=0.10,
            motion_threshold=0.02,
            min_document_area=0.1,
        )
    )
    captured = []
    while packet := source.read():
        _, candidate = detector.feed(packet)
        if candidate:
            captured.append(candidate)
    source.close()

    assert len(captured) == 1


def test_lamp_glare_pool_is_never_captured_as_a_receipt() -> None:
    """A pool of light is not a document: its outline is only an iso-brightness
    line through a smooth falloff, as shallow as the grain of the table."""
    settings = DetectionSettings(stable_seconds=0.1, target_analysis_fps=30)
    detector = ReceiptDetector(settings, profile="offline")

    captured = [
        candidate
        for index in range(40)
        if (
            candidate := detector.feed(
                FramePacket(index * 0.05, lamp_glare_on_dark_table(), index=index)
            )[1]
        )
        is not None
    ]

    assert captured == []


def test_receipt_under_that_same_lamp_glare_is_still_captured() -> None:
    settings = DetectionSettings(stable_seconds=0.1, target_analysis_fps=30)
    detector = ReceiptDetector(settings, profile="offline")

    captured = [
        candidate
        for index in range(40)
        if (
            candidate := detector.feed(
                FramePacket(index * 0.05, lamp_glare_on_dark_table(True), index=index)
            )[1]
        )
        is not None
    ]

    assert len(captured) == 1


def test_manual_capture_perspective_corrects_the_receipt_inside_the_selected_area(
    tmp_path: Path,
) -> None:
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Manual", settings)
    frame = receipt_frame("MANUAL")
    processor = ReceiptProcessor(repository)
    request = ManualCaptureRequest(2.0, frame, (0.15, 0.02, 0.70, 0.96))

    record = processor.save_manual(
        session.id, request, settings.output, settings.detection
    )

    stored = json.loads(record.edit_json)
    saved_corners = np.asarray(stored["auto_corners"], np.float32)
    expected = np.array([[190, 45], [610, 60], [635, 555], [170, 540]], np.float32)
    assert record.quality_flag is None
    assert np.abs(order_corners(saved_corners) - order_corners(expected)).max() <= 30.0
    processed = cv2.imread(record.processed_path, cv2.IMREAD_COLOR)
    assert processed is not None
    assert processed.mean() > 150.0


def test_manual_capture_without_a_detectable_receipt_saves_the_selection_flagged(
    tmp_path: Path,
) -> None:
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Manual", settings)
    frame = receipt_frame("MANUAL")
    processor = ReceiptProcessor(repository)
    request = ManualCaptureRequest(2.0, frame, (0.0, 0.0, 0.12, 0.9))

    record = processor.save_manual(
        session.id, request, settings.output, settings.detection
    )

    stored = json.loads(record.edit_json)
    assert record.quality_flag == "manual-crop"
    assert stored["auto_corners"] is None
    assert stored["crop"] == [0, 0, 96, 540]
    original = cv2.imread(record.original_path, cv2.IMREAD_COLOR)
    assert original.shape == frame.shape


def test_manual_capture_of_an_already_captured_receipt_joins_its_duplicate_deck(
    tmp_path: Path,
) -> None:
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Manual", settings)
    frame = receipt_frame("SAME")
    corners = np.array([[190, 45], [610, 60], [635, 555], [170, 540]], np.float32)
    processor = ReceiptProcessor(repository)
    processor.save_candidate(
        session.id,
        CaptureCandidate(1.0, frame, corners, 0.95, 0.9, difference_hash(frame)),
        settings.output,
    )

    manual = processor.save_manual(
        session.id,
        ManualCaptureRequest(3.0, frame, (0.15, 0.02, 0.70, 0.96)),
        settings.output,
        settings.detection,
    )

    records = repository.list_receipts(session.id)
    assert [item.sequence for item in records] == [1, 2]
    assert manual.duplicate_group is not None
    assert records[0].duplicate_group == records[1].duplicate_group


def test_manual_capture_crops_the_whole_receipt_not_a_stronger_inner_edge(
    tmp_path: Path,
) -> None:
    """The reported bug: a receipt whose printed content out-edges its own paper
    was cropped to a band of itself, because detection ran on the isolated
    selection where a fraction-of-frame area floor means far fewer pixels."""
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Manual", settings)
    frame = faint_receipt_with_strong_header()
    expected, _confidence = ReceiptDetector(
        settings.detection, profile="realtime"
    ).detect_document(frame)
    processor = ReceiptProcessor(repository)

    record = processor.save_manual(
        session.id,
        ManualCaptureRequest(2.0, frame, (0.39, 0.194, 0.235, 0.61)),
        settings.output,
        settings.detection,
    )

    saved = np.asarray(json.loads(record.edit_json)["auto_corners"], np.float32)
    assert np.abs(order_corners(saved) - order_corners(expected)).max() <= 4.0


def test_manual_capture_ignores_a_quad_covering_a_sliver_of_the_selection(
    tmp_path: Path,
) -> None:
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Manual", settings)
    frame = np.full((720, 1280, 3), 30, np.uint8)
    cv2.rectangle(frame, (400, 300), (460, 340), (240, 240, 235), -1)
    processor = ReceiptProcessor(repository)

    record = processor.save_manual(
        session.id,
        ManualCaptureRequest(2.0, frame, (0.25, 0.30, 0.35, 0.35)),
        settings.output,
        settings.detection,
    )

    stored = json.loads(record.edit_json)
    assert stored["auto_corners"] is None
    assert stored["crop"] == [320, 216, 448, 252]
    assert record.quality_flag == "manual-crop"


def test_manual_capture_keeps_the_drawn_area_when_detection_finds_a_fragment(
    tmp_path: Path,
) -> None:
    """The drawn area bounds the result. A quad that fills only part of it is a
    piece of the receipt, not the receipt, however confident detection is."""
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Manual", settings)
    processor = ReceiptProcessor(repository)

    record = processor.save_manual(
        session.id,
        ManualCaptureRequest(
            2.0, faint_receipt_with_strong_header(), (0.3125, 0.0833, 0.3906, 0.8333)
        ),
        settings.output,
        settings.detection,
    )

    stored = json.loads(record.edit_json)
    assert stored["auto_corners"] is None
    assert stored["crop"] == [400, 60, 500, 600]
    assert record.quality_flag == "manual-crop"


def test_exact_area_manual_capture_saves_the_selection_without_detecting(
    tmp_path: Path,
) -> None:
    """Exact-area mode keeps the drawn rectangle verbatim. Enhancement still runs;
    only the detect-and-straighten step is skipped."""
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Manual", settings)
    frame = receipt_frame("EXACT")
    processor = ReceiptProcessor(repository)
    region = (0.15, 0.02, 0.70, 0.96)

    detected = processor.save_manual(
        session.id,
        ManualCaptureRequest(1.0, frame, region),
        settings.output,
        settings.detection,
    )
    exact = processor.save_manual(
        session.id,
        ManualCaptureRequest(2.0, frame, region, exact_area=True),
        settings.output,
        settings.detection,
    )

    assert json.loads(detected.edit_json)["auto_corners"] is not None
    stored = json.loads(exact.edit_json)
    assert stored["auto_corners"] is None
    assert stored["crop"] == [120, 12, 560, 576]
    assert exact.quality_flag is None
    saved = cv2.imread(exact.processed_path, cv2.IMREAD_COLOR)
    assert saved.shape[1] / saved.shape[0] == pytest.approx(560 / 576, abs=0.02)


def solid(width: int, height: int, color: tuple[int, int, int]) -> np.ndarray:
    return np.full((height, width, 3), color, np.uint8)


def color_bounds(sheet: np.ndarray, color: tuple[int, int, int]) -> tuple[int, int]:
    """Width and height of the region painted in `color`."""
    mask = np.all(sheet == np.array(color, np.uint8), axis=2)
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    return int(columns[-1] - columns[0] + 1), int(rows[-1] - rows[0] + 1)


def combined_session(tmp_path: Path, count: int):
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Replay", settings)
    processor = ReceiptProcessor(repository)
    corners = np.array([[190, 45], [610, 60], [635, 555], [170, 540]], np.float32)
    for index in range(count):
        frame = receipt_frame(f"RECEIPT {index}")
        processor.save_candidate(
            session.id,
            CaptureCandidate(
                float(index), frame, corners, 0.95, 0.9, difference_hash(frame)
            ),
            settings.output,
        )
    return repository, processor, session, settings


def test_combined_grid_is_as_square_as_the_count_allows() -> None:
    assert grid_shape(9) == (3, 3)
    assert grid_shape(4) == (2, 2)
    assert grid_shape(3) == (2, 2)
    assert grid_shape(2) == (1, 2)
    assert grid_shape(6) == (2, 3)


def test_combined_sheet_keeps_every_receipt_whole_and_unstretched() -> None:
    tall = solid(60, 180, (0, 0, 255))
    sheet = compose_grid([tall, tall], 800)

    assert sheet.shape[1] == 800
    width, height = color_bounds(sheet, (0, 0, 255))
    # Both cells hold the same tall receipt, so the painted region spans one row.
    assert height / (width / 2) == pytest.approx(180 / 60, abs=0.15)


def test_combining_hides_the_sources_but_keeps_their_files(tmp_path: Path) -> None:
    repository, processor, session, settings = combined_session(tmp_path, 3)
    sources = repository.list_receipts(session.id)

    sheet = processor.combine_receipts(session.id, sources, settings.output)

    assert [item.id for item in repository.list_receipts(session.id)] == [sheet.id]
    assert all(Path(item.processed_path).exists() for item in sources)
    assert all(Path(item.original_path).exists() for item in sources)
    assert json.loads(sheet.edit_json)["combined_from"] == [
        item.id for item in sources
    ]
    size = Path(sheet.processed_path).stat().st_size
    assert settings.output.minimum_bytes <= size <= settings.output.maximum_bytes
    assert repository.get_session(session.id).receipt_count == 1


def test_uncombine_restores_the_sources_and_removes_the_sheet(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("scan_receipts.processing.send2trash", os.remove)
    repository, processor, session, settings = combined_session(tmp_path, 3)
    sources = repository.list_receipts(session.id)
    sheet = processor.combine_receipts(session.id, sources, settings.output)

    processor.uncombine(sheet)

    assert [item.id for item in repository.list_receipts(session.id)] == [
        item.id for item in sources
    ]
    assert not Path(sheet.processed_path).exists()
    assert repository.get_receipt(sheet.id).deleted


def test_deleting_a_combined_sheet_also_deletes_the_receipts_it_absorbed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("scan_receipts.processing.send2trash", os.remove)
    repository, processor, session, settings = combined_session(tmp_path, 3)
    sources = repository.list_receipts(session.id)
    sheet = processor.combine_receipts(session.id, sources, settings.output)

    processor.trash(sheet)

    assert repository.list_receipts(session.id) == []
    assert not any(Path(item.processed_path).exists() for item in sources)
    assert all(repository.get_receipt(item.id).deleted for item in sources)


def test_combining_refuses_more_receipts_than_the_configured_maximum(
    tmp_path: Path,
) -> None:
    repository, processor, session, settings = combined_session(tmp_path, 4)
    sources = repository.list_receipts(session.id)

    with pytest.raises(ValueError):
        processor.combine_receipts(session.id, sources, settings.output, maximum=3)

    with pytest.raises(ValueError):
        processor.combine_receipts(session.id, sources[:1], settings.output)


def test_a_combined_sheet_cannot_be_combined_again(tmp_path: Path) -> None:
    repository, processor, session, settings = combined_session(tmp_path, 4)
    sources = repository.list_receipts(session.id)
    sheet = processor.combine_receipts(session.id, sources[:2], settings.output)

    with pytest.raises(ValueError):
        processor.combine_receipts(
            session.id, [sheet, *repository.list_receipts(session.id)[1:]],
            settings.output,
        )

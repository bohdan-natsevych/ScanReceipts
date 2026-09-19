from __future__ import annotations

import json
import math
import os
import struct
import uuid
import zlib
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from send2trash import send2trash

from .database import Repository
from .detection import (
    ReceiptDetector,
    difference_hash,
    hamming_distance,
    likely_same_receipt,
    order_corners,
    perceptual_hash,
    perspective_crop,
)
from .models import (
    CaptureCandidate,
    DetectionSettings,
    ManualCaptureRequest,
    OutputSettings,
    ReceiptRecord,
)


def _expanded_corners(
    corners: np.ndarray, percent: float, shape: tuple[int, ...]
) -> np.ndarray:
    ordered = order_corners(corners)
    center = ordered.mean(axis=0)
    expanded = center + (ordered - center) * (1.0 + percent / 100.0 * 2.0)
    height, width = shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return expanded.astype(np.float32)


def process_image(
    original: np.ndarray,
    corners: np.ndarray | None,
    boundary_confidence: float,
    settings: OutputSettings,
    edits: dict | None = None,
) -> np.ndarray:
    edits = edits or {}
    image = original.copy()
    active_corners = (
        np.asarray(edits.get("corners", corners), dtype=np.float32)
        if (edits.get("corners") is not None or corners is not None)
        else None
    )
    if active_corners is not None:
        manual_corners = edits.get("corners") is not None
        if not manual_corners and boundary_confidence < 0.70:
            # CURSOR: low confidence -> try to recover the full outer paper
            # and keep extra background rather than risk cutting content (DET-7).
            outer, _confidence = ReceiptDetector(DetectionSettings()).detect_document(
                image
            )
            if outer is not None and cv2.contourArea(outer) > 1.25 * cv2.contourArea(
                active_corners
            ):
                active_corners = outer
        margin_percent = settings.crop_margin_percent
        if not manual_corners:
            margin_percent += max(0.0, 0.70 - boundary_confidence) * 20.0
        active_corners = _expanded_corners(active_corners, margin_percent, image.shape)
        if settings.perspective_correction:
            image = perspective_crop(image, active_corners)
        else:
            x, y, width, height = cv2.boundingRect(active_corners.astype(np.int32))
            image = image[y : y + height, x : x + width].copy()
    crop = edits.get("crop")
    if crop:
        x, y, width, height = [int(value) for value in crop]
        x, y = max(0, x), max(0, y)
        image = image[
            y : min(image.shape[0], y + height), x : min(image.shape[1], x + width)
        ].copy()
    rotation = (settings.rotation + int(edits.get("rotation", 0))) % 360
    if rotation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if settings.grayscale and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if settings.enhancement:
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        if image.ndim == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lightness, first, second = cv2.split(lab)
            image = cv2.cvtColor(
                cv2.merge((clahe.apply(lightness), first, second)),
                cv2.COLOR_LAB2BGR,
            )
        else:
            image = clahe.apply(image)
        alpha = 1.0 + settings.contrast / 100.0
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=settings.brightness)
        for _ in range(max(0, settings.sharpening)):
            blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
            image = cv2.addWeighted(image, 1.18, blurred, -0.18, 0)
    long_edge = max(image.shape[:2])
    if settings.minimum_long_edge > 0 and long_edge < settings.minimum_long_edge:
        # CURSOR: opt-in only; the default is 0 because interpolation cannot add
        # detail the sensor never captured, and measurably destroys what it did.
        scale = settings.minimum_long_edge / long_edge
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
        )
    if settings.max_width > 0 and image.shape[1] > settings.max_width:
        scale = settings.max_width / image.shape[1]
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    crop_relative = edits.get("crop_relative")
    if crop_relative:
        relative_x, relative_y, relative_width, relative_height = [
            float(value) for value in crop_relative
        ]
        height, width = image.shape[:2]
        x = max(0, round(relative_x * width))
        y = max(0, round(relative_y * height))
        crop_width = max(1, round(relative_width * width))
        crop_height = max(1, round(relative_height * height))
        image = image[
            y : min(height, y + crop_height),
            x : min(width, x + crop_width),
        ].copy()
    return image


# CLAUDE CODE: fallback sheet width when the output width is "original size".
COMBINE_SHEET_WIDTH = 2400
# CLAUDE CODE: a single freak-shaped receipt must not decide the sheet aspect
# for all the others; anything outside these bounds is letterboxed instead.
COMBINE_CELL_ASPECT = (0.25, 4.0)


def grid_shape(count: int) -> tuple[int, int]:
    """Rows and columns for the squarest grid holding `count` receipts."""
    columns = math.ceil(math.sqrt(max(1, count)))
    return math.ceil(max(1, count) / columns), columns


def compose_grid(images: list[np.ndarray], sheet_width: int) -> np.ndarray:
    """Lay receipts out on one white sheet, each scaled to fit its cell whole."""
    if not images:
        raise ValueError("Nothing to combine")
    rows, columns = grid_shape(len(images))
    cell_width = max(1, sheet_width // columns)
    aspect = float(
        np.clip(
            np.median([image.shape[0] / image.shape[1] for image in images]),
            *COMBINE_CELL_ASPECT,
        )
    )
    cell_height = max(1, round(cell_width * aspect))
    gutter = max(4, cell_width // 50)
    sheet = np.full(
        (rows * cell_height, columns * cell_width, 3), 255, np.uint8
    )
    for index, image in enumerate(images):
        tile = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = tile.shape[:2]
        scale = min(
            (cell_width - 2 * gutter) / width, (cell_height - 2 * gutter) / height
        )
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = cv2.resize(tile, target, interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x = column * cell_width + (cell_width - target[0]) // 2
        y = row * cell_height + (cell_height - target[1]) // 2
        sheet[y : y + target[1], x : x + target[0]] = resized
    return sheet


def _jpeg_comment_padding(data: bytes, target: int) -> bytes:
    if len(data) >= target or not data.endswith(b"\xff\xd9"):
        return data
    remaining = target - len(data)
    chunks = []
    while remaining > 0:
        payload_size = min(65531, max(0, remaining - 4))
        chunks.append(
            b"\xff\xfe" + struct.pack(">H", payload_size + 2) + b" " * payload_size
        )
        remaining -= max(1, payload_size + 4)
    return data[:-2] + b"".join(chunks) + data[-2:]


def _png_text_padding(data: bytes, target: int) -> bytes:
    if len(data) >= target or not data.endswith(b"IEND\xaeB`\x82"):
        return data
    payload_size = max(1, target - len(data) - 12)
    payload = b"MinimumUploadSize\0" + b" " * payload_size
    chunk_type = b"tEXt"
    chunk = struct.pack(">I", len(payload)) + chunk_type + payload
    chunk += struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    return data[:-12] + chunk + data[-12:]


def _pdf_comment_padding(data: bytes, target: int) -> bytes:
    if len(data) >= target:
        return data
    marker = data.rfind(b"%%EOF")
    if marker < 0:
        return data
    padding = bytearray()
    while len(data) + len(padding) < target:
        needed = target - len(data) - len(padding)
        padding.extend(b"%" + b" " * max(0, min(240, needed - 2)) + b"\n")
    return data[:marker] + bytes(padding) + data[marker:]


def _encode_pdf(image: np.ndarray, quality: int) -> bytes:
    if image.ndim == 2:
        pil_image = Image.fromarray(image, mode="L")
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mode="RGB")
    output = BytesIO()
    pil_image.save(output, format="PDF", resolution=300.0, quality=quality)
    return output.getvalue()


def _encode_once(
    image: np.ndarray, settings: OutputSettings, quality: int
) -> tuple[bytes, str]:
    is_pdf = settings.format.upper() == "PDF"
    is_png = settings.format.upper() == "PNG"
    extension = "pdf" if is_pdf else ("png" if is_png else "jpg")
    if is_pdf:
        return _encode_pdf(image, quality), extension
    parameters = (
        [cv2.IMWRITE_PNG_COMPRESSION, 3]
        if is_png
        else [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    ok, encoded = cv2.imencode(f".{extension}", image, parameters)
    if not ok:
        raise RuntimeError(f"Could not encode {extension.upper()} output")
    return encoded.tobytes(), extension


def encode_output(image: np.ndarray, settings: OutputSettings) -> tuple[bytes, str]:
    is_png = settings.format.upper() == "PNG"
    working = image
    quality = int(np.clip(settings.jpeg_quality, 30, 100))
    data, extension = _encode_once(working, settings, quality)
    for _ in range(8):
        if len(data) <= settings.maximum_bytes:
            break
        scale = max(0.55, (settings.maximum_bytes / len(data)) ** 0.5 * 0.94)
        working = cv2.resize(
            working, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        quality = max(45, quality - 8)
        data, extension = _encode_once(working, settings, quality)
    if len(data) > settings.maximum_bytes:
        raise RuntimeError(
            "Image cannot be compressed below the configured maximum size"
        )
    # CURSOR: SET-3 floor - raise quality, then resolution; pad only as
    # the last resort for content too plain to reach the floor honestly.
    # The ceiling stays binding throughout: a raise that would break it, or
    # push past the configured output width, is discarded instead of applied.
    floor = min(settings.minimum_bytes, settings.maximum_bytes)
    while len(data) < floor and not is_png and quality < 100:
        quality = min(100, quality + 10)
        raised, raised_extension = _encode_once(working, settings, quality)
        if len(raised) > settings.maximum_bytes:
            break
        data, extension = raised, raised_extension
    upscales = 0
    while len(data) < floor and upscales < 2:
        if settings.max_width > 0 and working.shape[1] * 1.25 > settings.max_width:
            break
        larger = cv2.resize(
            working, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_LANCZOS4
        )
        enlarged, enlarged_extension = _encode_once(larger, settings, quality)
        if len(enlarged) > settings.maximum_bytes:
            break
        working, data, extension = larger, enlarged, enlarged_extension
        upscales += 1
    if len(data) < floor:
        if extension == "pdf":
            data = _pdf_comment_padding(data, floor)
        elif extension == "png":
            data = _png_text_padding(data, floor)
        else:
            data = _jpeg_comment_padding(data, floor)
    return data, extension


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


# CLAUDE CODE: share of the drawn area a detected quad must fill to be taken as
# the document the user pointed at. Below it the quad is a piece of the receipt -
# a printed band, a table, the half the detector could see - and the drawn area
# is the better answer, so the area bounds the result rather than merely hinting.
MANUAL_REGION_COVERAGE = 0.60


def region_pixels(
    region: tuple[float, float, float, float] | None, width: int, height: int
) -> tuple[int, int, int, int]:
    """Clamp a relative (x, y, w, h) selection to whole pixels inside the frame."""
    if region is None:
        return 0, 0, width, height
    x = min(max(0, round(region[0] * width)), max(0, width - 1))
    y = min(max(0, round(region[1] * height)), max(0, height - 1))
    return (
        x,
        y,
        max(1, min(round(region[2] * width), width - x)),
        max(1, min(round(region[3] * height), height - y)),
    )


def _fills_region(corners: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    x, y, width, height = rect
    quad = np.ascontiguousarray(
        np.asarray(corners, np.float32).reshape(-1, 2), dtype=np.float32
    )
    region = np.ascontiguousarray(
        np.array(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float32,
        )
    )
    overlap, _ = cv2.intersectConvexConvex(quad, region)
    return overlap >= MANUAL_REGION_COVERAGE * width * height


def preview_path(original_path: str | Path) -> Path:
    original = Path(original_path)
    return original.with_name(original.name.replace("_original.png", "_preview.jpg"))


def write_preview(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        atomic_write(path, encoded.tobytes())


def trash_receipt_files(receipt: ReceiptRecord) -> None:
    for path in (
        receipt.processed_path,
        receipt.original_path,
        preview_path(receipt.original_path),
    ):
        if Path(path).exists():
            send2trash(str(path))


def combined_sources(receipt: ReceiptRecord) -> list[int]:
    """Ids of the receipts this one was combined from, empty when it is not a sheet."""
    try:
        stored = json.loads(receipt.edit_json or "{}")
    except ValueError:
        return []
    return [int(value) for value in stored.get("combined_from", [])]


class ReceiptProcessor:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    def save_candidate(
        self,
        session_id: str,
        candidate: CaptureCandidate,
        settings: OutputSettings,
        edits: dict | None = None,
    ) -> ReceiptRecord:
        processed = process_image(
            candidate.frame,
            candidate.corners,
            candidate.boundary_confidence,
            settings,
            edits,
        )
        encoded, extension = encode_output(processed, settings)
        sequence, processed_path, original_path = self.repository.next_receipt_paths(
            session_id, extension
        )
        ok, original_encoded = cv2.imencode(
            ".png", candidate.frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]
        )
        if not ok:
            raise RuntimeError("Could not preserve original frame")
        atomic_write(original_path, original_encoded.tobytes())
        atomic_write(processed_path, encoded)
        write_preview(preview_path(original_path), processed)
        priors = self.repository.list_receipts(session_id)
        duplicate_ids: list[int] = []
        matched_groups: set[str] = set()
        for prior in priors:
            prior_image = cv2.imread(prior.processed_path, cv2.IMREAD_COLOR)
            structural_match = bool(
                prior.content_hash
                and hamming_distance(
                    int(prior.content_hash, 16), candidate.content_hash
                )
                <= 128
            )
            visual_match = bool(
                prior_image is not None and likely_same_receipt(prior_image, processed)
            )
            if structural_match or visual_match:
                duplicate_ids.append(prior.id)
                if prior.duplicate_group:
                    matched_groups.add(prior.duplicate_group)
        # Merge complete existing decks. Reassigning only the directly matched
        # receipt leaves orphaned one-card "duplicate" groups behind.
        if matched_groups:
            duplicate_ids.extend(
                prior.id for prior in priors if prior.duplicate_group in matched_groups
            )
        duplicate_ids = list(dict.fromkeys(duplicate_ids))
        duplicate_group = (
            next(iter(matched_groups))
            if len(matched_groups) == 1
            else str(uuid.uuid4())
            if duplicate_ids
            else None
        )
        record = self.repository.add_receipt(
            session_id,
            sequence,
            candidate.timestamp,
            processed_path,
            original_path,
            candidate.content_hash,
            edit_json=json.dumps(
                {
                    "auto_corners": np.asarray(candidate.corners).tolist()
                    if candidate.corners is not None
                    else None,
                    "boundary_confidence": candidate.boundary_confidence,
                    **(edits or {}),
                }
            ),
            duplicate_group=duplicate_group,
            quality_flag=candidate.quality_flag,
        )
        if duplicate_group:
            self.repository.set_duplicate_group(
                [*duplicate_ids, record.id], duplicate_group
            )
        return record

    def combine_receipts(
        self,
        session_id: str,
        receipts: list[ReceiptRecord],
        settings: OutputSettings,
        maximum: int = 9,
    ) -> ReceiptRecord:
        """Lay several receipts out on one sheet and absorb them into it.

        The sources keep their files and rows so the sheet can be undone; they
        are hidden from the session by `combined_into` instead of being deleted.
        """
        if len(receipts) < 2:
            raise ValueError("Select at least two receipts to combine")
        if len(receipts) > maximum:
            raise ValueError(f"Select at most {maximum} receipts to combine")
        if any(combined_sources(receipt) for receipt in receipts):
            raise ValueError("Uncombine the combined receipt before combining it again")
        images = []
        for receipt in receipts:
            image = cv2.imread(receipt.processed_path, cv2.IMREAD_COLOR)
            if image is None:
                image = cv2.imread(receipt.original_path, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(receipt.processed_path)
            images.append(image)
        sheet = compose_grid(
            images, settings.max_width if settings.max_width > 0 else COMBINE_SHEET_WIDTH
        )
        content_hash = difference_hash(sheet)
        if settings.grayscale:
            sheet = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
        encoded, extension = encode_output(sheet, settings)
        sequence, processed_path, original_path = self.repository.next_receipt_paths(
            session_id, extension
        )
        ok, original_encoded = cv2.imencode(
            ".png", sheet, [cv2.IMWRITE_PNG_COMPRESSION, 1]
        )
        if not ok:
            raise RuntimeError("Could not preserve the combined sheet")
        atomic_write(original_path, original_encoded.tobytes())
        atomic_write(processed_path, encoded)
        write_preview(preview_path(original_path), sheet)
        rows, columns = grid_shape(len(receipts))
        record = self.repository.add_receipt(
            session_id,
            sequence,
            max(receipt.captured_at for receipt in receipts),
            processed_path,
            original_path,
            content_hash,
            edit_json=json.dumps(
                {
                    "combined_from": [receipt.id for receipt in receipts],
                    "combined_grid": [rows, columns],
                }
            ),
        )
        self.repository.set_combined_into(
            [receipt.id for receipt in receipts], record.id
        )
        return record

    def uncombine(self, receipt: ReceiptRecord) -> list[int]:
        """Release the receipts a sheet absorbed and discard the sheet itself."""
        members = self.repository.list_combined_members(receipt.id)
        if not members:
            raise ValueError("This receipt was not combined from others")
        self.repository.set_combined_into([item.id for item in members], None)
        trash_receipt_files(receipt)
        self.repository.mark_receipt_deleted(receipt.id)
        return [item.id for item in members]

    def trash(self, receipt: ReceiptRecord) -> list[int]:
        """Delete a receipt, and with a combined sheet everything it absorbed."""
        removed = [*self.repository.list_combined_members(receipt.id), receipt]
        for item in removed:
            trash_receipt_files(item)
            self.repository.mark_receipt_deleted(item.id)
        return [item.id for item in removed]

    def save_manual(
        self,
        session_id: str,
        request: ManualCaptureRequest,
        settings: OutputSettings,
        detection: DetectionSettings | None = None,
    ) -> ReceiptRecord:
        """Save a user-triggered capture through the automatic pipeline (WF-5).

        The selected region is an ROI hint: the receipt is detected inside it and
        perspective-corrected exactly like an automatic capture.
        """
        frame = request.frame
        detection = detection or DetectionSettings()
        rect = region_pixels(request.region, frame.shape[1], frame.shape[0])
        x, y, width, height = rect
        corners, confidence, from_region = (
            (None, 0.0, False)
            if request.exact_area
            else self._manual_corners(
                frame, rect, detection, request.region is not None
            )
        )
        edits: dict = {}
        quality_flag = None
        if request.exact_area:
            # CLAUDE CODE: the user asked for the rectangle they drew, so this is
            # the intended result rather than a detection failure - no flag.
            edits = {"crop": [x, y, width, height]}
        elif corners is None:
            # CLAUDE CODE: ACC-6 - a manual capture is never refused. With no
            # receipt found the selection itself is the crop, flagged so review
            # can see the geometry was never confirmed.
            edits = {"crop": [x, y, width, height]}
            quality_flag = "manual-crop"
        elif from_region:
            # CLAUDE CODE: a quad only the isolated selection could see is handed
            # over as an edit, so process_image takes its manual branch and its
            # whole-frame recovery cannot crop outside the region the user drew.
            edits = {"corners": corners.tolist()}
        chosen = (
            perspective_crop(frame, corners)
            if corners is not None
            else frame[y : y + height, x : x + width]
        )
        candidate = CaptureCandidate(
            timestamp=request.timestamp,
            frame=frame,
            corners=corners,
            boundary_confidence=confidence,
            score=1.0,
            content_hash=difference_hash(chosen),
            quality_flag=quality_flag,
        )
        return self.save_candidate(session_id, candidate, settings, edits)

    @staticmethod
    def _manual_corners(
        frame: np.ndarray,
        rect: tuple[int, int, int, int],
        detection: DetectionSettings,
        drawn: bool,
    ) -> tuple[np.ndarray | None, float, bool]:
        """Corners for a manual capture, and whether only the region found them.

        CLAUDE CODE: the whole frame is searched first, at the scale the
        automatic path uses. Detecting inside the cropped selection alone changes
        what the fraction-of-frame thresholds mean in pixels - that is how a
        printed band inside a receipt once outranked the receipt. Whatever is
        found must then fill the drawn area, because a detector that cannot see
        this receipt's outline returns a confident quad around part of it.
        """
        x, y, width, height = rect
        corners, confidence = ReceiptDetector(
            detection, profile="realtime"
        ).detect_document(frame)
        if not drawn:
            # Nothing was drawn to disagree with, so this is an automatic capture
            # that the user asked for by hand.
            return (
                (np.asarray(corners, np.float32), confidence, False)
                if corners is not None
                else (None, 0.0, False)
            )
        if corners is not None and _fills_region(corners, rect):
            return np.asarray(corners, np.float32), confidence, False
        corners, confidence = ReceiptDetector(
            detection, profile="offline"
        ).detect_document(frame[y : y + height, x : x + width])
        if corners is None:
            return None, 0.0, False
        corners = np.asarray(corners, np.float32) + np.array([x, y], dtype=np.float32)
        if not _fills_region(corners, rect):
            return None, 0.0, False
        return corners, confidence, True

    def render_edits(
        self,
        receipt: ReceiptRecord,
        settings: OutputSettings,
        edits: dict,
    ) -> ReceiptRecord:
        original = cv2.imread(receipt.original_path, cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(receipt.original_path)
        stored = json.loads(receipt.edit_json or "{}")
        auto_corners = stored.get("auto_corners")
        confidence = float(stored.get("boundary_confidence", 0.0))
        image = process_image(original, auto_corners, confidence, settings, edits)
        self._write_result(receipt, image, settings)
        merged = {**stored, **edits}
        self.repository.update_receipt_edits(receipt.id, json.dumps(merged))
        return self.repository.get_receipt(receipt.id)

    def restore_original(
        self, receipt: ReceiptRecord, settings: OutputSettings
    ) -> ReceiptRecord:
        # Restore means no auto crop or enhancement, while keeping a QuickBooks-compatible encoding.
        original = cv2.imread(receipt.original_path, cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(receipt.original_path)
        plain = OutputSettings(
            **{
                **asdict(settings),
                "perspective_correction": False,
                "enhancement": False,
            }
        )
        self._write_result(receipt, original, plain)
        self.repository.update_receipt_edits(receipt.id, json.dumps({"restored": True}))
        return self.repository.get_receipt(receipt.id)

    def reprocess_auto(
        self,
        receipt: ReceiptRecord,
        settings: OutputSettings,
        detection_settings: DetectionSettings | None = None,
    ) -> ReceiptRecord:
        original = cv2.imread(receipt.original_path, cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(receipt.original_path)
        stored = json.loads(receipt.edit_json or "{}")
        corners, confidence = ReceiptDetector(
            detection_settings or DetectionSettings(), profile="offline"
        ).detect_document(original)
        if corners is None:
            corners = stored.get("auto_corners")
            confidence = float(stored.get("boundary_confidence", 0.0))
        auto = {
            "auto_corners": np.asarray(corners).tolist()
            if corners is not None
            else None,
            "boundary_confidence": confidence,
        }
        image = process_image(
            original,
            auto["auto_corners"],
            auto["boundary_confidence"],
            settings,
        )
        self._write_result(receipt, image, settings)
        self.repository.update_receipt_edits(receipt.id, json.dumps(auto))
        return self.repository.get_receipt(receipt.id)

    def find_duplicate_groups(self, session_id: str) -> int:
        receipts = self.repository.list_receipts(session_id)
        fingerprints: dict[int, int] = {}
        images: dict[int, np.ndarray] = {}
        for receipt in receipts:
            image = cv2.imread(receipt.processed_path, cv2.IMREAD_COLOR)
            if image is not None:
                images[receipt.id] = image
                fingerprints[receipt.id] = perceptual_hash(image)

        parents = {receipt.id: receipt.id for receipt in receipts}

        def find(receipt_id: int) -> int:
            while parents[receipt_id] != receipt_id:
                parents[receipt_id] = parents[parents[receipt_id]]
                receipt_id = parents[receipt_id]
            return receipt_id

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        for index, left in enumerate(receipts):
            for right in receipts[index + 1 :]:
                if left.id not in fingerprints or right.id not in fingerprints:
                    continue
                left_image = images.get(left.id)
                right_image = images.get(right.id)
                if (
                    left_image is not None
                    and right_image is not None
                    and (
                        hamming_distance(fingerprints[left.id], fingerprints[right.id])
                        <= 10
                        or likely_same_receipt(left_image, right_image)
                    )
                ):
                    union(left.id, right.id)

        groups: dict[int, list[int]] = {}
        for receipt in receipts:
            groups.setdefault(find(receipt.id), []).append(receipt.id)
        duplicates = [members for members in groups.values() if len(members) > 1]
        self.repository.clear_duplicate_groups(session_id)
        for members in duplicates:
            self.repository.set_duplicate_group(members, str(uuid.uuid4()))
        return len(duplicates)

    def _write_result(
        self, receipt: ReceiptRecord, image: np.ndarray, settings: OutputSettings
    ) -> None:
        encoded, extension = encode_output(image, settings)
        current = Path(receipt.processed_path)
        target = current.with_suffix(f".{extension}")
        if target != current and target.exists():
            raise FileExistsError(target)
        atomic_write(target, encoded)
        write_preview(preview_path(receipt.original_path), image)
        if target != current:
            if current.exists():
                current.unlink()
            self.repository.update_receipt_path(receipt.id, target)

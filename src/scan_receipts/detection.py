from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .frame_store import FullFrameStore
from .models import (
    CaptureCandidate,
    DetectionMetrics,
    DetectionSettings,
    DetectorState,
    FramePacket,
)
from .presentation import PresentationTracker, majority_hash

log = logging.getLogger(__name__)


def order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    result = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    result[0] = points[np.argmin(sums)]  # top-left
    result[2] = points[np.argmax(sums)]  # bottom-right
    result[1] = points[np.argmin(differences)]  # top-right
    result[3] = points[np.argmax(differences)]  # bottom-left
    return result


def perspective_crop(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    corners = order_corners(corners)
    tl, tr, br, bl = corners
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 2 or height < 2:
        return image.copy()
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(
        image, transform, (width, height), borderMode=cv2.BORDER_REPLICATE
    )


def difference_hash(image: np.ndarray, size: int = 32) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(resized)
    # A Laplacian sign map retains small printed-content changes that an 8x8
    # luminance hash loses, while remaining insensitive to global brightness.
    structure = cv2.Laplacian(normalized, cv2.CV_32F)
    differences = structure > float(np.median(structure))
    result = 0
    for bit in differences.flat:
        result = (result << 1) | int(bit)
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def perceptual_hash(image: np.ndarray, size: int = 32, low_frequency: int = 8) -> int:
    """Compact DCT fingerprint robust to scale, lighting, and small crop changes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))[:low_frequency, :low_frequency]
    median = float(np.median(coefficients.flat[1:]))
    result = 0
    for bit in (coefficients > median).flat:
        result = (result << 1) | int(bit)
    return result


def likely_same_receipt(left: np.ndarray, right: np.ndarray) -> bool:
    """Recognize the same physical receipt after geometric alignment."""
    hash_match = hamming_distance(perceptual_hash(left), perceptual_hash(right)) <= 10

    def normalized_gray(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.resize(gray, (400, 600), interpolation=cv2.INTER_LINEAR)

    left_gray, right_gray = normalized_gray(left), normalized_gray(right)
    sift = cv2.SIFT_create(nfeatures=1500)
    left_points, left_descriptors = sift.detectAndCompute(left_gray, None)
    right_points, right_descriptors = sift.detectAndCompute(right_gray, None)
    if (
        left_descriptors is None
        or right_descriptors is None
        or len(left_points) < 20
        or len(right_points) < 20
    ):
        return hash_match
    matches = cv2.BFMatcher().knnMatch(left_descriptors, right_descriptors, k=2)
    good = [match for match, other in matches if match.distance < 0.72 * other.distance]
    if len(good) < 20:
        return hash_match
    left_locations = np.float32([left_points[item.queryIdx].pt for item in good])
    right_locations = np.float32([right_points[item.trainIdx].pt for item in good])
    transform, inlier_mask = cv2.findHomography(
        right_locations, left_locations, cv2.RANSAC, 7.0
    )
    if transform is None or inlier_mask is None:
        return hash_match
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good)
    if inlier_ratio < 0.65:
        return hash_match
    # When one view is a close crop and another contains the full camera
    # viewport, whole-image residuals are dominated by desk/background. A
    # large, overwhelmingly consistent homography is conclusive evidence of
    # the same physical receipt. Distinct preprinted forms in the regression
    # recordings produce far fewer and much less consistent inliers.
    if inliers >= 100 and inlier_ratio >= 0.84:
        return True
    aligned_right = cv2.warpPerspective(
        right_gray,
        transform,
        (left_gray.shape[1], left_gray.shape[0]),
        borderValue=255,
    )
    overlap = cv2.warpPerspective(
        np.full(right_gray.shape, 255, dtype=np.uint8),
        transform,
        (left_gray.shape[1], left_gray.shape[0]),
        borderValue=0,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized_left = clahe.apply(left_gray)
    normalized_right = clahe.apply(aligned_right)
    difference = cv2.absdiff(normalized_left, normalized_right)[30:-30, 20:-20]
    valid_overlap = overlap[30:-30, 20:-20] > 0
    if float(np.mean(valid_overlap)) < 0.50:
        return hash_match
    # Same printed forms are common. SIFT alignment alone mostly matches the
    # template, so require the handwritten/printed residual to remain small.
    changed_fraction = float(np.mean(difference[valid_overlap] > 25))
    return changed_fraction <= 0.22


SKIN_COVER_FRACTION = 0.32
SKIN_CORE_SCALE = 0.50


def skin_fraction(bgr: np.ndarray, corners: np.ndarray | None) -> float:
    """How much of the receipt's own middle a hand is covering, by colour.

    CLAUDE CODE: the existing occlusion test looks for a strong-change blob, so
    it sees a hand arriving and never sees one already resting on the page. An
    operator who holds a receipt flat while it settles therefore had the hand
    photographed as part of the receipt. Skin separates from paper on chroma
    alone even under this warm lamp, and pink carbon forms do not read as skin.

    CLAUDE CODE: measured over the whole quad this cannot tell a palm lying on
    a receipt from the finger that holds a flap up during a flip - both put
    0.21-0.50 of the quad in skin. Over the middle half, where the print is,
    they separate cleanly: a palm reads 0.38-0.90 and a finger at the edge
    0.06-0.27. What matters is whether the hand is hiding the receipt, and that
    is a question about the middle, not the border.
    """
    if corners is None or bgr.ndim != 3:
        return 0.0
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    chroma_red, chroma_blue = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin = (
        (chroma_red >= 135)
        & (chroma_red <= 180)
        & (chroma_blue >= 85)
        & (chroma_blue <= 135)
    ).astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ordered = order_corners(corners)
    centre = ordered.mean(axis=0)
    core = np.zeros(skin.shape, np.uint8)
    cv2.fillConvexPoly(
        core, (centre + (ordered - centre) * SKIN_CORE_SCALE).astype(np.int32), 1
    )
    area = int(np.count_nonzero(core))
    if not area:
        return 0.0
    return float(np.count_nonzero(skin & core)) / area


def structural_image(gray: np.ndarray) -> np.ndarray:
    """Local contrast + edges make motion insensitive to global AE/AWB shifts."""
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    x_gradient = cv2.Sobel(normalized, cv2.CV_16S, 1, 0, ksize=3)
    y_gradient = cv2.Sobel(normalized, cv2.CV_16S, 0, 1, ksize=3)
    magnitude = cv2.addWeighted(
        cv2.convertScaleAbs(x_gradient),
        0.5,
        cv2.convertScaleAbs(y_gradient),
        0.5,
        0,
    )
    # CURSOR: include local contrast so a filled flip or finger registers as
    # motion; Sobel-only energy lives on thin edges and misses uniform cover.
    return cv2.addWeighted(normalized, 0.5, magnitude, 0.5, 0)


@dataclass(frozen=True, slots=True)
class ReceiptContentEvidence:
    edge_density: float
    spatial_coverage: float
    line_count: int

    @property
    def is_receipt(self) -> bool:
        return content_weight(self) >= 0.35


def receipt_content_evidence(image: np.ndarray) -> ReceiptContentEvidence:
    """Measure text/line structure at the receipt's own aspect ratio."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape
    scale = 480.0 / max(height, width)
    normalized = cv2.resize(
        gray,
        (max(48, round(width * scale)), max(48, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    margin_y = max(4, normalized.shape[0] // 20)
    margin_x = max(4, normalized.shape[1] // 20)
    interior = normalized[margin_y:-margin_y, margin_x:-margin_x]
    edges = cv2.Canny(interior, 60, 140)
    edge_density = float(np.mean(edges > 0))
    cell_height = max(1, interior.shape[0] // 6)
    cell_width = max(1, interior.shape[1] // 6)
    cells = [
        edges[y : y + cell_height, x : x + cell_width]
        for y in range(0, cell_height * 6, cell_height)
        for x in range(0, cell_width * 6, cell_width)
    ]
    spatial_coverage = sum(float(np.mean(cell > 0)) >= 0.004 for cell in cells) / len(
        cells
    )
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        35,
        minLineLength=max(20, interior.shape[1] // 6),
        maxLineGap=8,
    )
    return ReceiptContentEvidence(
        edge_density=edge_density,
        spatial_coverage=spatial_coverage,
        line_count=0 if lines is None else len(lines),
    )


def _ramp(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def content_weight(evidence: ReceiptContentEvidence) -> float:
    """Continuous measure of print spread across the crop, in [0, 1].

    CLAUDE CODE: coverage carries this. A raw edge count cannot: the grain of a
    dark stone table under camera noise runs at 8-13% edge pixels, several
    times over any threshold a printed receipt needs, so density and lines only
    have to clear a floor while the distribution of that structure decides.
    """
    return _content_weight(evidence, COVERAGE_FLOOR)


COVERAGE_FLOOR = 0.60
LOOSE_COVERAGE_FLOOR = 0.45


def _content_weight(evidence: ReceiptContentEvidence, coverage_floor: float) -> float:
    marked = max(
        min(evidence.edge_density / 0.004, 1.0),
        min(evidence.line_count / 6.0, 1.0),
    )
    return _ramp(evidence.spatial_coverage, coverage_floor, 0.95) * (0.3 + 0.7 * marked)


def paper_mask(blurred: np.ndarray) -> np.ndarray | None:
    """Split paper from the desk by where the two brightness classes actually sit.

    CLAUDE CODE: a fixed bright threshold cannot do this. Under the lamp this
    application is used with, paper peaks around 130-155 grey, so a constant
    written for a brighter room never fires; a constant low enough for this room
    swallows the desk elsewhere. Otsu puts the cut between the two classes
    wherever they are, and the separation check rejects the split it still
    returns for a scene holding no paper, where both classes are desk grain.
    """
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = blurred[mask > 0]
    dark = blurred[mask == 0]
    if bright.size < 64 or dark.size < 64:
        return None
    if float(np.median(bright)) - float(np.median(dark)) < 30.0:
        return None
    return cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2
    )


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.magnitude(
        cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3),
    )


def edge_relief(
    gray: np.ndarray, corners: np.ndarray, magnitude: np.ndarray | None = None
) -> float:
    """How strongly the quad's edge separates what is inside it from what is out.

    CLAUDE CODE: paper has a boundary - intensity steps across it. A pool of
    lamp light has none, so the contour found on one is an iso-brightness line
    through a smooth falloff and scores near 1 however large or bright it is.

    Two readings of that step, and the boundary only has to satisfy one. The
    gradient reading asks whether the outline is steeper than the grain inside
    it; that is the older one, and it fails on the pre-printed forms this
    application is aimed at, whose ruled grid is as steep as their own edge -
    receipt 24's flipped side scored 0.86 with the boundary plainly visible.
    The step reading compares brightness just inside the quad against just
    outside it, measured against the interior's own variation. A ruled form on
    a dark desk cannot cancel that, and a lamp pool cannot fake it.
    """
    if magnitude is None:
        magnitude = gradient_magnitude(gray)
    quad = corners.astype(np.int32)
    filled = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(filled, quad, 255)
    span = max(3, round(min(gray.shape) * 0.03) | 1)
    kernel = np.ones((span, span), np.uint8)
    eroded = cv2.erode(filled, kernel)
    dilated = cv2.dilate(filled, kernel)
    outline = dilated - eroded
    on_outline = magnitude[outline > 0]
    inside = magnitude[eroded > 0]
    if on_outline.size < 20 or inside.size < 50:
        return 0.0
    gradient_relief = float(np.percentile(on_outline, 90)) / (
        float(np.percentile(inside, 75)) + 1e-3
    )

    # CLAUDE CODE: brightness alone cannot tell a boundary from a lamp pool -
    # both are brighter inside than out. What separates them is abruptness, so
    # the crossing is measured against the change over the same distance one
    # band further in. Paper barely changes across that inner band and drops to
    # the desk across the outer one; a pool of light changes by the same amount
    # across both, because it is one smooth falloff, and scores near 1.
    deep = cv2.erode(eroded, kernel)
    inner_band = gray[(eroded - deep) > 0]
    edge_inside = gray[(filled - eroded) > 0]
    edge_outside = gray[(dilated - filled) > 0]
    if inner_band.size < 50 or edge_inside.size < 50 or edge_outside.size < 50:
        return gradient_relief
    edge_level = float(np.median(edge_inside))
    across = abs(edge_level - float(np.median(edge_outside)))
    within = abs(edge_level - float(np.median(inner_band)))
    step_relief = across / (within + 2.0)
    return max(gradient_relief, step_relief)


def document_weight(evidence: ReceiptContentEvidence, relief: float) -> float:
    """Receipt-likeness in [0, 1]; the document gate is >= 0.35.

    CLAUDE CODE: relief multiplies rather than adds because a document without
    an outline is not a document. Interior structure corroborates a boundary,
    it cannot stand in for one - on a grainy table the grain alone scores well
    enough to carry a lamp pool past any additive gate. The floor keeps
    outline-less quads ordered against each other for ranking; it is too low to
    reach the gate on its own.

    CLAUDE CODE: an abrupt outline also decides where coverage starts counting.
    A rectangle drawn round a card and the sheet behind it is two thirds paper
    and a third desk, so a floor of 0.60 scores it zero and the scene is dropped
    - the silent discard ACC-6 forbids, on evidence that otherwise reads as a
    receipt. How sharply the boundary steps is evidence about the paper that
    does not depend on how tightly the rectangle sits on it, so a strong step
    lowers the floor. It cannot lower the ceiling: loose framing still scores
    below tight framing, and `_emit_best_in_window` flags what it emits.
    """
    coverage_floor = COVERAGE_FLOOR - (COVERAGE_FLOOR - LOOSE_COVERAGE_FLOOR) * _ramp(
        relief, 6.0, 12.0
    )
    return _content_weight(evidence, coverage_floor) * (
        0.15 + 0.85 * _ramp(relief, 1.8, 3.0)
    )


def receipt_content_score(image: np.ndarray) -> float:
    """Compatibility metric for diagnostics and tests."""
    return receipt_content_evidence(image).edge_density


def active_viewport(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Exclude true-black letterboxing from portrait virtual cameras."""
    height, width = frame.shape[:2]
    sampled = frame[::8, ::8].mean(axis=2)

    def bounds(values: np.ndarray, full_size: int) -> tuple[int, int]:
        active = np.flatnonzero(
            (values.mean(axis=0) > 3.0) | (values.std(axis=0) > 3.0)
        )
        if active.size == 0:
            return 0, full_size
        start = max(0, int(active[0]) * 8 - 8)
        end = min(full_size, (int(active[-1]) + 1) * 8 + 8)
        if end - start < full_size * 0.2 or end - start > full_size * 0.92:
            return 0, full_size
        return start, end

    x0, x1 = bounds(sampled, width)
    y0, y1 = bounds(sampled.T, height)
    return x0, y0, x1 - x0, y1 - y0


@dataclass(slots=True)
class _Evaluation:
    timestamp: float
    analysis: np.ndarray
    metrics: DetectionMetrics
    frame_index: int = -1
    corners_small: np.ndarray | None = None
    content_score: float | None = None


class ReceiptDetector:
    """One stateful pipeline used by live and offline profiles."""

    def __init__(self, settings: DetectionSettings, profile: str = "realtime") -> None:
        self.settings = settings
        self.profile = profile
        self.analysis_width = (
            settings.analysis_width
            if profile == "realtime"
            else max(1400, settings.analysis_width)
        )
        self._previous_structure: np.ndarray | None = None
        self._previous_corners_small: np.ndarray | None = None
        from .background import BackgroundModel

        self._background = BackgroundModel()
        self._sharpness_history: deque[tuple[float, float]] = deque()
        self._stable_since: float | None = None
        self._last_analysis_at = -math.inf
        self._last_capture_at: float | None = None
        self._tracker = PresentationTracker(
            settings.content_change_threshold, settings.long_transition_seconds
        )
        self._recent_hashes: deque[int] = deque(maxlen=3)
        self._no_document_since: float | None = None
        self._motion_started: float | None = None
        self._last_document_at: float | None = None
        self._presentation_start: float | None = None
        self._captured_current = False
        self._pending_window: tuple[float | None, float] | None = None
        self._pending_fingerprint: int | None = None
        self._last_candidate_crop: np.ndarray | None = None
        self._analysis_buffer: deque[_Evaluation] = deque()
        self.frames = FullFrameStore(
            settings.full_resolution_seconds, settings.compressed_buffer_seconds
        )
        self._next_index = 0
        self._last_metrics = DetectionMetrics()

    def reset(self) -> None:
        log.debug("Resetting the %s detector", self.profile)
        self.__init__(self.settings, self.profile)

    def detect_document(self, frame: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Detect full-resolution corners for manual extraction and diagnostics."""
        analysis, scale, offset = self._downscale(frame)
        gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
        corners, confidence, _, _, _ = self._detect_document(
            gray, allow_bright_fallback=True
        )
        return (corners / scale + offset if corners is not None else None), confidence

    def feed(
        self, packet: FramePacket
    ) -> tuple[DetectionMetrics, CaptureCandidate | None]:
        interval = 1.0 / max(1.0, self.settings.target_analysis_fps)
        if (
            self.profile == "realtime"
            and packet.timestamp - self._last_analysis_at + 1e-6 < interval
        ):
            return self._last_metrics, None
        self._last_analysis_at = packet.timestamp

        if packet.index < 0:
            packet = FramePacket(packet.timestamp, packet.frame, index=self._next_index)
        self._next_index = packet.index + 1
        self.frames.put(packet)

        analysis, scale, offset = self._downscale(packet.frame)
        gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
        corners_small, confidence, visibility, clipping, content_score = (
            self._detect_document(gray, allow_bright_fallback=True)
        )
        corners_full = (
            corners_small / scale + offset if corners_small is not None else None
        )
        structure = structural_image(gray)
        difference = self._compensated_difference(structure)
        motion = self._motion_score(difference, corners_small)
        quad_iou = self._quad_iou(
            corners_small, self._previous_corners_small, gray.shape
        )
        occluded = self._occlusion_detected(difference, corners_small)
        skin = skin_fraction(analysis, corners_small)
        document_mask = None
        if corners_small is not None:
            document_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillConvexPoly(document_mask, corners_small.astype(np.int32), 255)
        self._background.update(gray, document_mask, motion)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if corners_small is not None:
            self._sharpness_history.append((packet.timestamp, sharpness))
        focus_settled = self._focus_settled(packet.timestamp)
        score = self._frame_score(
            sharpness, motion, confidence, visibility, clipping, corners_small
        )
        metrics = DetectionMetrics(
            timestamp=packet.timestamp,
            motion=motion,
            sharpness=sharpness,
            boundary_confidence=confidence,
            visibility=visibility,
            clipping=clipping,
            score=score,
            quad_iou=quad_iou,
            occluded=occluded,
            skin=skin,
            focus_settled=focus_settled,
            corners=corners_full,
        )
        evaluation = _Evaluation(
            packet.timestamp,
            gray.copy(),
            metrics,
            frame_index=packet.index,
            corners_small=corners_small,
            content_score=content_score,
        )
        candidate = self._advance(evaluation, packet)
        self._previous_structure = structure
        self._previous_corners_small = (
            corners_small if corners_small is not None else None
        )
        self._last_metrics = metrics
        return metrics, candidate

    @staticmethod
    def _active_viewport(frame: np.ndarray) -> tuple[int, int, int, int]:
        return active_viewport(frame)

    def _downscale(self, frame: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        x, y, view_width, view_height = self._active_viewport(frame)
        view = frame[y : y + view_height, x : x + view_width]
        width = view.shape[1]
        if width <= self.analysis_width:
            return view, 1.0, np.array([x, y], dtype=np.float32)
        scale = self.analysis_width / width
        return (
            cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
            scale,
            np.array([x, y], dtype=np.float32),
        )

    def _detect_document(
        self,
        gray: np.ndarray,
        allow_bright_fallback: bool = False,
    ) -> tuple[np.ndarray | None, float, float, float, float | None]:
        """Return the winning quad plus the content weight it was ranked with.

        CURSOR: the content weight is None when the returned quad never went
        through fused ranking (bright fallback), so callers must score it.
        """
        height, width = gray.shape
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median = float(np.median(blurred))
        edges = cv2.Canny(
            blurred, int(max(20, median * 0.55)), int(min(240, median * 1.35))
        )
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
        )
        if self.settings.exclusion_rect:
            x, y, w, h = self.settings.exclusion_rect
            edges[
                int(y * height) : int((y + h) * height),
                int(x * width) : int((x + w) * width),
            ] = 0
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(width * height)
        candidates: list[tuple[float, np.ndarray, float, float]] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
            area = float(cv2.contourArea(contour))
            area_ratio = area / frame_area
            if area_ratio < self.settings.min_document_area or area_ratio > 0.97:
                continue
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approximation) != 4 or not cv2.isContourConvex(approximation):
                rectangle = cv2.boxPoints(cv2.minAreaRect(contour))
                rectangularity = area / max(
                    1.0, cv2.contourArea(rectangle.astype(np.float32))
                )
                if rectangularity < 0.65:
                    continue
                points = rectangle
                corner_quality = 0.55
            else:
                points = approximation.reshape(4, 2).astype(np.float32)
                corner_quality = 1.0
            ordered = order_corners(points)
            distances = np.minimum.reduce(
                [
                    ordered[:, 0],
                    ordered[:, 1],
                    width - 1 - ordered[:, 0],
                    height - 1 - ordered[:, 1],
                ]
            )
            clipping = float(np.mean(distances < max(5, min(width, height) * 0.012)))
            confidence = min(
                1.0, 0.35 + area_ratio * 0.5 + corner_quality * 0.25 - clipping * 0.25
            )
            rank = confidence * (0.08 + min(area_ratio, 0.30))
            candidates.append((rank, ordered, confidence, clipping))
        mask = self._paper_mask(gray)
        candidates.extend(self._paper_quads(gray, mask))
        bright = self._detect_bright_document(gray) if allow_bright_fallback else None
        foreground = self._background.foreground_quad(
            gray, self.settings.min_document_area, self.settings.exclusion_rect
        )
        if foreground is not None:
            fg_points, fg_confidence, fg_visibility, fg_clipping = foreground
            candidates.append(
                (
                    fg_confidence * (0.08 + min(fg_visibility, 0.30)),
                    fg_points,
                    fg_confidence,
                    fg_clipping,
                )
            )
        if not candidates:
            return (
                (*bright, None) if bright is not None else (None, 0.0, 0.0, 0.0, None)
            )
        # CLAUDE CODE: every quad is re-ranked by how much of it is paper, not
        # only the ones the brightness split produced. A rectangle drawn round a
        # receipt lying at an angle, or round a receipt and its neighbour, is
        # mostly desk, and the edge that matters then lies in the dark where
        # there is no boundary to measure. Preferring the tighter quad fixes the
        # crop and the boundary reading together.
        candidates = [
            (rank * self._paper_fill_weight(points, mask), points, confidence, clipping)
            for rank, points, confidence, clipping in candidates
        ]
        candidates.sort(key=lambda item: item[0], reverse=True)
        best: tuple[float, np.ndarray, float, float, float] | None = None
        magnitude = gradient_magnitude(gray)
        for rank, points, confidence, clipping in candidates[:5]:
            # CURSOR: the fused score never exceeds the rank and the list is
            # rank-ordered, so once no remaining candidate can win, the content
            # evaluation - the costliest step per frame - is skipped (DET-2).
            if best is not None and rank <= best[0]:
                break
            crop = perspective_crop(gray, points)
            content = document_weight(
                receipt_content_evidence(crop), edge_relief(gray, points, magnitude)
            )
            fused = rank * (0.3 + 0.7 * content)
            if best is None or fused > best[0]:
                best = (fused, points, confidence, clipping, content)
        _, points, confidence, clipping, content = best
        visibility = float(cv2.contourArea(points) / frame_area)
        if bright is not None:
            bright_points, bright_confidence, bright_visibility, bright_clipping = (
                bright
            )
            bright_content = document_weight(
                receipt_content_evidence(perspective_crop(gray, bright_points)),
                edge_relief(gray, bright_points, magnitude),
            )
            if bright_content >= 0.35 and bright_content > content:
                return (
                    bright_points,
                    bright_confidence,
                    bright_visibility,
                    bright_clipping,
                    bright_content,
                )
            if bright_visibility > visibility * 1.55:
                return (
                    bright_points,
                    bright_confidence,
                    bright_visibility,
                    bright_clipping,
                    None,
                )
        return points, confidence, visibility, clipping, content

    def _paper_mask(self, gray: np.ndarray) -> np.ndarray | None:
        """The paper-versus-desk split, blanked inside the excluded region."""
        mask = paper_mask(cv2.GaussianBlur(gray, (9, 9), 0))
        if mask is None or not self.settings.exclusion_rect:
            return mask
        height, width = gray.shape
        x, y, w, h = self.settings.exclusion_rect
        mask[
            int(y * height) : int((y + h) * height),
            int(x * width) : int((x + w) * width),
        ] = 0
        return mask

    @staticmethod
    def _paper_fill_weight(points: np.ndarray, mask: np.ndarray | None) -> float:
        """Rank multiplier in [0.35, 1.0] for the paper fraction of a quad."""
        if mask is None:
            return 1.0
        covered = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(covered, points.astype(np.int32), 255)
        quad_pixels = int(np.count_nonzero(covered))
        if not quad_pixels:
            return 1.0
        fill = float(np.count_nonzero(covered & mask)) / quad_pixels
        return 0.35 + 0.65 * _ramp(fill, 0.55, 0.90)

    def _paper_quads(
        self, gray: np.ndarray, mask: np.ndarray | None
    ) -> list[tuple[float, np.ndarray, float, float]]:
        """Rank paper-versus-desk regions alongside the edge-derived quads.

        CLAUDE CODE: Canny outlines a receipt only where paper meets desk. A
        hand along one side joins the two into a single contour whose rectangle
        fills barely 40% of its own box, so it never survives the rectangularity
        gate and the receipt reaches the content ranking not at all. A
        brightness split keeps the region solid. These quads are ranked with the
        other candidates rather than overriding them, so a desk blob still loses
        on content.
        """
        if mask is None:
            return []
        height, width = gray.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(width * height)
        quads: list[tuple[float, np.ndarray, float, float]] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            if float(cv2.contourArea(contour)) / frame_area < (
                self.settings.min_document_area
            ):
                continue
            points = order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))
            quads.extend(self._rank_quad(points, frame_area, width, height))
        return quads

    def _rank_quad(
        self, points: np.ndarray, frame_area: float, width: int, height: int
    ) -> list[tuple[float, np.ndarray, float, float]]:
        area_ratio = float(cv2.contourArea(points)) / frame_area
        if area_ratio < self.settings.min_document_area or area_ratio > 0.90:
            return []
        distances = np.minimum.reduce(
            [
                points[:, 0],
                points[:, 1],
                width - 1 - points[:, 0],
                height - 1 - points[:, 1],
            ]
        )
        clipping = float(np.mean(distances < max(5, min(width, height) * 0.012)))
        confidence = min(1.0, 0.35 + area_ratio * 0.5 + 0.55 * 0.25 - clipping * 0.25)
        rank = confidence * (0.08 + min(area_ratio, 0.30))
        return [(rank, points, confidence, clipping)]

    def _detect_bright_document(
        self, gray: np.ndarray
    ) -> tuple[np.ndarray, float, float, float] | None:
        """Recover a faint outer paper edge when an internal box wins Canny."""
        height, width = gray.shape
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        threshold_value = int(np.clip(np.percentile(blurred, 75), 180, 200))
        _, mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2
        )
        if self.settings.exclusion_rect:
            x, y, w, h = self.settings.exclusion_rect
            mask[
                int(y * height) : int((y + h) * height),
                int(x * width) : int((x + w) * width),
            ] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(width * height)
        best: tuple[float, np.ndarray, float, float, float] | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / frame_area
            if area_ratio < self.settings.min_document_area or area_ratio > 0.65:
                continue
            points = order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))
            rectangle_area = max(1.0, float(cv2.contourArea(points)))
            rectangularity = area / rectangle_area
            if rectangularity < 0.68:
                continue
            distances = np.minimum.reduce(
                [
                    points[:, 0],
                    points[:, 1],
                    width - 1 - points[:, 0],
                    height - 1 - points[:, 1],
                ]
            )
            clipping = float(np.mean(distances < max(5, min(width, height) * 0.012)))
            rank = area_ratio * rectangularity
            confidence = min(
                0.78,
                0.48
                + min(0.12, area_ratio * 0.35)
                + rectangularity * 0.18
                - clipping * 0.20,
            )
            if best is None or rank > best[0]:
                best = (rank, points, confidence, area_ratio, clipping)
        if best is None:
            return None
        _, points, confidence, visibility, clipping = best
        return points, confidence, visibility, clipping

    def _compensated_difference(self, structure: np.ndarray) -> np.ndarray | None:
        """Structural diff after removing rigid shift; handheld shake is not motion."""
        previous = self._previous_structure
        if previous is None or previous.shape != structure.shape:
            return None
        small = (256, 192)
        current_small = cv2.resize(structure, small).astype(np.float32)
        previous_small = cv2.resize(previous, small).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(previous_small, current_small)
        shift_x = round(dx * structure.shape[1] / small[0])
        shift_y = round(dy * structure.shape[0] / small[1])
        if abs(shift_x) > 40 or abs(shift_y) > 40:
            shift_x = shift_y = 0
        aligned = previous
        if shift_x or shift_y:
            translation = np.array(
                [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32
            )
            aligned = cv2.warpAffine(
                previous,
                translation,
                (previous.shape[1], previous.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE,
            )
        difference = cv2.absdiff(structure, aligned)
        # CURSOR: the strip the shift uncovers has no counterpart in the
        # previous frame, so scoring it would read a camera pan as receipt
        # motion or as an occluding blob near the frame border.
        if shift_x > 0:
            difference[:, :shift_x] = 0
        elif shift_x < 0:
            difference[:, shift_x:] = 0
        if shift_y > 0:
            difference[:shift_y, :] = 0
        elif shift_y < 0:
            difference[shift_y:, :] = 0
        return difference

    @staticmethod
    def _quad_iou(
        a: np.ndarray | None, b: np.ndarray | None, shape: tuple[int, int]
    ) -> float:
        if a is None and b is None:
            return 1.0
        if a is None or b is None:
            return 0.0
        mask_a = np.zeros(shape, dtype=np.uint8)
        mask_b = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask_a, a.astype(np.int32), 1)
        cv2.fillConvexPoly(mask_b, b.astype(np.int32), 1)
        union = int(np.count_nonzero(mask_a | mask_b))
        if union == 0:
            return 0.0
        return float(np.count_nonzero(mask_a & mask_b)) / union

    def _occlusion_detected(
        self, difference: np.ndarray | None, corners_small: np.ndarray | None
    ) -> bool:
        """A compact strong-change blob inside the quad means fingers or a page edge."""
        if difference is None or corners_small is None:
            return False
        mask = np.zeros(difference.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, corners_small.astype(np.int32), 255)
        quad_area = int(np.count_nonzero(mask))
        if quad_area == 0:
            return False
        strong = ((difference > 40) & (mask > 0)).astype(np.uint8)
        strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(strong)
        return any(
            stats[label, cv2.CC_STAT_AREA] > 0.02 * quad_area
            for label in range(1, count)
        )

    def _motion_score(
        self, difference: np.ndarray | None, corners: np.ndarray | None
    ) -> float:
        if difference is None:
            return 1.0
        mask = np.zeros(difference.shape, dtype=np.uint8)
        if corners is not None:
            cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
        else:
            margin_y, margin_x = difference.shape[0] // 10, difference.shape[1] // 10
            mask[margin_y : -margin_y or None, margin_x : -margin_x or None] = 255
        if self.settings.exclusion_rect:
            x, y, w, h = self.settings.exclusion_rect
            height, width = mask.shape
            mask[
                int(y * height) : int((y + h) * height),
                int(x * width) : int((x + w) * width),
            ] = 0
        pixels = difference[mask > 0]
        if pixels.size == 0:
            return 1.0
        return float(np.mean(pixels > 24))

    @staticmethod
    def _frame_score(
        sharpness: float,
        motion: float,
        confidence: float,
        visibility: float,
        clipping: float,
        corners: np.ndarray | None,
    ) -> float:
        sharp = min(1.0, math.log1p(max(0.0, sharpness)) / math.log(1001.0))
        stillness = max(0.0, 1.0 - motion * 8.0)
        perspective = 0.5
        if corners is not None:
            tl, tr, br, bl = order_corners(corners)
            horizontal = min(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)) / max(
                1.0, max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
            )
            vertical = min(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)) / max(
                1.0, max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
            )
            perspective = float((horizontal + vertical) / 2)
        return (
            sharp * 0.30
            + stillness * 0.22
            + confidence * 0.18
            + min(1.0, visibility / 0.55) * 0.10
            + perspective * 0.08
            # Completeness matters more than a sharp partial page. This term
            # makes a later full view replace an early clipped one in-buffer.
            + (1.0 - clipping) * 0.12
        )

    def _focus_settled(self, now: float) -> bool:
        while self._sharpness_history and self._sharpness_history[0][0] < now - 0.6:
            self._sharpness_history.popleft()
        if len(self._sharpness_history) < 4:
            return True
        values = [value for _, value in self._sharpness_history]
        top = max(values)
        if top <= 0:
            return True
        return (top - min(values)) / top <= 0.5

    def flush(self) -> CaptureCandidate | None:
        """Emit a still-window candidate that was waiting when capture ended."""
        # CURSOR: only an open, still-uncaptured presentation may be flushed; a
        # presentation that already ended was committed at its own boundary.
        if self._captured_current:
            return None
        if (
            self._last_document_at is None
            or self._last_analysis_at - self._last_document_at > 0.6
        ):
            return None
        candidate = self._emit_best_in_window(force=True)
        if candidate is not None:
            self._tracker.confirm_capture(candidate.content_hash)
            self._captured_current = True
        return candidate

    def flush_all(self) -> list[CaptureCandidate]:
        """Commit any valid still-window candidate still held at end-of-stream."""
        candidate = self.flush()
        if candidate is not None:
            log.info("End-of-stream flush released a held receipt")
        return [candidate] if candidate is not None else []

    def _full_resolution_sharpness(
        self, evaluation: _Evaluation
    ) -> tuple[FramePacket | None, float]:
        packet = self.frames.get(evaluation.frame_index)
        if packet is None or evaluation.metrics.corners is None:
            return None, -1.0
        crop = perspective_crop(packet.frame, evaluation.metrics.corners)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return packet, float(np.mean(gx * gx + gy * gy))

    def _emit_best_in_window(
        self,
        force: bool = False,
        window_start: float | None = None,
        window_end: float | None = None,
    ) -> CaptureCandidate | None:
        """Pick the best buffered frame of one presentation window.

        CURSOR: the window defaults to the current still interval; committing a
        presentation that already ended passes its own start and end so frames
        of the receipt now on the desk cannot be emitted in its place.
        """
        floor = (
            self._last_capture_at if self._last_capture_at is not None else -math.inf
        )
        still_start = self._stable_since if window_start is None else window_start
        end = math.inf if window_end is None else window_end
        quality_flag: str | None = None
        eligible = [
            item
            for item in self._analysis_buffer
            if still_start is not None
            and still_start <= item.timestamp <= end
            and item.timestamp > floor
            and item.metrics.corners is not None
            and item.metrics.motion <= self.settings.motion_threshold
            and not item.metrics.occluded
            and item.metrics.skin < SKIN_COVER_FRACTION
            and item.metrics.focus_settled
        ]
        if not eligible:
            if not force:
                return None
            # CURSOR: never drop a presentation silently (ACC-6); emit the
            # least-moving frame flagged for review instead.
            fallback = [
                item
                for item in self._analysis_buffer
                if item.timestamp > floor
                and (window_start is None or item.timestamp >= window_start)
                and item.timestamp <= end
                and item.metrics.corners is not None
                and item.metrics.content_score >= 0.35
            ]
            if not fallback:
                return None
            # CLAUDE CODE: this path is why hands survived the settle check -
            # it is the one that must emit something, so it took the steadiest
            # frame whatever was lying on the page. A hand holds very still, so
            # "steadiest" chose it. Prefer any uncovered frame of the same
            # window; only when the receipt was never seen bare does a covered
            # one go through, and then it says so.
            bare = [
                item for item in fallback if item.metrics.skin < SKIN_COVER_FRACTION
            ]
            if bare:
                eligible = [min(bare, key=lambda item: item.metrics.motion)]
                quality_flag = "motion-blur"
            else:
                eligible = [min(fallback, key=lambda item: item.metrics.skin)]
                quality_flag = "hand-covered"
        ranked = sorted(eligible, key=lambda item: item.metrics.score, reverse=True)
        best: _Evaluation | None = None
        best_packet: FramePacket | None = None
        best_sharpness = -1.0
        for evaluation in ranked[:3]:
            packet, sharpness = self._full_resolution_sharpness(evaluation)
            if packet is not None and sharpness > best_sharpness:
                best, best_packet, best_sharpness = evaluation, packet, sharpness
        if best is None or best_packet is None:
            return None
        chosen_crop = (
            perspective_crop(best.analysis, best.corners_small)
            if best.corners_small is not None
            else best.analysis
        )
        content_hash = difference_hash(chosen_crop)
        if (
            quality_flag is None
            and receipt_content_evidence(chosen_crop).spatial_coverage < COVERAGE_FLOOR
        ):
            # CLAUDE CODE: the quad reaches past the paper - two sheets with
            # desk between them, or a hand alongside. document_weight lets an
            # abrupt outline carry this to capture rather than drop it, so the
            # cost has to land here, where review can see it.
            quality_flag = "loose-crop"
        candidate = CaptureCandidate(
            timestamp=best_packet.timestamp,
            frame=best_packet.frame,
            corners=np.asarray(best.metrics.corners).copy(),
            boundary_confidence=best.metrics.boundary_confidence,
            score=best.metrics.score,
            content_hash=content_hash,
            quality_flag=quality_flag,
        )
        self._last_candidate_crop = (
            perspective_crop(best.analysis, best.corners_small)
            if best.corners_small is not None
            else best.analysis.copy()
        )
        self._last_capture_at = best.timestamp
        log.debug(
            "Capture candidate at %.3f, score %.3f, boundary %.3f%s",
            candidate.timestamp,
            candidate.score,
            candidate.boundary_confidence,
            f", flagged {quality_flag}" if quality_flag else "",
        )
        return candidate

    def _advance(
        self, evaluation: _Evaluation, current_packet: FramePacket
    ) -> CaptureCandidate | None:
        metrics = evaluation.metrics
        now = evaluation.timestamp
        self._analysis_buffer.append(evaluation)
        self._trim(now)

        crop = None
        content = 0.0
        if evaluation.corners_small is not None:
            # CURSOR: content gate and fingerprint both run at analysis
            # scale; full resolution is touched only for the selected frame.
            crop = perspective_crop(evaluation.analysis, evaluation.corners_small)
            content = (
                evaluation.content_score
                if evaluation.content_score is not None
                else document_weight(
                    receipt_content_evidence(crop),
                    edge_relief(evaluation.analysis, evaluation.corners_small),
                )
            )
        metrics.content_score = content

        if evaluation.corners_small is None or content < 0.35:
            if self._has_paper_presence(evaluation.analysis):
                self._note_transition(now)
                metrics.state = DetectorState.SETTLING
                metrics.message = "Receipt visible; waiting for a complete boundary"
                return None
            if metrics.motion > min(0.05, self.settings.motion_threshold):
                # CLAUDE CODE: a hand or folded page can hide every usable
                # boundary while the placement is still being adjusted. Only
                # a quiet empty scene is evidence that the receipt was removed.
                self._no_document_since = None
                self._note_transition(now)
                metrics.state = DetectorState.MOTION
                metrics.message = "Waiting for the empty scene to settle"
                return None
            metrics.state = DetectorState.NO_DOCUMENT
            if self._no_document_since is None:
                self._no_document_since = now
            # CURSOR: _last_document_at doubles as "a presentation is still
            # open"; it is cleared below once this one has been closed out.
            if now - self._no_document_since >= 0.25 and (
                self._last_document_at is not None
            ):
                committed = self._commit_on_absence()
                self._tracker.note_document_lost()
                self._stable_since = None
                self._motion_started = None
                self._presentation_start = None
                self._last_document_at = None
                self._captured_current = False
                self._last_candidate_crop = None
                self._recent_hashes.clear()
                self._sharpness_history.clear()
                if committed is not None:
                    metrics.state = DetectorState.CAPTURED
                    metrics.message = "Receipt left the frame; best frame kept"
                return committed
            return None

        self._no_document_since = None
        self._last_document_at = now

        # CURSOR: skip IoU-as-motion on the first framed quad; there is no
        # previous geometry to compare against.
        moving = (
            metrics.motion > self.settings.motion_threshold
            or (
                metrics.quad_iou < self.settings.quad_stability_iou
                and self._previous_corners_small is not None
            )
            or metrics.occluded
        )
        if moving:
            self._note_transition(now)
            metrics.state = DetectorState.MOTION
            metrics.message = "Waiting for receipt motion to settle"
            return None

        # CURSOR: unlike _stable_since this survives a brief wobble, so it
        # marks where the presentation began rather than the current still run.
        if self._presentation_start is None:
            self._presentation_start = now

        if metrics.skin >= SKIN_COVER_FRACTION:
            # CLAUDE CODE: an operator holding a receipt flat is not presenting
            # it yet, so this frame starts no still window and contributes no
            # fingerprint. Both matter: the settle timer then runs from when the
            # hand leaves rather than expiring under it, and a stretch that was
            # never seen bare leaves `_recent_hashes` empty, so
            # `_close_presentation` does not remember it as a page of its own -
            # it is the same page that settles bare a moment later, not a
            # second receipt. A receipt only ever seen covered still reaches
            # review: `_commit_on_absence` works from `_presentation_start`,
            # which is set above, and flags what it emits (ACC-6).
            self._stable_since = None
            metrics.state = DetectorState.SETTLING
            metrics.message = "Hand covering the receipt"
            return None

        if self._stable_since is None:
            self._stable_since = now
        self._recent_hashes.append(difference_hash(crop))

        stable_for = now - self._stable_since
        if stable_for < self.settings.stable_seconds:
            metrics.state = DetectorState.SETTLING
            metrics.message = f"Stable for {stable_for:.1f}s"
            return None
        # CURSOR: the transition only counts as over once the page has held
        # still for a full stable window. Clearing it on the first quiet frame
        # let a hesitation mid-flip split one long flip into two short ones, so
        # an identical-looking next page never reached long_transition_seconds.
        self._motion_started = None

        fingerprint = majority_hash(list(self._recent_hashes))
        metrics.content_hash = fingerprint
        if self._pending_window is not None:
            committed = self._close_pending_presentation(fingerprint)
            if committed is not None:
                metrics.state = DetectorState.CAPTURED
                metrics.message = "Earlier page emitted for review"
                return committed

        if not metrics.focus_settled:
            metrics.state = DetectorState.SETTLING
            metrics.message = "Autofocus settling"
            return None

        decision = self._tracker.evaluate(fingerprint)
        if not decision.is_new_presentation:
            metrics.state = DetectorState.CAPTURED
            metrics.message = "Continuous presentation already captured"
            return None
        if (
            self._last_candidate_crop is not None
            and self._last_capture_at is not None
            and now - self._last_capture_at > 2.0
            and decision.changed_bits is not None
            and decision.changed_bits > self.settings.content_change_threshold
            and likely_same_receipt(self._last_candidate_crop, crop)
        ):
            # CLAUDE CODE: one placement can have several stable plateaus while
            # a fold is opened or a hand is removed. Refreshing the fingerprint
            # prevents that geometry change from creating another saved image.
            self._tracker.confirm_capture(fingerprint)
            self._captured_current = True
            metrics.state = DetectorState.CAPTURED
            metrics.message = "Continuous presentation already captured"
            return None
        self._captured_current = False
        if (
            self._last_capture_at is not None
            and now - self._last_capture_at < self.settings.minimum_capture_interval
        ):
            metrics.state = DetectorState.CAPTURED
            metrics.message = "Continuous presentation already captured"
            return None

        candidate = self._emit_best_in_window()
        if candidate is None:
            metrics.state = DetectorState.SETTLING
            metrics.message = f"Stable for {stable_for:.1f}s"
            return None
        self._tracker.confirm_capture(fingerprint)
        self._captured_current = True
        metrics.state = DetectorState.CAPTURED
        metrics.message = f"Best-frame score {candidate.score:.2f}"
        return candidate

    def _note_transition(self, now: float) -> None:
        """Record that the scene is between presentations rather than settled.

        CLAUDE CODE: every branch that ends a still window has to agree on this,
        and one of them did not. A page flipped off the one beneath it turns its
        blank back to the lens, which drops the quad and sends those frames down
        the no-boundary branch; that branch left `_stable_since` pointing before
        the flip, so the page underneath was judged to have been still the whole
        time and the tracker never saw a transition to make it new.
        """
        # CURSOR: sharpness is only comparable within one still window; a
        # scene change across motion is not the lens hunting for focus.
        self._sharpness_history.clear()
        if self._motion_started is None:
            self._close_presentation(now)
            self._motion_started = now
        self._tracker.note_transition(now - self._motion_started)
        self._stable_since = None
        self._recent_hashes.clear()

    def _close_presentation(self, now: float) -> None:
        """Remember a presentation that a transition ends before any capture.

        ACC-6 forbids losing a distinct receipt, so its window is kept until
        either the same page settles again or a different page proves it gone.
        Only a presentation with at least one still frame is remembered: without
        a fingerprint there is nothing to tell those two outcomes apart.
        """
        if not self._captured_current and self._recent_hashes:
            start = (
                self._pending_window[0]
                if self._pending_window is not None
                else self._presentation_start
            )
            self._pending_window = (start, now)
            self._pending_fingerprint = majority_hash(list(self._recent_hashes))
        self._presentation_start = None

    def _close_pending_presentation(self, fingerprint: int) -> CaptureCandidate | None:
        """Emit the remembered presentation once a different page has settled."""
        window = self._pending_window
        pending_fingerprint = self._pending_fingerprint
        self._pending_window = None
        self._pending_fingerprint = None
        if window is None or pending_fingerprint is None:
            return None
        distinct = (
            self._tracker.long_transition_seen
            or hamming_distance(fingerprint, pending_fingerprint)
            > self.settings.content_change_threshold
        )
        if not distinct:
            return None
        return self._emit_best_in_window(
            force=True, window_start=window[0], window_end=window[1]
        )

    def _commit_on_absence(self) -> CaptureCandidate | None:
        """Emit whatever the receipt that just left the frame never produced."""
        pending = self._pending_window
        self._pending_window = None
        self._pending_fingerprint = None
        if pending is not None:
            start = pending[0]
        elif self._captured_current:
            return None
        else:
            start = self._presentation_start
        if self._last_document_at is None:
            return None
        return self._emit_best_in_window(
            force=True, window_start=start, window_end=self._last_document_at
        )

    def _trim(self, now: float) -> None:
        analysis_cutoff = now - self.settings.buffer_seconds
        while (
            self._analysis_buffer
            and self._analysis_buffer[0].timestamp < analysis_cutoff
        ):
            self._analysis_buffer.popleft()

    def _has_paper_presence(self, gray: np.ndarray) -> bool:
        """Keep a presentation alive while its outer contour is briefly obscured."""
        bright = self._detect_bright_document(gray)
        if bright is None:
            return False
        points, _confidence, _visibility, _clipping = bright
        evidence = receipt_content_evidence(perspective_crop(gray, points))
        return evidence.spatial_coverage >= 0.50 and edge_relief(gray, points) >= 3.0

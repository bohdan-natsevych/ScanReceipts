from __future__ import annotations

import cv2
import numpy as np

from .detection import order_corners


class BackgroundModel:
    """Slow running average of the empty desk; a document is its foreground."""

    def __init__(self, alpha: float = 0.02) -> None:
        self.alpha = alpha
        self._background: np.ndarray | None = None

    def reset(self) -> None:
        self._background = None

    def update(
        self,
        gray: np.ndarray,
        document_mask: np.ndarray | None,
        motion: float,
    ) -> None:
        if self._background is None or self._background.shape != gray.shape:
            self._background = gray.astype(np.float32)
            return
        if motion > 0.10:
            return
        update_mask = np.full(gray.shape, 255, dtype=np.uint8)
        if document_mask is not None:
            update_mask[document_mask > 0] = 0
        cv2.accumulateWeighted(
            gray.astype(np.float32), self._background, self.alpha, mask=update_mask
        )

    def foreground_quad(
        self,
        gray: np.ndarray,
        min_area_ratio: float,
        exclusion_rect: tuple[float, float, float, float] | None = None,
    ) -> tuple[np.ndarray, float, float, float] | None:
        if self._background is None or self._background.shape != gray.shape:
            return None
        height, width = gray.shape
        background = self._background.astype(np.uint8)
        difference = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(difference, 14, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2
        )
        # CURSOR: the excluded region is blanked here for the same reason the
        # edge and bright paths blank it - a quad found there is not a receipt.
        if exclusion_rect:
            x, y, w, h = exclusion_rect
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
            if area_ratio < min_area_ratio or area_ratio > 0.90:
                continue
            points = order_corners(cv2.boxPoints(cv2.minAreaRect(contour)))
            rectangle_area = max(1.0, float(cv2.contourArea(points)))
            rectangularity = area / rectangle_area
            if rectangularity < 0.60:
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
            confidence = min(0.72, 0.42 + rectangularity * 0.25 - clipping * 0.20)
            rank = area_ratio * rectangularity
            if best is None or rank > best[0]:
                best = (rank, points, confidence, area_ratio, clipping)
        if best is None:
            return None
        _, points, confidence, visibility, clipping = best
        return points, confidence, visibility, clipping

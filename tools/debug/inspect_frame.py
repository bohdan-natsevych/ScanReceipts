"""Explain the detector's decision on individual frames of a recording.

Prints the winning quad's content evidence term by term - which one starves the
document gate is usually the whole answer - and writes the crop it scored.

    python tools/debug/inspect_frame.py 9d9b84d2 44.5 45.0 --out C:/tmp/flip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from session_replay import build_detector, find_session, iter_frames

from scan_receipts.detection import (
    active_viewport,
    content_weight,
    document_weight,
    edge_relief,
    gradient_magnitude,
    perspective_crop,
    receipt_content_evidence,
)
from scan_receipts.models import FramePacket


def report_candidates(detector, gray, winner, offset, directory) -> None:
    """Print every quad `_detect_document` ranked, in the order it ranked them.

    The winner alone cannot answer "why this crop?" - the question is always what
    it beat, and whether the quad you wanted was in the list at all.
    """
    from scan_receipts.detection import ReceiptDetector

    seen: list = []
    original = ReceiptDetector._paper_fill_weight

    def spy(points, mask):
        weight = original(points, mask)
        seen.append((np.asarray(points, np.float32).copy(), weight, mask))
        return weight

    ReceiptDetector._paper_fill_weight = staticmethod(spy)
    try:
        detector._detect_document(gray, allow_bright_fallback=True)
    finally:
        ReceiptDetector._paper_fill_weight = staticmethod(original)

    magnitude = gradient_magnitude(gray)
    frame_area = float(gray.shape[0] * gray.shape[1])
    print(f"    {len(seen)} candidates ranked:")
    rows = []
    for points, fill_weight, mask in seen:
        crop = perspective_crop(gray, points)
        evidence = receipt_content_evidence(crop)
        relief = edge_relief(gray, points, magnitude)
        rows.append(
            {
                "area": float(cv2.contourArea(points)) / frame_area,
                "fill": fill_weight,
                "document": document_weight(evidence, relief),
                "relief": relief,
                "coverage": evidence.spatial_coverage,
                "outside": outside_paper_fraction(points, mask),
                "points": points,
                "won": float(np.abs(np.sort(points, 0) - np.sort(winner, 0)).max())
                < 2.0,
            }
        )
    for index, row in enumerate(sorted(rows, key=lambda item: -item["area"])):
        print(
            f"      {'*' if row['won'] else ' '} area={row['area']:.3f}"
            f" fill_w={row['fill']:.2f} coverage={row['coverage']:.2f}"
            f" relief={row['relief']:5.2f} doc={row['document']:.3f}"
            f" outside_paper={row['outside']:.2f}"
        )
        if directory is not None:
            cv2.imwrite(
                str(directory / f"{offset:07.2f}_cand{index:02d}.png"),
                perspective_crop(gray, row["points"]),
            )


def outside_paper_fraction(points: np.ndarray, mask) -> float:
    """How much of a thin band just outside the quad is also paper.

    A real document edge has desk outside it. A rule printed across a form has
    more of the same sheet, so this reads near 1 for a quad carved out of a
    larger piece of paper.
    """
    if mask is None:
        return float("nan")
    inner = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(inner, points.astype(np.int32), 255)
    grown = cv2.dilate(inner, np.ones((13, 13), np.uint8))
    band = grown - inner
    total = int(np.count_nonzero(band))
    if not total:
        return float("nan")
    return float(np.count_nonzero(band & mask)) / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("offsets", nargs="+", type=float)
    parser.add_argument("--out", help="directory to write the scored crops into")
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="list every ranked quad, not only the winner",
    )
    args = parser.parse_args()

    wanted = sorted(args.offsets)
    directory = Path(args.out) if args.out else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)

    # CLAUDE CODE: fed from the start so the background model and the plateau
    # state match what the detector holds at the requested moment.
    detector = build_detector()
    for frame in iter_frames(find_session(args.session)):
        metrics, _ = detector.feed(
            FramePacket(frame.timestamp, frame.image, index=frame.index)
        )
        if not wanted or frame.offset < wanted[0]:
            continue
        wanted.pop(0)
        x, y, view_width, view_height = active_viewport(frame.image)
        view = frame.image[y : y + view_height, x : x + view_width]
        scale = 800 / view.shape[1]
        gray = cv2.cvtColor(
            cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(
            f"--- {frame.offset:.2f} {metrics.state.value} sharpness={sharpness:.0f}"
            f" p99_grey={np.percentile(gray, 99):.0f}"
        )
        if metrics.corners is None:
            print("    no quad")
            continue
        quad = (metrics.corners - np.array([x, y], np.float32)) * scale
        crop = perspective_crop(gray, quad)
        evidence = receipt_content_evidence(crop)
        relief = edge_relief(gray, quad, gradient_magnitude(gray))
        print(
            f"    vis={metrics.visibility:.3f} conf={metrics.boundary_confidence:.2f}"
            f" edge_density={evidence.edge_density:.4f} coverage={evidence.spatial_coverage:.2f}"
            f" lines={evidence.line_count} relief={relief:.2f}"
            f" content_weight={content_weight(evidence):.3f}"
            f" document_weight={document_weight(evidence, relief):.3f} (gate 0.35)"
        )
        if directory is not None:
            cv2.imwrite(str(directory / f"{frame.offset:07.2f}.png"), crop)
        if args.candidates:
            report_candidates(detector, gray, quad, frame.offset, directory)
        if not wanted:
            break


if __name__ == "__main__":
    main()

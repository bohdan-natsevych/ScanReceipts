"""Save each candidate a replay produces, so captures can be judged by eye.

python tools/debug/dump_candidates.py 92495b7e out_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from session_replay import find_session, replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("output")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sheet", help="also write a single contact sheet here")
    args = parser.parse_args()

    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    started_at, candidates = replay(find_session(args.session), args.stride)

    cells = []
    for position, candidate in enumerate(candidates, start=1):
        offset = candidate.timestamp - started_at
        image = candidate.frame
        if candidate.corners is not None:
            quad = candidate.corners.astype(np.int32)
            image = image.copy()
            cv2.polylines(image, [quad], True, (0, 255, 255), 3)
        cv2.imwrite(str(directory / f"{position:02d}_{offset:06.2f}.jpg"), image)
        scale = 300 / max(image.shape[:2])
        cell = cv2.resize(image, None, fx=scale, fy=scale)
        canvas = np.zeros((300, 300, 3), np.uint8)
        canvas[: cell.shape[0], : cell.shape[1]] = cell
        cv2.putText(
            canvas,
            f"{offset:.1f}",
            (5, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cells.append(canvas)

    print(f"{len(cells)} candidates -> {directory}")
    if args.sheet and cells:
        cols = 6
        rows = (len(cells) + cols - 1) // cols
        sheet = np.zeros((rows * 300, cols * 300, 3), np.uint8)
        for position, cell in enumerate(cells):
            row, column = divmod(position, cols)
            sheet[row * 300 : (row + 1) * 300, column * 300 : (column + 1) * 300] = cell
        cv2.imwrite(args.sheet, sheet)
        print(args.sheet)


if __name__ == "__main__":
    main()

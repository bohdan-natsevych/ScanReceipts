"""Tile saved receipt images into one sheet, to compare a run against the recording.

python tools/debug/image_grid.py "%USERPROFILE%/Documents/Receipts/2026-09-01/Session_006" out.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CELL = 260


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("output")
    parser.add_argument("--glob", default="*.jpg")
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    cells = []
    for path in sorted(Path(args.directory).glob(args.glob)):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = CELL / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
        cell = np.zeros((CELL, CELL, 3), np.uint8)
        cell[: image.shape[0], : image.shape[1]] = image
        cv2.putText(
            cell, path.stem[-8:], (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
        cells.append(cell)

    if not cells:
        raise SystemExit("no images matched")
    rows = (len(cells) + args.cols - 1) // args.cols
    sheet = np.zeros((rows * CELL, args.cols * CELL, 3), np.uint8)
    for position, cell in enumerate(cells):
        row, column = divmod(position, args.cols)
        sheet[row * CELL : (row + 1) * CELL, column * CELL : (column + 1) * CELL] = cell
    cv2.imwrite(args.output, sheet)
    print(f"{args.output} cells={len(cells)}")


if __name__ == "__main__":
    main()

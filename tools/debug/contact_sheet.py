"""Render a labelled contact sheet from a recording, to eyeball what was presented.

python tools/debug/contact_sheet.py 92495b7e out.png --step 2 --from 20 --to 34
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from session_replay import find_session, iter_frames

CELL = (320, 180)
PORTRAIT_CELL = (180, 250)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("output")
    parser.add_argument("--step", type=float, default=2.0, help="seconds between cells")
    parser.add_argument("--from", dest="start", type=float, default=0.0)
    parser.add_argument("--to", dest="end", type=float, default=float("inf"))
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument(
        "--viewport",
        action="store_true",
        help="crop the letterboxing the detector also strips",
    )
    args = parser.parse_args()

    cell_size = PORTRAIT_CELL if args.viewport else CELL
    cells = []
    next_offset = args.start
    for frame in iter_frames(find_session(args.session)):
        if not args.start <= frame.offset <= args.end or frame.offset < next_offset:
            continue
        image = frame.image
        if args.viewport:
            from scan_receipts.detection import active_viewport

            x, y, width, height = active_viewport(image)
            image = image[y : y + height, x : x + width]
        cell = cv2.resize(image, cell_size)
        cv2.putText(
            cell,
            f"{frame.offset:.1f}",
            (5, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cells.append(cell)
        next_offset = frame.offset + args.step

    if not cells:
        raise SystemExit("no frames in the requested range")
    rows = (len(cells) + args.cols - 1) // args.cols
    sheet = np.zeros((rows * cell_size[1], args.cols * cell_size[0], 3), np.uint8)
    for position, cell in enumerate(cells):
        row, column = divmod(position, args.cols)
        sheet[
            row * cell_size[1] : (row + 1) * cell_size[1],
            column * cell_size[0] : (column + 1) * cell_size[0],
        ] = cell
    cv2.imwrite(args.output, sheet)
    print(f"{args.output} cells={len(cells)}")


if __name__ == "__main__":
    main()

"""Propose label windows for a recording, from the pixels alone.

Hand-labelling a session by scrubbing a contact sheet is slow and picks windows
that are wrong at the edges. This segments the recording into the stretches where
something is on the desk and nothing is moving, which is what a label window is
meant to bracket, and prints them ready to paste into a `labels.json`.

Nothing here uses the detector. That is the point: ground truth derived from the
thing under test would only ever agree with it.

    python tools/debug/segment_timeline.py 38efd941
    python tools/debug/segment_timeline.py 38efd941 --min-still 0.5 --json
    python tools/debug/segment_timeline.py 38efd941 --sheet plateaus.png
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from session_replay import find_session, iter_frames

WORK_WIDTH = 160


def viewport_of(root) -> tuple[int, int, int, int]:
    """Letterbox bounds, taken from the brightest frame in the first few seconds."""
    from scan_receipts.detection import active_viewport

    best, brightest = None, -1.0
    for frame in iter_frames(root):
        level = float(frame.image[::16, ::16].mean())
        if level > brightest:
            brightest, best = level, frame.image
        if frame.offset > 20.0:
            break
    return active_viewport(best)


def measure(root) -> list[tuple[float, float, float, int]]:
    """Per frame: offset, viewport brightness, change since the last frame, index."""
    x, y, width, height = viewport_of(root)
    scale = WORK_WIDTH / width
    readings: list[tuple[float, float, float, int]] = []
    previous: np.ndarray | None = None
    for frame in iter_frames(root):
        crop = frame.image[y : y + height, x : x + width]
        small = cv2.resize(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
            (WORK_WIDTH, max(1, int(height * scale))),
        ).astype(np.float32)
        change = 0.0 if previous is None else float(np.abs(small - previous).mean())
        readings.append((frame.offset, float(small.mean()), change, frame.index))
        previous = small
    return readings


def plateaus(
    readings: list[tuple[float, float, float, int]],
    min_still: float,
    change_limit: float,
    occupied_margin: float,
) -> list[dict[str, float]]:
    levels = np.array([item[1] for item in readings])
    empty_level = float(np.percentile(levels, 5))
    floor = empty_level + occupied_margin

    runs: list[dict[str, float]] = []
    start: tuple[float, int] | None = None
    for offset, level, change, index in readings:
        settled = level > floor and change < change_limit
        if settled and start is None:
            start = (offset, index)
        elif not settled and start is not None:
            if offset - start[0] >= min_still:
                runs.append({"start": start[0], "end": offset, "index": start[1]})
            start = None
    if start is not None and readings[-1][0] - start[0] >= min_still:
        runs.append({"start": start[0], "end": readings[-1][0], "index": start[1]})
    for run in runs:
        run["middle"] = round((run["start"] + run["end"]) / 2.0, 2)
        run["start"] = round(run["start"], 2)
        run["end"] = round(run["end"], 2)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("--min-still", type=float, default=0.45)
    parser.add_argument("--change-limit", type=float, default=1.6)
    parser.add_argument("--occupied-margin", type=float, default=6.0)
    parser.add_argument("--json", action="store_true", help="emit labels.json stub")
    parser.add_argument("--sheet", help="write one frame per plateau to this PNG")
    parser.add_argument("--cell-width", type=int, default=180)
    parser.add_argument("--cols", type=int, default=12)
    args = parser.parse_args()

    root = find_session(args.session)
    readings = measure(root)
    runs = plateaus(readings, args.min_still, args.change_limit, args.occupied_margin)

    if args.json:
        print(
            json.dumps(
                {
                    "session": root.name[:8],
                    "stride": 1,
                    "expected": [
                        {
                            "id": f"r{number:02d}",
                            "window": [run["start"], run["end"]],
                            "note": "",
                        }
                        for number, run in enumerate(runs, 1)
                    ],
                },
                indent=2,
            )
        )
    else:
        for number, run in enumerate(runs, 1):
            print(
                f"{number:3d}  {run['start']:6.2f} - {run['end']:6.2f}"
                f"  ({run['end'] - run['start']:.2f}s)"
            )
        print(f"{len(runs)} still stretches")

    if args.sheet:
        wanted = {run["index"]: number for number, run in enumerate(runs, 1)}
        x, y, width, height = viewport_of(root)
        cell_width = args.cell_width
        cell_height = round(cell_width * height / width)
        cells = []
        for frame in iter_frames(root):
            if frame.index not in wanted:
                continue
            cell = cv2.resize(
                frame.image[y : y + height, x : x + width], (cell_width, cell_height)
            )
            cv2.putText(
                cell,
                f"{wanted[frame.index]}:{frame.offset:.1f}",
                (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * cell_width / 180,
                (0, 0, 255),
                2,
            )
            cells.append(cell)
        columns = args.cols
        rows = (len(cells) + columns - 1) // columns
        sheet = np.zeros((rows * cell_height, columns * cell_width, 3), np.uint8)
        for position, cell in enumerate(cells):
            row, column = divmod(position, columns)
            sheet[
                row * cell_height : (row + 1) * cell_height,
                column * cell_width : (column + 1) * cell_width,
            ] = cell
        cv2.imwrite(args.sheet, sheet)
        print(f"{args.sheet} cells={len(cells)}")


if __name__ == "__main__":
    main()

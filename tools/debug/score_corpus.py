"""Score the detector against every labelled recording and print one table.

The corpus tests answer "did session 006 change?". This answers the question that
actually decides whether a detection change ships: across every labelled session,
did recall go up, and did the noise beside it go up more?

A label file names the receipts an operator presented and the window in which a
capture of each is acceptable:

    {
      "session": "Session_011",
      "stride": 1,
      "expected": [
        {"id": "r01", "window": [5.0, 9.0], "note": "long grocery receipt"},
        {"id": "r02", "window": [10.0, 13.0], "note": "same receipt, flipped"}
      ]
    }

It lives beside the recording as `labels.json`, and a copy is kept in the
repository under `corpus/labels/<uuid>.json` so the hand-labelling survives video
retention. Either location is read; the one beside the recording wins.

    python tools/debug/score_corpus.py                       # every labelled session
    python tools/debug/score_corpus.py 38efd941 92495b7e     # named sessions
    python tools/debug/score_corpus.py --baseline ../baseline/src
    python tools/debug/score_corpus.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from session_replay import find_session, list_sessions, replay

REPO_LABELS = Path(__file__).resolve().parents[2] / "corpus" / "labels"


def load_labels(root: Path) -> dict[str, Any] | None:
    for path in (root / "labels.json", REPO_LABELS / f"{root.name}.json"):
        if path.exists():
            labels = json.loads(path.read_text(encoding="utf-8"))
            labels.setdefault("session", root.name[:8])
            labels.setdefault("stride", 1)
            return labels
    return None


def labelled_sessions() -> list[Path]:
    return [root for root in list_sessions() if load_labels(root) is not None]


def match(offsets: list[float], expected: list[dict[str, Any]]) -> dict[str, Any]:
    """Assign each capture to a receipt, then say what was missed and what was noise.

    A capture inside an unclaimed window is that receipt; a second capture of the
    same one is a duplicate, which review can collapse; a capture inside no window
    is a stray, which review cannot do anything with. The three are counted apart
    because they cost the operator very different amounts.
    """
    claimed: dict[int, float] = {}
    duplicates: list[float] = []
    strays: list[float] = []
    for offset in offsets:
        inside = [
            position
            for position, item in enumerate(expected)
            if item["window"][0] <= offset <= item["window"][1]
        ]
        if not inside:
            strays.append(offset)
            continue
        free = [position for position in inside if position not in claimed]
        if not free:
            duplicates.append(offset)
            continue
        chosen = min(
            free,
            key=lambda position: abs(offset - sum(expected[position]["window"]) / 2.0),
        )
        claimed[chosen] = offset
    return {
        "hit": len(claimed),
        "duplicates": [round(value, 2) for value in duplicates],
        "strays": [round(value, 2) for value in strays],
        "missed": [
            item.get("id", str(position))
            for position, item in enumerate(expected)
            if position not in claimed
        ],
    }


def score_session(root: Path, stride_override: int | None = None) -> dict[str, Any]:
    labels = load_labels(root)
    if labels is None:
        raise SystemExit(f"no labels for {root}")
    stride = stride_override or int(labels["stride"])
    started_at, candidates = replay(root, stride)
    offsets = [candidate.timestamp - started_at for candidate in candidates]
    result = match(offsets, labels["expected"])
    result.update(
        name=labels["session"],
        uuid=root.name,
        stride=stride,
        expected=len(labels["expected"]),
        captured=len(offsets),
        flagged=sum(1 for item in candidates if item.quality_flag),
        offsets=[round(value, 2) for value in offsets],
    )
    return result


COUNTS = ("expected", "hit", "duplicates", "strays", "flagged", "captured")


def build_report(roots: list[Path], stride: int | None) -> dict[str, Any]:
    sessions = [score_session(root, stride) for root in roots]
    total = {
        key: sum(
            len(item[key]) if isinstance(item[key], list) else item[key]
            for item in sessions
        )
        for key in COUNTS
    }
    return {"sessions": sessions, "total": total}


def run_baseline(baseline_src: Path, argv: list[str]) -> dict[str, Any]:
    """Re-run this script against another checkout's sources in a fresh process.

    Two versions of `scan_receipts` cannot share an interpreter, and PYTHONPATH is
    set here rather than in a shell because a Windows path in that variable is
    mangled by MSYS on its way through one.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(baseline_src.resolve())
    finished = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *argv, "--json", "-"],
        env=environment,
        capture_output=True,
        text=True,
    )
    if finished.returncode != 0:
        raise SystemExit(f"baseline run failed:\n{finished.stderr}")
    return json.loads(finished.stdout)


def delta(current: int, before: int | None, lower_is_better: bool) -> str:
    if before is None or current == before:
        return ""
    change = current - before
    better = change < 0 if lower_is_better else change > 0
    return f" {'+' if change > 0 else ''}{change}{'' if better else '!'}"


def count(item: dict[str, Any], key: str) -> int:
    value = item[key]
    return len(value) if isinstance(value, list) else value


def print_report(report: dict[str, Any], baseline: dict[str, Any] | None) -> None:
    before: dict[str, dict[str, Any]] = {}
    if baseline is not None:
        before = {item["uuid"]: item for item in baseline["sessions"]}

    def row(name: str, item: dict[str, Any], old: dict[str, Any] | None) -> str:
        recall = f"{item['hit']}/{item['expected']}"
        cells = [f"{recall + delta(item['hit'], old and old['hit'], False):>12}"]
        for key, width in (("duplicates", 8), ("strays", 9), ("flagged", 8)):
            text = str(count(item, key)) + delta(
                count(item, key), old and count(old, key), True
            )
            cells.append(f"{text:>{width}}")
        return f"{name:<14}" + "".join(cells)

    print(f"{'session':<14}{'recall':>12}{'dup':>8}{'stray':>9}{'flag':>8}  missed")
    for item in report["sessions"]:
        old = before.get(item["uuid"])
        print(f"{row(item['name'], item, old)}  {','.join(item['missed']) or '-'}")

    print("-" * 72)
    print(row("TOTAL", report["total"], baseline and baseline["total"]))
    if baseline is not None:
        print("(a '!' marks a number that moved the wrong way)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session", nargs="*", help="path, <date>/<uuid>, or UUID prefix"
    )
    parser.add_argument("--stride", type=int, help="override the per-session stride")
    parser.add_argument(
        "--baseline", type=Path, help="src directory of a checkout to compare against"
    )
    parser.add_argument("--json", help="write the report here; '-' for stdout")
    parser.add_argument(
        "--offsets", action="store_true", help="also print the capture timeline"
    )
    parser.add_argument("--list", action="store_true", help="list labelled recordings")
    args = parser.parse_args()

    if args.list:
        for root in labelled_sessions():
            labels = load_labels(root)
            print(f"{labels['session']:<14}{root.parent.name}/{root.name}")
        return

    if args.session:
        roots = [find_session(spec) for spec in args.session]
        missing = [root.name for root in roots if load_labels(root) is None]
        if missing:
            raise SystemExit(f"no labels for {', '.join(missing)}")
    else:
        roots = labelled_sessions()
    if not roots:
        raise SystemExit(f"no labelled recordings; write one under {REPO_LABELS}")

    report = build_report(roots, args.stride)

    if args.json == "-":
        json.dump(report, sys.stdout)
        return

    baseline = None
    if args.baseline is not None:
        baseline = run_baseline(args.baseline, [str(root) for root in roots])
    print_report(report, baseline)
    if args.offsets:
        print()
        for item in report["sessions"]:
            print(f"{item['name']:<14}n={item['captured']:<3} {item['offsets']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

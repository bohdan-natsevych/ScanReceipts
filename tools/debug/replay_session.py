"""Print the candidate timeline a recording produces.

python tools/debug/replay_session.py 92495b7e --stride 1
python tools/debug/replay_session.py --list
"""

from __future__ import annotations

import argparse

from session_replay import find_session, list_sessions, replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session", nargs="*", help="path, <date>/<uuid>, or UUID prefix"
    )
    parser.add_argument("--stride", type=int, default=1, help="feed every Nth frame")
    parser.add_argument("--list", action="store_true", help="list known recordings")
    args = parser.parse_args()

    if args.list or not args.session:
        for path in list_sessions():
            print(f"{path.parent.name}/{path.name}")
        return

    for spec in args.session:
        root = find_session(spec)
        started_at, candidates = replay(root, args.stride)
        offsets = [round(item.timestamp - started_at, 2) for item in candidates]
        flags = {item.quality_flag for item in candidates if item.quality_flag}
        print(f"{root.name} stride={args.stride} n={len(offsets)} {offsets}")
        if flags:
            print(f"  quality flags: {sorted(flags)}")


if __name__ == "__main__":
    main()

"""Print per-frame detector state over a time window of a recording.

This is the workhorse for "why was nothing captured here?" questions: it shows the
gate values (motion, content weight, visibility, boundary confidence) alongside the
internal plateau/pending bookkeeping for every fed frame.

    python tools/debug/trace_session.py 92495b7e --from 20 --to 34
    python tools/debug/trace_session.py 92495b7e --from 64 --to 80 --stride 2
"""

from __future__ import annotations

import argparse

from session_replay import find_session, replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("--from", dest="start", type=float, default=0.0)
    parser.add_argument("--to", dest="end", type=float, default=float("inf"))
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    def on_frame(frame, metrics, candidate, detector) -> None:
        if not args.start <= frame.offset <= args.end:
            return
        pending = detector._pending_window
        pending_text = f"{pending[0]:.1f}-{pending[1]:.1f}" if pending else "-"
        stable = detector._stable_since
        print(
            f"{frame.offset:6.2f} {metrics.state.value:22s}"
            f" motion={metrics.motion:5.3f} content={metrics.content_score:4.2f}"
            f" vis={metrics.visibility:5.3f} conf={metrics.boundary_confidence:4.2f}"
            f" iou={metrics.quad_iou:4.2f} occl={int(metrics.occluded)}"
            f" stable={'-' if stable is None else f'{stable - frame.timestamp + frame.offset:.1f}'}"
            f" pend={pending_text} hashes={len(detector._recent_hashes)}"
            f" cand={'YES' if candidate is not None else ''} {metrics.message}"
        )

    started_at, candidates = replay(
        find_session(args.session), args.stride, on_frame=on_frame
    )
    offsets = [round(item.timestamp - started_at, 2) for item in candidates]
    print(f"candidates n={len(offsets)} {offsets}")


if __name__ == "__main__":
    main()

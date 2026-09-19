"""Corpus regression: replay every labelled recording and score it (ACC-1, ACC-6).

The windows these assertions run on live in `corpus/labels/<session-uuid>.json`, not
here, and `tools/debug/score_corpus.py` owns the rule that turns captures into recall,
duplicates and strays. Both are shared with the harness, so a number in this file and a
number on the harness table can never drift apart into meaning different things.

The budgets below are what the detector scores today. Raising a `hit` floor is the
point of a detection change; raising a `duplicates` or `strays` budget is a cost being
accepted, and belongs in the commit message of whatever change accepts it.

A green run on a machine without the recordings proves nothing: every case skips.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "debug"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from score_corpus import load_labels, score_session  # noqa: E402
from session_replay import replay, videos_root  # noqa: E402

from scan_receipts.detection import SKIN_COVER_FRACTION, skin_fraction  # noqa: E402

pytestmark = pytest.mark.corpus

# uuid: (minimum receipts found, duplicate budget, stray budget)
BUDGETS = {
    "2026-08-31/b98d0106-17ef-4280-8cb4-1b47763f3830": (10, 0, 0),
    "2026-09-01/92495b7e-0d36-4f9b-b437-aa9437e2a0dd": (12, 2, 0),
    "2026-09-01/d5dee043-32de-49d8-8e6c-47454110ec9a": (9, 1, 0),
    "2026-09-01/38efd941-6da4-45cb-9a67-5392b234c671": (15, 5, 1),
    "2026-09-02/591045ad-b11f-4166-9603-4260374df0a5": (20, 1, 0),
}


@pytest.mark.parametrize("relative", sorted(BUDGETS))
def test_labelled_session_meets_its_budget(relative: str) -> None:
    root = videos_root() / relative
    if not root.exists():
        pytest.skip(f"{relative} is part of the local corpus")
    labels = load_labels(root)
    assert labels is not None, f"no labels for {relative}"

    minimum_hits, duplicate_budget, stray_budget = BUDGETS[relative]
    result = score_session(root)
    name = result["name"]

    assert result["hit"] >= minimum_hits, (
        f"{name} found {result['hit']}/{result['expected']} receipts, "
        f"expected at least {minimum_hits}; missed {', '.join(result['missed'])}"
    )
    assert len(result["duplicates"]) <= duplicate_budget, (
        f"{name} captured {len(result['duplicates'])} second copies "
        f"at {result['duplicates']}, budget {duplicate_budget}"
    )
    assert len(result["strays"]) <= stray_budget, (
        f"{name} captured {len(result['strays'])} frames matching no receipt "
        f"at {result['strays']}, budget {stray_budget}"
    )


def test_session_011_captures_both_flips_and_all_three_small_cards() -> None:
    """The two failures the operator reported, named individually.

    A flip turns a blank page back to the lens, and a small card is presented with
    a sheet behind it; both used to be dropped, and a total that happened to stay
    the same would hide either coming back. Every capture here also has to sit
    inside its own window, so a stray landing in the right second cannot stand in
    for the receipt that went missing.
    """
    relative = "2026-09-01/38efd941-6da4-45cb-9a67-5392b234c671"
    root = videos_root() / relative
    if not root.exists():
        pytest.skip(f"{relative} is part of the local corpus")

    # r06 and r08: the thermal receipt folded up off the carbon form beneath it.
    # r09, r10, r11: the three Seair cards, each behind or beside another sheet.
    reported = {"r06", "r08", "r09", "r10", "r11"}
    missed = reported.intersection(score_session(root)["missed"])
    assert not missed, f"reported failures back: {', '.join(sorted(missed))}"


def test_session_003_never_saves_an_unflagged_hand() -> None:
    """A hand on the page is either waited out or declared - never passed off.

    Every receipt in this session is placed under a flat hand and then released,
    which used to produce two captures: a photograph of the hand, then the page.
    A hand held still is not motion, so the change-based occlusion test never saw
    it. A capture whose quad is mostly skin is now either deferred until the hand
    leaves or - when the page was never seen bare - emitted `hand-covered`, so
    review can find every one of them in a single pass.
    """
    relative = "2026-09-02/591045ad-b11f-4166-9603-4260374df0a5"
    root = videos_root() / relative
    if not root.exists():
        pytest.skip(f"{relative} is part of the local corpus")

    started_at, candidates = replay(root)
    covered = [
        (candidate.timestamp - started_at, fraction, candidate.quality_flag)
        for candidate in candidates
        if (fraction := skin_fraction(candidate.frame, candidate.corners))
        >= SKIN_COVER_FRACTION
    ]

    unflagged = [item for item in covered if item[2] != "hand-covered"]
    assert not unflagged, (
        "captures with a hand over the receipt and no flag: "
        + ", ".join(
            f"{offset:.2f}s (skin {value:.2f})" for offset, value, _ in unflagged
        )
    )
    assert len(covered) <= 3, (
        f"{len(covered)} hand-covered captures, budget 3: "
        + ", ".join(f"{offset:.2f}s" for offset, _, _ in covered)
    )


@pytest.mark.xfail(
    reason="known defect, no guard found that keeps recall - see the docstring",
    strict=False,
)
def test_session_010_saves_nothing_while_the_desk_is_empty() -> None:
    """Two receipts were saved of a bare desk with hands moving over it.

    `_commit_on_absence` is the ACC-6 net for a receipt only ever seen under a
    hand: whatever the presentation held is emitted when the document goes away.
    A pair of hands over an empty desk formed the same shape of presentation -
    a framed quad past the content gate - and was committed the same way, with
    no fingerprint ever recorded and no paper ever detected in the scene.

    Refusing to commit when nothing in the presentation ever registered as paper
    (`_has_paper_presence` false throughout, `_recent_hashes` empty) does remove
    both captures, and costs 9 receipts on the corpus: recall 56/65 -> 47/65,
    Session_003 20->16, Session_006 12->8, Session_004 9->8. The receipts it
    drops are the ACC-6 case this path exists for - pages only ever seen under a
    hand - which register no paper presence either, so that signal does not
    separate the two populations. Left failing rather than paid for at that
    price; it flips to XPASS when a signal that does separate them is found.
    """
    relative = "2026-09-02/e5cdecb3-081e-402b-a66b-03e1af4447ae"
    root = videos_root() / relative
    if not root.exists():
        pytest.skip(f"{relative} is part of the local corpus")

    started_at, candidates = replay(root)
    empty_desk = [
        offset
        for candidate in candidates
        if 54.0 <= (offset := candidate.timestamp - started_at) <= 58.5
    ]

    assert not empty_desk, (
        "captures of an empty desk at "
        + ", ".join(f"{offset:.2f}s" for offset in empty_desk)
    )

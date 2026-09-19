# Detector debugging tools

Reusable scripts for investigating capture behaviour against recorded sessions.
Recordings live under `%LOCALAPPDATA%\ScanReceipts\Videos\<date>\<session-uuid>`;
every script accepts a full path, a `<date>/<uuid>` pair, or a bare UUID prefix.

Run them from the repository root with `tools/debug` on the path:

```
python tools/debug/replay_session.py --list
python tools/debug/replay_session.py 92495b7e
python tools/debug/replay_session.py 92495b7e --stride 5      # sparse live sampling
python tools/debug/trace_session.py 92495b7e --from 20 --to 34
python tools/debug/contact_sheet.py 92495b7e sheet.png --step 2 --from 20 --to 34
python tools/debug/image_grid.py "$env:USERPROFILE/Documents/Receipts/2026-09-01/Session_006" saved.png
```

| Script | Answers |
| --- | --- |
| `score_corpus.py` | Is this change net positive across every labelled session? |
| `segment_timeline.py` | Where are the settled presentations, so I can label them? |
| `replay_session.py` | Which candidates does a recording produce, and when? |
| `trace_session.py` | Which gate rejected a frame the detector should have kept? |
| `contact_sheet.py` | What was actually presented to the camera in this window? |
| `image_grid.py` | What did a live run save, compared to the recording? |
| `session_replay.py` | Shared library: session lookup, frame iteration, replay loop. |

`replay_session.py` mirrors what `tests/test_recorded_sessions.py` asserts, so use it
to derive expected windows before writing or changing a session test.

## Scoring a detection change

`score_corpus.py` is the one to run before and after touching detection. It replays
every labelled recording and prints, per session and in total, how many of the
presented receipts reached review (`recall`), how many second copies of one receipt
came with them (`dup`), how many captures matched no receipt at all (`stray`), and how
many were flagged. `--baseline` re-runs the whole thing against another checkout in a
subprocess and marks every number that moved the wrong way.

```
python tools/debug/score_corpus.py                       # every labelled session
python tools/debug/score_corpus.py 38efd941 --offsets    # one, with its timeline
git worktree add ../baseline HEAD
python tools/debug/score_corpus.py --baseline ../baseline/src
```

Pass `--baseline` a path rather than setting `PYTHONPATH` in a shell: a Windows path
in that variable is mangled on its way through MSYS, and the run then silently scores
the working tree twice.

## Labelling a recording

Ground truth lives in `corpus/labels/<session-uuid>.json` (or beside the recording as
`labels.json`, which wins if both exist): one entry per receipt the operator presented,
with the window in which a capture of it is acceptable.

```
python tools/debug/segment_timeline.py 38efd941 --sheet plateaus.png --cell-width 460 --cols 7
python tools/debug/segment_timeline.py 38efd941 --json > corpus/labels/<uuid>.json
```

`segment_timeline.py` finds the stretches where something is on the desk and nothing is
moving, using pixels only - no detector - and `--sheet` renders one full-viewport frame
per stretch so each can be identified before the windows are named. Merge the stretches
that show the same page and write the note that says which page it was; a regression
then reports `r06,r11` rather than a count.

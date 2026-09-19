# Scan Receipts

A local Windows desktop application that watches a camera, finds stable receipts,
selects the best recent frame, corrects/crops it, and saves one QuickBooks-ready
image per receipt. Recorded video files use the identical capture and detection
pipeline, which makes repeatable testing possible without a camera.

## Run

```powershell
uv sync
uv run scan-receipts
```

`uv sync` creates the project environment from `uv.lock` and includes the
development tools. Run the checks with `uv run pytest` and
`uv run ruff check src tests tools`.

Use **Video file...** in the Source list to replay a recording. Receipt images
default to `Documents\Receipts`; recordings and SQLite metadata live under
`%LOCALAPPDATA%\ScanReceipts`. All locations are visible in Settings.

## Detection workflow

Start a session, put one receipt under the selected camera, and hold it reasonably
steady. The status changes from motion/settling to captured and the counter
increments. Replace it immediately; processing happens on a separate worker.
Stop the session to drain pending work, review/edit each result, then explicitly
confirm the session successful.

The live diagnostics overlay is intentionally enabled by default. Camera and
lighting setups vary, so motion, stability, sharpness, boundary confidence, and
the detector state remain visible and thresholds can be tuned in Settings.

## Hardware and corpus validation

The code cannot certify camera-driver behavior without the target devices. Run
the required CAM-5 spike on 2-3 cameras (including Phone Link), then complete the
manual checks written into the report:

```powershell
uv run python tools\camera_spike.py --output camera_spike_report.json
```

Use **Video file...** to replay each reference-corpus recording through the same
detector. **Thorough Recovery Pass** runs that detector with its full-resolution
offline profile and presents possible missing receipts for explicit acceptance
or rejection; it never auto-adds uncertain results.

# Scan Receipts

A local Windows desktop application that watches a camera, finds stable receipts,
selects the best recent frame, corrects/crops it, and saves one QuickBooks-ready
image per receipt. Recorded video files use the identical capture and detection
pipeline, which makes repeatable testing possible without a camera.

## Install

Download `ScanReceipts-Setup.exe` from the
[latest release](https://github.com/bohdan-natsevych/ScanReceipts/releases/latest)
and run it. It installs for the current Windows user only, needs no
Administrator rights, and adds a **Scan Receipts** entry to the Start menu.

The installer is not code signed, so Windows SmartScreen shows "Windows
protected your PC" the first time. Choose **More info**, then **Run anyway**.

To move to a newer version later, open the app and use **Check for updates** on
the Scan tab. The app downloads the same installer, replaces itself, and starts
again. Receipts in `Documents\Receipts` and everything under
`%LOCALAPPDATA%\ScanReceipts` are never touched by an install, an update, or an
uninstall.

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

## Releasing

Merging a pull request into `master` publishes a new installer. There is nothing
to bump by hand.

GitHub Actions reads the latest release tag, adds one to the patch number
(`v0.1.2` becomes `0.1.3`; the first release is `0.1.0`), stamps that string into
the frozen application, builds `ScanReceipts-Setup.exe` with PyInstaller and Inno
Setup, and publishes it as the new tag. Nothing is committed back to the
repository, and `version` in `pyproject.toml` is not the source of truth.

The workflow does not run the tests. Run them before you merge:

```powershell
uv run pytest
uv run ruff check src tests tools
```

To build an installer locally without publishing:

```powershell
uv run pyinstaller --noconfirm packaging\scan_receipts.spec
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DAppVersion=0.0.0 packaging\ScanReceipts.iss
```

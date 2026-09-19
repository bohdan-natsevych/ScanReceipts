from __future__ import annotations

from pathlib import Path

from scan_receipts.update import ReleaseInfo, UpdateError
from scan_receipts.workers import UpdateWorker


def release(version: str = "0.1.4") -> ReleaseInfo:
    return ReleaseInfo(
        version=version,
        tag=f"v{version}",
        download_url=(
            "https://github.com/owner/repo/releases/download/"
            f"v{version}/ScanReceipts-Setup.exe"
        ),
        page_url=f"https://github.com/owner/repo/releases/tag/v{version}",
    )


def collect(worker: UpdateWorker) -> dict[str, list]:
    seen: dict[str, list] = {
        "up_to_date": [],
        "update_found": [],
        "progress": [],
        "downloaded": [],
        "failed": [],
    }
    for name, sink in seen.items():
        getattr(worker, name).connect(sink.append)
    return seen


def test_a_newer_release_is_announced(qtbot, monkeypatch) -> None:
    monkeypatch.setattr("scan_receipts.workers.latest_release", lambda: release())
    worker = UpdateWorker("0.1.3")
    seen = collect(worker)

    worker.check()

    assert seen["update_found"] == [release()]
    assert seen["up_to_date"] == []


def test_the_same_version_reports_up_to_date(qtbot, monkeypatch) -> None:
    monkeypatch.setattr("scan_receipts.workers.latest_release", lambda: release("0.1.3"))
    worker = UpdateWorker("0.1.3")
    seen = collect(worker)

    worker.check()

    assert seen["up_to_date"] == ["0.1.3"]
    assert seen["update_found"] == []


def test_a_lookup_failure_becomes_a_message(qtbot, monkeypatch) -> None:
    def explode() -> ReleaseInfo:
        raise UpdateError("Could not reach GitHub: offline")

    monkeypatch.setattr("scan_receipts.workers.latest_release", explode)
    worker = UpdateWorker("0.1.3")
    seen = collect(worker)

    worker.check()

    assert seen["failed"] == ["Could not reach GitHub: offline"]


def test_downloading_reports_progress_and_the_finished_path(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    installer = tmp_path / "ScanReceipts-Setup.exe"
    installer.write_bytes(b"stub")

    def download(_release, progress=None):
        if progress is not None:
            progress(50)
            progress(100)
        return installer

    monkeypatch.setattr("scan_receipts.workers.download_installer", download)
    worker = UpdateWorker("0.1.3")
    seen = collect(worker)

    worker.download(release())

    assert seen["progress"] == [50, 100]
    assert seen["downloaded"] == [installer]


def test_a_download_failure_becomes_a_message(qtbot, monkeypatch) -> None:
    def explode(_release, progress=None):
        raise UpdateError("Could not download the installer: reset")

    monkeypatch.setattr("scan_receipts.workers.download_installer", explode)
    worker = UpdateWorker("0.1.3")
    seen = collect(worker)

    worker.download(release())

    assert seen["failed"] == ["Could not download the installer: reset"]
    assert seen["downloaded"] == []

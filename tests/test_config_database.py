from __future__ import annotations

import faulthandler
import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from scan_receipts.config import SettingsStore, default_settings
from scan_receipts.database import Repository
from scan_receipts.diagnostics import configure_logging, fault_path
from scan_receipts.models import SessionStatus
from scan_receipts.storage import delete_session_video
from scan_receipts.version import APP_VERSION


def configured(tmp_path: Path):
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    return settings


def test_settings_round_trip(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    settings = configured(tmp_path)
    settings.output.jpeg_quality = 87
    settings.detection.exclusion_rect = (0.0, 0.0, 0.1, 0.2)
    store.save(settings)

    loaded = store.load()

    assert loaded.output.jpeg_quality == 87
    assert loaded.detection.exclusion_rect == (0.0, 0.0, 0.1, 0.2)


def test_sessions_get_unique_folders_and_interrupted_sessions_recover(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "history.sqlite3")
    settings = configured(tmp_path)
    first = repository.create_session("Test camera", settings)
    second = repository.create_session("Test camera", settings)

    assert first.receipt_folder != second.receipt_folder
    assert Path(first.receipt_folder).name == "Session_001"
    assert Path(second.receipt_folder).name == "Session_002"

    repository.recover_interrupted_sessions()
    recovered = repository.get_session(first.id)
    assert recovered is not None
    assert recovered.status is SessionStatus.NEEDS_REVIEW
    assert "Interrupted" in recovered.processing_status


def test_session_can_use_an_exact_user_selected_folder(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "history.sqlite3")
    chosen = tmp_path / "Bookkeeping" / "August receipts"

    session = repository.create_session(
        "Test camera", configured(tmp_path), receipt_folder=chosen
    )

    assert Path(session.receipt_folder) == chosen.resolve()
    assert (chosen / "Originals").is_dir()


def test_delete_session_keeps_receipts_but_removes_external_video_and_data(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "history.sqlite3")
    settings = configured(tmp_path)
    session = repository.create_session("Test camera", settings)
    marker = Path(session.receipt_folder) / "keep-me.txt"
    marker.write_text("receipt", encoding="utf-8")
    imported_video = tmp_path / "Imported" / "source.mp4"
    imported_video.parent.mkdir()
    imported_video.write_bytes(b"video")
    repository.set_video_path(session.id, str(imported_video))

    delete_session_video(imported_video, settings.video_root)
    repository.remove_session_history(session.id)

    assert repository.get_session(session.id) is None
    assert not imported_video.exists()
    assert marker.read_text(encoding="utf-8") == "receipt"


def test_video_cleanup_only_recursively_deletes_managed_directories(
    tmp_path: Path,
) -> None:
    video_root = tmp_path / "Videos"
    managed_session = video_root / "session-id"
    managed_session.mkdir(parents=True)
    (managed_session / "recording.avi").write_bytes(b"video")

    delete_session_video(managed_session, video_root)

    assert not managed_session.exists()

    external_directory = tmp_path / "Imported"
    external_directory.mkdir()
    try:
        delete_session_video(external_directory, video_root)
    except RuntimeError as error:
        assert "external directory" in str(error)
    else:
        raise AssertionError("External directories must never be recursively deleted")
    assert external_directory.exists()


def test_review_flag_defaults_off_and_round_trips(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Test camera", configured(tmp_path))
    folder = Path(session.receipt_folder)
    receipt = repository.add_receipt(
        session.id, 1, 0.0, folder / "Receipt_0001.jpg", folder / "Receipt_0001.png", 1
    )

    assert receipt.review_flag is False

    repository.set_review_flag(receipt.id, True)
    assert repository.get_receipt(receipt.id).review_flag is True

    repository.set_review_flag(receipt.id, False)

def test_logging_writes_to_a_rotating_file_capped_at_ten_fifty_megabyte_files(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    log = configure_logging(tmp_path / "logs")

    logging.getLogger("scan_receipts.test").info("hello from the session")

    handler = next(
        item
        for item in logging.getLogger("scan_receipts").handlers
        if isinstance(item, RotatingFileHandler)
    )
    assert handler.maxBytes == 50 * 1024 * 1024
    assert handler.backupCount == 10
    assert "hello from the session" in log.read_text(encoding="utf-8")


def test_info_is_the_default_level_and_debug_can_be_turned_on(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    quiet = configure_logging(tmp_path / "quiet")
    logging.getLogger("scan_receipts.test").debug("noisy detail")
    assert "noisy detail" not in quiet.read_text(encoding="utf-8")

    verbose = configure_logging(tmp_path / "verbose", level="DEBUG")
    logging.getLogger("scan_receipts.test").debug("noisy detail")

    assert "noisy detail" in verbose.read_text(encoding="utf-8")


def test_the_log_level_can_be_raised_without_a_new_build(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setenv("SCANRECEIPTS_LOG_LEVEL", "DEBUG")

    log = configure_logging(tmp_path / "logs")
    logging.getLogger("scan_receipts.test").debug("env driven detail")

    assert "env driven detail" in log.read_text(encoding="utf-8")


def test_every_run_is_marked_in_the_log_with_its_version(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)

    log = configure_logging(tmp_path / "logs")

    written = log.read_text(encoding="utf-8")
    assert "started" in written
    assert APP_VERSION in written


def test_unhandled_exceptions_are_logged_and_reported(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    shown: list[str] = []
    log = configure_logging(tmp_path / "logs", notify=shown.append)

    try:
        raise ValueError("duplicate deck exploded")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    written = log.read_text(encoding="utf-8")
    assert "duplicate deck exploded" in written
    assert "test_unhandled_exceptions_are_logged_and_reported" in written
    assert shown and "duplicate deck exploded" in shown[0]


def test_worker_thread_exceptions_are_logged_too(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setattr(threading, "excepthook", threading.__excepthook__)
    log = configure_logging(tmp_path / "logs")

    worker = threading.Thread(target=lambda: 1 / 0, name="detection")
    worker.start()
    worker.join()

    written = log.read_text(encoding="utf-8")
    assert "ZeroDivisionError" in written
    assert "detection" in written


def test_native_crashes_are_armed_with_the_fault_handler(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    was_enabled = faulthandler.is_enabled()

    try:
        configure_logging(tmp_path / "logs")

        assert faulthandler.is_enabled()
        assert fault_path(tmp_path / "logs").exists()
    finally:
        if not was_enabled:
            faulthandler.disable()


def test_an_unusable_log_level_falls_back_instead_of_blocking_startup(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setenv("SCANRECEIPTS_LOG_LEVEL", "VERBOSE")

    log = configure_logging(tmp_path / "logs")

    assert logging.getLogger("scan_receipts").level == logging.INFO
    written = log.read_text(encoding="utf-8")
    assert "VERBOSE" in written
    assert "INFO" in written


def test_a_log_level_is_read_whatever_its_case_or_spacing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setenv("SCANRECEIPTS_LOG_LEVEL", "  debug ")

    configure_logging(tmp_path / "logs")

    assert logging.getLogger("scan_receipts").level == logging.DEBUG

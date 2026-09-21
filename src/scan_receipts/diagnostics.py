from __future__ import annotations

import faulthandler
import logging
import os
import sys
import threading
import traceback
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

from .config import app_data_dir
from .version import APP_VERSION

LOGGER_NAME = "scan_receipts"
MAXIMUM_LOG_BYTES = 50 * 1024 * 1024
LOG_BACKUPS = 10
LEVEL_VARIABLE = "SCANRECEIPTS_LOG_LEVEL"
FORMAT = "%(asctime)s %(levelname)-8s [%(threadName)s] %(name)s: %(message)s"

_fault_file = None


def log_directory(directory: Path | None = None) -> Path:
    return Path(directory) if directory is not None else app_data_dir() / "logs"


def log_path(directory: Path | None = None) -> Path:
    return log_directory(directory) / "scan_receipts.log"


def fault_path(directory: Path | None = None) -> Path:
    return log_directory(directory) / "scan_receipts.fault.log"


def configure_logging(
    directory: Path | None = None,
    level: int | str | None = None,
    notify: Callable[[str], None] | None = None,
) -> Path:
    """Start this run's log and route every kind of failure into it.

    CLAUDE CODE: a frozen windowed build has no console, so anything printed is
    discarded. Three separate paths can end a run and each needs its own hook:
    an unhandled exception on the GUI thread, one on a worker thread, and a
    native fault in Qt or OpenCV that never reaches Python at all.
    """
    path = log_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_level(level))
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = RotatingFileHandler(
        path,
        maxBytes=MAXIMUM_LOG_BYTES,
        backupCount=LOG_BACKUPS,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)
    # CLAUDE CODE: running from source still deserves a console; the frozen build
    # has no usable stderr, and logging would raise on every record if given one.
    if sys.stderr is not None:
        logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.propagate = False

    _install_hooks(logger, notify)
    _arm_fault_handler(fault_path(directory))
    logger.info(
        "Scan Receipts %s started, logging at %s to %s",
        APP_VERSION,
        logging.getLevelName(logger.level),
        path,
    )
    return path


def _level(level: int | str | None) -> int | str:
    if level is not None:
        return level
    return os.environ.get(LEVEL_VARIABLE, "INFO").upper()


def _install_hooks(
    logger: logging.Logger, notify: Callable[[str], None] | None
) -> None:
    def report(
        kind: type[BaseException],
        value: BaseException,
        stack: TracebackType | None,
        thread: str | None = None,
    ) -> None:
        where = f" in {thread}" if thread else ""
        logger.critical("Unhandled exception%s", where, exc_info=(kind, value, stack))
        if notify is not None:
            notify("".join(traceback.format_exception(kind, value, stack)))

    sys.excepthook = lambda kind, value, stack: report(kind, value, stack)
    threading.excepthook = lambda args: report(
        args.exc_type, args.exc_value, args.exc_traceback, args.thread.name
    )


def install_qt_message_handler() -> None:
    """Send Qt's own diagnostics to the log instead of a discarded stderr.

    CLAUDE CODE: Qt ends the process itself for a fatal condition - a QThread
    destroyed while still running, a re-entrant paint - by printing one line and
    calling abort. No Python hook sees that, and in a windowed build the line is
    thrown away, which leaves a crash with no cause. This is the only place that
    line can be captured, and it must be written before abort runs.
    """
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    logger = logging.getLogger(f"{LOGGER_NAME}.qt")
    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(mode, context, message: str) -> None:
        logger.log(levels.get(mode, logging.INFO), "%s", message)
        for target in logger.handlers or logging.getLogger(LOGGER_NAME).handlers:
            target.flush()

    qInstallMessageHandler(handler)


def _arm_fault_handler(path: Path) -> None:
    """Dump the stack of a native crash, which no Python hook can see.

    CLAUDE CODE: an access violation inside Qt kills the process outright, so the
    only record possible is the one the C handler writes on its way down. The
    file stays open for the life of the process because faulthandler keeps the
    descriptor, not the object.
    """
    global _fault_file
    if _fault_file is not None:
        _fault_file.close()
    _fault_file = path.open("a", encoding="utf-8", buffering=1)
    _fault_file.write(f"--- run started, Scan Receipts {APP_VERSION} ---\n")
    faulthandler.enable(file=_fault_file, all_threads=True)

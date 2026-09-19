from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from .config import app_data_dir


def log_path(directory: Path | None = None) -> Path:
    base = Path(directory) if directory is not None else app_data_dir() / "logs"
    return base / "scan_receipts.log"


MAXIMUM_LOG_BYTES = 1_000_000


def install_crash_logging(
    directory: Path | None = None,
    notify: Callable[[str], None] | None = None,
    maximum_bytes: int = MAXIMUM_LOG_BYTES,
) -> Path:
    """Route unhandled exceptions to a log file, and optionally to the user.

    CLAUDE CODE: a frozen windowed build has no console, so a traceback printed
    by Qt goes nowhere and a failed action looks like a crash with no evidence
    behind it. PySide6 prints unhandled slot exceptions through sys.excepthook,
    so replacing it is what captures them.
    """
    path = log_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = sys.excepthook

    def hook(
        kind: type[BaseException],
        value: BaseException,
        stack: TracebackType | None,
    ) -> None:
        report = "".join(traceback.format_exception(kind, value, stack))
        stamp = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
        _append(path, f"{stamp}\n{report}\n", maximum_bytes)
        if notify is not None:
            notify(report)
        previous(kind, value, stack)

    sys.excepthook = hook
    return path


def _append(path: Path, entry: str, maximum_bytes: int) -> None:
    """Add one entry, keeping this log and the one before it under the cap.

    CLAUDE CODE: a repeating failure writes a traceback per occurrence, so an
    unbounded log is a disk leak on the machine it is meant to help diagnose.
    """
    encoded = entry.encode("utf-8")
    written = path.stat().st_size if path.exists() else 0
    if written and written + len(encoded) > maximum_bytes:
        path.replace(path.with_name(path.name + ".1"))
    with path.open("ab") as handle:
        handle.write(encoded[-maximum_bytes:])

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from .diagnostics import configure_logging, install_qt_message_handler, log_path
from .ui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Scan Receipts")
    app.setOrganizationName("ScanReceipts")

    def report_crash(report: str) -> None:
        QMessageBox.critical(
            None,
            "Scan Receipts hit an unexpected error",
            f"{report.strip().splitlines()[-1]}\n\nThe full details are in "
            f"{log_path()}. Send that file with your bug report.",
        )

    configure_logging(notify=report_crash)
    install_qt_message_handler()
    log = logging.getLogger(__name__)
    try:
        window = MainWindow()
        window.show()
        code = app.exec()
    except Exception as error:
        log.critical("Scan Receipts could not start", exc_info=True)
        QMessageBox.critical(None, "Scan Receipts could not start", str(error))
        return 1
    # CLAUDE CODE: the closing line is the evidence that the run ended on its own.
    # A log that stops without it means the process was killed from underneath.
    log.info("Scan Receipts exited cleanly with code %s", code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

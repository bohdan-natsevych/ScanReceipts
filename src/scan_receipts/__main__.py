from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from .diagnostics import install_crash_logging, log_path
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

    install_crash_logging(notify=report_crash)
    try:
        window = MainWindow()
        window.show()
        return app.exec()
    except Exception as error:
        QMessageBox.critical(None, "Scan Receipts could not start", str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

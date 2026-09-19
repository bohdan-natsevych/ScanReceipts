from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from .ui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Scan Receipts")
    app.setOrganizationName("ScanReceipts")
    try:
        window = MainWindow()
        window.show()
        return app.exec()
    except Exception as error:
        QMessageBox.critical(None, "Scan Receipts could not start", str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

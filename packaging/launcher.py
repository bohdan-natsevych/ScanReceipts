"""Frozen entry point. PyInstaller cannot use the package's own __main__."""

from scan_receipts.__main__ import main

raise SystemExit(main())

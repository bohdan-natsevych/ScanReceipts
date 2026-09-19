# -*- mode: python ; coding: utf-8 -*-
"""Onedir freeze of Scan Receipts, built by .github/workflows/release.yml."""

from pathlib import Path

project_root = Path(SPECPATH).parent

analysis = Analysis(
    [str(project_root / "packaging" / "launcher.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=[],
    hiddenimports=["scan_receipts", "comtypes"],
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest", "pytestqt", "tests", "tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="ScanReceipts",
    console=False,
    debug=False,
    strip=False,
    upx=False,
)
collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="ScanReceipts",
)

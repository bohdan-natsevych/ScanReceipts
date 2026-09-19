from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "tools" / "stamp_version.py"


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def test_stamping_replaces_only_the_version_literal(tmp_path: Path) -> None:
    target = tmp_path / "version.py"
    target.write_text(
        '"""Doc."""\n\nAPP_VERSION = "0.1.0"\nOTHER = "0.1.0"\n', encoding="utf-8"
    )

    result = run("0.1.7", str(target))

    assert result.returncode == 0, result.stderr
    assert 'APP_VERSION = "0.1.7"' in target.read_text(encoding="utf-8")
    assert 'OTHER = "0.1.0"' in target.read_text(encoding="utf-8")


def test_stamping_fails_when_the_literal_is_missing(tmp_path: Path) -> None:
    target = tmp_path / "version.py"
    target.write_text("VERSION = '0.1.0'\n", encoding="utf-8")

    result = run("0.1.7", str(target))

    assert result.returncode != 0
    assert "APP_VERSION" in result.stderr


def test_the_shipped_version_module_can_be_stamped() -> None:
    module = REPOSITORY / "src" / "scan_receipts" / "version.py"
    text = module.read_text(encoding="utf-8")

    assert text.count("APP_VERSION = ") == 1

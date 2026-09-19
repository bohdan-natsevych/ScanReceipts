"""Rewrite APP_VERSION before a release build. The result is never committed."""

from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT_TARGET = (
    Path(__file__).resolve().parents[1] / "src" / "scan_receipts" / "version.py"
)
ASSIGNMENT = re.compile(r'^APP_VERSION = "[^"]*"$', re.MULTILINE)


def stamp(version: str, target: Path = DEFAULT_TARGET) -> str:
    text = target.read_text(encoding="utf-8")
    updated, count = ASSIGNMENT.subn(f'APP_VERSION = "{version}"', text)
    if count != 1:
        raise SystemExit(
            f"Expected exactly one APP_VERSION assignment in {target}, found {count}"
        )
    target.write_text(updated, encoding="utf-8")
    return version


def main(argv: list[str]) -> int:
    if not 2 <= len(argv) <= 3:
        raise SystemExit("usage: stamp_version.py <version> [path]")
    target = Path(argv[2]) if len(argv) == 3 else DEFAULT_TARGET
    print(stamp(argv[1], target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

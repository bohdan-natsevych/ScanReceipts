from __future__ import annotations

from pathlib import Path

INSTALLER_SCRIPT = (
    Path(__file__).resolve().parents[1] / "packaging" / "ScanReceipts.iss"
)


def script_text() -> str:
    return INSTALLER_SCRIPT.read_text(encoding="utf-8")


def test_the_installer_is_per_user_and_never_asks_for_administrator() -> None:
    text = script_text()

    assert "PrivilegesRequired=lowest" in text
    assert r"DefaultDirName={localappdata}\Programs\ScanReceipts" in text


def test_the_installer_produces_the_asset_name_the_updater_looks_for() -> None:
    assert "OutputBaseFilename=ScanReceipts-Setup" in script_text()


def test_the_installer_never_names_a_user_data_folder() -> None:
    text = script_text().lower()

    assert "{userdocs}" not in text
    assert r"{localappdata}\scanreceipts" not in text


def test_the_installer_relaunches_the_app_after_a_silent_update() -> None:
    text = script_text()

    assert "[Run]" in text
    assert "skipifsilent" not in text

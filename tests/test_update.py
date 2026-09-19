from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest

from scan_receipts.update import (
    INSTALLER_ARGUMENTS,
    INSTALLER_ASSET,
    ReleaseInfo,
    UpdateError,
    download_installer,
    is_newer,
    latest_release,
    parse_version,
    run_installer,
)


def release_payload(tag: str = "v0.1.4", asset: str = INSTALLER_ASSET) -> bytes:
    return json.dumps(
        {
            "tag_name": tag,
            "html_url": f"https://github.com/owner/repo/releases/tag/{tag}",
            "assets": [
                {"name": "other.zip", "browser_download_url": "https://github.com/a.zip"},
                {
                    "name": asset,
                    "browser_download_url": (
                        f"https://github.com/owner/repo/releases/download/{tag}/{asset}"
                    ),
                },
            ],
        }
    ).encode("utf-8")


def responder(payload: bytes):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_exception) -> bool:
            return False

    def open_url(_request, timeout=0):
        return Response(payload)

    return open_url


def test_versions_compare_numerically_not_as_text() -> None:
    assert parse_version("v0.1.10") == (0, 1, 10)
    assert is_newer("0.1.10", "0.1.9")
    assert not is_newer("0.1.9", "0.1.10")
    assert not is_newer("0.1.9", "0.1.9")


def test_an_unrecognised_version_is_an_update_error() -> None:
    with pytest.raises(UpdateError):
        parse_version("nightly")


def test_the_latest_release_is_read_from_the_named_installer_asset() -> None:
    info = latest_release(responder(release_payload()))

    assert info == ReleaseInfo(
        version="0.1.4",
        tag="v0.1.4",
        download_url=(
            "https://github.com/owner/repo/releases/download/v0.1.4/ScanReceipts-Setup.exe"
        ),
        page_url="https://github.com/owner/repo/releases/tag/v0.1.4",
    )


def test_a_release_without_the_installer_is_an_error_not_up_to_date() -> None:
    with pytest.raises(UpdateError, match=INSTALLER_ASSET):
        latest_release(responder(release_payload(asset="ScanReceipts.zip")))


def test_malformed_json_is_an_update_error() -> None:
    with pytest.raises(UpdateError):
        latest_release(responder(b"not json"))


def test_a_network_failure_is_an_update_error() -> None:
    def open_url(_request, timeout=0):
        raise OSError("no route to host")

    with pytest.raises(UpdateError, match="GitHub"):
        latest_release(open_url)


def test_an_asset_hosted_outside_github_is_refused() -> None:
    payload = json.loads(release_payload())
    payload["assets"][1]["browser_download_url"] = (
        f"https://example.com/{INSTALLER_ASSET}"
    )

    with pytest.raises(UpdateError, match=r"example\.com"):
        latest_release(responder(json.dumps(payload).encode("utf-8")))


def installer_release(url: str = "") -> ReleaseInfo:
    return ReleaseInfo(
        version="0.1.4",
        tag="v0.1.4",
        download_url=url
        or (
            "https://github.com/owner/repo/releases/download/v0.1.4/ScanReceipts-Setup.exe"
        ),
        page_url="https://github.com/owner/repo/releases/tag/v0.1.4",
    )


def byte_responder(payload: bytes, length: str | None = None):
    class Response(io.BytesIO):
        def __init__(self) -> None:
            super().__init__(payload)
            self.headers = {"Content-Length": length or str(len(payload))}

        def __enter__(self):
            return self

        def __exit__(self, *_exception) -> bool:
            return False

    def open_url(_request, timeout=0):
        return Response()

    return open_url


def test_the_installer_is_written_to_disk_with_progress(
    tmp_path: Path, monkeypatch
) -> None:
    # CLAUDE CODE: gettempdir() caches, so the env var alone would be ignored.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    reported: list[int] = []

    target = download_installer(
        installer_release(), byte_responder(b"installer bytes"), reported.append
    )

    assert target.read_bytes() == b"installer bytes"
    assert target.name == INSTALLER_ASSET
    assert reported[-1] == 100


def test_a_failed_download_leaves_no_partial_file(tmp_path: Path, monkeypatch) -> None:
    # CLAUDE CODE: gettempdir() caches, so the env var alone would be ignored.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def open_url(_request, timeout=0):
        raise OSError("connection reset")

    with pytest.raises(UpdateError):
        download_installer(installer_release(), open_url)

    assert not list(tmp_path.rglob(INSTALLER_ASSET))


def test_a_download_url_outside_github_is_refused(tmp_path: Path, monkeypatch) -> None:
    # CLAUDE CODE: gettempdir() caches, so the env var alone would be ignored.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    with pytest.raises(UpdateError, match=r"evil\.test"):
        download_installer(
            installer_release("https://evil.test/ScanReceipts-Setup.exe"),
            byte_responder(b"payload"),
        )


def test_the_installer_runs_silently_and_is_never_waited_on(tmp_path: Path) -> None:
    installer = tmp_path / INSTALLER_ASSET
    installer.write_bytes(b"stub")
    launched: list[list[str]] = []

    run_installer(installer, popen=lambda command, **_kwargs: launched.append(command))

    assert launched == [[str(installer), *INSTALLER_ARGUMENTS]]
    assert "/SILENT" in INSTALLER_ARGUMENTS


def test_a_missing_installer_is_an_update_error(tmp_path: Path) -> None:
    def popen(_command, **_kwargs):
        raise OSError("not found")

    with pytest.raises(UpdateError):
        run_installer(tmp_path / INSTALLER_ASSET, popen=popen)

from __future__ import annotations

import io
import json

import pytest

from scan_receipts.update import (
    INSTALLER_ASSET,
    ReleaseInfo,
    UpdateError,
    is_newer,
    latest_release,
    parse_version,
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
        headers: dict[str, str] = {}

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

    with pytest.raises(UpdateError, match="example.com"):
        latest_release(responder(json.dumps(payload).encode("utf-8")))

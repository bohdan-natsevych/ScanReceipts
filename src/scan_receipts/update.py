"""Look up and install newer releases published by the project's CI.

Deliberately free of Qt so the network and subprocess behaviour is testable
without a running application.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

GITHUB_REPO = "bohdan-natsevych/ScanReceipts"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"
INSTALLER_ASSET = "ScanReceipts-Setup.exe"
INSTALLER_ARGUMENTS = ("/SILENT", "/NOCANCEL", "/NORESTART", "/SUPPRESSMSGBOXES")
DOWNLOAD_FOLDER = "ScanReceipts-Update"
CHUNK = 256 * 1024
USER_AGENT = "ScanReceipts-Updater"

# CLAUDE CODE: the updater executes what it downloads, so the host is pinned
# rather than trusted from the release payload.
GITHUB_HOSTS = frozenset(
    {
        "github.com",
        "api.github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)

_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


class UpdateError(RuntimeError):
    """A version check, download, or install could not be completed."""


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    tag: str
    download_url: str
    page_url: str


def parse_version(text: str) -> tuple[int, int, int]:
    match = _VERSION.match(text.strip())
    if match is None:
        raise UpdateError(f"Unrecognised version: {text!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def is_newer(candidate: str, installed: str) -> bool:
    return parse_version(candidate) > parse_version(installed)


def require_github_https(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in GITHUB_HOSTS:
        raise UpdateError(f"Refusing to download an installer from {url}")
    return url


def latest_release(open_url=urllib.request.urlopen) -> ReleaseInfo:
    request = urllib.request.Request(
        LATEST_RELEASE_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT},
    )
    try:
        with open_url(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as error:
        raise UpdateError(f"Could not reach GitHub: {error}") from error

    tag = str(payload.get("tag_name") or "")
    if not tag:
        raise UpdateError("The latest release has no tag")
    page_url = str(payload.get("html_url") or RELEASES_PAGE)

    for asset in payload.get("assets") or []:
        if asset.get("name") == INSTALLER_ASSET:
            url = require_github_https(str(asset.get("browser_download_url") or ""))
            return ReleaseInfo(tag.lstrip("v"), tag, url, page_url)

    raise UpdateError(f"Release {tag} does not include {INSTALLER_ASSET}")


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def download_directory() -> Path:
    return Path(tempfile.gettempdir()) / DOWNLOAD_FOLDER


def download_installer(
    release: ReleaseInfo,
    open_url=urllib.request.urlopen,
    progress: Callable[[int], None] | None = None,
) -> Path:
    require_github_https(release.download_url)
    directory = download_directory()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / INSTALLER_ASSET
    request = urllib.request.Request(
        release.download_url, headers={"User-Agent": USER_AGENT}
    )
    try:
        with open_url(request, timeout=30) as response:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            with target.open("wb") as handle:
                while chunk := response.read(CHUNK):
                    handle.write(chunk)
                    received += len(chunk)
                    if progress is not None and total:
                        progress(min(100, int(received * 100 / total)))
    except (OSError, ValueError) as error:
        target.unlink(missing_ok=True)
        raise UpdateError(f"Could not download the installer: {error}") from error
    return target


def run_installer(installer: Path, popen=subprocess.Popen) -> None:
    """Start the installer detached; the caller quits so it can replace the app."""
    try:
        popen([str(installer), *INSTALLER_ARGUMENTS], close_fds=True)
    except OSError as error:
        raise UpdateError(f"Could not start the installer: {error}") from error

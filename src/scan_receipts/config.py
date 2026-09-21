from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import TypeVar

from .models import AppSettings, DetectionSettings, OutputSettings

T = TypeVar("T")


def app_data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / "ScanReceipts"


def default_settings() -> AppSettings:
    documents = Path(os.environ.get("USERPROFILE", Path.home())) / "Documents"
    data = app_data_dir()
    return AppSettings(
        receipt_root=str(documents / "Receipts"),
        video_root=str(data / "Videos"),
    )


def _filtered(cls: type[T], values: dict) -> T:
    names = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in names})


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or app_data_dir() / "settings.json"

    def load(self) -> AppSettings:
        defaults = default_settings()
        if not self.path.exists():
            return defaults
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            output = _filtered(OutputSettings, raw.pop("output", {}))
            detection_raw = raw.pop("detection", {})
            if isinstance(detection_raw.get("exclusion_rect"), list):
                detection_raw["exclusion_rect"] = tuple(detection_raw["exclusion_rect"])
            detection = _filtered(DetectionSettings, detection_raw)
            loaded = _filtered(AppSettings, raw)
            loaded.output = output
            loaded.detection = detection
            return loaded
        except (OSError, ValueError, TypeError):
            return defaults

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(self.path)

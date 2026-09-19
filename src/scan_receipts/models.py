from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class SessionStatus(StrEnum):
    SCANNING = "Scanning"
    PROCESSING = "Processing"
    UNCONFIRMED = "Unconfirmed"
    NEEDS_REVIEW = "Needs review"
    SUCCESSFUL = "Successful"


class DetectorState(StrEnum):
    READY = "READY"
    MOTION = "FLIPPING - MOTION"
    SETTLING = "NEW RECEIPT - SETTLING"
    CAPTURED = "CAPTURED"
    NO_DOCUMENT = "READY - NO RECEIPT"
    MANUAL = "MANUAL ONLY - AUTO OFF"


@dataclass(slots=True)
class OutputSettings:
    format: str = "JPEG"
    jpeg_quality: int = 92
    max_width: int = 2400
    # CLAUDE CODE: 0 means never interpolate. A default that upscaled every
    # receipt to a 1600px long edge cost 2-4x of measured sharpness for no
    # gain - the pixels were invented, and the outputs already cleared the
    # SET-3 byte floor several times over. encode_output still raises
    # resolution when an output genuinely falls under that floor.
    minimum_long_edge: int = 0
    grayscale: bool = False
    crop_margin_percent: float = 2.0
    rotation: int = 0
    perspective_correction: bool = True
    enhancement: bool = True
    sharpening: int = 1
    brightness: int = 0
    contrast: int = 0
    minimum_bytes: int = 50 * 1024
    maximum_bytes: int = 20 * 1024 * 1024


@dataclass(slots=True)
class DetectionSettings:
    analysis_width: int = 800
    buffer_seconds: int = 15
    full_resolution_seconds: float = 3.0
    compressed_buffer_seconds: float = 20.0
    target_analysis_fps: float = 15.0
    stable_seconds: float = 0.30
    minimum_capture_interval: float = 1.25
    motion_threshold: float = 0.24
    content_change_threshold: int = 128
    long_transition_seconds: float = 0.5
    quad_stability_iou: float = 0.80
    min_document_area: float = 0.015
    exclusion_rect: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class AppSettings:
    receipt_root: str
    video_root: str
    recording_mode: str = "until_confirmed"
    video_retention: str = "delete"
    history_cleanup_days: int = 0
    debug_overlay: bool = True
    autofocus: bool = True
    focus_lock: bool = True
    exposure_lock: bool = True
    white_balance_lock: bool = True
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: int = 30
    camera_controls: dict[str, float] = field(default_factory=dict)
    # CLAUDE CODE: "detect" straightens the receipt found in the drawn area;
    # "exact" keeps the drawn rectangle as it is. Enhancement runs either way.
    manual_capture_mode: str = "detect"
    combine_maximum: int = 9
    output: OutputSettings = field(default_factory=OutputSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CameraDescriptor:
    id: str
    name: str
    backend: str
    kind: str = "camera"


@dataclass(slots=True)
class CameraCapability:
    name: str
    supported: bool
    value: float | None = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass(slots=True)
class FramePacket:
    timestamp: float
    frame: Any
    analysis: Any | None = None
    index: int = -1


@dataclass(slots=True)
class ManualCaptureRequest:
    """A user-triggered capture: one full-resolution frame plus the region of
    it the user selected, in relative (x, y, width, height) coordinates."""

    timestamp: float
    frame: Any
    region: tuple[float, float, float, float] | None = None
    exact_area: bool = False


@dataclass(slots=True)
class DetectionMetrics:
    timestamp: float = 0.0
    state: DetectorState = DetectorState.READY
    motion: float = 0.0
    sharpness: float = 0.0
    boundary_confidence: float = 0.0
    visibility: float = 0.0
    clipping: float = 0.0
    score: float = 0.0
    content_score: float = 0.0
    quad_iou: float = 1.0
    occluded: bool = False
    skin: float = 0.0
    focus_settled: bool = True
    corners: Any | None = None
    content_hash: int | None = None
    message: str = ""


@dataclass(slots=True)
class CaptureCandidate:
    timestamp: float
    frame: Any
    corners: Any | None
    boundary_confidence: float
    score: float
    content_hash: int
    quality_flag: str | None = None


@dataclass(slots=True)
class SessionRecord:
    id: str
    started_at: str
    ended_at: str | None
    camera_name: str
    settings_json: str
    receipt_folder: str
    video_path: str | None
    status: SessionStatus
    processing_status: str
    receipt_count: int = 0


@dataclass(slots=True)
class ReceiptRecord:
    id: int
    session_id: str
    sequence: int
    captured_at: float
    filename: str
    original_path: str
    processed_path: str
    edit_json: str
    content_hash: str
    duplicate_group: str | None
    deleted: bool
    quality_flag: str | None = None
    review_flag: bool = False
    combined_into: int | None = None


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

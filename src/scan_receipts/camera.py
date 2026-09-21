from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

try:
    import cv2
except (
    ImportError
):  # A clear startup error is preferable to an obscure attribute error.
    cv2 = None  # type: ignore[assignment]

from .models import AppSettings, CameraCapability, CameraDescriptor, FramePacket

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


class FrameSource(ABC):
    descriptor: CameraDescriptor

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> FramePacket | None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def capabilities(self) -> list[CameraCapability]: ...

    @property
    @abstractmethod
    def fps(self) -> float: ...

    @property
    def is_replay(self) -> bool:
        return False

    # CURSOR: optional hardware hooks, not abstract - sources without camera
    # controls (replay, video file) inherit the no-op deliberately.
    def lock_after_settle(self) -> None:  # noqa: B027
        """Lock auto controls once the scene has settled; default no-op."""

    def boost_low_light(self) -> None:  # noqa: B027
        """Raise exposure on a persistently dark scene; default no-op."""


class SceneSettler:
    """Let AF/AE/AWB converge at session start, then lock them (CAM-5/CAM-6)."""

    def __init__(
        self, source: FrameSource, timeout: float = 3.0, quiet_seconds: float = 1.0
    ) -> None:
        self.source = source
        self.timeout = timeout
        self.quiet_seconds = quiet_seconds
        self._history: list[tuple[float, float, float]] = []
        self._started: float | None = None
        self._done = source.is_replay

    @property
    def settled(self) -> bool:
        return self._done

    def observe(self, packet) -> bool:
        if self._done:
            return True
        require_opencv()
        now = packet.timestamp
        if self._started is None:
            self._started = now
        gray = cv2.cvtColor(
            cv2.resize(packet.frame, (320, 240), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        luma = float(np.median(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self._history.append((now, luma, sharpness))
        self._history = [
            item for item in self._history if item[0] >= now - self.quiet_seconds
        ]
        quiet = False
        if len(self._history) >= 4 and (
            self._history[-1][0] - self._history[0][0] >= self.quiet_seconds * 0.9
        ):
            lumas = [item[1] for item in self._history]
            sharps = [item[2] for item in self._history]
            luma_spread = (max(lumas) - min(lumas)) / max(1.0, max(lumas))
            sharp_spread = (max(sharps) - min(sharps)) / max(1.0, max(sharps))
            quiet = luma_spread <= 0.05 and sharp_spread <= 0.15
        if quiet or now - self._started >= self.timeout:
            self.source.lock_after_settle()
            self._done = True
        return self._done


class LowLightMonitor:
    """Detect a persistently dark scene and trigger a one-time response (CAM-8)."""

    def __init__(
        self, threshold: float = 60.0, checks: int = 3, interval: float = 1.0
    ) -> None:
        self.threshold = threshold
        self.checks = checks
        self.interval = interval
        self._dark_streak = 0
        self._last_check = -1e9
        self._triggered = False

    def observe(self, packet) -> bool:
        if self._triggered or packet.timestamp - self._last_check < self.interval:
            return False
        self._last_check = packet.timestamp
        sampled = packet.frame[::8, ::8]
        luma = float(np.median(sampled))
        if luma < self.threshold:
            self._dark_streak += 1
        else:
            self._dark_streak = 0
        if self._dark_streak >= self.checks:
            self._triggered = True
            return True
        return False


def require_opencv() -> None:
    if cv2 is None:
        raise CameraError("OpenCV is not installed. Run: pip install -e .")


class OpenCVSource(FrameSource):
    _PROPERTIES: ClassVar[dict[str, str]] = {
        "autofocus": "CAP_PROP_AUTOFOCUS",
        "focus": "CAP_PROP_FOCUS",
        "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
        "exposure": "CAP_PROP_EXPOSURE",
        "auto_white_balance": "CAP_PROP_AUTO_WB",
        "white_balance": "CAP_PROP_WB_TEMPERATURE",
        "brightness": "CAP_PROP_BRIGHTNESS",
        "contrast": "CAP_PROP_CONTRAST",
        "zoom": "CAP_PROP_ZOOM",
    }

    def __init__(
        self,
        camera_index: int,
        settings: AppSettings,
        name: str | None = None,
        backend: int | None = None,
    ) -> None:
        self.camera_index = camera_index
        self.settings = settings
        self.backend = backend
        backend_name = _backend_label(backend)
        self.descriptor = CameraDescriptor(
            id=f"camera:{backend or 0}:{camera_index}",
            name=name or f"Camera {camera_index + 1}",
            backend=f"OpenCV / {backend_name}",
        )
        self._capture: Any | None = None
        self._opened_at = 0.0

    def open(self) -> None:
        require_opencv()
        backend = self.backend
        if backend is None:
            backend = cv2.CAP_MSMF if hasattr(cv2, "CAP_MSMF") else cv2.CAP_ANY
        log.info(
            "Opening %s (index %s) at %sx%s @ %s fps",
            self.descriptor.name,
            self.camera_index,
            self.settings.camera_width,
            self.settings.camera_height,
            self.settings.camera_fps,
        )
        capture = cv2.VideoCapture(self.camera_index, backend)
        if not capture.isOpened():
            capture.release()
            log.error("Could not open %s", self.descriptor.name)
            raise CameraError(f"Could not open {self.descriptor.name}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)
        capture.set(cv2.CAP_PROP_FPS, self.settings.camera_fps)
        if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
            capture.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.settings.autofocus else 0)
        self._capture = capture
        self._opened_at = time.monotonic()
        log.info(
            "%s delivers %sx%s",
            self.descriptor.name,
            capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        self._apply_controls()

    def _apply_controls(self) -> None:
        if self._capture is None:
            return
        for name, value in self.settings.camera_controls.items():
            property_name = self._PROPERTIES.get(name)
            if property_name and hasattr(cv2, property_name):
                self._capture.set(getattr(cv2, property_name), value)

    def lock_after_settle(self) -> None:
        if self._capture is None:
            return
        if self.settings.focus_lock and hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
            self._capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if self.settings.exposure_lock and hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
            # CURSOR: MSMF/DirectShow drivers differ (0.25 vs 0); best-effort.
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        if self.settings.white_balance_lock and hasattr(cv2, "CAP_PROP_AUTO_WB"):
            self._capture.set(cv2.CAP_PROP_AUTO_WB, 0)

    def boost_low_light(self) -> None:
        if self._capture is None:
            return
        try:
            from .dshow_controls import control_ranges

            ranges = control_ranges(self.descriptor.name)
        except Exception:
            log.debug("DirectShow control ranges unavailable", exc_info=True)
            ranges = {}
        exposure = ranges.get("exposure")
        if exposure is not None and hasattr(cv2, "CAP_PROP_EXPOSURE"):
            target = min(exposure.maximum, exposure.current + 2 * max(1, exposure.step))
            self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._capture.set(cv2.CAP_PROP_EXPOSURE, float(target))
        elif hasattr(cv2, "CAP_PROP_EXPOSURE"):
            current = self._capture.get(cv2.CAP_PROP_EXPOSURE)
            self._capture.set(cv2.CAP_PROP_EXPOSURE, current + 1)

    def read(self) -> FramePacket | None:
        if self._capture is None:
            raise CameraError("Source is not open")
        ok, frame = self._capture.read()
        if not ok:
            return None
        return FramePacket(timestamp=time.monotonic(), frame=frame)

    def close(self) -> None:
        if self._capture is not None:
            log.info("Closing %s", self.descriptor.name)
            self._capture.release()
            self._capture = None

    @property
    def fps(self) -> float:
        if self._capture is None:
            return float(self.settings.camera_fps)
        measured = self._capture.get(cv2.CAP_PROP_FPS)
        return measured if measured > 0 else float(self.settings.camera_fps)

    def capabilities(self) -> list[CameraCapability]:
        if self._capture is None:
            return []
        capabilities = []
        for name, constant in self._PROPERTIES.items():
            if not hasattr(cv2, constant):
                capabilities.append(CameraCapability(name, False))
                continue
            value = float(self._capture.get(getattr(cv2, constant)))
            capabilities.append(CameraCapability(name, value != -1.0, value=value))
        capabilities.extend(
            [
                CameraCapability(
                    "width", True, self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                ),
                CameraCapability(
                    "height", True, self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                ),
                CameraCapability("fps", True, self._capture.get(cv2.CAP_PROP_FPS)),
            ]
        )
        try:
            from .dshow_controls import control_ranges, merge_capabilities

            capabilities = merge_capabilities(
                capabilities, control_ranges(self.descriptor.name)
            )
        except Exception:
            log.debug("DirectShow capabilities unavailable", exc_info=True)
        return capabilities


class VideoFileSource(FrameSource):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.descriptor = CameraDescriptor(
            id=f"video:{self.path}",
            name=f"Replay: {self.path.name}",
            backend="OpenCV video replay",
            kind="video",
        )
        self._capture: Any | None = None
        self._fps = 30.0
        self._start_wall = 0.0
        self._first_media = 0.0

    @property
    def is_replay(self) -> bool:
        return True

    def open(self) -> None:
        require_opencv()
        log.info("Opening video file %s", self.path)
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            log.error("Could not open video %s", self.path)
            raise CameraError(f"Could not open video: {self.path}")
        self._capture = capture
        self._fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._start_wall = time.monotonic()
        self._first_media = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    def read(self) -> FramePacket | None:
        if self._capture is None:
            raise CameraError("Source is not open")
        ok, frame = self._capture.read()
        if not ok:
            return None
        media_time = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        target = self._start_wall + max(0.0, media_time - self._first_media)
        remaining = target - time.monotonic()
        if remaining > 0:
            time.sleep(min(remaining, 1.0 / max(self._fps, 1.0)))
        return FramePacket(timestamp=time.monotonic(), frame=frame)

    def close(self) -> None:
        if self._capture is not None:
            log.info("Closing video file %s", self.path)
            self._capture.release()
            self._capture = None

    @property
    def fps(self) -> float:
        return self._fps

    def capabilities(self) -> list[CameraCapability]:
        if self._capture is None:
            return []
        return [
            CameraCapability(
                "width", True, self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            ),
            CameraCapability(
                "height", True, self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            ),
            CameraCapability("fps", True, self._fps),
        ]


def _backend_label(backend: int | None) -> str:
    if cv2 is None or backend in (None, getattr(cv2, "CAP_ANY", 0)):
        return "Automatic backend"
    labels = {
        getattr(cv2, "CAP_MSMF", -1): "Media Foundation",
        getattr(cv2, "CAP_DSHOW", -2): "DirectShow",
    }
    return labels.get(backend, f"Backend {backend}")


def _parse_dshow_device_names(output: str) -> list[str]:
    names = []
    for line in output.splitlines():
        match = re.search(r'"([^"]+)" \(video\)\s*$', line.strip())
        if match:
            names.append(match.group(1))
    return names


def windows_camera_names() -> list[str]:
    """Return DirectShow display names when FFmpeg is available."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-list_devices",
                "true",
                "-f",
                "dshow",
                "-i",
                "dummy",
            ],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=8,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return _parse_dshow_device_names(result.stdout + "\n" + result.stderr)
    except (OSError, subprocess.SubprocessError):
        log.warning("Could not enumerate DirectShow devices", exc_info=True)
        return []


def enumerate_cameras(
    settings: AppSettings, maximum: int = 10
) -> list[CameraDescriptor]:
    """Probe Windows camera indexes. No resolution or device-type assumptions are made."""
    if cv2 is None:
        return []
    found: list[CameraDescriptor] = []
    names = windows_camera_names()
    backends = []
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if not backends:
        backends.append(cv2.CAP_ANY)
    logging = getattr(getattr(cv2, "utils", None), "logging", None)
    previous_log_level = logging.getLogLevel() if logging else None
    if logging:
        logging.setLogLevel(logging.LOG_LEVEL_ERROR)
    try:
        for index in range(maximum):
            for backend in dict.fromkeys(backends):
                capture = cv2.VideoCapture(index, backend)
                opened = capture.isOpened()
                capture.release()
                if opened:
                    found.append(
                        CameraDescriptor(
                            id=f"camera:{backend}:{index}",
                            name=names[index]
                            if index < len(names)
                            else f"Camera {index + 1}",
                            backend=f"OpenCV / {_backend_label(backend)}",
                        )
                    )
                    break
    finally:
        if logging and previous_log_level is not None:
            logging.setLogLevel(previous_log_level)
    return found


def descriptor_source(
    descriptor: CameraDescriptor, settings: AppSettings
) -> OpenCVSource:
    """Create a source that retains the backend which discovered the device."""
    parts = descriptor.id.split(":")
    if len(parts) == 3:
        backend, index = int(parts[1]), int(parts[2])
    else:  # Read older descriptors defensively.
        backend, index = None, int(parts[-1])
    return OpenCVSource(index, settings, descriptor.name, backend)


class SegmentedRecorder:
    """Short MJPEG AVI segments keep completed video readable after a crash."""

    def __init__(self, folder: Path, fps: float, segment_seconds: float = 60.0) -> None:
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.fps = max(1.0, fps)
        self.segment_seconds = segment_seconds
        self._writer: Any | None = None
        self._segment_started = 0.0
        existing = []
        for path in self.folder.glob("segment_*.avi"):
            try:
                existing.append(int(path.stem.rsplit("_", 1)[1]))
            except ValueError:
                log.debug("Ignoring unnumbered segment %s", path)
                continue
        self._sequence = max(existing, default=0)
        self.paths: list[Path] = []
        self._index_file = (self.folder / "frame_index.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )
        self._frame_in_segment = 0

    def write(self, packet: FramePacket) -> None:
        require_opencv()
        height, width = packet.frame.shape[:2]
        if (
            self._writer is None
            or packet.timestamp - self._segment_started >= self.segment_seconds
        ):
            self._close_writer()
            self._sequence += 1
            path = self.folder / f"segment_{self._sequence:04d}.avi"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self.fps,
                (width, height),
            )
            if not writer.isOpened():
                writer.release()
                raise CameraError(f"Could not create recording segment: {path}")
            self._writer = writer
            self._segment_started = packet.timestamp
            self._frame_in_segment = 0
            self.paths.append(path)
        self._writer.write(packet.frame)
        self._index_file.write(
            json.dumps(
                {
                    "segment": self._sequence,
                    "frame": self._frame_in_segment,
                    "timestamp": packet.timestamp,
                }
            )
            + "\n"
        )
        self._frame_in_segment += 1

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def close(self) -> None:
        self._close_writer()
        if not self._index_file.closed:
            self._index_file.close()

"""Report what QtMultimedia can get out of a camera that OpenCV cannot.

Every recorded session asked for 1920x1080 and received 1280x720, and the frames
of 2026-09-01 carry roughly a thirtieth of the detail of 2026-08-31 on the same
device. OpenCV cannot say whether a larger format exists, because Media
Foundation silently substitutes one it likes; QtMultimedia enumerates the real
formats instead. This spike answers three questions and writes them to JSON:

  1. Which formats does the device actually advertise?
  2. What does QtMultimedia deliver when the largest is requested - resolution,
     pixel format, frame rate, and measured sharpness?
  3. What does OpenCV deliver for the same device, measured the same way?

Run it with the camera live and a receipt under the lamp:

    python tools/qt_camera_spike.py --seconds 5
    python tools/qt_camera_spike.py --device "My Z Fold7" --frames-dir C:/tmp/qtspike
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QTimer
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import (
    QCamera,
    QCameraDevice,
    QMediaCaptureSession,
    QMediaDevices,
    QVideoFrame,
    QVideoSink,
)


def sharpness(image: np.ndarray) -> float:
    """Laplacian variance - the measure that separated the good sessions from the bad."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def describe_formats(device: QCameraDevice) -> list[dict[str, Any]]:
    formats = []
    for camera_format in device.videoFormats():
        size = camera_format.resolution()
        formats.append(
            {
                "width": size.width(),
                "height": size.height(),
                "pixel_format": camera_format.pixelFormat().name.decode()
                if isinstance(camera_format.pixelFormat().name, bytes)
                else str(camera_format.pixelFormat()),
                "min_fps": camera_format.minFrameRate(),
                "max_fps": camera_format.maxFrameRate(),
            }
        )
    return formats


def largest_format(device: QCameraDevice):
    formats = device.videoFormats()
    if not formats:
        return None
    return max(
        formats,
        key=lambda item: (
            item.resolution().width() * item.resolution().height(),
            item.maxFrameRate(),
        ),
    )


def frame_to_array(frame: QVideoFrame) -> np.ndarray | None:
    image = frame.toImage()
    if image.isNull():
        return None
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height = image.width(), image.height()
    buffer = np.frombuffer(
        image.constBits(), dtype=np.uint8, count=height * image.bytesPerLine()
    )
    rows = buffer.reshape(height, image.bytesPerLine())[:, : width * 3]
    return cv2.cvtColor(rows.reshape(height, width, 3), cv2.COLOR_RGB2BGR)


def probe_qt(
    device: QCameraDevice, seconds: float, frames_dir: Path | None
) -> dict[str, Any]:
    application = QCoreApplication.instance() or QCoreApplication([])
    camera = QCamera(device)
    chosen = largest_format(device)
    if chosen is not None:
        camera.setCameraFormat(chosen)
    sink = QVideoSink()
    session = QMediaCaptureSession()
    session.setCamera(camera)
    session.setVideoSink(sink)

    samples: list[dict[str, Any]] = []
    started_at = time.monotonic()

    def on_frame(frame: QVideoFrame) -> None:
        array = frame_to_array(frame)
        if array is None:
            return
        index = len(samples)
        samples.append(
            {
                "at": time.monotonic() - started_at,
                "width": array.shape[1],
                "height": array.shape[0],
                "sharpness": sharpness(array),
            }
        )
        if frames_dir is not None and index % 15 == 0:
            cv2.imwrite(str(frames_dir / f"qt_{index:04d}.png"), array)

    sink.videoFrameChanged.connect(on_frame)
    camera.start()
    QTimer.singleShot(int(seconds * 1000), application.quit)
    application.exec()
    camera.stop()

    elapsed = max(1e-6, samples[-1]["at"] if samples else seconds)
    values = [item["sharpness"] for item in samples]
    return {
        "requested_format": None
        if chosen is None
        else {
            "width": chosen.resolution().width(),
            "height": chosen.resolution().height(),
            "max_fps": chosen.maxFrameRate(),
        },
        "error": camera.errorString() or None,
        "frames": len(samples),
        "delivered_fps": round(len(samples) / elapsed, 2),
        "delivered_size": None
        if not samples
        else f"{samples[-1]['width']}x{samples[-1]['height']}",
        "sharpness_p50": None if not values else round(float(np.median(values)), 1),
        "sharpness_p90": None
        if not values
        else round(float(np.percentile(values, 90)), 1),
    }


BACKENDS = {
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "any": cv2.CAP_ANY,
}


def probe_opencv(
    index: int,
    seconds: float,
    width: int,
    height: int,
    frames_dir: Path | None,
    backend_name: str,
) -> dict[str, Any]:
    # CLAUDE CODE: indexes are per backend, so this has to match the one the
    # application opens with or the comparison is against a different device.
    capture = cv2.VideoCapture(index, BACKENDS[backend_name])
    if not capture.isOpened():
        return {"error": f"could not open index {index} on {backend_name}"}
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    values: list[float] = []
    started_at = time.monotonic()
    position = 0
    while time.monotonic() - started_at < seconds:
        ok, frame = capture.read()
        if not ok:
            break
        values.append(sharpness(frame))
        if frames_dir is not None and position % 15 == 0:
            cv2.imwrite(str(frames_dir / f"cv_{position:04d}.png"), frame)
        position += 1
    elapsed = max(1e-6, time.monotonic() - started_at)
    result = {
        "backend": backend_name,
        "requested": f"{width}x{height}",
        "delivered_size": f"{int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}"
        f"x{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        "reported_fps": capture.get(cv2.CAP_PROP_FPS),
        "frames": len(values),
        "delivered_fps": round(len(values) / elapsed, 2),
        "sharpness_p50": None if not values else round(float(np.median(values)), 1),
        "sharpness_p90": None
        if not values
        else round(float(np.percentile(values, 90)), 1),
    }
    capture.release()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="substring of the camera description")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--opencv-index", type=int, default=0)
    parser.add_argument(
        "--opencv-backend",
        choices=sorted(BACKENDS),
        default="dshow",
        help="must match the backend the application opens the camera with",
    )
    parser.add_argument("--requested-width", type=int, default=1920)
    parser.add_argument("--requested-height", type=int, default=1080)
    parser.add_argument("--frames-dir", type=Path, help="write sample frames here")
    parser.add_argument("--output", type=Path, default=Path("qt_camera_spike.json"))
    parser.add_argument(
        "--list-only", action="store_true", help="enumerate formats and stop"
    )
    args = parser.parse_args()

    QCoreApplication.instance() or QCoreApplication([])
    devices = QMediaDevices.videoInputs()
    if not devices:
        print("QtMultimedia sees no video inputs")
        return 1

    report: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "devices": [
            {
                "description": device.description(),
                "id": bytes(device.id()).decode(errors="replace"),
                "is_default": device.isDefault(),
                "formats": describe_formats(device),
            }
            for device in devices
        ],
    }

    for entry in report["devices"]:
        sizes = sorted(
            {(item["width"], item["height"]) for item in entry["formats"]},
            reverse=True,
        )
        print(f"{entry['description']}  ({len(entry['formats'])} formats)")
        for width, height in sizes[:8]:
            rates = [
                item["max_fps"]
                for item in entry["formats"]
                if (item["width"], item["height"]) == (width, height)
            ]
            print(f"    {width}x{height} up to {max(rates):.0f} fps")

    if args.list_only:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")
        return 0

    selected = devices[0]
    if args.device:
        matches = [d for d in devices if args.device.lower() in d.description().lower()]
        if not matches:
            print(f"no device matches {args.device!r}")
            return 1
        selected = matches[0]

    frames_dir = args.frames_dir
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nprobing {selected.description()} for {args.seconds}s ...")
    report["qt"] = probe_qt(selected, args.seconds, frames_dir)
    print(json.dumps(report["qt"], indent=2))

    print(f"\nprobing OpenCV index {args.opencv_index} for {args.seconds}s ...")
    report["opencv"] = probe_opencv(
        args.opencv_index,
        args.seconds,
        args.requested_width,
        args.requested_height,
        frames_dir,
        args.opencv_backend,
    )
    print(json.dumps(report["opencv"], indent=2))

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

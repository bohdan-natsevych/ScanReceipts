from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from scan_receipts.camera import FrameSource
from scan_receipts.config import default_settings
from scan_receipts.controller import SessionController
from scan_receipts.database import Repository
from scan_receipts.models import (
    CameraDescriptor,
    DetectionMetrics,
    DetectorState,
    FramePacket,
)
from scan_receipts.workers import CaptureWorker, DetectionWorker, sharpest_packet


class FakeReplaySource(FrameSource):
    descriptor = CameraDescriptor("fake", "Fake camera", "Test", "video")

    def __init__(self) -> None:
        self.index = 0

    def open(self) -> None:
        self.index = 0

    def read(self) -> FramePacket | None:
        if self.index >= 8:
            return None
        frame = np.full((240, 320, 3), 127, np.uint8)
        packet = FramePacket(self.index / self.fps, frame)
        self.index += 1
        return packet

    def close(self) -> None:
        return None

    def capabilities(self) -> list:
        return []

    @property
    def fps(self) -> float:
        return 30.0

    @property
    def is_replay(self) -> bool:
        return True


def test_controller_relays_qimage_preview_signal(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    settings.recording_mode = "never"
    controller = SessionController(Repository(tmp_path / "history.sqlite3"), settings)
    loop = QEventLoop()
    previews = []
    controller.preview_ready.connect(previews.append)
    controller.session_finished.connect(loop.quit)
    QTimer.singleShot(5000, loop.quit)

    controller.start(FakeReplaySource())
    loop.exec()

    assert previews
    assert previews[0].width() == 320
    if controller.active:
        controller.stop()
        app.processEvents()


class RecordingDetector:
    def __init__(self, *_args, **_kwargs) -> None:
        self.timestamps = []

    def feed(self, packet: FramePacket):
        self.timestamps.append(packet.timestamp)
        return DetectionMetrics(timestamp=packet.timestamp), None

    def flush_all(self) -> list:
        return []

    def reset(self) -> None:
        return None


def test_capture_worker_analyzes_preview_handoff_before_new_source_frames() -> None:
    source = FakeReplaySource()
    detector = RecordingDetector()
    initial = [
        FramePacket(-0.4 + index * 0.1, np.full((240, 320, 3), 220, np.uint8))
        for index in range(4)
    ]
    worker = CaptureWorker(source, detector, False, None, initial)

    worker.run()

    assert detector.timestamps[:4] == [packet.timestamp for packet in initial]
    assert detector.timestamps[4:]
    assert all(timestamp >= 0 for timestamp in detector.timestamps[4:])


class FakeLiveSource(FakeReplaySource):
    descriptor = CameraDescriptor("live", "Live camera", "Test", "camera")

    def read(self) -> FramePacket | None:
        time.sleep(0.01)
        value = self.index % 255
        frame = np.full((240, 320, 3), value, np.uint8)
        packet = FramePacket(time.monotonic(), frame)
        self.index += 1
        return packet

    @property
    def is_replay(self) -> bool:
        return False


class ImmediateSettler:
    def __init__(self, _source: FrameSource) -> None:
        pass

    def observe(self, _packet: FramePacket) -> bool:
        return True


def test_live_preview_and_metrics_arrive_before_session_stops(
    tmp_path: Path, monkeypatch
) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr("scan_receipts.controller.ReceiptDetector", RecordingDetector)
    monkeypatch.setattr("scan_receipts.workers.SceneSettler", ImmediateSettler)
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    settings.recording_mode = "never"
    controller = SessionController(Repository(tmp_path / "history.sqlite3"), settings)
    previews = []
    metrics = []
    observed_while_active = {}
    loop = QEventLoop()
    controller.preview_ready.connect(previews.append)
    controller.metrics_ready.connect(metrics.append)
    controller.session_finished.connect(loop.quit)

    def observe_and_stop() -> None:
        observed_while_active["previews"] = len(previews)
        observed_while_active["metrics"] = len(metrics)
        observed_while_active["colors"] = {
            image.pixelColor(0, 0).red() for image in previews
        }
        controller.stop()

    QTimer.singleShot(500, observe_and_stop)
    QTimer.singleShot(5000, loop.quit)
    controller.start(FakeLiveSource())
    loop.exec()

    assert observed_while_active["previews"] >= 2
    assert observed_while_active["metrics"] >= 2
    assert len(observed_while_active["colors"]) >= 2
    if controller.active:
        controller.stop()
        app.processEvents()


class PausableDetector(RecordingDetector):
    """Records the order of the calls the pause path must make."""

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self.events: list[str] = []

    def feed(self, packet: FramePacket):
        self.events.append("feed")
        return super().feed(packet)

    def flush_all(self) -> list:
        self.events.append("flush_all")
        return ["pending-receipt"]

    def reset(self) -> None:
        self.events.append("reset")


def test_capture_worker_feeds_the_detector_nothing_while_auto_capture_is_off() -> None:
    source = FakeReplaySource()
    detector = PausableDetector()
    worker = CaptureWorker(source, detector, False, None)
    previews: list = []
    metrics: list = []
    worker.preview_ready.connect(previews.append)
    worker.metrics_ready.connect(metrics.append)
    worker.set_auto_enabled(False)

    worker.run()

    assert "feed" not in detector.events
    assert previews
    assert {item.state for item in metrics} == {DetectorState.MANUAL}


def test_pausing_detection_emits_the_pending_receipt_before_resetting() -> None:
    detector = PausableDetector()
    worker = DetectionWorker(detector, False)
    candidates: list = []
    worker.candidate_ready.connect(
        candidates.append, Qt.ConnectionType.DirectConnection
    )
    worker.start()

    worker.submit(FramePacket(0.0, np.full((240, 320, 3), 127, np.uint8)))
    worker.pause()
    worker.finish()
    worker.join(5.0)

    assert candidates[:1] == ["pending-receipt"]
    assert detector.events[:3] == ["feed", "flush_all", "reset"]


def sharpness_frame(sharp_half: str) -> np.ndarray:
    frame = np.full((240, 320, 3), 40, np.uint8)
    for x in range(0, 320, 8):
        cv2.line(frame, (x, 0), (x, 239), (230, 230, 230), 2)
    half = frame[:, :160] if sharp_half == "left" else frame[:, 160:]
    other = frame[:, 160:] if sharp_half == "left" else frame[:, :160]
    other[:] = cv2.GaussianBlur(other, (0, 0), 6.0)
    assert half.size
    return frame


def test_sharpest_packet_is_chosen_by_the_selected_region_only() -> None:
    left = FramePacket(0.0, sharpness_frame("left"))
    right = FramePacket(0.1, sharpness_frame("right"))

    assert sharpest_packet([left, right], (0.55, 0.0, 0.45, 1.0)) is right
    assert sharpest_packet([left, right], (0.0, 0.0, 0.45, 1.0)) is left


class ManualTriggerSource(FakeReplaySource):
    def __init__(self) -> None:
        super().__init__()
        self.worker: CaptureWorker | None = None

    def read(self) -> FramePacket | None:
        packet = super().read()
        if packet is not None and self.index == 4 and self.worker is not None:
            self.worker.request_manual_capture((0.25, 0.25, 0.5, 0.5))
        return packet


def test_manual_capture_is_served_from_the_recent_frames_while_auto_is_off() -> None:
    source = ManualTriggerSource()
    worker = CaptureWorker(source, PausableDetector(), False, None)
    source.worker = worker
    requests: list = []
    worker.manual_capture_ready.connect(requests.append)
    worker.set_auto_enabled(False)

    worker.run()

    assert len(requests) == 1
    assert requests[0].region == (0.25, 0.25, 0.5, 0.5)
    assert requests[0].frame.shape == (240, 320, 3)


def test_manual_capture_reaches_the_receipt_folder_while_auto_capture_is_off(
    tmp_path: Path, monkeypatch
) -> None:
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr("scan_receipts.controller.ReceiptDetector", RecordingDetector)
    monkeypatch.setattr("scan_receipts.workers.SceneSettler", ImmediateSettler)
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    settings.recording_mode = "never"
    repository = Repository(tmp_path / "history.sqlite3")
    controller = SessionController(repository, settings)
    loop = QEventLoop()
    controller.receipt_saved.connect(lambda _receipt: loop.quit())
    controller.session_started.connect(
        lambda _session: controller.set_auto_capture(False)
    )
    QTimer.singleShot(300, lambda: controller.request_manual_capture((0.2, 0.2, 0.6, 0.6)))
    QTimer.singleShot(5000, loop.quit)

    controller.start(FakeLiveSource())
    loop.exec()

    assert controller.auto_capture_enabled is False
    receipts = repository.list_receipts(controller.session.id)
    assert len(receipts) == 1
    assert receipts[0].quality_flag == "manual-crop"
    with sqlite3.connect(tmp_path / "history.sqlite3") as connection:
        events = {
            row[0]
            for row in connection.execute(
                "SELECT event_type FROM events WHERE session_id = ?",
                (controller.session.id,),
            )
        }
    assert "auto_capture_disabled" in events
    assert "manual_capture" in events
    if controller.active:
        controller.stop()
        app.processEvents()

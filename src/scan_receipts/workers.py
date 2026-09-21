from __future__ import annotations

import contextlib
import logging
import math
import queue
import threading
import time
import traceback
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage

from .camera import FrameSource, LowLightMonitor, SceneSettler, SegmentedRecorder
from .detection import ReceiptDetector
from .models import (
    CaptureCandidate,
    DetectionMetrics,
    DetectionSettings,
    DetectorState,
    FramePacket,
    ManualCaptureRequest,
    OutputSettings,
)
from .processing import ReceiptProcessor, region_pixels
from .update import (
    ReleaseInfo,
    UpdateError,
    download_installer,
    is_newer,
    latest_release,
)

log = logging.getLogger(__name__)

# CLAUDE CODE: how far back a manual capture may reach for a steadier frame.
# Long enough to outlast the shake of pressing the button, short enough that the
# saved receipt is still the one the user was looking at.
MANUAL_FRAME_WINDOW = 0.5

# CLAUDE CODE: queued out of band so the detection thread runs the flush and the
# reset itself; doing either from the UI thread would race the frame it is
# analysing.
_PAUSE = object()


def sharpest_packet(
    packets: Sequence[FramePacket],
    region: tuple[float, float, float, float] | None,
) -> FramePacket | None:
    """Least motion-blurred of the recent frames, judged inside the selection.

    CLAUDE CODE: scored on a strided sample rather than the full frame - ranking
    fifteen full-resolution frames on the capture thread would stall the
    recorder, and blur costs energy at every scale, so the order survives it.
    """
    best: FramePacket | None = None
    best_score = -1.0
    for packet in packets:
        frame = packet.frame
        x, y, width, height = region_pixels(region, frame.shape[1], frame.shape[0])
        sample = frame[y : y + height : 4, x : x + width : 4]
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) if sample.ndim == 3 else sample
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if score > best_score:
            best, best_score = packet, score
    return best


def frame_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(
        rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
    ).copy()


class SourcePreviewWorker(QObject):
    """Open a selected source for preview only; never detects or records."""

    preview_ready = Signal(QImage)
    packet_ready = Signal(object)
    opened = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, source: FrameSource) -> None:
        super().__init__()
        self.source = source
        # CLAUDE CODE: an Event, not a bool, because stop() is called from the
        # GUI thread and run() from this one. run() used to begin by setting the
        # flag True, which quietly undid a stop that had already arrived while
        # the thread was still starting - the preview then held the camera for
        # the rest of the session.
        self._stop = threading.Event()

    @Slot()
    def run(self) -> None:
        failed_reads = 0
        try:
            if self._stop.is_set():
                log.debug("Preview was stopped before it opened the source")
                return
            self.source.open()
            # CLAUDE CODE: open() takes seconds on a webcam, and a stop asked
            # for during it has to be honoured the moment it returns - before a
            # single frame is read.
            if self._stop.is_set():
                log.debug("Preview was stopped while the source was opening")
                return
            self.opened.emit(self.source.capabilities())
            while not self._stop.is_set():
                packet = self.source.read()
                if packet is None:
                    failed_reads += 1
                    if self.source.is_replay or failed_reads >= 30:
                        break
                    time.sleep(0.02)
                    continue
                failed_reads = 0
                self.packet_ready.emit(packet)
                self.preview_ready.emit(frame_to_qimage(packet.frame))
        except Exception as error:
            log.error("Preview source failed", exc_info=True)
            if not self._stop.is_set():
                self.failed.emit(str(error))
        finally:
            self._stop.set()
            self.source.close()
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._stop.set()


class UpdateWorker(QObject):
    """Checks GitHub and downloads the installer without blocking the UI."""

    up_to_date = Signal(str)
    update_found = Signal(object)
    progress = Signal(int)
    downloaded = Signal(object)
    failed = Signal(str)

    def __init__(self, installed_version: str) -> None:
        super().__init__()
        self.installed_version = installed_version

    @Slot()
    def check(self) -> None:
        try:
            release = latest_release()
            newer = is_newer(release.version, self.installed_version)
        except UpdateError as error:
            self.failed.emit(str(error))
            return
        if newer:
            self.update_found.emit(release)
        else:
            self.up_to_date.emit(self.installed_version)

    @Slot(object)
    def download(self, release: ReleaseInfo) -> None:
        try:
            installer = download_installer(release, progress=self.progress.emit)
        except UpdateError as error:
            self.failed.emit(str(error))
            return
        self.downloaded.emit(installer)


class DetectionWorker(QObject):
    """Runs the detector on its own thread; analysis frames are droppable,
    recorded frames never are (DET-4)."""

    metrics_ready = Signal(object)
    candidate_ready = Signal(object)
    preview_ready = Signal(QImage)

    # CURSOR: a detector that raises on every frame would otherwise freeze the
    # preview and drop every receipt in silence; end the session instead.
    MAX_CONSECUTIVE_FAILURES = 10

    def __init__(self, detector: ReceiptDetector, debug_overlay: bool) -> None:
        super().__init__()
        self.detector = detector
        self.debug_overlay = debug_overlay
        self._queue: queue.Queue[FramePacket | object | None] = queue.Queue(maxsize=4)
        self._pause_requested = threading.Event()
        # CURSOR: the capture loop polls this instead of receiving a signal; it
        # is blocked inside run() for the whole session, so a queued slot call
        # would only arrive once the session it was meant to end is already over.
        self._broken = threading.Event()
        self._failure_message = ""
        self._thread = threading.Thread(
            target=self._run, name="receipt-detection", daemon=True
        )

    @property
    def broken(self) -> bool:
        return self._broken.is_set()

    @property
    def failure_message(self) -> str:
        return self._failure_message

    def start(self) -> None:
        self._thread.start()

    def submit(self, packet: FramePacket) -> None:
        self._put_dropping_oldest(packet)

    def finish(self) -> None:
        self._put_dropping_oldest(None)

    def pause(self) -> None:
        """Commit whatever the detector was holding, then reset it (WF-6)."""
        if self._pause_requested.is_set():
            return
        self._pause_requested.set()
        self._put_dropping_oldest(_PAUSE)

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)

    def _put_dropping_oldest(self, item: FramePacket | object | None) -> None:
        # CURSOR: never block. A dead detection thread leaves a full queue
        # behind, and a blocking put of the stop sentinel would hang the
        # capture thread - and with it the whole session - forever.
        # CLAUDE CODE: only frames are ever evicted here - the capture thread is
        # the sole producer and stops submitting frames the moment it queues a
        # pause, so a control message can never reach the head of the queue.
        while self._thread.is_alive():
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                log.debug("Detection queue full, dropping the oldest frame")
                with contextlib.suppress(queue.Empty):
                    self._queue.get_nowait()

    def _fail(self, message: str) -> None:
        log.error("Detection worker failing the session: %s", message)
        self._failure_message = message
        self._broken.set()

    def _emit_flushed(self, failure: str) -> None:
        try:
            for candidate in self.detector.flush_all():
                self.candidate_ready.emit(candidate)
        except Exception:
            # CURSOR: the end-of-stream candidate is a real receipt; losing it
            # without a word is the silent discard ACC-6 forbids.
            self._fail(f"{failure}\n{traceback.format_exc(limit=3)}")

    def _run(self) -> None:
        consecutive_failures = 0
        analyzed_at = -math.inf
        while True:
            packet = self._queue.get()
            if packet is None:
                break
            if packet is _PAUSE:
                # CLAUDE CODE: a presentation the detector had already committed
                # to is still a receipt; pausing must not swallow it (ACC-6).
                self._emit_flushed(
                    "Detection failed while emitting the receipt held at pause"
                )
                self.detector.reset()
                self._pause_requested.clear()
                consecutive_failures = 0
                analyzed_at = -math.inf
                continue
            try:
                metrics, candidate = self.detector.feed(packet)
            except Exception:
                consecutive_failures += 1
                log.warning(
                    "Detector raised on frame %s (%d in a row)",
                    packet.index,
                    consecutive_failures,
                    exc_info=True,
                )
                if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    self._fail(
                        f"Detection failed on {consecutive_failures} frames in a row\n"
                        f"{traceback.format_exc(limit=3)}"
                    )
                    return
                continue
            # CURSOR: the detector returns its previous metrics for frames it
            # skips to hold the analysis rate. Only a fresh analysis may clear
            # the streak, otherwise every skipped frame masks a broken detector
            # and the limit is never reached at camera rates above the analysis
            # rate.
            if metrics.timestamp > analyzed_at:
                analyzed_at = metrics.timestamp
                consecutive_failures = 0
            self.metrics_ready.emit(metrics)
            preview = packet.frame.copy()
            if self.debug_overlay:
                CaptureWorker._draw_overlay(preview, metrics)
            self.preview_ready.emit(frame_to_qimage(preview))
            if candidate is not None:
                self.candidate_ready.emit(candidate)
        try:
            for candidate in self.detector.flush_all():
                self.candidate_ready.emit(candidate)
        except Exception:
            # CURSOR: the end-of-stream candidate is a real receipt; losing it
            # without a word is the silent discard ACC-6 forbids.
            self._fail(
                f"Detection failed while emitting the final receipt\n"
                f"{traceback.format_exc(limit=3)}"
            )


class CaptureWorker(QObject):
    preview_ready = Signal(QImage)
    metrics_ready = Signal(object)
    candidate_ready = Signal(object)
    opened = Signal(object)
    ended = Signal(bool, str)
    recording_warning = Signal(str)
    low_light = Signal(str)
    manual_capture_ready = Signal(object)

    def __init__(
        self,
        source: FrameSource,
        detector: ReceiptDetector,
        debug_overlay: bool,
        recording_folder: Path | None,
        initial_packets: list[FramePacket] | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.detector = detector
        self.debug_overlay = debug_overlay
        self.recording_folder = recording_folder
        self.initial_packets = initial_packets or []
        self._running = False
        self._auto_enabled = threading.Event()
        self._auto_enabled.set()
        self._manual_lock = threading.Lock()
        self._manual_pending: list[
            tuple[tuple[float, float, float, float] | None, bool]
        ] = []
        self._recent: deque[FramePacket] = deque(maxlen=60)
        self._detection = DetectionWorker(detector, debug_overlay)
        self._detection.metrics_ready.connect(
            self.metrics_ready.emit, Qt.ConnectionType.DirectConnection
        )
        self._detection.candidate_ready.connect(
            self.candidate_ready.emit, Qt.ConnectionType.DirectConnection
        )
        self._detection.preview_ready.connect(
            self.preview_ready.emit, Qt.ConnectionType.DirectConnection
        )

    @Slot()
    def run(self) -> None:
        recorder: SegmentedRecorder | None = None
        detection_started = False
        self._running = True
        failed_reads = 0
        outcome: tuple[bool, str] | None = None
        try:
            self.source.open()
            settler = SceneSettler(self.source)
            low_light_monitor = LowLightMonitor()
            if self.recording_folder is not None and not self.source.is_replay:
                recorder = SegmentedRecorder(self.recording_folder, self.source.fps)
            self.opened.emit(self.source.capabilities())
            self._frame_index = 0
            auto_running = self._auto_enabled.is_set()
            for initial in self.initial_packets:
                packet = FramePacket(
                    initial.timestamp, initial.frame.copy(), index=self._frame_index
                )
                self._frame_index += 1
                if recorder is not None:
                    recorder.write(packet)
                self._remember(packet)
                if not auto_running:
                    self._emit_paused(packet)
                    continue
                metrics, candidate = self.detector.feed(packet)
                self.metrics_ready.emit(metrics)
                preview = packet.frame.copy()
                if self.debug_overlay:
                    self._draw_overlay(preview, metrics)
                self.preview_ready.emit(frame_to_qimage(preview))
                if candidate is not None:
                    self.candidate_ready.emit(candidate)
            self._detection.start()
            detection_started = True
            while self._running:
                if self._detection.broken:
                    # The outcome is read off the detection worker in finally,
                    # so the failure reaches the session exactly once.
                    return
                packet = self.source.read()
                if packet is None:
                    failed_reads += 1
                    if self.source.is_replay:
                        outcome = (False, "Replay completed")
                        return
                    if failed_reads >= 30:
                        outcome = (
                            True,
                            "Camera disconnected or stopped returning frames",
                        )
                        return
                    time.sleep(0.02)
                    continue
                failed_reads = 0
                if recorder is not None:
                    try:
                        recorder.write(packet)
                    except Exception as error:
                        log.error(
                            "Recorder failed, continuing without it", exc_info=True
                        )
                        self.recording_warning.emit(str(error))
                        recorder.close()
                        recorder = None
                if not settler.observe(packet):
                    metrics = DetectionMetrics(
                        timestamp=packet.timestamp,
                        state=DetectorState.SETTLING,
                        message="Camera settling before lock",
                    )
                    preview = packet.frame.copy()
                    if self.debug_overlay:
                        self._draw_overlay(preview, metrics)
                    self.metrics_ready.emit(metrics)
                    self.preview_ready.emit(frame_to_qimage(preview))
                    continue
                # CURSOR: only judge brightness after the settler has locked, so
                # auto-exposure convergence is never mistaken for a dark scene.
                if not self.source.is_replay and low_light_monitor.observe(packet):
                    self.source.boost_low_light()
                    self.low_light.emit(
                        "Scene is dark - exposure raised. "
                        "Add light if captures stay dim."
                    )
                packet = FramePacket(
                    packet.timestamp, packet.frame, index=self._frame_index
                )
                self._frame_index += 1
                self._remember(packet)
                self._serve_manual_requests()
                if self._auto_enabled.is_set():
                    auto_running = True
                    self._detection.submit(packet)
                    continue
                if auto_running:
                    auto_running = False
                    self._detection.pause()
                self._emit_paused(packet)
        except Exception as error:
            log.error("Capture loop failed", exc_info=True)
            outcome = (True, f"{error}\n{traceback.format_exc(limit=3)}")
        finally:
            # CURSOR: the detection thread still holds queued frames and its
            # end-of-stream candidates, so drain it before announcing the end -
            # nothing may reach the session after it is reported finished.
            # CLAUDE CODE: a capture the user asked for before stopping is not
            # the session's to discard (ACC-6).
            self._serve_manual_requests()
            if detection_started:
                self._detection.finish()
                self._detection.join(10.0)
            self._running = False
            if recorder is not None:
                recorder.close()
            self.source.close()
            # CURSOR: a detector that gave up mid-session or while emitting the
            # final receipt outranks a clean end-of-stream outcome.
            if self._detection.broken and (outcome is None or not outcome[0]):
                outcome = (True, self._detection.failure_message)
            if outcome is not None:
                self.ended.emit(*outcome)

    @Slot(bool)
    def set_auto_enabled(self, enabled: bool) -> None:
        """Turn automatic capture on or off mid-session (WF-6)."""
        if enabled:
            self._auto_enabled.set()
        else:
            self._auto_enabled.clear()

    @Slot(object, bool)
    def request_manual_capture(
        self,
        region: tuple[float, float, float, float] | None = None,
        exact_area: bool = False,
    ) -> None:
        """Ask for the next loop pass to hand back a frame to capture (WF-5)."""
        with self._manual_lock:
            self._manual_pending.append((region, exact_area))

    def _remember(self, packet: FramePacket) -> None:
        self._recent.append(packet)
        cutoff = packet.timestamp - MANUAL_FRAME_WINDOW
        while len(self._recent) > 1 and self._recent[0].timestamp < cutoff:
            self._recent.popleft()

    def _serve_manual_requests(self) -> None:
        with self._manual_lock:
            pending = list(self._manual_pending)
            self._manual_pending.clear()
        for region, exact_area in pending:
            chosen = sharpest_packet(self._recent, region)
            if chosen is None:
                continue
            self.manual_capture_ready.emit(
                ManualCaptureRequest(
                    chosen.timestamp, chosen.frame.copy(), region, exact_area
                )
            )

    def _emit_paused(self, packet: FramePacket) -> None:
        metrics = DetectionMetrics(
            timestamp=packet.timestamp,
            state=DetectorState.MANUAL,
            message="Automatic capture is off",
        )
        self.metrics_ready.emit(metrics)
        preview = packet.frame.copy()
        if self.debug_overlay:
            self._draw_overlay(preview, metrics)
        self.preview_ready.emit(frame_to_qimage(preview))

    @staticmethod
    def _draw_overlay(frame: np.ndarray, metrics: DetectionMetrics) -> None:
        if metrics.corners is not None:
            cv2.polylines(
                frame, [np.asarray(metrics.corners, np.int32)], True, (30, 220, 80), 3
            )
        lines = [
            metrics.state.value,
            f"Motion {metrics.motion:.3f}  Stable {'YES' if metrics.motion < 0.035 else 'NO'}",
            f"Sharp {metrics.sharpness:.0f}  Boundary {metrics.boundary_confidence:.2f}",
            f"Visibility {metrics.visibility:.2f}  Score {metrics.score:.2f}",
        ]
        for index, line in enumerate(lines):
            y = 30 + index * 27
            cv2.putText(
                frame,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    @Slot()
    def stop(self) -> None:
        self._running = False


class ProcessingWorker(QObject):
    receipt_saved = Signal(object)
    failed = Signal(str)
    drained = Signal()

    def __init__(
        self,
        processor: ReceiptProcessor,
        session_id: str,
        settings: OutputSettings,
        detection: DetectionSettings | None = None,
    ) -> None:
        super().__init__()
        self.processor = processor
        self.session_id = session_id
        self.settings = settings
        self.detection = detection
        self._queue: queue.Queue[
            CaptureCandidate | ManualCaptureRequest | None
        ] = queue.Queue()
        self._finish_requested = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="receipt-processing", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, candidate: CaptureCandidate) -> None:
        self._queue.put(candidate)

    def enqueue_manual(self, request: ManualCaptureRequest) -> None:
        self._queue.put(request)

    def finish_when_empty(self) -> None:
        self._finish_requested.set()
        self._queue.put(None)

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    if self._finish_requested.is_set() and self._queue.empty():
                        break
                    continue
                if isinstance(item, ManualCaptureRequest):
                    record = self.processor.save_manual(
                        self.session_id, item, self.settings, self.detection
                    )
                else:
                    record = self.processor.save_candidate(
                        self.session_id, item, self.settings
                    )
                self.receipt_saved.emit(record)
            except Exception as error:
                log.error("Could not save a capture", exc_info=True)
                self.failed.emit(f"{error}\n{traceback.format_exc(limit=3)}")
            finally:
                self._queue.task_done()
        log.info("Processing queue drained")
        self.drained.emit()

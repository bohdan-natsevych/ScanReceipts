from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage

from .camera import FrameSource, VideoFileSource
from .database import Repository
from .detection import ReceiptDetector
from .models import (
    AppSettings,
    CaptureCandidate,
    DetectionMetrics,
    FramePacket,
    ManualCaptureRequest,
    SessionRecord,
    SessionStatus,
)
from .processing import ReceiptProcessor
from .workers import CaptureWorker, ProcessingWorker


class SessionController(QObject):
    preview_ready = Signal(object)
    metrics_ready = Signal(object)
    receipt_saved = Signal(object)
    session_started = Signal(object)
    session_finished = Signal(object)
    capabilities_ready = Signal(object)
    error = Signal(str)
    notice = Signal(str)

    def __init__(self, repository: Repository, settings: AppSettings) -> None:
        super().__init__()
        self.repository = repository
        self.settings = settings
        self.session: SessionRecord | None = None
        self._capture_thread: QThread | None = None
        self._capture_worker: CaptureWorker | None = None
        self._processing: ProcessingWorker | None = None
        self._stopping = False
        self._last_state = None
        self._had_error = False
        self._had_flag = False
        self._auto_capture = True

    @property
    def auto_capture_enabled(self) -> bool:
        return self._auto_capture

    @property
    def active(self) -> bool:
        return self.session is not None and self.session.status in {
            SessionStatus.SCANNING,
            SessionStatus.PROCESSING,
        }

    def start(
        self,
        source: FrameSource,
        receipt_folder: str | Path | None = None,
        initial_packets: list[FramePacket] | None = None,
    ) -> None:
        if self.active:
            raise RuntimeError("A session is already active")
        self.session = self.repository.create_session(
            source.descriptor.name, self.settings, receipt_folder
        )
        if isinstance(source, VideoFileSource):
            self.repository.set_video_path(self.session.id, str(source.path))
            self.session.video_path = str(source.path)
        self._launch(source, "session_started", initial_packets)

    def resume(self, session_id: str, source: FrameSource) -> None:
        if self.active:
            raise RuntimeError("A session is already active")
        self.session = self.repository.resume_session(session_id)
        self._launch(source, "session_resumed")

    def _launch(
        self,
        source: FrameSource,
        event_type: str,
        initial_packets: list[FramePacket] | None = None,
    ) -> None:
        if self.session is None:
            raise RuntimeError("No session was allocated")
        processor = ReceiptProcessor(self.repository)
        self._processing = ProcessingWorker(
            processor, self.session.id, self.settings.output, self.settings.detection
        )
        self._processing.receipt_saved.connect(self._on_receipt_saved)
        self._processing.failed.connect(self._on_processing_error)
        self._processing.drained.connect(self._on_drained)

        detector = ReceiptDetector(self.settings.detection, profile="realtime")
        recording_folder = (
            Path(self.session.video_path)
            if self.session.video_path and not isinstance(source, VideoFileSource)
            else None
        )
        worker = CaptureWorker(
            source,
            detector,
            self.settings.debug_overlay,
            recording_folder,
            initial_packets,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.preview_ready.connect(self._on_preview)
        worker.metrics_ready.connect(self._on_metrics)
        worker.candidate_ready.connect(self._on_candidate)
        worker.opened.connect(self._on_opened)
        worker.recording_warning.connect(self._on_recording_warning)
        worker.manual_capture_ready.connect(self._on_manual_capture)
        worker.low_light.connect(self.notice.emit)
        worker.ended.connect(self._on_source_ended)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._capture_stopped)
        worker.set_auto_enabled(self._auto_capture)
        self._capture_worker = worker
        self._capture_thread = thread
        self._stopping = False
        self._had_error = False
        self._had_flag = False
        thread.start()
        self._processing.start()
        self.repository.log_event(
            self.session.id, event_type, {"camera": source.descriptor.name}
        )
        self.session_started.emit(self.session)

    @Slot(QImage)
    def _on_preview(self, image: QImage) -> None:
        self.preview_ready.emit(image)

    @Slot(object)
    def _on_opened(self, capabilities: list) -> None:
        values = {item.name: item.value for item in capabilities}
        if self.session:
            self.repository.update_actual_capture(
                self.session.id,
                int(values.get("width") or 0),
                int(values.get("height") or 0),
                float(values.get("fps") or 0.0),
            )
        self.capabilities_ready.emit(capabilities)

    def stop(self) -> None:
        if not self.session or self._stopping:
            return
        self._stopping = True
        self.repository.update_session_status(
            self.session.id,
            SessionStatus.PROCESSING,
            "Draining processing queue",
            ended=True,
        )
        self.session.status = SessionStatus.PROCESSING
        if self._capture_worker is not None:
            self._capture_worker.stop()
        if self._capture_thread is not None:
            self._capture_thread.quit()
        if self._capture_thread is None or not self._capture_thread.isRunning():
            self._capture_stopped()

    @Slot()
    def _capture_stopped(self) -> None:
        if self._stopping and self._processing is not None:
            self._processing.finish_when_empty()

    @Slot(object)
    def _on_candidate(self, candidate: CaptureCandidate) -> None:
        if not self.session or not self._processing:
            return
        self.repository.log_event(
            self.session.id,
            "candidate_selected",
            {
                "timestamp": candidate.timestamp,
                "score": candidate.score,
                "boundary_confidence": candidate.boundary_confidence,
            },
        )
        self._processing.enqueue(candidate)

    def set_auto_capture(self, enabled: bool) -> None:
        """Switch automatic capture on or off; manual capture is unaffected (WF-6)."""
        if enabled == self._auto_capture:
            return
        self._auto_capture = enabled
        if self._capture_worker is not None:
            self._capture_worker.set_auto_enabled(enabled)
        if self.session:
            self.repository.log_event(
                self.session.id,
                "auto_capture_enabled" if enabled else "auto_capture_disabled",
                {},
            )

    def request_manual_capture(
        self,
        region: tuple[float, float, float, float] | None = None,
        exact_area: bool = False,
    ) -> None:
        """Capture the selected region of the live frame right now (WF-5)."""
        if self._capture_worker is not None:
            self._capture_worker.request_manual_capture(region, exact_area)

    @Slot(object)
    def _on_manual_capture(self, request: ManualCaptureRequest) -> None:
        if not self.session or not self._processing:
            return
        self.repository.log_event(
            self.session.id,
            "manual_capture",
            {
                "timestamp": request.timestamp,
                "region": request.region,
                "exact_area": request.exact_area,
            },
        )
        self._processing.enqueue_manual(request)

    @Slot(object)
    def _on_metrics(self, metrics: DetectionMetrics) -> None:
        if self.session and metrics.state != self._last_state:
            self.repository.log_event(
                self.session.id,
                "detector_state",
                {
                    "state": metrics.state.value,
                    "motion": metrics.motion,
                    "sharpness": metrics.sharpness,
                    "boundary": metrics.boundary_confidence,
                },
            )
            self._last_state = metrics.state
        self.metrics_ready.emit(metrics)

    @Slot(object)
    def _on_receipt_saved(self, receipt: object) -> None:
        if self.session:
            refreshed = self.repository.get_session(self.session.id)
            if refreshed:
                self.session = refreshed
        if getattr(receipt, "quality_flag", None):
            self._had_flag = True
        self.receipt_saved.emit(receipt)

    @Slot(str)
    def _on_processing_error(self, message: str) -> None:
        self._had_error = True
        if self.session:
            self.repository.log_event(
                self.session.id, "processing_error", {"message": message}
            )
            self.repository.update_session_status(
                self.session.id,
                SessionStatus.NEEDS_REVIEW,
                "Processing error - review required",
            )
        self.error.emit(message)

    @Slot()
    def _on_drained(self) -> None:
        if not self.session:
            return
        current = self.repository.get_session(self.session.id)
        flagged_only = self._had_flag and not self._had_error
        target = (
            SessionStatus.NEEDS_REVIEW
            if self._had_error
            or self._had_flag
            or (current and current.status == SessionStatus.NEEDS_REVIEW)
            else SessionStatus.UNCONFIRMED
        )
        self.repository.update_session_status(
            self.session.id,
            target,
            "Complete - low-quality capture(s) to review"
            if flagged_only
            else "Complete",
            ended=True,
        )
        self.session = self.repository.get_session(self.session.id)
        self.repository.log_event(self.session.id, "processing_drained", {})
        self.session_finished.emit(self.session)
        self._cleanup_threads()

    @Slot(bool, str)
    def _on_source_ended(self, disconnected: bool, message: str) -> None:
        if not self.session or self._stopping:
            return
        if disconnected:
            self._had_error = True
            self.repository.update_session_status(
                self.session.id,
                SessionStatus.NEEDS_REVIEW,
                "Camera disconnected - captures and partial video preserved",
                ended=True,
            )
            self.error.emit(message)
        else:
            self.notice.emit(message)
        self.stop()

    @Slot(str)
    def _on_recording_warning(self, message: str) -> None:
        if self.session:
            self.repository.log_event(
                self.session.id, "recording_error", {"message": message}
            )
        self.notice.emit(f"Recording stopped, but receipt capture continues: {message}")

    def _cleanup_threads(self) -> None:
        if self._capture_thread is not None:
            self._capture_thread.quit()
            self._capture_thread.wait(1000)
            self._capture_thread.deleteLater()
        self._capture_thread = None
        self._capture_worker = None
        self._processing = None
        self._stopping = False

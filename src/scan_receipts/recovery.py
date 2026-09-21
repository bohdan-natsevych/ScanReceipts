from __future__ import annotations

import json
import logging
from bisect import bisect_left
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from .database import Repository
from .detection import (
    ReceiptDetector,
    difference_hash,
    perspective_crop,
)
from .models import (
    AppSettings,
    CaptureCandidate,
    FramePacket,
    ReceiptRecord,
    SessionRecord,
)
from .processing import ReceiptProcessor
from .workers import frame_to_qimage

SAME_PRESENTATION_SECONDS = 1.0


log = logging.getLogger(__name__)


def matches_existing_capture(
    candidate: CaptureCandidate,
    existing_hashes: list[int],
    existing_timestamps: list[float],
) -> bool:
    """Match only the same point in the recording, never hash alone.

    Preprinted receipt forms can have very similar structural hashes while
    containing different handwritten values. Suppressing one here would make
    it impossible for the user to recover or review it. Possible visual
    duplicates are deliberately retained and surfaced by the duplicate deck.
    """
    del existing_hashes
    return any(
        abs(candidate.timestamp - timestamp) <= SAME_PRESENTATION_SECONDS
        for timestamp in existing_timestamps
    )


def video_files(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        return [source]
    return sorted(source.glob("segment_*.avi")) if source.is_dir() else []


class VideoTimeline:
    def __init__(self, path: str | Path) -> None:
        self.root = Path(path)
        self.files = video_files(path)
        if not self.files:
            raise FileNotFoundError(f"No readable video found at {path}")
        self.starts: list[int] = []
        self.counts: list[int] = []
        self.fps_values: list[float] = []
        total = 0
        for filename in self.files:
            capture = cv2.VideoCapture(str(filename))
            if not capture.isOpened():
                capture.release()
                continue
            self.starts.append(total)
            count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.counts.append(count)
            self.fps_values.append(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            total += count
            capture.release()
        self.total_frames = total
        self.timestamps = self._read_timestamps()

    def _read_timestamps(self) -> list[float]:
        index = self.root / "frame_index.jsonl" if self.root.is_dir() else Path()
        if not self.root.is_dir() or not index.exists():
            return []
        values = []
        try:
            for line in index.read_text(encoding="utf-8").splitlines():
                values.append(float(json.loads(line)["timestamp"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            log.warning("Frame index %s is unusable", index, exc_info=True)
            return []
        return values if len(values) >= self.total_frames else []

    def read(self, index: int):
        index = max(0, min(index, self.total_frames - 1))
        file_index = max(0, bisect_left(self.starts, index + 1) - 1)
        local = index - self.starts[file_index]
        capture = cv2.VideoCapture(str(self.files[file_index]))
        capture.set(cv2.CAP_PROP_POS_FRAMES, local)
        ok, frame = capture.read()
        capture.release()
        return frame if ok else None

    def timestamp(self, index: int) -> float:
        if self.timestamps:
            return self.timestamps[min(index, len(self.timestamps) - 1)]
        elapsed = 0.0
        for start, count, fps in zip(
            self.starts, self.counts, self.fps_values, strict=True
        ):
            if index < start + count:
                return elapsed + (index - start) / fps
            elapsed += count / fps
        return elapsed

    def frame_for_timestamp(self, timestamp: float) -> int | None:
        if not self.timestamps:
            return None
        position = bisect_left(self.timestamps, timestamp)
        return max(0, min(position, self.total_frames - 1))

    def fps_at(self, index: int) -> float:
        file_index = max(0, bisect_left(self.starts, index + 1) - 1)
        return self.fps_values[file_index]


class VideoViewerDialog(QDialog):
    receipt_added = Signal(object)

    def __init__(
        self,
        session: SessionRecord,
        receipts: list[ReceiptRecord],
        repository: Repository,
        settings: AppSettings,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Session Video Viewer")
        self.resize(980, 720)
        self.session = session
        self.receipts = receipts
        self.repository = repository
        self.settings = settings
        self.timeline = VideoTimeline(session.video_path or "")
        self.position = 0
        self.current_frame = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(800, 540)
        self.preview.setStyleSheet("background:#111;")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(0, self.timeline.total_frames - 1))
        self.slider.valueChanged.connect(self.show_frame)
        self.position_label = QLabel()
        self.play = QPushButton("Play")
        previous = QPushButton("Previous Frame")
        following = QPushButton("Next Frame")
        previous_receipt = QPushButton("Previous Receipt")
        next_receipt = QPushButton("Next Receipt")
        capture = QPushButton("Capture Current Frame as Receipt")
        self.play.clicked.connect(self.toggle_play)
        previous.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        following.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        previous_receipt.clicked.connect(lambda: self.jump_receipt(-1))
        next_receipt.clicked.connect(lambda: self.jump_receipt(1))
        capture.clicked.connect(self.capture_current)
        buttons = QHBoxLayout()
        for widget in (
            self.play,
            previous,
            following,
            previous_receipt,
            next_receipt,
            capture,
        ):
            buttons.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.slider)
        layout.addWidget(self.position_label)
        layout.addLayout(buttons)
        self.show_frame(0)

    def show_frame(self, index: int) -> None:
        frame = self.timeline.read(index)
        if frame is None:
            return
        self.position = index
        self.current_frame = frame
        pixmap = QPixmap.fromImage(frame_to_qimage(frame)).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)
        self.position_label.setText(
            f"Frame {index + 1} of {self.timeline.total_frames}    "
            f"Timestamp {self.timeline.timestamp(index):.3f}s"
        )

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play.setText("Play")
        else:
            interval = round(1000 / max(1.0, self.timeline.fps_at(self.position)))
            self.timer.start(interval)
            self.play.setText("Pause")

    def next_frame(self) -> None:
        if self.position >= self.timeline.total_frames - 1:
            self.toggle_play()
            return
        self.slider.setValue(self.position + 1)

    def jump_receipt(self, direction: int) -> None:
        mapped = sorted(
            frame
            for receipt in self.receipts
            if (frame := self.timeline.frame_for_timestamp(receipt.captured_at))
            is not None
        )
        if not mapped:
            QMessageBox.information(
                self,
                "Receipt timestamps unavailable",
                "Frame-accurate jumps are available for recordings created by this version. "
                "Use the slider for an imported replay video.",
            )
            return
        choices = (
            [frame for frame in mapped if frame > self.position]
            if direction > 0
            else [frame for frame in mapped if frame < self.position]
        )
        if choices:
            self.slider.setValue(min(choices) if direction > 0 else max(choices))

    def capture_current(self) -> None:
        if self.current_frame is None:
            return
        detector = ReceiptDetector(self.settings.detection, profile="offline")
        corners, confidence = detector.detect_document(self.current_frame)
        sample = (
            self.current_frame
            if corners is None
            else perspective_crop(self.current_frame, corners)
        )
        candidate = CaptureCandidate(
            timestamp=self.timeline.timestamp(self.position),
            frame=self.current_frame.copy(),
            corners=corners,
            boundary_confidence=confidence,
            score=1.0,
            content_hash=difference_hash(sample),
        )
        try:
            receipt = ReceiptProcessor(self.repository).save_candidate(
                self.session.id,
                candidate,
                self.settings.output,
            )
            self.receipts.append(receipt)
            self.receipt_added.emit(receipt)
            QMessageBox.information(self, "Receipt captured", receipt.filename)
        except Exception as error:
            log.error("Manual capture from the recording failed", exc_info=True)
            QMessageBox.critical(self, "Could not capture frame", str(error))


class OfflineRecoveryWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        timeline: VideoTimeline,
        settings: AppSettings,
        existing_hashes: list[int],
        existing_timestamps: list[float],
    ) -> None:
        super().__init__()
        self.timeline = timeline
        self.settings = settings
        self.existing_hashes = existing_hashes
        self.existing_timestamps = existing_timestamps
        self.cancelled = False

    @Slot()
    def run(self) -> None:
        detector = ReceiptDetector(self.settings.detection, profile="offline")
        candidates: list[CaptureCandidate] = []
        try:
            index = 0
            for filename in self.timeline.files:
                capture = cv2.VideoCapture(str(filename))
                while not self.cancelled:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    _, candidate = detector.feed(
                        FramePacket(self.timeline.timestamp(index), frame)
                    )
                    if candidate is not None and not matches_existing_capture(
                        candidate,
                        [
                            *self.existing_hashes,
                            *(item.content_hash for item in candidates),
                        ],
                        self.existing_timestamps,
                    ):
                        candidates.append(candidate)
                    if index % 30 == 0:
                        self.progress.emit(index, self.timeline.total_frames)
                    index += 1
                capture.release()
                if self.cancelled:
                    break
            for final_candidate in detector.flush_all():
                if not matches_existing_capture(
                    final_candidate,
                    [
                        *self.existing_hashes,
                        *(item.content_hash for item in candidates),
                    ],
                    self.existing_timestamps,
                ):
                    candidates.append(final_candidate)
            log.info("Offline recovery found %d candidate(s)", len(candidates))
            self.finished.emit(candidates)
        except Exception as error:
            log.error("Offline recovery failed", exc_info=True)
            self.failed.emit(str(error))


class RecoveryDialog(QDialog):
    receipts_added = Signal()

    def __init__(
        self,
        session: SessionRecord,
        repository: Repository,
        settings: AppSettings,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Thorough Offline Recovery")
        self.resize(900, 620)
        self.session = session
        self.repository = repository
        self.settings = settings
        self.candidates: list[CaptureCandidate] = []
        self.status = QLabel(
            "Analyzing the complete recording with the offline detector..."
        )
        self.cards = QListWidget()
        self.cards.setViewMode(QListWidget.ViewMode.IconMode)
        self.cards.setIconSize(QPixmap(180, 180).size())
        self.cards.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        accept = QPushButton("Accept Selected as Receipts")
        reject = QPushButton("Reject Selected")
        close = QPushButton("Close")
        accept.clicked.connect(self.accept_selected)
        reject.clicked.connect(self.reject_selected)
        close.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addWidget(accept)
        buttons.addWidget(reject)
        buttons.addStretch()
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.cards, 1)
        layout.addLayout(buttons)
        self._thread = QThread(self)
        existing_receipts = repository.list_receipts(session.id)
        existing_hashes = [
            int(item.content_hash, 16)
            for item in existing_receipts
            if item.content_hash
        ]
        self._worker = OfflineRecoveryWorker(
            VideoTimeline(session.video_path or ""),
            settings,
            existing_hashes,
            [item.captured_at for item in existing_receipts],
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(
            lambda done, total: self.status.setText(
                f"Analyzing frame {done:,} of {total:,}..."
            )
        )
        self._worker.finished.connect(self.show_candidates)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self.failed)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

    @Slot(object)
    def show_candidates(self, candidates: list[CaptureCandidate]) -> None:
        self.candidates = candidates
        self.status.setText(
            f"{len(candidates)} possible missing receipt(s). Nothing is added until you accept it."
        )
        for index, candidate in enumerate(candidates):
            item = QListWidgetItem(f"Possible missing receipt {index + 1}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setIcon(QIcon(QPixmap.fromImage(frame_to_qimage(candidate.frame))))
            self.cards.addItem(item)

    @Slot(str)
    def failed(self, message: str) -> None:
        self.status.setText("Recovery failed")
        QMessageBox.critical(self, "Offline recovery failed", message)

    def accept_selected(self) -> None:
        processor = ReceiptProcessor(self.repository)
        accepted = []
        for item in self.cards.selectedItems():
            candidate = self.candidates[item.data(Qt.ItemDataRole.UserRole)]
            try:
                processor.save_candidate(
                    self.session.id, candidate, self.settings.output
                )
                accepted.append(item)
            except Exception as error:
                log.error("Could not accept a recovered candidate", exc_info=True)
                QMessageBox.critical(self, "Could not accept candidate", str(error))
        for item in accepted:
            self.cards.takeItem(self.cards.row(item))
        if accepted:
            self.receipts_added.emit()

    def reject_selected(self) -> None:
        for item in self.cards.selectedItems():
            self.cards.takeItem(self.cards.row(item))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._thread.isRunning():
            self._worker.cancelled = True
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)

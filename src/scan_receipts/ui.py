from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QItemSelectionModel,
    QObject,
    QPoint,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QIcon,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from send2trash import send2trash

from .camera import VideoFileSource, descriptor_source, enumerate_cameras
from .config import SettingsStore
from .controller import SessionController
from .database import Repository
from .models import (
    AppSettings,
    CameraDescriptor,
    FramePacket,
    ReceiptRecord,
    SessionRecord,
    SessionStatus,
)
from .processing import ReceiptProcessor, combined_sources, preview_path
from .recovery import RecoveryDialog, VideoViewerDialog
from .storage import delete_session_video
from .update import ReleaseInfo, is_frozen, run_installer
from .version import APP_VERSION
from .workers import SourcePreviewWorker, UpdateWorker


def reveal(path: str | Path) -> None:
    path = Path(path)
    target = path if path.is_dir() else path.parent
    os.startfile(str(target))  # type: ignore[attr-defined]


def copy_icon() -> QIcon:
    pixmap = QPixmap(18, 18)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#7d8790"), 1.5))
    painter.drawRoundedRect(QRect(6, 2, 9, 11), 1, 1)
    painter.drawRoundedRect(QRect(3, 5, 9, 11), 1, 1)
    painter.end()
    return QIcon(pixmap)


def copied_icon() -> QIcon:
    pixmap = QPixmap(18, 18)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#43c463"), 2.4))
    painter.drawLine(3, 9, 7, 13)
    painter.drawLine(7, 13, 15, 4)
    painter.end()
    return QIcon(pixmap)


def combined_suffix(repository: Repository, receipt: ReceiptRecord) -> str:
    """Names the receipts a combined sheet would take with it when deleted."""
    members = repository.list_combined_members(receipt.id)
    return f" and the {len(members)} receipts it combines" if members else ""


def duplicate_deck_icon(receipts: list[ReceiptRecord]) -> QIcon:
    canvas = QPixmap(170, 112)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    shown = receipts[:3]
    for index, receipt in enumerate(reversed(shown)):
        image = QPixmap(receipt.processed_path)
        card = image.scaled(
            78,
            98,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = 12 + (len(shown) - index - 1) * 17
        y = 7 + index * 2
        painter.fillRect(
            QRect(x - 2, y - 2, card.width() + 4, card.height() + 4), QColor("white")
        )
        painter.drawPixmap(x, y, card)
    painter.setBrush(QColor("#d83b3b"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(QRect(128, 5, 34, 34))
    painter.setPen(QColor("white"))
    painter.drawText(
        QRect(128, 5, 34, 34),
        Qt.AlignmentFlag.AlignCenter,
        str(len(receipts)),
    )
    painter.end()
    return QIcon(canvas)


class SmoothScroll(QObject):
    """Glides a scroll area to the wheel's target instead of jumping to it.

    A wheel notch is a single large step, so even pixel-scrolling views land in
    visible lurches. Animating the scroll bar between the current value and the
    accumulated target is what makes the movement read as continuous.
    """

    WHEEL_PIXELS = 110
    DURATION_MS = 190

    def __init__(self, area: QAbstractScrollArea) -> None:
        super().__init__(area)
        self.area = area
        self._target: int | None = None
        self._animation = QPropertyAnimation(self)
        self._animation.setPropertyName(b"value")
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.setDuration(self.DURATION_MS)
        area.viewport().installEventFilter(self)

    def active_bar(self) -> QScrollBar:
        """The bar a wheel notch should move: vertical unless only sideways room."""
        vertical = self.area.verticalScrollBar()
        if vertical.maximum() > vertical.minimum():
            return vertical
        return self.area.horizontalScrollBar()

    def scroll_by(self, pixels: int) -> None:
        bar = self.active_bar()
        running = (
            self._animation.state() == QAbstractAnimation.State.Running
            and self._animation.targetObject() is bar
            and self._target is not None
        )
        # CLAUDE CODE: a second notch mid-glide extends the pending target rather
        # than restarting from where the animation happens to be.
        base = self._target if running else bar.value()
        self._target = max(bar.minimum(), min(bar.maximum(), base + pixels))
        self._animation.stop()
        self._animation.setTargetObject(bar)
        self._animation.setStartValue(bar.value())
        self._animation.setEndValue(self._target)
        self._animation.start()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Type.Wheel:
            return super().eventFilter(watched, event)
        # CLAUDE CODE: touchpads and Ctrl+wheel zoom already deliver their own
        # fine-grained deltas; only notched wheels need the animation.
        if not event.pixelDelta().isNull():
            return False
        if event.modifiers() != Qt.KeyboardModifier.NoModifier:
            return False
        notches = event.angleDelta().y() / 120
        bar = self.active_bar()
        if not notches or bar.maximum() <= bar.minimum():
            return False
        self.scroll_by(int(-notches * self.WHEEL_PIXELS))
        event.accept()
        return True


def install_smooth_scroll(area: QAbstractScrollArea, step: int = 16) -> SmoothScroll:
    """Pixel scrolling plus animated wheel movement for one scrollable widget."""
    if isinstance(area, QAbstractItemView):
        area.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        area.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    for bar in (area.verticalScrollBar(), area.horizontalScrollBar()):
        bar.setSingleStep(step)
        bar.setPageStep(step * 10)
    return SmoothScroll(area)


class DuplicateDeckDialog(QDialog):
    changed = Signal()

    def __init__(
        self,
        repository: Repository,
        group: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repository = repository
        self.group = group
        self.selected_receipt: ReceiptRecord | None = None
        self.card_buttons: dict[int, QToolButton] = {}
        self.setWindowTitle("Review possible duplicates")
        self.setMinimumSize(900, 620)
        self.resize(1200, 760)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.cards = QWidget()
        self.cards_layout = QHBoxLayout(self.cards)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.cards)
        self.smooth_scroll = install_smooth_scroll(scroll)
        self.selection = QLabel("Click a receipt card to select the copy to save.")
        self.keep_selected_button = QPushButton(
            "Save selected receipt and remove other copies"
        )
        self.keep_selected_button.setEnabled(False)
        self.keep_selected_button.clicked.connect(self.keep_selected)
        self.delete_selected_button = QPushButton("Delete selected copy")
        self.delete_selected_button.setEnabled(False)
        self.delete_selected_button.clicked.connect(self.delete_selected)
        self.keep_all_button = QPushButton("Keep all - these are not duplicates")
        self.keep_all_button.clicked.connect(self.mark_all_independent)
        self.delete_all_button = QPushButton("Remove all copies")
        self.delete_all_button.setToolTip(
            "Send every receipt in this group to the Recycle Bin"
        )
        self.delete_all_button.clicked.connect(self.delete_all)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        actions = QHBoxLayout()
        actions.addWidget(self.selection, 1)
        actions.addWidget(self.keep_all_button)
        actions.addWidget(self.delete_all_button)
        actions.addWidget(self.delete_selected_button)
        actions.addWidget(self.keep_selected_button)
        actions.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(scroll, 1)
        layout.addLayout(actions)
        self.reload()

    def group_receipts(self) -> list[ReceiptRecord]:
        return [
            receipt
            for session in self.repository.list_sessions()
            for receipt in self.repository.list_receipts(session.id)
            if receipt.duplicate_group == self.group
        ]

    def reload(self) -> None:
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        receipts = self.group_receipts()
        self.selected_receipt = None
        self.card_buttons.clear()
        self.selection.setText("Click a receipt card to select the copy to save.")
        self.keep_selected_button.setEnabled(False)
        self.delete_selected_button.setEnabled(False)
        if len(receipts) < 2:
            self.accept()
            return
        self.summary.setText(
            f"{len(receipts)} receipts may be duplicates. Click one complete card to "
            "select it; the blue outline shows which copy will be saved. Nothing is "
            "removed until you press the save button and confirm."
        )
        for receipt in receipts:
            card = QToolButton(self.cards)
            card.setCheckable(True)
            card.setAutoExclusive(True)
            card.setMinimumWidth(310)
            card.setMaximumWidth(360)
            card.setMinimumHeight(540)
            card.setText(receipt.filename)
            card.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            card.setIconSize(QSize(300, 470))
            card.setCursor(Qt.CursorShape.PointingHandCursor)
            card.setAccessibleName(f"Select {receipt.filename} to save")
            card.setToolTip(f"Select {receipt.filename} as the copy to save")
            card.setStyleSheet(
                "QToolButton { border: 2px solid #555; border-radius: 6px; "
                "padding: 8px; background: #242424; }"
                "QToolButton:hover { border-color: #8aa9c7; background: #2b3035; }"
                "QToolButton:checked { border: 4px solid #1686d9; "
                "background: #17344d; color: white; }"
            )
            pixmap = QPixmap(receipt.processed_path)
            if not pixmap.isNull():
                card.setIcon(
                    QIcon(
                        pixmap.scaled(
                            300,
                            470,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                )
            card.clicked.connect(
                lambda _checked=False, chosen=receipt: self.select_card(chosen)
            )
            self.card_buttons[receipt.id] = card
            self.cards_layout.addWidget(card)
        self.cards_layout.addStretch()

    def select_card(self, receipt: ReceiptRecord) -> None:
        self.selected_receipt = receipt
        for receipt_id, card in self.card_buttons.items():
            card.setChecked(receipt_id == receipt.id)
        self.selection.setText(f"Selected to save: {receipt.filename}")
        self.keep_selected_button.setEnabled(True)
        self.delete_selected_button.setEnabled(True)

    def keep_selected(self) -> None:
        if self.selected_receipt is not None:
            self.keep_only(self.selected_receipt)

    def delete_selected(self) -> None:
        if self.selected_receipt is not None:
            self.delete_one(self.selected_receipt)

    def delete_all(self) -> None:
        receipts = self.group_receipts()
        if (
            QMessageBox.question(
                self,
                "Remove every copy",
                f"Send all {len(receipts)} copies and their preserved originals to "
                "the Recycle Bin? This can be undone from the Recycle Bin.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        for receipt in receipts:
            self.trash_receipt(self.repository, receipt)
        self.changed.emit()
        self.accept()

    def mark_all_independent(self) -> None:
        for receipt in self.group_receipts():
            self.repository.resolve_duplicate_member(receipt.id)
        self.changed.emit()
        self.accept()

    @staticmethod
    def trash_receipt(repository: Repository, receipt: ReceiptRecord) -> None:
        ReceiptProcessor(repository).trash(receipt)

    def keep_only(self, chosen: ReceiptRecord) -> None:
        others = [item for item in self.group_receipts() if item.id != chosen.id]
        if (
            QMessageBox.question(
                self,
                "Save this receipt and remove other copies?",
                f"Save {chosen.filename} and send the other {len(others)} copy/copies "
                "to the Recycle Bin? This can be undone from the Recycle Bin.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        for receipt in others:
            self.trash_receipt(self.repository, receipt)
        self.changed.emit()
        self.accept()

    def delete_one(self, receipt: ReceiptRecord) -> None:
        if (
            QMessageBox.question(
                self,
                "Delete duplicate copy",
                f"Send {receipt.filename} and its preserved original to the Recycle Bin?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.trash_receipt(self.repository, receipt)
        self.changed.emit()
        self.reload()

    def mark_independent(self, receipt: ReceiptRecord) -> None:
        self.repository.resolve_duplicate_member(receipt.id)
        self.changed.emit()
        self.reload()


class PreviewLabel(QLabel):
    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 240)
        self.setStyleSheet("background:#17191c;color:#aeb5bd;border-radius:6px;")
        self._image: QImage | None = None

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._render()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._image is not None:
            pixmap = QPixmap.fromImage(self._image).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(pixmap)


class SelectableLabel(PreviewLabel):
    """A preview the user can drag a rectangle on, reported in source pixels."""

    selection_changed = Signal(object)

    def __init__(self, placeholder: str) -> None:
        super().__init__(placeholder)
        self._origin: QPoint | None = None
        self._selection = QRect()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.pixmap() is not None:
            self._origin = event.position().toPoint()
            self._selection = QRect(self._origin, QSize())
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._origin is not None:
            self._selection = QRect(
                self._origin, event.position().toPoint()
            ).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._origin is not None:
            self._selection = QRect(
                self._origin, event.position().toPoint()
            ).normalized()
            self._origin = None
            crop = self.source_crop()
            if crop:
                self.selection_changed.emit(crop)
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if not self._selection.isEmpty():
            painter = QPainter(self)
            painter.setPen(QPen(Qt.GlobalColor.cyan, 2, Qt.PenStyle.DashLine))
            painter.drawRect(self._selection)

    def source_crop(self) -> list[int] | None:
        pixmap = self.pixmap()
        if (
            self._image is None
            or pixmap is None
            or self._selection.width() < 8
            or self._selection.height() < 8
        ):
            return None
        left = (self.width() - pixmap.width()) // 2
        top = (self.height() - pixmap.height()) // 2
        visible = self._selection.intersected(
            QRect(left, top, pixmap.width(), pixmap.height())
        )
        if visible.isEmpty():
            return None
        sx = self._image.width() / pixmap.width()
        sy = self._image.height() / pixmap.height()
        return [
            round((visible.x() - left) * sx),
            round((visible.y() - top) * sy),
            round(visible.width() * sx),
            round(visible.height() * sy),
        ]

    def source_crop_relative(self) -> list[float] | None:
        crop = self.source_crop()
        if crop is None or self._image is None:
            return None
        return [
            crop[0] / self._image.width(),
            crop[1] / self._image.height(),
            crop[2] / self._image.width(),
            crop[3] / self._image.height(),
        ]

    def clear_crop(self) -> None:
        self._selection = QRect()
        self.update()


class LivePreviewLabel(SelectableLabel):
    """The scanning preview. Unlike review, the selection outlives every new
    frame so one chosen area serves a whole stack of manual captures."""

    def resizeEvent(self, event) -> None:  # noqa: N802
        # CLAUDE CODE: the rectangle is held in widget pixels, so a resize would
        # silently move it over a different part of the receipt.
        self.clear_crop()
        super().resizeEvent(event)


class CropLabel(SelectableLabel):
    def __init__(self) -> None:
        super().__init__("Select a receipt to review")
        self.setMinimumSize(1, 1)
        self._viewport_size = QSize(420, 300)
        self._zoom = 1.0

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._zoom = 1.0
        self.clear_crop()
        self._render()

    def set_viewport_size(self, size: QSize) -> None:
        self._viewport_size = QSize(max(1, size.width()), max(1, size.height()))
        self._render()

    def zoom_by(self, multiplier: float) -> int:
        self._zoom = min(6.0, max(0.25, self._zoom * multiplier))
        self.clear_crop()
        self._render()
        return round(self._zoom * 100)

    def reset_zoom(self) -> int:
        self._zoom = 1.0
        self.clear_crop()
        self._render()
        return 100

    def resizeEvent(self, event) -> None:  # noqa: N802
        QLabel.resizeEvent(self, event)

    def _render(self) -> None:
        if self._image is None:
            return
        fitted = self._image.size().scaled(
            self._viewport_size,
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        target = QSize(
            max(1, round(fitted.width() * self._zoom)),
            max(1, round(fitted.height() * self._zoom)),
        )
        pixmap = QPixmap.fromImage(self._image).scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)
        self.resize(pixmap.size())


class ImageScrollArea(QScrollArea):
    def __init__(self, image: CropLabel) -> None:
        super().__init__()
        self.image = image
        self.setWidget(image)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 220)
        # CLAUDE CODE: the default single step is one pixel, so a drag on the bar
        # crawls; the wheel is animated on top of that.
        self.smooth_scroll = install_smooth_scroll(self, step=24)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.image.set_viewport_size(self.viewport().size())


class CaptureStrip(QListWidget):
    """Newest-first cards for the receipts captured in the running session."""

    flag_toggled = Signal(int, bool)
    delete_requested = Signal(int)
    enlarge_requested = Signal(int)
    combine_selection_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setViewMode(QListWidget.ViewMode.ListMode)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setSpacing(4)
        self.setFixedWidth(250)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.smooth_scroll = install_smooth_scroll(self)
        self.preview_buttons: dict[int, QToolButton] = {}
        self.flag_buttons: dict[int, QToolButton] = {}
        self.delete_buttons: dict[int, QToolButton] = {}
        self.combine_buttons: dict[int, QToolButton] = {}
        self._thumbnails: dict[int, QPixmap] = {}
        self._combine_ticks: set[int] = set()
        self._order: list[int] = []
        self._placeholder = "Captured receipts appear here"
        self.set_receipts([])

    def clear_captures(self) -> None:
        self._thumbnails.clear()
        self._combine_ticks.clear()
        self.set_receipts([])

    def combine_selection(self) -> list[int]:
        """Ticked receipt ids in capture order, so the sheet reads as scanned."""
        return [item for item in self._order if item in self._combine_ticks]

    def _combine_toggled(self, receipt_id: int, checked: bool) -> None:
        if checked:
            self._combine_ticks.add(receipt_id)
        else:
            self._combine_ticks.discard(receipt_id)
        self.combine_selection_changed.emit()

    def set_receipts(self, receipts: list[ReceiptRecord]) -> None:
        self.clear()
        self.preview_buttons.clear()
        self.flag_buttons.clear()
        self.delete_buttons.clear()
        self.combine_buttons.clear()
        self._order = [receipt.id for receipt in receipts]
        self._combine_ticks &= set(self._order)
        if not receipts:
            item = QListWidgetItem(self._placeholder)
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            item.setForeground(QColor("#aeb5bd"))
            self.addItem(item)
            return
        for receipt in reversed(receipts):
            # CLAUDE CODE: the view paints item text under the card widget, and the
            # card is transparent wherever a small thumbnail leaves room, so the
            # filename must live on the card alone.
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, receipt.id)
            item.setData(Qt.ItemDataRole.ToolTipRole, receipt.filename)
            card = self._card(receipt)
            item.setSizeHint(card.sizeHint())
            self.addItem(item)
            self.setItemWidget(item, card)

    def _card(self, receipt: ReceiptRecord) -> QWidget:
        card = QWidget(self)
        preview = QToolButton(card)
        preview.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        preview.setIconSize(QSize(190, 190))
        preview.setCursor(Qt.CursorShape.PointingHandCursor)
        preview.setText(receipt.filename)
        preview.setToolTip(f"Show {receipt.filename} full size")
        preview.setAccessibleName(f"Show {receipt.filename} full size")
        thumbnail = self._thumbnail(receipt)
        if not thumbnail.isNull():
            preview.setIcon(QIcon(thumbnail))
        preview.clicked.connect(
            lambda _checked=False, receipt_id=receipt.id: self.enlarge_requested.emit(
                receipt_id
            )
        )
        flag = QToolButton(card)
        flag.setCheckable(True)
        flag.setChecked(receipt.review_flag)
        flag.setText("Flagged" if receipt.review_flag else "Flag")
        flag.setToolTip("Flag this receipt to check during session review")
        flag.setAccessibleName(f"Flag {receipt.filename} for review")
        flag.toggled.connect(
            lambda checked, receipt_id=receipt.id: self.flag_toggled.emit(
                receipt_id, checked
            )
        )
        combine = QToolButton(card)
        combine.setCheckable(True)
        combine.setText("Combine")
        combine.setChecked(receipt.id in self._combine_ticks)
        combine.setToolTip("Tick to include this capture in a combined sheet")
        combine.setAccessibleName(f"Combine {receipt.filename}")
        combine.toggled.connect(
            lambda checked, receipt_id=receipt.id: self._combine_toggled(
                receipt_id, checked
            )
        )
        delete = QToolButton(card)
        delete.setText("Delete")
        delete.setToolTip("Send this capture to the Recycle Bin")
        delete.setAccessibleName(f"Delete {receipt.filename}")
        delete.clicked.connect(
            lambda _checked=False, receipt_id=receipt.id: self.delete_requested.emit(
                receipt_id
            )
        )
        buttons = QHBoxLayout()
        buttons.addWidget(flag)
        buttons.addWidget(combine)
        buttons.addWidget(delete)
        buttons.addStretch()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(preview)
        layout.addLayout(buttons)
        card.setStyleSheet(self._card_style(receipt))
        self.preview_buttons[receipt.id] = preview
        self.flag_buttons[receipt.id] = flag
        self.delete_buttons[receipt.id] = delete
        self.combine_buttons[receipt.id] = combine
        return card

    @staticmethod
    def _card_style(receipt: ReceiptRecord) -> str:
        if receipt.review_flag:
            return "QWidget { border:2px solid #1686d9; border-radius:6px; background:#17344d; }"
        if receipt.duplicate_group:
            return "QWidget { border:2px solid #a67c00; border-radius:6px; background:#5a4300; }"
        return "QWidget { border:1px solid #3a3f45; border-radius:6px; }"

    def _thumbnail(self, receipt: ReceiptRecord) -> QPixmap:
        cached = self._thumbnails.get(receipt.id)
        if cached is not None:
            return cached
        pixmap = QPixmap(receipt.processed_path)
        if pixmap.isNull():
            pixmap = QPixmap(str(preview_path(receipt.original_path)))
        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                QSize(190, 190),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._thumbnails[receipt.id] = pixmap
        return pixmap


class ScanPage(QWidget):
    capabilities_ready = Signal(object)
    settings_changed = Signal()
    request_download = Signal(object)

    IDLE_COUNTER = "Receipts captured: 0"
    IDLE_STATE = "READY"
    IDLE_DIAGNOSTICS = "Motion --  |  Sharpness --  |  Boundary --"

    def __init__(self, controller: SessionController, settings: AppSettings) -> None:
        super().__init__()
        self.controller = controller
        self.settings = settings
        self._sources: list[CameraDescriptor] = []
        self._video_path: Path | None = None
        self._resume_session_id: str | None = None
        self._session_folder: Path | None = None
        self._preview_thread: QThread | None = None
        self._preview_worker: SourcePreviewWorker | None = None
        self._update_thread: QThread | None = None
        self._update_worker: UpdateWorker | None = None
        self._update_progress_dialog: QProgressDialog | None = None
        self._preview_packets: deque[FramePacket] = deque()
        self._session_id: str | None = None

        self.source_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh cameras")
        self.video_button = QPushButton("Video file...")
        self.folder_button = QPushButton("Choose session folder...")
        self.clear_folder_button = QPushButton("Use automatic folder")
        self.folder_value = QLabel()
        self.folder_value.setWordWrap(True)
        self.start_button = QPushButton("Start Session")
        self.stop_button = QPushButton("Stop Session")
        self.stop_button.setEnabled(False)
        self.version_label = QLabel(f"v{APP_VERSION}")
        self.version_label.setStyleSheet("color:#666;")
        self.update_button = QPushButton("Check for updates")
        self.update_button.setToolTip(
            "Ask GitHub whether a newer version has been published."
        )
        self.auto_check = QCheckBox("Automatic capture")
        self.auto_check.setChecked(controller.auto_capture_enabled)
        self.auto_check.setToolTip(
            "Turn off to stop detecting receipts by itself and capture only "
            "with the button below."
        )
        self.capture_button = QPushButton("Capture Area")
        self.capture_button.setShortcut(QKeySequence("Ctrl+Return"))
        self.capture_button.setToolTip(
            "Capture the selected area now (Ctrl+Return). Drag on the preview to "
            "choose an area; with none chosen the whole frame is captured."
        )
        self.clear_selection_button = QPushButton("Clear area")
        for button in (self.capture_button, self.clear_selection_button):
            button.setEnabled(False)
        self.detect_area = QRadioButton("Detect + straighten")
        self.detect_area.setToolTip(
            "Find the receipt inside the drawn area and correct its perspective."
        )
        self.exact_area = QRadioButton("Exact area")
        self.exact_area.setToolTip(
            "Keep the drawn rectangle exactly as it is. Image enhancement still runs."
        )
        self.area_mode = QButtonGroup(self)
        self.area_mode.addButton(self.detect_area)
        self.area_mode.addButton(self.exact_area)
        self.exact_area.setChecked(settings.manual_capture_mode == "exact")
        self.detect_area.setChecked(not self.exact_area.isChecked())
        self.counter = QLabel(self.IDLE_COUNTER)
        self.counter.setStyleSheet("font-size:24px;font-weight:600;")
        self.state = QLabel(self.IDLE_STATE)
        self.state.setStyleSheet("font-size:18px;color:#2e7d32;")
        self.preview = LivePreviewLabel(
            "Choose a camera or replay video, then start a session"
        )
        self.diagnostics = QLabel(self.IDLE_DIAGNOSTICS)
        self.captures = CaptureStrip()
        self.combine_button = QPushButton("Combine selected")
        self.combine_button.setToolTip(
            "Lay the ticked captures out on one sheet and keep that as the capture"
        )
        self.combine_button.setEnabled(False)
        self.capture_dialog: QDialog | None = None
        self.processor = ReceiptProcessor(controller.repository)

        controls = QHBoxLayout()
        for widget in (
            QLabel("Source:"),
            self.source_combo,
            self.refresh_button,
            self.video_button,
            self.start_button,
            self.stop_button,
        ):
            controls.addWidget(widget)
        controls.addStretch()
        controls.addWidget(self.version_label)
        controls.addWidget(self.update_button)
        manual_controls = QHBoxLayout()
        manual_controls.addWidget(self.auto_check)
        manual_controls.addWidget(self.capture_button)
        manual_controls.addWidget(self.clear_selection_button)
        manual_controls.addSpacing(12)
        manual_controls.addWidget(QLabel("Area:"))
        manual_controls.addWidget(self.detect_area)
        manual_controls.addWidget(self.exact_area)
        manual_controls.addStretch()
        folder_controls = QHBoxLayout()
        folder_controls.addWidget(QLabel("Receipt folder:"))
        folder_controls.addWidget(self.folder_value, 1)
        folder_controls.addWidget(self.folder_button)
        folder_controls.addWidget(self.clear_folder_button)
        feedback = QHBoxLayout()
        feedback.addWidget(self.counter)
        feedback.addStretch()
        feedback.addWidget(self.state)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(manual_controls)
        layout.addLayout(folder_controls)
        layout.addLayout(feedback)
        capture_column = QVBoxLayout()
        capture_column.setContentsMargins(0, 0, 0, 0)
        capture_column.addWidget(self.captures, 1)
        capture_column.addWidget(self.combine_button)
        preview_row = QHBoxLayout()
        preview_row.addWidget(self.preview, 1)
        preview_row.addLayout(capture_column)
        layout.addLayout(preview_row, 1)
        layout.addWidget(self.diagnostics)

        self.refresh_button.clicked.connect(self.refresh_sources)
        self.video_button.clicked.connect(self.choose_video)
        self.folder_button.clicked.connect(self.choose_session_folder)
        self.clear_folder_button.clicked.connect(self.clear_session_folder)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(controller.stop)
        self.update_button.clicked.connect(self.check_for_updates)
        self.auto_check.toggled.connect(controller.set_auto_capture)
        self.capture_button.clicked.connect(self.capture_selected_area)
        self.clear_selection_button.clicked.connect(self.preview.clear_crop)
        self.exact_area.toggled.connect(self._area_mode_changed)
        controller.preview_ready.connect(self.preview.set_image)
        controller.metrics_ready.connect(self.on_metrics)
        controller.receipt_saved.connect(self.on_saved)
        controller.session_started.connect(self.on_started)
        controller.session_finished.connect(self.on_finished)
        self.captures.flag_toggled.connect(self.set_capture_flag)
        self.captures.delete_requested.connect(self.delete_capture)
        self.captures.enlarge_requested.connect(self.show_capture)
        self.captures.combine_selection_changed.connect(self._combine_selection_changed)
        self.combine_button.clicked.connect(self.combine_captures)
        self.source_combo.currentIndexChanged.connect(self.start_source_preview)
        self._update_folder_value()
        QTimer.singleShot(0, self.refresh_sources)

    def refresh_sources(self) -> None:
        self.stop_source_preview()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self._sources = enumerate_cameras(self.settings)
        for descriptor in self._sources:
            self.source_combo.addItem(
                f"{descriptor.name} ({descriptor.backend})", descriptor
            )
        self.source_combo.addItem("Video file...", None)
        if not self._sources:
            self.source_combo.setCurrentIndex(self.source_combo.count() - 1)
        self.source_combo.blockSignals(False)
        QTimer.singleShot(0, self.start_source_preview)

    def choose_video(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open recorded receipt session",
            "",
            "Videos (*.mp4 *.mkv *.avi *.mov *.wmv);;All files (*)",
        )
        if filename:
            self._video_path = Path(filename)
            self.source_combo.setItemText(
                self.source_combo.count() - 1, f"Replay: {self._video_path.name}"
            )
            self.source_combo.setCurrentIndex(self.source_combo.count() - 1)
            self.start_source_preview()

    def choose_session_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose this session's receipt folder",
            str(self._session_folder or Path(self.settings.receipt_root)),
        )
        if selected:
            self._session_folder = Path(selected)
            self._update_folder_value()

    def clear_session_folder(self) -> None:
        self._session_folder = None
        self._update_folder_value()

    def _update_folder_value(self) -> None:
        text = (
            str(self._session_folder)
            if self._session_folder
            else f"Automatic under {self.settings.receipt_root}"
        )
        self.folder_value.setText(text)
        self.folder_value.setToolTip(text)
        self.clear_folder_button.setEnabled(self._session_folder is not None)

    def _selected_source(self):
        descriptor = self.source_combo.currentData()
        if descriptor is not None:
            return descriptor_source(descriptor, self.settings)
        if self._video_path is not None:
            return VideoFileSource(self._video_path)
        return None

    def start_source_preview(self, _index: int | None = None) -> None:
        if self.controller.active:
            return
        self._preview_packets.clear()
        self.stop_source_preview()
        source = self._selected_source()
        if source is None:
            return
        worker = SourcePreviewWorker(source)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.preview_ready.connect(self.preview.set_image)
        worker.packet_ready.connect(self._remember_preview_packet)
        worker.opened.connect(self._preview_opened)
        worker.failed.connect(
            lambda message: self.diagnostics.setText(f"Preview unavailable: {message}")
        )
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            lambda preview_thread=thread: self._preview_finished(preview_thread)
        )
        thread.finished.connect(thread.deleteLater)
        self._preview_worker = worker
        self._preview_thread = thread
        thread.start()

    def _remember_preview_packet(self, packet: FramePacket) -> None:
        self._preview_packets.append(
            FramePacket(packet.timestamp, packet.frame.copy(), index=packet.index)
        )
        cutoff = packet.timestamp - 0.75
        while self._preview_packets and self._preview_packets[0].timestamp < cutoff:
            self._preview_packets.popleft()

    def _preview_finished(self, thread: QThread) -> None:
        if self._preview_thread is thread:
            self._preview_worker = None
            self._preview_thread = None

    def _preview_opened(self, capabilities: list) -> None:
        values = {item.name: item.value for item in capabilities}
        width = int(values.get("width") or 0)
        height = int(values.get("height") or 0)
        fps = float(values.get("fps") or 0.0)
        self.diagnostics.setText(
            f"Preview source (actual): {width}x{height} @ {fps:g} fps — not recording"
        )
        self.capabilities_ready.emit(capabilities)

    def stop_source_preview(self) -> None:
        worker, thread = self._preview_worker, self._preview_thread
        self._preview_worker = None
        self._preview_thread = None
        if worker is not None:
            worker.stop()
        if thread is not None:
            thread.quit()
            if thread.isRunning():
                thread.wait(3000)

    def check_for_updates(self) -> None:
        if self._update_thread is not None:
            return
        self.update_button.setEnabled(False)
        worker = UpdateWorker(APP_VERSION)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.check)
        self.request_download.connect(worker.download)
        worker.up_to_date.connect(self._update_not_needed)
        worker.update_found.connect(self._update_available)
        worker.progress.connect(self._update_progress)
        worker.downloaded.connect(self._installer_ready)
        worker.failed.connect(self._update_failed)
        self._update_worker = worker
        self._update_thread = thread
        thread.start()

    def _finish_update(self) -> None:
        worker, thread = self._update_worker, self._update_thread
        self._update_worker = None
        self._update_thread = None
        if self._update_progress_dialog is not None:
            self._update_progress_dialog.close()
            self._update_progress_dialog = None
        if thread is not None:
            thread.quit()
            if thread.isRunning():
                thread.wait(3000)
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()
        # CLAUDE CODE: a session may have started during the check; Start being
        # disabled is the single source of truth for "a session is running".
        self.update_button.setEnabled(self.start_button.isEnabled())

    def _update_not_needed(self, version: str) -> None:
        self._finish_update()
        QMessageBox.information(
            self,
            "Scan Receipts is up to date",
            f"You are running the latest version ({version}).",
        )

    def _update_available(self, release: ReleaseInfo) -> None:
        if not is_frozen():
            self._finish_update()
            QMessageBox.information(
                self,
                "Update available",
                f"Version {release.version} has been published.\n\n"
                "This is a source checkout, so the installer will not be run. "
                f"See {release.page_url}",
            )
            return
        answer = QMessageBox.question(
            self,
            "Update available",
            f"Version {release.version} is available (you have {APP_VERSION}).\n\n"
            "Scan Receipts will close while it installs and then start again. "
            "Your receipts and settings are not affected.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._finish_update()
            return
        dialog = QProgressDialog("Downloading the update...", "", 0, 100, self)
        dialog.setCancelButton(None)
        dialog.setWindowTitle("Scan Receipts")
        dialog.setMinimumDuration(0)
        dialog.setValue(0)
        self._update_progress_dialog = dialog
        self.request_download.emit(release)

    def _update_progress(self, percent: int) -> None:
        if self._update_progress_dialog is not None:
            self._update_progress_dialog.setValue(percent)

    def _installer_ready(self, installer: Path) -> None:
        self._finish_update()
        try:
            run_installer(installer)
        except Exception as error:
            QMessageBox.warning(self, "Could not install the update", str(error))
            return
        QApplication.quit()

    def _update_failed(self, message: str) -> None:
        self._finish_update()
        QMessageBox.warning(self, "Could not check for updates", message)

    def start(self) -> None:
        descriptor = self.source_combo.currentData()
        if descriptor is None:
            if self._video_path is None:
                self.choose_video()
            if self._video_path is None:
                return
        self.stop_source_preview()
        source = self._selected_source()
        if source is None:
            return
        try:
            if self._resume_session_id:
                self.controller.resume(self._resume_session_id, source)
                self._resume_session_id = None
                self.start_button.setText("Start Session")
            else:
                initial_packets = (
                    list(self._preview_packets) if descriptor is not None else None
                )
                self.controller.start(source, self._session_folder, initial_packets)
        except Exception as error:
            QMessageBox.critical(self, "Could not start session", str(error))
            self.start_source_preview()

    def _area_mode_changed(self, exact: bool) -> None:
        self.settings.manual_capture_mode = "exact" if exact else "detect"
        self.settings_changed.emit()

    def capture_selected_area(self) -> None:
        """Capture the drawn area, or the whole frame when nothing is drawn."""
        selection = self.preview.source_crop_relative()
        self.controller.request_manual_capture(
            tuple(selection) if selection is not None else None,
            self.exact_area.isChecked(),
        )
        self.preview.clear_crop()

    def _set_manual_controls_enabled(self, enabled: bool) -> None:
        self.capture_button.setEnabled(enabled)
        self.clear_selection_button.setEnabled(enabled)

    def reset_captures(self) -> None:
        """Put the capture side of the page back to how it looks before a session."""
        self.close_capture_dialog()
        self._session_id = None
        self.captures.clear_captures()
        self.combine_button.setEnabled(False)
        self.counter.setText(self.IDLE_COUNTER)
        self.state.setText(self.IDLE_STATE)
        self.diagnostics.setText(self.IDLE_DIAGNOSTICS)

    def on_started(self, session: SessionRecord) -> None:
        self.reset_captures()
        self._session_id = session.id
        self._set_manual_controls_enabled(True)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.source_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.video_button.setEnabled(False)
        self.folder_button.setEnabled(False)
        self.clear_folder_button.setEnabled(False)
        self.update_button.setEnabled(False)

    def on_finished(self, session: SessionRecord) -> None:
        self._set_manual_controls_enabled(False)
        self.preview.clear_crop()
        self.reset_captures()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.source_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.video_button.setEnabled(True)
        self.folder_button.setEnabled(True)
        self.update_button.setEnabled(True)
        self._session_folder = None
        self._update_folder_value()
        QTimer.singleShot(0, self.start_source_preview)

    def on_saved(self, receipt: ReceiptRecord) -> None:
        self._session_id = receipt.session_id
        self.state.setText(f"CAPTURED - {self.refresh_captures()}")

    def refresh_captures(self) -> int:
        if not self._session_id:
            return 0
        receipts = self.controller.repository.list_receipts(self._session_id)
        self.captures.set_receipts(receipts)
        self._combine_selection_changed()
        groups: dict[str, int] = {}
        for record in receipts:
            if record.duplicate_group:
                groups[record.duplicate_group] = groups.get(record.duplicate_group, 0) + 1
        to_review = sum(max(0, members - 1) for members in groups.values())
        confirmed = len(receipts) - to_review
        suffix = f" (+{to_review} to review)" if to_review else ""
        self.counter.setText(f"Receipts captured: {confirmed}{suffix}")
        return confirmed

    def _combine_selection_changed(self) -> None:
        self.combine_button.setEnabled(len(self.captures.combine_selection()) >= 2)

    def combine_captures(self) -> None:
        selected = self.captures.combine_selection()
        if not self._session_id or len(selected) < 2:
            return
        try:
            self.processor.combine_receipts(
                self._session_id,
                [self.controller.repository.get_receipt(item) for item in selected],
                self.settings.output,
                self.settings.combine_maximum,
            )
        except Exception as error:
            QMessageBox.information(self, "Combine receipts", str(error))
            return
        self.refresh_captures()

    def set_capture_flag(self, receipt_id: int, flagged: bool) -> None:
        self.controller.repository.set_review_flag(receipt_id, flagged)
        self.refresh_captures()

    def show_capture(self, receipt_id: int) -> None:
        receipt = self.controller.repository.get_receipt(receipt_id)
        image = QImage(receipt.processed_path)
        if image.isNull():
            image = QImage(str(preview_path(receipt.original_path)))
        self.close_capture_dialog()
        dialog = QDialog(self)
        dialog.setWindowTitle(receipt.filename)
        dialog.setModal(False)
        dialog.resize(self.width() * 7 // 10, self.height() * 7 // 10)
        label = PreviewLabel(receipt.filename, dialog)
        if not image.isNull():
            label.set_image(image)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label)
        self.capture_dialog = dialog
        dialog.finished.connect(lambda _result: setattr(self, "capture_dialog", None))
        dialog.show()

    def close_capture_dialog(self) -> None:
        if self.capture_dialog is not None:
            self.capture_dialog.close()
            self.capture_dialog = None

    def delete_capture(self, receipt_id: int) -> None:
        receipt = self.controller.repository.get_receipt(receipt_id)
        if (
            QMessageBox.question(
                self,
                "Delete this capture",
                f"Send {receipt.filename}{combined_suffix(self.controller.repository, receipt)} "
                "and the preserved originals to the Recycle Bin?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.processor.trash(receipt)
        self.refresh_captures()

    def on_metrics(self, metrics) -> None:
        self.state.setText(metrics.state.value)
        self.diagnostics.setText(
            f"Motion {metrics.motion:.3f}  |  Sharpness {metrics.sharpness:.0f}  |  "
            f"Boundary {metrics.boundary_confidence:.2f}  |  Score {metrics.score:.2f}"
        )

    def prepare_resume(self, session_id: str) -> None:
        self._resume_session_id = session_id
        self.start_button.setText("Resume This Session")

    def shutdown(self) -> None:
        self.close_capture_dialog()
        self.stop_source_preview()


def _readable_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


SELECTION_STYLE = """
QListWidget::item:selected {
    background: #1d4f7c;
    color: #ffffff;
    border: 1px solid #67b0f0;
    border-radius: 4px;
}
QListWidget::item:selected:!active {
    background: #24537d;
    color: #ffffff;
}
QListWidget::item:hover:!selected { background: rgba(255,255,255,0.06); }
"""


class SessionsPage(QWidget):
    resume_requested = Signal(str)

    def __init__(self, repository: Repository, settings: AppSettings) -> None:
        super().__init__()
        self.repository = repository
        self.settings = settings
        self.processor = ReceiptProcessor(repository)
        self.current_session: SessionRecord | None = None
        self.current_receipt: ReceiptRecord | None = None

        self.sessions = QListWidget()
        self.sessions.setMinimumWidth(260)
        self.sessions.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.details = QLabel("Select a session")
        self.details.setWordWrap(True)
        self.folder_path = QLabel("—")
        self.recording_path = QLabel("—")
        for path_label in (self.folder_path, self.recording_path):
            path_label.setWordWrap(True)
            path_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            path_label.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
        self.copy_folder = QToolButton()
        self.copy_folder.setIcon(copy_icon())
        self.copy_folder.setToolTip("Copy receipt folder path")
        self.copy_folder.setAutoRaise(True)
        self.open_folder = QToolButton()
        self.open_folder.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        self.open_folder.setToolTip("Open receipt folder")
        self.open_folder.setAutoRaise(True)
        self.copy_recording = QToolButton()
        self.copy_recording.setIcon(copy_icon())
        self.copy_recording.setToolTip("Copy recording path")
        self.copy_recording.setAutoRaise(True)
        self.open_recording = QToolButton()
        self.open_recording.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        )
        self.open_recording.setToolTip("Open recording location")
        self.open_recording.setAutoRaise(True)
        self.receipts = QListWidget()
        self.receipts.setViewMode(QListWidget.ViewMode.ListMode)
        self.receipts.setIconSize(QSize(96, 96))
        self.receipts.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.receipts.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.receipts.setMinimumWidth(280)
        self.receipts.setAlternatingRowColors(True)
        self.receipts.setSpacing(2)
        self.receipts.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_scrolling = [
            install_smooth_scroll(view) for view in (self.sessions, self.receipts)
        ]
        for view in (self.sessions, self.receipts):
            view.setStyleSheet(SELECTION_STYLE)
        self.image = CropLabel()
        self.image_scroll = ImageScrollArea(self.image)
        self.zoom_out = QPushButton("-")
        self.zoom_out.setToolTip("Zoom out")
        self.zoom_in = QPushButton("+")
        self.zoom_in.setToolTip("Zoom in")
        self.zoom_reset = QPushButton("Fit")
        self.zoom_reset.setToolTip("Fit the complete receipt")
        self.zoom_value = QLabel("100%")
        self.edit_status = QLabel("")
        self.edit_status.setStyleSheet("color: #68c878; padding: 0 6px;")

        self.duplicate_decks = QWidget()
        self.duplicate_decks_layout = QHBoxLayout(self.duplicate_decks)
        self.duplicate_decks_layout.setContentsMargins(2, 2, 2, 2)
        self.duplicate_decks_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.duplicate_decks_scroll = QScrollArea()
        self.duplicate_decks_scroll.setWidgetResizable(True)
        self.duplicate_decks_scroll.setWidget(self.duplicate_decks)
        self.duplicate_decks_scroll.setMaximumHeight(168)
        self.duplicate_decks_scroll.setVisible(False)

        self.confirm = QPushButton("Confirm")
        self.confirm.setToolTip("Confirm this session captured every expected receipt")
        self.needs_review = QPushButton("Needs review")
        self.delete_session_button = QPushButton("Delete Session")
        self.discard_button = QPushButton("Discard Session")
        self.discard_button.setToolTip(
            "Remove everything for the selected session(s): receipt images, "
            "preserved originals, recording and history. Files go to the Recycle Bin."
        )
        self.discard_button.setEnabled(False)
        self.delete_video = QPushButton("Delete Session Video")
        self.delete_images = QPushButton("Delete Receipt Images")
        self.inspect_video = QPushButton("View video")
        self.recover_video = QPushButton("Recover")
        self.recover_video.setToolTip("Run a thorough recovery pass on the recording")
        self.resume_session = QPushButton("Resume")
        self.rename = QPushButton("Rename")
        self.rotate_left = QPushButton("Rotate left")
        self.rotate_right = QPushButton("Rotate right")
        self.apply_crop = QPushButton("Apply Crop")
        self.restore = QPushButton("Restore Original")
        self.reprocess = QPushButton("Reprocess")
        self.reprocess.setToolTip(
            "Discard manual edits, redetect the receipt, and apply current output enhancements."
        )
        self.combine_button = QPushButton("Combine")
        self.combine_button.setToolTip(
            "Lay the selected receipts out on one sheet and keep that as the capture"
        )
        self.uncombine_button = QPushButton("Uncombine")
        self.uncombine_button.setToolTip(
            "Discard this combined sheet and bring back the receipts it holds"
        )
        self.find_duplicates = QPushButton("Duplicates")
        self.find_duplicates.setToolTip("Find possible duplicate receipts")
        self.delete_selected = QPushButton("Delete")
        self.not_duplicate = QPushButton("Not duplicate")
        self.open_original = QPushButton("Open original")

        command_buttons = (
            self.confirm,
            self.needs_review,
            self.inspect_video,
            self.recover_video,
            self.find_duplicates,
            self.resume_session,
            self.rename,
            self.rotate_left,
            self.rotate_right,
            self.reprocess,
            self.combine_button,
            self.uncombine_button,
            self.delete_selected,
            self.not_duplicate,
            self.open_original,
        )
        for button in command_buttons:
            button.setFlat(True)
            button.setMaximumHeight(30)
            button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

        self.session_more = QToolButton()
        self.session_more.setText("Session actions")
        self.session_more.setToolTip("More session actions")
        self.session_more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        session_more_menu = QMenu(self.session_more)
        session_more_menu.addAction("Discard session (everything)", self.discard_sessions)
        session_more_menu.addAction("Delete session", self.delete_session)
        session_more_menu.addAction("Delete session video", self.remove_video)
        session_more_menu.addAction("Delete receipt images", self.remove_all_images)
        self.session_more.setMenu(session_more_menu)

        self.receipt_more = QToolButton()
        self.receipt_more.setText("More actions")
        self.receipt_more.setToolTip("More receipt actions")
        self.receipt_more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        receipt_more_menu = QMenu(self.receipt_more)
        receipt_more_menu.addAction("Apply selected crop", self.crop_receipt)
        receipt_more_menu.addAction("Restore original", self.restore_receipt)
        receipt_more_menu.addAction("Mark not duplicate", self.mark_not_duplicate)
        self.receipt_more.setMenu(receipt_more_menu)

        session_commands = QHBoxLayout()
        session_commands.setContentsMargins(0, 0, 0, 0)
        session_commands.setSpacing(2)
        for widget in (
            self.confirm,
            self.needs_review,
            self.inspect_video,
            self.recover_video,
            self.find_duplicates,
            self.resume_session,
            self.discard_button,
            self.session_more,
        ):
            session_commands.addWidget(widget)
        session_commands.addStretch()

        paths = QGridLayout()
        paths.addWidget(QLabel("Receipt folder:"), 0, 0)
        paths.addWidget(self.folder_path, 0, 1)
        paths.addWidget(self.copy_folder, 0, 2)
        paths.addWidget(self.open_folder, 0, 3)
        paths.addWidget(QLabel("Recording:"), 1, 0)
        paths.addWidget(self.recording_path, 1, 1)
        paths.addWidget(self.copy_recording, 1, 2)
        paths.addWidget(self.open_recording, 1, 3)

        receipt_commands = QHBoxLayout()
        receipt_commands.setContentsMargins(0, 0, 0, 0)
        receipt_commands.setSpacing(2)
        for widget in (
            self.open_original,
            self.rename,
            self.rotate_left,
            self.rotate_right,
            self.reprocess,
            self.combine_button,
            self.uncombine_button,
            self.delete_selected,
            self.receipt_more,
        ):
            receipt_commands.addWidget(widget)
        receipt_commands.addStretch()
        receipt_commands.addWidget(self.edit_status)
        zoom_controls = QHBoxLayout()
        zoom_controls.setContentsMargins(0, 0, 0, 0)
        zoom_controls.addStretch()
        zoom_controls.addWidget(self.zoom_out)
        zoom_controls.addWidget(self.zoom_in)
        zoom_controls.addWidget(self.zoom_reset)
        zoom_controls.addWidget(self.zoom_value)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addLayout(zoom_controls)
        preview_layout.addWidget(self.image_scroll, 1)
        receipt_browser = QSplitter(Qt.Orientation.Horizontal)
        receipt_browser.addWidget(self.receipts)
        receipt_browser.addWidget(preview)
        receipt_browser.setStretchFactor(0, 0)
        receipt_browser.setStretchFactor(1, 1)
        receipt_browser.setSizes([260, 700])

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 4, 6, 4)
        right_layout.setSpacing(4)
        right_layout.addLayout(session_commands)
        right_layout.addWidget(self.details)
        right_layout.addLayout(paths)
        right_layout.addWidget(self.duplicate_decks_scroll)
        right_layout.addLayout(receipt_commands)
        right_layout.addWidget(receipt_browser, 1)
        splitter = QSplitter()
        splitter.addWidget(self.sessions)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self.sessions.currentItemChanged.connect(self.select_session)
        self.receipts.currentItemChanged.connect(self.select_receipt)
        delete_receipts = QShortcut(QKeySequence.StandardKey.Delete, self.receipts)
        delete_receipts.setContext(Qt.ShortcutContext.WidgetShortcut)
        delete_receipts.activated.connect(self.remove_selected)
        discard_sessions = QShortcut(QKeySequence.StandardKey.Delete, self.sessions)
        discard_sessions.setContext(Qt.ShortcutContext.WidgetShortcut)
        discard_sessions.activated.connect(self.discard_sessions)
        self.receipts.customContextMenuRequested.connect(self.show_receipt_context_menu)
        self.open_folder.clicked.connect(
            lambda: self.current_session and reveal(self.current_session.receipt_folder)
        )
        self.copy_folder.clicked.connect(self.copy_receipt_folder_path)
        self.copy_recording.clicked.connect(self.copy_recording_path)
        self.open_recording.clicked.connect(
            lambda: (
                self.current_session
                and self.current_session.video_path
                and reveal(self.current_session.video_path)
            )
        )
        self.confirm.clicked.connect(self.confirm_successful)
        self.needs_review.clicked.connect(self.mark_needs_review)
        self.delete_session_button.clicked.connect(self.delete_session)
        self.discard_button.clicked.connect(self.discard_sessions)
        self.sessions.itemSelectionChanged.connect(self._session_selection_changed)
        self.delete_video.clicked.connect(self.remove_video)
        self.delete_images.clicked.connect(self.remove_all_images)
        self.inspect_video.clicked.connect(self.show_video)
        self.recover_video.clicked.connect(self.run_recovery)
        self.find_duplicates.clicked.connect(self.scan_duplicates)
        self.resume_session.clicked.connect(self.request_resume)
        self.rename.clicked.connect(self.rename_receipt)
        self.rotate_left.clicked.connect(lambda: self.rotate(-90))
        self.rotate_right.clicked.connect(lambda: self.rotate(90))
        self.apply_crop.clicked.connect(self.crop_receipt)
        self.restore.clicked.connect(self.restore_receipt)
        self.reprocess.clicked.connect(self.reprocess_receipt)
        self.combine_button.clicked.connect(self.combine_receipts)
        self.uncombine_button.clicked.connect(self.uncombine_receipt)
        self.delete_selected.clicked.connect(self.remove_selected)
        self.not_duplicate.clicked.connect(self.mark_not_duplicate)
        self.open_original.clicked.connect(self.show_original)
        self.zoom_in.clicked.connect(lambda: self.set_zoom(1.25))
        self.zoom_out.clicked.connect(lambda: self.set_zoom(0.8))
        self.zoom_reset.clicked.connect(self.reset_zoom)
        self.refresh()

    def refresh(self, select_id: str | None = None) -> None:
        current = select_id or (
            self.current_session.id if self.current_session else None
        )
        self.sessions.clear()
        target = None
        for session in self.repository.list_sessions():
            receipts = self.repository.list_receipts(session.id)
            group_sizes: dict[str, int] = {}
            for receipt in receipts:
                if receipt.duplicate_group:
                    group_sizes[receipt.duplicate_group] = (
                        group_sizes.get(receipt.duplicate_group, 0) + 1
                    )
            to_review = sum(max(0, size - 1) for size in group_sizes.values())
            confirmed = len(receipts) - to_review
            review_suffix = f" (+{to_review} to review)" if to_review else ""
            started = datetime.fromisoformat(session.started_at).strftime(
                "%b %d, %Y %I:%M %p"
            )
            video = (
                "Video available"
                if session.video_path and Path(session.video_path).exists()
                else "No video"
            )
            item = QListWidgetItem(
                f"{started}\n{confirmed} receipts{review_suffix} - "
                f"{session.status.value}\n{video} - {session.processing_status}"
            )
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            self.sessions.addItem(item)
            if session.id == current:
                target = item
        if target:
            self.sessions.setCurrentItem(target)

    def select_session(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        self.current_session = self.repository.get_session(
            item.data(Qt.ItemDataRole.UserRole)
        )
        if not self.current_session:
            return
        session = self.current_session
        self.resume_session.setEnabled(session.status == SessionStatus.NEEDS_REVIEW)
        settings = json.loads(session.settings_json)
        actual = settings.get("actual_capture") or {}
        if not actual:
            receipts = self.repository.list_receipts(session.id)
            if receipts:
                original = QImage(receipts[0].original_path)
                if not original.isNull():
                    actual = {
                        "width": original.width(),
                        "height": original.height(),
                        "fps": settings.get("camera_fps"),
                    }
        requested = (
            f"{settings.get('camera_width')}x{settings.get('camera_height')} @ "
            f"{settings.get('camera_fps')} fps"
        )
        delivered = (
            f"{actual.get('width')}x{actual.get('height')} @ {actual.get('fps'):g} fps"
            if actual.get("width") and actual.get("height") and actual.get("fps")
            else "Unavailable"
        )
        receipts = self.repository.list_receipts(session.id)
        group_sizes: dict[str, int] = {}
        for receipt in receipts:
            if receipt.duplicate_group:
                group_sizes[receipt.duplicate_group] = (
                    group_sizes.get(receipt.duplicate_group, 0) + 1
                )
        to_review = sum(max(0, size - 1) for size in group_sizes.values())
        confirmed = len(receipts) - to_review
        receipt_summary = (
            f"{confirmed} (+{to_review} to review)" if to_review else str(confirmed)
        )
        self.details.setText(
            f"Status: {session.status.value}    Receipts: {receipt_summary}\n"
            f"Camera: {session.camera_name}\n"
            f"Source delivered: {delivered}    Requested: {requested}\n"
            f"Output: {settings.get('output', {}).get('format', 'JPEG')}"
        )
        self.folder_path.setText(session.receipt_folder)
        self.folder_path.setToolTip(session.receipt_folder)
        recording = session.video_path or "No recording"
        self.recording_path.setText(recording)
        self.recording_path.setToolTip(recording)
        self.copy_recording.setEnabled(bool(session.video_path))
        self.open_recording.setEnabled(
            bool(session.video_path and Path(session.video_path).exists())
        )
        self.current_receipt = None
        self.load_receipts()

    def load_receipts(
        self,
        selected_ids: list[int] | None = None,
        current_id: int | None = None,
    ) -> None:
        if selected_ids is None:
            selected_ids = [
                item.data(Qt.ItemDataRole.UserRole)
                for item in self.receipts.selectedItems()
            ]
        if current_id is None and self.current_receipt:
            current_id = self.current_receipt.id
        selected = set(selected_ids)
        self.receipts.blockSignals(True)
        self.receipts.clear()
        if not self.current_session:
            self.receipts.blockSignals(False)
            self.duplicate_decks_scroll.setVisible(False)
            return
        current_item = None
        receipts = self.repository.list_receipts(self.current_session.id)
        group_members: dict[str, list[ReceiptRecord]] = {}
        for receipt in receipts:
            if receipt.duplicate_group:
                group_members.setdefault(receipt.duplicate_group, []).append(receipt)
        valid_duplicate_groups = {
            group for group, members in group_members.items() if len(members) >= 2
        }
        duplicate_role = int(Qt.ItemDataRole.UserRole) + 1
        for receipt in receipts:
            item = QListWidgetItem(receipt.filename)
            item.setData(Qt.ItemDataRole.UserRole, receipt.id)
            pixmap = QPixmap(receipt.processed_path)
            if pixmap.isNull():
                pixmap = QPixmap(str(preview_path(receipt.original_path)))
            if not pixmap.isNull():
                item.setIcon(QIcon(pixmap))
            if receipt.duplicate_group in valid_duplicate_groups:
                item.setData(duplicate_role, receipt.duplicate_group)
                item.setText(f"POSSIBLE DUPLICATE\n{receipt.filename}")
                item.setToolTip(f"Duplicate review group: {receipt.duplicate_group}")
                item.setBackground(QColor("#5a4300"))
                item.setForeground(QColor("#ffe08a"))
            if receipt.review_flag:
                item.setText(f"FLAGGED\n{item.text()}")
                item.setToolTip("Flagged during scanning - check this receipt")
            self.receipts.addItem(item)
            item.setSelected(receipt.id in selected)
            if receipt.id == current_id:
                current_item = item
        if current_item is None and self.receipts.count():
            current_item = self.receipts.item(0)
        if current_item is not None:
            self.receipts.setCurrentItem(
                current_item,
                QItemSelectionModel.SelectionFlag.NoUpdate,
            )
            self.receipts.scrollToItem(
                current_item, QAbstractItemView.ScrollHint.PositionAtCenter
            )
        self.receipts.blockSignals(False)
        self.refresh_duplicate_decks(receipts)
        self.select_receipt(current_item)

    def show_feedback(self, message: str) -> None:
        self.edit_status.setText(message)

    def copy_receipt_folder_path(self) -> None:
        if not self.current_session:
            return
        QApplication.clipboard().setText(self.current_session.receipt_folder)
        self.show_copy_success(self.copy_folder, "Copy receipt folder path")
        self.show_feedback("Receipt folder path copied")

    def copy_recording_path(self) -> None:
        if not self.current_session or not self.current_session.video_path:
            return
        QApplication.clipboard().setText(self.current_session.video_path)
        self.show_copy_success(self.copy_recording, "Copy recording path")
        self.show_feedback("Recording path copied")

    @staticmethod
    def show_copy_success(button: QToolButton, original_tooltip: str) -> None:
        button.setIcon(copied_icon())
        button.setToolTip("Copied")
        button.setStyleSheet("QToolButton { background: #214d2c; }")

        def restore() -> None:
            button.setIcon(copy_icon())
            button.setToolTip(original_tooltip)
            button.setStyleSheet("")

        QTimer.singleShot(1800, restore)

    def refresh_duplicate_decks(self, receipts: list[ReceiptRecord]) -> None:
        while self.duplicate_decks_layout.count():
            item = self.duplicate_decks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        groups: dict[str, list[ReceiptRecord]] = {}
        for receipt in receipts:
            if receipt.duplicate_group:
                groups.setdefault(receipt.duplicate_group, []).append(receipt)
        groups = {
            group: members for group, members in groups.items() if len(members) >= 2
        }
        self.duplicate_decks_scroll.setVisible(bool(groups))
        for group, members in groups.items():
            deck = QToolButton()
            deck.setIcon(duplicate_deck_icon(members))
            deck.setIconSize(QSize(170, 112))
            deck.setText(f"Review {len(members)} possible duplicates")
            deck.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            deck.setCursor(Qt.CursorShape.PointingHandCursor)
            deck.setToolTip("Open this duplicate deck and compare every receipt")
            deck.clicked.connect(
                lambda _checked=False, group_id=group: self.open_duplicate_deck(
                    group_id
                )
            )
            self.duplicate_decks_layout.addWidget(deck)
        self.duplicate_decks_layout.addStretch()

    def open_duplicate_deck(self, group: str) -> None:
        dialog = DuplicateDeckDialog(self.repository, group, self)
        dialog.changed.connect(self._refresh_current)
        dialog.exec()
        self._refresh_current()

    def open_first_duplicate_deck(self) -> None:
        if not self.current_session:
            return
        groups: dict[str, int] = {}
        for receipt in self.repository.list_receipts(self.current_session.id):
            if receipt.duplicate_group:
                groups[receipt.duplicate_group] = (
                    groups.get(receipt.duplicate_group, 0) + 1
                )
        group = next((key for key, size in groups.items() if size >= 2), None)
        if group:
            self.open_duplicate_deck(group)

    def select_duplicate_receipts(self) -> None:
        self.open_first_duplicate_deck()

    def show_receipt_context_menu(self, position: QPoint) -> None:
        if not self.receipts.itemAt(position):
            return
        menu = QMenu(self)
        menu.addAction("Open original", self.show_original)
        menu.addAction("Rename", self.rename_receipt)
        menu.addSeparator()
        menu.addAction("Rotate left", lambda: self.rotate(-90))
        menu.addAction("Rotate right", lambda: self.rotate(90))
        menu.addAction("Reprocess from original", self.reprocess_receipt)
        menu.addAction("Combine selected", self.combine_receipts)
        menu.addAction("Uncombine", self.uncombine_receipt)
        menu.addAction("Apply selected crop", self.crop_receipt)
        menu.addAction("Restore original", self.restore_receipt)
        menu.addSeparator()
        menu.addAction("Mark not duplicate", self.mark_not_duplicate)
        menu.addAction("Flag / unflag for review", self.toggle_review_flag)
        menu.addAction("Delete selected", self.remove_selected)
        menu.exec(self.receipts.viewport().mapToGlobal(position))

    def select_receipt(self, item: QListWidgetItem | None) -> None:
        self.current_receipt = (
            self.repository.get_receipt(item.data(Qt.ItemDataRole.UserRole))
            if item
            else None
        )
        if self.current_receipt:
            image = QImage(self.current_receipt.processed_path)
            if image.isNull():
                image = QImage(str(preview_path(self.current_receipt.original_path)))
            if not image.isNull():
                self.image.set_image(image)
                self.zoom_value.setText("100%")

    def selected_receipts(self) -> list[ReceiptRecord]:
        items = self.receipts.selectedItems()
        if not items and self.receipts.currentItem():
            items = [self.receipts.currentItem()]
        return [
            self.repository.get_receipt(item.data(Qt.ItemDataRole.UserRole))
            for item in items
        ]

    def set_zoom(self, multiplier: float) -> None:
        self.zoom_value.setText(f"{self.image.zoom_by(multiplier)}%")

    def reset_zoom(self) -> None:
        self.zoom_value.setText(f"{self.image.reset_zoom()}%")

    def _edits(self) -> dict:
        if not self.current_receipt:
            return {}
        return self._edits_for(self.current_receipt)

    @staticmethod
    def _edits_for(receipt: ReceiptRecord) -> dict:
        stored = json.loads(receipt.edit_json or "{}")
        return {
            key: value
            for key, value in stored.items()
            if key in {"crop", "crop_relative", "rotation", "corners"}
        }

    def _render(self, edits: dict) -> None:
        if not self.current_receipt:
            return
        try:
            self.current_receipt = self.processor.render_edits(
                self.current_receipt, self.settings.output, edits
            )
            receipt_id = self.current_receipt.id
            self.load_receipts([receipt_id], receipt_id)
            self.edit_status.setText("Saved")
        except Exception as error:
            QMessageBox.critical(self, "Could not edit receipt", str(error))

    def rotate(self, amount: int) -> None:
        receipts = self.selected_receipts()
        if not receipts:
            return
        current_id = self.current_receipt.id if self.current_receipt else receipts[0].id
        try:
            for receipt in receipts:
                edits = self._edits_for(receipt)
                edits["rotation"] = (int(edits.get("rotation", 0)) + amount) % 360
                self.processor.render_edits(receipt, self.settings.output, edits)
            selected_ids = [receipt.id for receipt in receipts]
            self.current_receipt = self.repository.get_receipt(current_id)
            self.load_receipts(selected_ids, current_id)
            self.edit_status.setText(f"Rotated {len(receipts)} receipt(s)")
        except Exception as error:
            QMessageBox.critical(self, "Could not rotate receipt", str(error))

    def crop_receipt(self) -> None:
        crop = self.image.source_crop_relative()
        if not crop:
            QMessageBox.information(
                self, "Crop", "Drag a rectangle over the image first."
            )
            return
        edits = self._edits()
        edits.pop("crop", None)
        edits["crop_relative"] = crop
        self._render(edits)

    def _refuse_on_combined(self, title: str) -> bool:
        """A combined sheet holds several receipts, so redetecting one crops it away."""
        if self.current_receipt and combined_sources(self.current_receipt):
            QMessageBox.information(
                self, title, "Uncombine this receipt before editing it this way."
            )
            return True
        return False

    def restore_receipt(self) -> None:
        if self.current_receipt:
            try:
                receipt_id = self.current_receipt.id
                self.processor.restore_original(
                    self.current_receipt, self.settings.output
                )
                self.current_receipt = self.repository.get_receipt(receipt_id)
                self.load_receipts([receipt_id], receipt_id)
                self.edit_status.setText("Restored preserved camera frame")
            except Exception as error:
                QMessageBox.critical(self, "Could not restore original", str(error))

    def reprocess_receipt(self) -> None:
        if self._refuse_on_combined("Reprocess receipt"):
            return
        receipts = self.selected_receipts()
        if receipts:
            try:
                current_id = (
                    self.current_receipt.id if self.current_receipt else receipts[0].id
                )
                for receipt in receipts:
                    self.processor.reprocess_auto(
                        receipt,
                        self.settings.output,
                        self.settings.detection,
                    )
                selected_ids = [receipt.id for receipt in receipts]
                self.current_receipt = self.repository.get_receipt(current_id)
                self.load_receipts(selected_ids, current_id)
                self.edit_status.setText(
                    f"Redetected and enhanced {len(receipts)} receipt(s) from originals"
                )
            except Exception as error:
                QMessageBox.critical(self, "Could not reprocess receipt", str(error))

    def rename_receipt(self) -> None:
        if not self.current_receipt:
            return
        current = Path(self.current_receipt.processed_path)
        name, ok = QInputDialog.getText(
            self, "Rename receipt", "Unique filename:", text=current.name
        )
        if not ok or not name or name == current.name:
            return
        if Path(name).name != name:
            QMessageBox.warning(self, "Invalid name", "Enter a filename, not a path.")
            return
        target = current.with_name(name)
        if target.exists():
            QMessageBox.warning(
                self,
                "Name already exists",
                "No file was overwritten. Choose a unique name.",
            )
            return
        current.rename(target)
        self.repository.update_receipt_path(self.current_receipt.id, target)
        self.current_receipt = self.repository.get_receipt(self.current_receipt.id)
        self.load_receipts([self.current_receipt.id], self.current_receipt.id)

    def show_original(self) -> None:
        if not self.current_receipt:
            return
        image = QImage(self.current_receipt.original_path)
        if not image.isNull():
            self.image.set_image(image)

    def _neighbour_after_removal(self, removed: set[int]) -> int | None:
        """The receipt to select once `removed` is gone - never back to the top."""
        order = [
            self.receipts.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.receipts.count())
        ]
        survivors = [item for item in order if item not in removed]
        if not survivors:
            return None
        positions = [order.index(item) for item in removed if item in order]
        if not positions:
            return survivors[-1]
        last = max(positions)
        following = next(
            (item for item in order[last + 1 :] if item not in removed), None
        )
        return following if following is not None else survivors[-1]

    def remove_selected(self) -> None:
        items = self.receipts.selectedItems()
        if not items:
            return
        if (
            QMessageBox.question(
                self,
                "Delete selected receipt images",
                f"Send {len(items)} selected receipt image(s), everything they "
                "combine and their preserved originals to the Recycle Bin?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        removed = {item.data(Qt.ItemDataRole.UserRole) for item in items}
        survivor = self._neighbour_after_removal(removed)
        for receipt_id in removed:
            self.processor.trash(self.repository.get_receipt(receipt_id))
        self.current_receipt = (
            self.repository.get_receipt(survivor) if survivor is not None else None
        )
        self._refresh_current()
        if survivor is not None:
            # CLAUDE CODE: refresh() reselects the session, which reloads the
            # receipt list from scratch; say again which one to land on.
            self.load_receipts([survivor], survivor)

    def combine_receipts(self) -> None:
        receipts = self.selected_receipts()
        if not self.current_session or len(receipts) < 2:
            QMessageBox.information(
                self, "Combine receipts", "Select at least two receipts to combine."
            )
            return
        try:
            sheet = self.processor.combine_receipts(
                self.current_session.id,
                receipts,
                self.settings.output,
                self.settings.combine_maximum,
            )
        except Exception as error:
            QMessageBox.information(self, "Combine receipts", str(error))
            return
        self.current_receipt = sheet
        self._refresh_current()
        self.load_receipts([sheet.id], sheet.id)
        self.edit_status.setText(f"Combined {len(receipts)} receipts")

    def uncombine_receipt(self) -> None:
        receipt = self.current_receipt
        if receipt is None:
            return
        try:
            restored = self.processor.uncombine(receipt)
        except Exception as error:
            QMessageBox.information(self, "Uncombine receipt", str(error))
            return
        self.current_receipt = self.repository.get_receipt(restored[0])
        self._refresh_current()
        self.load_receipts(restored, restored[0])
        self.edit_status.setText(f"Restored {len(restored)} receipts")

    def toggle_review_flag(self) -> None:
        items = self.receipts.selectedItems()
        if not items:
            return
        selected_ids = [item.data(Qt.ItemDataRole.UserRole) for item in items]
        receipts = [self.repository.get_receipt(item_id) for item_id in selected_ids]
        flagged = not all(receipt.review_flag for receipt in receipts)
        for receipt in receipts:
            self.repository.set_review_flag(receipt.id, flagged)
        current_id = self.current_receipt.id if self.current_receipt else None
        self.load_receipts(selected_ids, current_id)
        verb = "Flagged" if flagged else "Unflagged"
        self.edit_status.setText(f"{verb} {len(selected_ids)} receipt(s) for review")

    def mark_not_duplicate(self) -> None:
        items = self.receipts.selectedItems()
        selected_ids = [item.data(Qt.ItemDataRole.UserRole) for item in items]
        for item in items:
            self.repository.resolve_duplicate_member(
                item.data(Qt.ItemDataRole.UserRole)
            )
        current_id = self.current_receipt.id if self.current_receipt else None
        self.load_receipts(selected_ids, current_id)
        self.edit_status.setText(
            f"Marked {len(selected_ids)} receipt(s) as not duplicate"
        )

    def scan_duplicates(self) -> None:
        if not self.current_session:
            return
        groups = self.processor.find_duplicate_groups(self.current_session.id)
        self.load_receipts()
        if groups:
            self.select_duplicate_receipts()
        self.show_feedback(f"Found {groups} possible duplicate group(s)")

    def confirm_successful(self) -> None:
        if not self.current_session:
            return
        group_sizes: dict[str, int] = {}
        flagged = 0
        for item in self.repository.list_receipts(self.current_session.id):
            if item.duplicate_group:
                group_sizes[item.duplicate_group] = (
                    group_sizes.get(item.duplicate_group, 0) + 1
                )
            if item.review_flag:
                flagged += 1
        duplicate_groups = {group for group, size in group_sizes.items() if size >= 2}
        warning = (
            f"\n\nThere are {len(duplicate_groups)} unresolved possible-duplicate group(s)."
            if duplicate_groups
            else ""
        )
        if flagged:
            warning += f"\n\nThere are {flagged} receipt(s) flagged for checking."
        policy = self.settings.video_retention
        managed_video = self._managed_video()
        action = {
            "delete": "The session video will be permanently deleted.",
            "ask": "You will be asked whether to delete the session video.",
            "keep": "The session video will be kept.",
        }.get(policy, "The session video will be kept.")
        if self.current_session.video_path and not managed_video:
            action = "The imported replay source is outside managed storage and will be kept."
        if (
            QMessageBox.question(
                self,
                "Confirm Successful",
                "Confirm that every expected receipt was captured successfully.\n\n"
                + action
                + warning,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        if policy == "delete" and managed_video:
            self._delete_video_files()
        elif (
            policy == "ask"
            and managed_video
            and QMessageBox.question(
                self,
                "Delete session video?",
                "Permanently delete the recovery video now?",
            )
            == QMessageBox.StandardButton.Yes
        ):
            self._delete_video_files()
        self.repository.update_session_status(
            self.current_session.id, SessionStatus.SUCCESSFUL, "Confirmed"
        )
        self._refresh_current()

    def mark_needs_review(self) -> None:
        if self.current_session:
            self.repository.update_session_status(
                self.current_session.id,
                SessionStatus.NEEDS_REVIEW,
                "User marked missing or uncertain receipts",
            )
            self._refresh_current()

    def _delete_video_files(self) -> None:
        if not self.current_session or not self.current_session.video_path:
            return
        delete_session_video(
            self.current_session.video_path,
            self.settings.video_root,
        )
        self.repository.set_video_path(self.current_session.id, None)

    def _managed_video(self) -> bool:
        if not self.current_session or not self.current_session.video_path:
            return False
        path = Path(self.current_session.video_path).resolve()
        root = Path(self.settings.video_root).resolve()
        return path != root and root in path.parents

    def show_video(self) -> None:
        if (
            not self.current_session
            or not self.current_session.video_path
            or not Path(self.current_session.video_path).exists()
        ):
            QMessageBox.information(
                self, "No session video", "This session has no available video."
            )
            return
        try:
            dialog = VideoViewerDialog(
                self.current_session,
                self.repository.list_receipts(self.current_session.id),
                self.repository,
                self.settings,
                self,
            )
            dialog.receipt_added.connect(self._refresh_current)
            dialog.exec()
        except Exception as error:
            QMessageBox.critical(self, "Could not inspect video", str(error))

    def run_recovery(self) -> None:
        if (
            not self.current_session
            or not self.current_session.video_path
            or not Path(self.current_session.video_path).exists()
        ):
            QMessageBox.information(
                self,
                "No session video",
                "A retained session video is required for recovery.",
            )
            return
        try:
            self.repository.update_session_status(
                self.current_session.id,
                SessionStatus.NEEDS_REVIEW,
                "Offline recovery in progress",
            )
            dialog = RecoveryDialog(
                self.current_session, self.repository, self.settings, self
            )
            dialog.receipts_added.connect(self._refresh_current)
            dialog.exec()
            self.repository.update_session_status(
                self.current_session.id,
                SessionStatus.NEEDS_REVIEW,
                "Recovery candidates reviewed",
            )
            self._refresh_current()
        except Exception as error:
            QMessageBox.critical(self, "Could not start recovery", str(error))

    def request_resume(self) -> None:
        if self.current_session:
            self.resume_requested.emit(self.current_session.id)

    def remove_video(self) -> None:
        if not self.current_session or not self.current_session.video_path:
            return
        if (
            QMessageBox.question(
                self,
                "Delete Session Video",
                "Permanently delete this session's source video?\n\n"
                f"Video: {self.current_session.video_path}\n\n"
                "Recognized receipt files will not be touched.",
            )
            == QMessageBox.StandardButton.Yes
        ):
            try:
                self._delete_video_files()
                self._refresh_current()
            except Exception as error:
                QMessageBox.critical(self, "Could not delete video", str(error))

    def remove_all_images(self) -> None:
        if not self.current_session:
            return
        if (
            QMessageBox.question(
                self,
                "Delete Receipt Images",
                "Send all receipt images and preserved originals in this session to the Recycle Bin? "
                "Session history and video will remain.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        for receipt in self.repository.list_receipts(self.current_session.id):
            for path in (
                receipt.processed_path,
                receipt.original_path,
                preview_path(receipt.original_path),
            ):
                if Path(path).exists():
                    send2trash(path)
            self.repository.mark_receipt_deleted(receipt.id)
        self._refresh_current()

    def delete_session(self) -> None:
        sessions = self.selected_sessions()
        if not sessions:
            return
        recordings = [session for session in sessions if session.video_path]
        label = f"{len(sessions)} sessions" if len(sessions) > 1 else "this session"
        if (
            QMessageBox.question(
                self,
                "Delete Session" + ("s" if len(sessions) > 1 else ""),
                f"Permanently delete {label}?\n\n"
                "The following data will be removed:\n"
                "- The source video, including an imported replay video\n"
                "- All database data for the selected session(s)\n\n"
                f"Recordings affected: {len(recordings)}\n\n"
                "Recognized receipt files will NOT be deleted.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        try:
            for session in sessions:
                if session.video_path:
                    delete_session_video(
                        session.video_path, self.settings.video_root
                    )
                    self.repository.set_video_path(session.id, None)
                self.repository.remove_session_history(session.id)
        except Exception as error:
            QMessageBox.critical(self, "Could not delete session", str(error))
            return
        self.current_session = None
        self.current_receipt = None
        self.refresh()

    def selected_sessions(self) -> list[SessionRecord]:
        """Every checked session, newest first, falling back to the current one."""
        chosen = [
            session
            for item in self.sessions.selectedItems()
            if (session := self.repository.get_session(item.data(Qt.ItemDataRole.UserRole)))
        ]
        if chosen:
            return chosen
        return [self.current_session] if self.current_session else []

    def _session_selection_changed(self) -> None:
        count = len(self.sessions.selectedItems())
        self.discard_button.setText(
            f"Discard {count} Sessions" if count > 1 else "Discard Session"
        )
        self.discard_button.setEnabled(count > 0 or self.current_session is not None)

    def _session_files(self, session: SessionRecord) -> list[Path]:
        paths: list[Path] = []
        for receipt in self.repository.list_receipts(session.id):
            paths.extend(
                (
                    Path(receipt.processed_path),
                    Path(receipt.original_path),
                    preview_path(receipt.original_path),
                )
            )
        return [path for path in paths if path.exists()]

    def discard_sessions(self) -> None:
        """Remove a session outright: images, originals, recording and history."""
        sessions = self.selected_sessions()
        if not sessions:
            return
        files = {session.id: self._session_files(session) for session in sessions}
        total_files = sum(len(items) for items in files.values())
        total_bytes = sum(
            path.stat().st_size for items in files.values() for path in items
        )
        recordings = sum(1 for session in sessions if session.video_path)
        label = (
            f"{len(sessions)} sessions"
            if len(sessions) > 1
            else sessions[0].receipt_folder
        )
        if (
            QMessageBox.question(
                self,
                "Discard session" + ("s" if len(sessions) > 1 else ""),
                f"Discard {label}?\n\n"
                f"- {total_files} receipt file(s), {_readable_size(total_bytes)}\n"
                f"- {recordings} recording(s)\n"
                "- All history rows for these sessions\n\n"
                "Files go to the Recycle Bin; history rows are removed outright.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        failures: list[str] = []
        for session in sessions:
            try:
                for path in files[session.id]:
                    send2trash(str(path))
                if session.video_path:
                    delete_session_video(
                        session.video_path, self.settings.video_root, to_trash=True
                    )
                self.repository.remove_session_history(session.id)
            except Exception as error:
                failures.append(f"{session.receipt_folder}: {error}")
        self.current_session = None
        self.current_receipt = None
        self.refresh()
        if failures:
            QMessageBox.warning(
                self, "Some sessions were not fully discarded", "\n".join(failures)
            )
        else:
            self.show_feedback(f"Discarded {len(sessions)} session(s)")

    def _refresh_current(self) -> None:
        if self.current_session:
            session_id = self.current_session.id
            self.current_session = self.repository.get_session(session_id)
            self.refresh(session_id)


class SettingsPage(QWidget):
    settings_saved = Signal(object)

    def __init__(
        self, settings: AppSettings, store: SettingsStore, database_path: Path
    ) -> None:
        super().__init__()
        self.settings = settings
        self.store = store
        self.receipt_root = QLineEdit(settings.receipt_root)
        self.video_root = QLineEdit(settings.video_root)
        self.recording = QComboBox()
        self.recording.addItems(
            ["Record until confirmed", "Always record", "Never record"]
        )
        self.recording.setCurrentIndex(
            {"until_confirmed": 0, "always": 1, "never": 2}.get(
                settings.recording_mode, 0
            )
        )
        self.retention = QComboBox()
        self.retention.addItems(["Delete automatically", "Ask before deleting", "Keep"])
        self.retention.setCurrentIndex(
            {"delete": 0, "ask": 1, "keep": 2}.get(settings.video_retention, 0)
        )
        self.history_cleanup = QComboBox()
        self.history_cleanup.addItems(
            ["Never", "After 7 days", "After 30 days", "After 90 days"]
        )
        self.history_cleanup.setCurrentIndex(
            {0: 0, 7: 1, 30: 2, 90: 3}.get(settings.history_cleanup_days, 0)
        )
        self.format = QComboBox()
        self.format.addItems(["JPEG", "PNG", "PDF"])
        self.format.setCurrentText(settings.output.format)
        self.quality = QSpinBox()
        self.quality.setRange(30, 100)
        self.quality.setValue(settings.output.jpeg_quality)
        self.output_width = QSpinBox()
        self.output_width.setRange(0, 10000)
        self.output_width.setValue(settings.output.max_width)
        self.minimum_long_edge = QSpinBox()
        self.minimum_long_edge.setRange(0, 6000)
        self.minimum_long_edge.setSpecialValueText("Do not upscale")
        self.minimum_long_edge.setSuffix(" px")
        self.minimum_long_edge.setValue(settings.output.minimum_long_edge)
        self.output_rotation = QComboBox()
        self.output_rotation.addItems(["0°", "90°", "180°", "270°"])
        self.output_rotation.setCurrentIndex((settings.output.rotation % 360) // 90)
        self.grayscale = QCheckBox()
        self.grayscale.setChecked(settings.output.grayscale)
        self.perspective = QCheckBox()
        self.perspective.setChecked(settings.output.perspective_correction)
        self.enhancement = QCheckBox()
        self.enhancement.setChecked(settings.output.enhancement)
        self.margin = QDoubleSpinBox()
        self.margin.setRange(0, 20)
        self.margin.setSuffix(" %")
        self.margin.setValue(settings.output.crop_margin_percent)
        self.sharpen = QSpinBox()
        self.sharpen.setRange(0, 3)
        self.sharpen.setValue(settings.output.sharpening)
        self.brightness = QSpinBox()
        self.brightness.setRange(-100, 100)
        self.brightness.setValue(settings.output.brightness)
        self.contrast = QSpinBox()
        self.contrast.setRange(-100, 100)
        self.contrast.setValue(settings.output.contrast)
        self.camera_width = QSpinBox()
        self.camera_width.setRange(160, 10000)
        self.camera_width.setValue(settings.camera_width)
        self.camera_height = QSpinBox()
        self.camera_height.setRange(120, 10000)
        self.camera_height.setValue(settings.camera_height)
        self.camera_fps = QSpinBox()
        self.camera_fps.setRange(1, 240)
        self.camera_fps.setValue(settings.camera_fps)
        self.autofocus = QCheckBox()
        self.autofocus.setChecked(settings.autofocus)
        self.focus_lock = QCheckBox()
        self.focus_lock.setChecked(settings.focus_lock)
        self.exposure_lock = QCheckBox()
        self.exposure_lock.setChecked(settings.exposure_lock)
        self.white_balance_lock = QCheckBox()
        self.white_balance_lock.setChecked(settings.white_balance_lock)
        self.combine_maximum = QSpinBox()
        self.combine_maximum.setRange(2, 16)
        self.combine_maximum.setValue(settings.combine_maximum)
        self.overlay = QCheckBox()
        self.overlay.setChecked(settings.debug_overlay)
        self.buffer = QSpinBox()
        self.buffer.setRange(10, 30)
        self.buffer.setValue(settings.detection.buffer_seconds)
        self.stability = QDoubleSpinBox()
        self.stability.setRange(0.03, 2)
        self.stability.setDecimals(2)
        self.stability.setSingleStep(0.01)
        self.stability.setValue(settings.detection.stable_seconds)
        self.motion = QDoubleSpinBox()
        self.motion.setRange(0.005, 0.5)
        self.motion.setDecimals(3)
        self.motion.setSingleStep(0.005)
        self.motion.setValue(settings.detection.motion_threshold)
        self.capability_status = QLabel(
            "Camera controls are enabled after the selected device reports support."
        )
        self.capability_status.setWordWrap(True)
        self.control_widgets: dict[str, QDoubleSpinBox] = {}
        for name in (
            "focus",
            "exposure",
            "brightness",
            "contrast",
            "white_balance",
            "zoom",
        ):
            widget = QDoubleSpinBox()
            widget.setRange(-100000, 100000)
            widget.setDecimals(2)
            widget.setValue(float(settings.camera_controls.get(name, 0.0)))
            widget.setEnabled(False)
            self.control_widgets[name] = widget
        save = QPushButton("Save Settings")
        save.clicked.connect(self.save)

        form = QFormLayout()
        entries = [
            ("Receipt output folder", self.receipt_root),
            ("Separate video folder", self.video_root),
            ("Metadata database", QLabel(str(database_path))),
            ("Settings file", QLabel(str(store.path))),
            ("Installed version", QLabel(APP_VERSION)),
            ("Recording mode", self.recording),
            ("On successful confirmation", self.retention),
            ("Successful-session history cleanup", self.history_cleanup),
            ("Output format", self.format),
            ("JPEG quality", self.quality),
            ("Maximum output width (0 = original)", self.output_width),
            (
                "Minimum receipt long edge (high-quality upscale)",
                self.minimum_long_edge,
            ),
            ("Default rotation", self.output_rotation),
            ("Grayscale", self.grayscale),
            ("Perspective correction", self.perspective),
            ("Image enhancement", self.enhancement),
            ("Safe crop margin", self.margin),
            ("Sharpening level", self.sharpen),
            ("Brightness correction", self.brightness),
            ("Contrast correction", self.contrast),
            ("Requested camera width", self.camera_width),
            ("Requested camera height", self.camera_height),
            ("Requested frame rate", self.camera_fps),
            ("Autofocus", self.autofocus),
            ("One-shot autofocus then lock", self.focus_lock),
            ("Lock exposure", self.exposure_lock),
            ("Lock white balance", self.white_balance_lock),
            ("Maximum receipts per combined capture", self.combine_maximum),
            ("Diagnostics overlay", self.overlay),
            ("Analysis buffer", self.buffer),
            ("Stable seconds", self.stability),
            ("Motion threshold", self.motion),
            ("Reported camera capabilities", self.capability_status),
        ]
        entries.extend(
            (f"Manual {name.replace('_', ' ')}", widget)
            for name, widget in self.control_widgets.items()
        )
        for label, widget in entries:
            form.addRow(label, widget)
        contents = QWidget()
        contents_layout = QVBoxLayout(contents)
        contents_layout.addLayout(form)
        contents_layout.addWidget(save)
        contents_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(contents)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)

    def save(self) -> None:
        values = self.settings
        values.receipt_root = self.receipt_root.text().strip()
        values.video_root = self.video_root.text().strip()
        values.recording_mode = ["until_confirmed", "always", "never"][
            self.recording.currentIndex()
        ]
        values.video_retention = ["delete", "ask", "keep"][
            self.retention.currentIndex()
        ]
        values.history_cleanup_days = [0, 7, 30, 90][
            self.history_cleanup.currentIndex()
        ]
        values.output.format = self.format.currentText()
        values.output.jpeg_quality = self.quality.value()
        values.output.max_width = self.output_width.value()
        values.output.minimum_long_edge = self.minimum_long_edge.value()
        values.output.rotation = self.output_rotation.currentIndex() * 90
        values.output.grayscale = self.grayscale.isChecked()
        values.output.perspective_correction = self.perspective.isChecked()
        values.output.enhancement = self.enhancement.isChecked()
        values.output.crop_margin_percent = self.margin.value()
        values.output.sharpening = self.sharpen.value()
        values.output.brightness = self.brightness.value()
        values.output.contrast = self.contrast.value()
        values.camera_width = self.camera_width.value()
        values.camera_height = self.camera_height.value()
        values.camera_fps = self.camera_fps.value()
        values.autofocus = self.autofocus.isChecked()
        values.focus_lock = self.focus_lock.isChecked()
        values.exposure_lock = self.exposure_lock.isChecked()
        values.white_balance_lock = self.white_balance_lock.isChecked()
        values.combine_maximum = self.combine_maximum.value()
        values.debug_overlay = self.overlay.isChecked()
        values.detection.buffer_seconds = self.buffer.value()
        values.detection.stable_seconds = self.stability.value()
        values.detection.motion_threshold = self.motion.value()
        values.camera_controls = {
            name: widget.value()
            for name, widget in self.control_widgets.items()
            if widget.isEnabled()
        }
        self.store.save(values)
        Path(values.receipt_root).mkdir(parents=True, exist_ok=True)
        Path(values.video_root).mkdir(parents=True, exist_ok=True)
        self.settings_saved.emit(values)
        QMessageBox.information(
            self, "Settings saved", "New sessions will use these settings."
        )

    def apply_capabilities(self, capabilities: list) -> None:
        reported = {item.name: item for item in capabilities}
        if "autofocus" not in reported:
            return  # A replay source has no camera controls.
        self.autofocus.setEnabled(reported.get("autofocus").supported)
        self.focus_lock.setEnabled(reported.get("autofocus").supported)
        self.exposure_lock.setEnabled(
            bool(reported.get("auto_exposure") and reported["auto_exposure"].supported)
        )
        self.white_balance_lock.setEnabled(
            bool(
                reported.get("auto_white_balance")
                and reported["auto_white_balance"].supported
            )
        )
        supported = []
        for name, widget in self.control_widgets.items():
            capability = reported.get(name)
            widget.setEnabled(bool(capability and capability.supported))
            if capability and capability.supported:
                supported.append(name.replace("_", " "))
                if capability.value is not None:
                    widget.setValue(capability.value)
        width = int(reported.get("width").value) if reported.get("width") else 0
        height = int(reported.get("height").value) if reported.get("height") else 0
        fps = float(reported.get("fps").value) if reported.get("fps") else 0.0
        self.capability_status.setText(
            f"Actual source: {width}x{height} @ {fps:g} fps. Supported controls: "
            + (", ".join(supported) if supported else "basic video only")
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Scan Receipts")
        self.setMinimumSize(760, 480)
        self.resize(1100, 680)
        self.store = SettingsStore()
        self.settings = self.store.load()
        self.repository = Repository()
        self.repository.cleanup_successful_history(self.settings.history_cleanup_days)
        self.controller = SessionController(self.repository, self.settings)
        self.tabs = QTabWidget()
        self.scan_page = ScanPage(self.controller, self.settings)
        self.sessions_page = SessionsPage(self.repository, self.settings)
        self.settings_page = SettingsPage(
            self.settings, self.store, self.repository.path
        )
        self.tabs.addTab(self.scan_page, "Scan")
        self.tabs.addTab(self.sessions_page, "Sessions & Review")
        self.tabs.addTab(self.settings_page, "Settings")
        self.setCentralWidget(self.tabs)
        self.controller.session_started.connect(self._session_started)
        self.controller.receipt_saved.connect(self._receipt_saved)
        self.controller.session_finished.connect(self._session_finished)
        self.controller.error.connect(
            lambda message: QMessageBox.critical(self, "Capture problem", message)
        )
        self.controller.notice.connect(
            lambda message: self.statusBar().showMessage(message, 10000)
        )
        self.settings_page.settings_saved.connect(self._settings_saved)
        self.sessions_page.resume_requested.connect(self._resume_session)
        self.controller.capabilities_ready.connect(
            self.settings_page.apply_capabilities
        )
        self.scan_page.settings_changed.connect(
            lambda: self.store.save(self.settings)
        )
        self.scan_page.capabilities_ready.connect(self.settings_page.apply_capabilities)

    def _session_started(self, session: SessionRecord) -> None:
        self.sessions_page.refresh(session.id)

    def _receipt_saved(self, receipt: ReceiptRecord) -> None:
        self.sessions_page.refresh(receipt.session_id)

    def _session_finished(self, session: SessionRecord) -> None:
        self.sessions_page.refresh(session.id)
        self.tabs.setCurrentWidget(self.sessions_page)

    def _settings_saved(self, settings: AppSettings) -> None:
        self.controller.settings = settings
        self.scan_page.settings = settings
        self.sessions_page.settings = settings
        self.scan_page._update_folder_value()

    def _resume_session(self, session_id: str) -> None:
        self.scan_page.prepare_resume(session_id)
        self.tabs.setCurrentWidget(self.scan_page)
        self.statusBar().showMessage(
            "Choose the reconnected camera, then click Resume This Session.", 15000
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.controller.active:
            answer = QMessageBox.question(
                self,
                "Session is still scanning",
                "Stop the session and safely drain pending receipt processing before closing?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.controller.stop()
            QMessageBox.information(
                self,
                "Finishing session",
                "The window will remain open until pending captures are saved. Close it again afterward.",
            )
            event.ignore()
            return
        self.scan_page.shutdown()
        event.accept()

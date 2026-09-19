from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QItemSelectionModel, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
)

from scan_receipts.config import SettingsStore, default_settings
from scan_receipts.controller import SessionController
from scan_receipts.database import Repository
from scan_receipts.models import CaptureCandidate
from scan_receipts.processing import ReceiptProcessor
from scan_receipts.ui import (
    DuplicateDeckDialog,
    MainWindow,
    ScanPage,
    SelectableLabel,
    SessionsPage,
    SettingsPage,
    SmoothScroll,
    install_smooth_scroll,
)
from scan_receipts.version import APP_VERSION

pytestmark = pytest.mark.gui


def receipt_frame() -> np.ndarray:
    frame = np.full((600, 800, 3), 35, np.uint8)
    corners = np.array([[190, 45], [610, 60], [635, 555], [170, 540]], np.int32)
    cv2.fillConvexPoly(frame, corners, (245, 245, 238))
    cv2.polylines(frame, [corners], True, (255, 255, 255), 4)
    for y in range(140, 500, 45):
        cv2.line(frame, (230, y), (570, y), (70, 70, 70), 3)
    return frame


def test_batch_rotation_keeps_selection_and_current_preview(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Test camera", settings)
    processor = ReceiptProcessor(repository)
    frame = receipt_frame()
    corners = np.array([[190, 45], [610, 60], [635, 555], [170, 540]], np.float32)
    for index in range(2):
        processor.save_candidate(
            session.id,
            CaptureCandidate(index * 2.0, frame, corners, 0.95, 0.9, index),
            settings.output,
        )

    page = SessionsPage(repository, settings)
    qtbot.addWidget(page)
    page.refresh(session.id)
    first, second = page.receipts.item(0), page.receipts.item(1)
    first.setSelected(True)
    second.setSelected(True)
    page.receipts.setCurrentItem(first, QItemSelectionModel.SelectionFlag.NoUpdate)
    current_id = first.data(Qt.ItemDataRole.UserRole)

    page.rotate(-90)

    assert page.current_receipt is not None
    assert page.current_receipt.id == current_id
    assert len(page.receipts.selectedItems()) == 2
    assert all(
        '"rotation": 270' in repository.get_receipt(receipt.id).edit_json
        for receipt in repository.list_receipts(session.id)
    )
    assert (
        page.folder_path.textInteractionFlags()
        == Qt.TextInteractionFlag.NoTextInteraction
    )
    assert not page.duplicate_decks_scroll.isHidden()
    deck = page.duplicate_decks_layout.itemAt(0).widget()
    assert deck is not None
    assert deck.text() == "Review 2 possible duplicates"

    copy_icon_key = page.copy_folder.icon().cacheKey()
    page.copy_folder.click()

    assert QApplication.clipboard().text() == session.receipt_folder
    assert page.edit_status.text() == "Receipt folder path copied"
    assert page.copy_folder.icon().cacheKey() != copy_icon_key
    assert page.copy_folder.toolTip() == "Copied"
    assert page.receipts.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu

    records = repository.list_receipts(session.id)
    group = records[0].duplicate_group
    assert group
    dialog = DuplicateDeckDialog(repository, group)
    qtbot.addWidget(dialog)
    assert dialog.cards_layout.count() - 1 == 2
    assert not dialog.keep_selected_button.isEnabled()
    chosen = records[1]
    qtbot.mouseClick(dialog.card_buttons[chosen.id], Qt.MouseButton.LeftButton)
    assert dialog.selected_receipt is not None
    assert dialog.selected_receipt.id == chosen.id
    assert dialog.card_buttons[chosen.id].isChecked()
    assert dialog.keep_selected_button.isEnabled()
    assert chosen.filename in dialog.selection.text()
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        DuplicateDeckDialog,
        "trash_receipt",
        staticmethod(lambda repo, receipt: repo.mark_receipt_deleted(receipt.id)),
    )

    qtbot.mouseClick(dialog.keep_selected_button, Qt.MouseButton.LeftButton)

    assert [item.id for item in repository.list_receipts(session.id)] == [chosen.id]

    processor.save_candidate(
        session.id,
        CaptureCandidate(10.0, frame, corners, 0.95, 0.9, 10),
        settings.output,
    )
    records = repository.list_receipts(session.id)
    group = records[0].duplicate_group
    assert group
    keep_all_dialog = DuplicateDeckDialog(repository, group)
    qtbot.addWidget(keep_all_dialog)
    qtbot.mouseClick(keep_all_dialog.keep_all_button, Qt.MouseButton.LeftButton)

    records = repository.list_receipts(session.id)
    assert len(records) == 2
    assert all(item.duplicate_group is None for item in records)


def test_duplicate_deck_removes_every_copy_in_the_group(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    repository, session, _settings = saved_session(tmp_path)
    receipts = repository.list_receipts(session.id)
    group = receipts[0].duplicate_group
    assert group
    dialog = DuplicateDeckDialog(repository, group)
    qtbot.addWidget(dialog)
    trashed: list[str] = []
    monkeypatch.setattr(
        "scan_receipts.processing.send2trash", lambda path: trashed.append(str(path))
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )

    qtbot.mouseClick(dialog.delete_all_button, Qt.MouseButton.LeftButton)

    assert repository.list_receipts(session.id) == []
    for receipt in receipts:
        assert receipt.processed_path in trashed
        assert receipt.original_path in trashed


def test_review_refreshes_when_active_session_saves_receipts(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr("scan_receipts.ui.enumerate_cameras", lambda _settings: [])
    window = MainWindow()
    qtbot.addWidget(window)
    refresh = Mock()
    window.sessions_page.refresh = refresh
    session = window.repository.create_session("Test camera", window.settings)

    window.controller.session_started.emit(session)
    window.controller.receipt_saved.emit(SimpleNamespace(session_id=session.id))

    assert refresh.call_args_list[0].args == (session.id,)
    assert refresh.call_args_list[1].args == (session.id,)


def saved_session(tmp_path: Path, captures: int = 2):
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    repository = Repository(tmp_path / "history.sqlite3")
    session = repository.create_session("Test camera", settings)
    processor = ReceiptProcessor(repository)
    frame = receipt_frame()
    corners = np.array([[190, 45], [610, 60], [635, 555], [170, 540]], np.float32)
    for index in range(captures):
        processor.save_candidate(
            session.id,
            CaptureCandidate(index * 2.0, frame, corners, 0.95, 0.9, index),
            settings.output,
        )
    return repository, session, settings


def scanning_page(qtbot, tmp_path: Path, monkeypatch, captures: int = 2):
    monkeypatch.setattr("scan_receipts.ui.enumerate_cameras", lambda _settings: [])
    repository, session, settings = saved_session(tmp_path, captures)
    page = ScanPage(SessionController(repository, settings), settings)
    qtbot.addWidget(page)
    page.on_saved(repository.list_receipts(session.id)[-1])
    return page, repository, session


def test_capture_strip_lists_saved_receipts_newest_first(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session = scanning_page(qtbot, tmp_path, monkeypatch)

    saved = repository.list_receipts(session.id)
    strip_ids = [
        page.captures.item(row).data(Qt.ItemDataRole.UserRole)
        for row in range(page.captures.count())
    ]

    assert strip_ids == [receipt.id for receipt in reversed(saved)]
    assert not page.captures.preview_buttons[saved[-1].id].icon().isNull()


def test_capture_strip_flags_a_receipt_for_later_review(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    newest = repository.list_receipts(session.id)[-1]

    qtbot.mouseClick(page.captures.flag_buttons[newest.id], Qt.MouseButton.LeftButton)

    assert repository.get_receipt(newest.id).review_flag is True
    assert page.captures.flag_buttons[newest.id].isChecked()

    qtbot.mouseClick(page.captures.flag_buttons[newest.id], Qt.MouseButton.LeftButton)

    assert repository.get_receipt(newest.id).review_flag is False


def test_capture_strip_delete_trashes_files_and_updates_the_counter(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    newest = repository.list_receipts(session.id)[-1]
    trashed: list[str] = []
    monkeypatch.setattr(
        "scan_receipts.processing.send2trash", lambda path: trashed.append(str(path))
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )

    qtbot.mouseClick(page.captures.delete_buttons[newest.id], Qt.MouseButton.LeftButton)

    assert newest.id not in [item.id for item in repository.list_receipts(session.id)]
    assert newest.processed_path in trashed
    assert newest.original_path in trashed
    assert page.captures.delete_buttons.keys() == {
        item.id for item in repository.list_receipts(session.id)
    }
    assert page.counter.text() == "Receipts captured: 1"


def test_capture_strip_opens_the_capture_full_size(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    newest = repository.list_receipts(session.id)[-1]

    qtbot.mouseClick(
        page.captures.preview_buttons[newest.id], Qt.MouseButton.LeftButton
    )

    dialog = page.capture_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert not dialog.isModal()
    assert newest.filename in dialog.windowTitle()


def test_capture_cards_keep_the_filename_off_the_list_background(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    newest = repository.list_receipts(session.id)[-1]

    assert page.captures.item(0).text() == ""
    assert page.captures.preview_buttons[newest.id].text() == newest.filename


def test_stopping_a_session_returns_the_scan_page_to_its_idle_state(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    page.on_started(session)
    page.on_saved(repository.list_receipts(session.id)[-1])
    for receipt in repository.list_receipts(session.id):
        page.captures.combine_buttons[receipt.id].setChecked(True)
    assert page.captures.count() == 2
    assert page.combine_button.isEnabled()

    page.on_finished(session)

    assert page.captures.count() == 1
    assert page.captures.item(0).text() == "Captured receipts appear here"
    assert page.captures.preview_buttons == {}
    assert page.captures.combine_selection() == []
    assert not page.combine_button.isEnabled()
    assert page.counter.text() == "Receipts captured: 0"
    assert page.state.text() == "READY"
    assert page.diagnostics.text() == ScanPage.IDLE_DIAGNOSTICS
    assert "Automatic under" in page.folder_value.text()


def test_review_page_toggles_and_shows_the_review_flag(qtbot, tmp_path: Path) -> None:
    repository, session, settings = saved_session(tmp_path)
    page = SessionsPage(repository, settings)
    qtbot.addWidget(page)
    page.refresh(session.id)
    flagged = repository.list_receipts(session.id)[0]
    page.receipts.item(0).setSelected(True)

    page.toggle_review_flag()

    assert repository.get_receipt(flagged.id).review_flag is True
    assert "FLAGGED" in page.receipts.item(0).text()

    page.toggle_review_flag()

    assert repository.get_receipt(flagged.id).review_flag is False
    assert "FLAGGED" not in page.receipts.item(0).text()


def test_confirm_successful_warns_about_flagged_receipts(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    repository, session, settings = saved_session(tmp_path)
    page = SessionsPage(repository, settings)
    qtbot.addWidget(page)
    page.refresh(session.id)
    repository.set_review_flag(repository.list_receipts(session.id)[0].id, True)
    asked: list[str] = []

    def question(_parent, _title, message, *args, **kwargs):
        asked.append(message)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", question)

    page.confirm_successful()

    assert "1 receipt(s) flagged for checking" in asked[0]


def drag(qtbot, widget, start: QPoint, end: QPoint) -> None:
    qtbot.mousePress(widget, Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(widget, end)
    qtbot.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=end)


def test_dragged_selection_maps_to_relative_frame_coordinates(qtbot) -> None:
    label = SelectableLabel("preview")
    qtbot.addWidget(label)
    label.resize(480, 360)
    label.set_image(QImage(800, 600, QImage.Format.Format_RGB888))

    drag(qtbot, label, QPoint(120, 90), QPoint(360, 270))

    x, y, width, height = label.source_crop_relative()
    assert (round(x, 2), round(y, 2)) == (0.25, 0.25)
    assert (round(width, 2), round(height, 2)) == (0.5, 0.5)


def test_manual_capture_controls_are_disabled_until_a_session_starts(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, session = scanning_page(qtbot, tmp_path, monkeypatch)

    assert not page.capture_button.isEnabled()

    page.on_started(session)
    assert page.capture_button.isEnabled()

    page.on_finished(session)
    assert not page.capture_button.isEnabled()


def test_capture_button_without_a_selection_captures_the_whole_frame(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    page.on_started(session)
    requested: list = []
    monkeypatch.setattr(
        page.controller,
        "request_manual_capture",
        lambda region, exact_area=False: requested.append(region),
    )

    qtbot.mouseClick(page.capture_button, Qt.MouseButton.LeftButton)

    assert requested == [None]


def test_the_selection_clears_once_it_has_been_captured(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    page.on_started(session)
    page.preview.resize(480, 360)
    page.preview.set_image(QImage(800, 600, QImage.Format.Format_RGB888))
    drag(qtbot, page.preview, QPoint(120, 90), QPoint(360, 270))
    requested: list = []
    monkeypatch.setattr(
        page.controller,
        "request_manual_capture",
        lambda region, exact_area=False: requested.append(region),
    )

    qtbot.mouseClick(page.capture_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(page.capture_button, Qt.MouseButton.LeftButton)

    assert requested[0] is not None
    assert requested[1] is None
    assert page.preview.source_crop_relative() is None


def test_auto_capture_checkbox_switches_automatic_capture_on_the_controller(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    page.on_started(session)
    switched: list = []
    monkeypatch.setattr(page.controller, "set_auto_capture", switched.append)

    page.auto_check.setChecked(False)
    page.auto_check.setChecked(True)

    assert switched == [False, True]


def test_exact_area_mode_is_sent_with_the_capture_and_remembered(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    page.on_started(session)
    requested: list = []
    monkeypatch.setattr(
        page.controller,
        "request_manual_capture",
        lambda region, exact_area=False: requested.append((region, exact_area)),
    )

    qtbot.mouseClick(page.capture_button, Qt.MouseButton.LeftButton)
    page.exact_area.setChecked(True)
    qtbot.mouseClick(page.capture_button, Qt.MouseButton.LeftButton)

    assert [item[1] for item in requested] == [False, True]
    assert page.settings.manual_capture_mode == "exact"


def review_page(qtbot, tmp_path: Path, captures: int = 4):
    repository, session, settings = saved_session(tmp_path, captures)
    page = SessionsPage(repository, settings)
    qtbot.addWidget(page)
    page.refresh(session.id)
    return page, repository, session


def test_deleting_a_receipt_keeps_the_selection_on_its_neighbour(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session = review_page(qtbot, tmp_path, captures=4)
    monkeypatch.setattr("scan_receipts.ui.send2trash", lambda _path: None)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )
    third = page.receipts.item(2)
    page.receipts.setCurrentItem(third)
    doomed = third.data(Qt.ItemDataRole.UserRole)
    survivor = page.receipts.item(3).data(Qt.ItemDataRole.UserRole)
    third.setSelected(True)

    page.remove_selected()

    assert doomed not in [item.id for item in repository.list_receipts(session.id)]
    assert page.current_receipt is not None
    assert page.current_receipt.id == survivor


def test_sessions_can_be_selected_together_and_discarded(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    repository, first, settings = saved_session(tmp_path, 2)
    second = repository.create_session("Test camera", settings)
    page = SessionsPage(repository, settings)
    qtbot.addWidget(page)
    page.refresh(first.id)
    trashed: list[str] = []
    monkeypatch.setattr(
        "scan_receipts.ui.send2trash", lambda path: trashed.append(str(path))
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )
    receipts = repository.list_receipts(first.id)
    for row in range(page.sessions.count()):
        page.sessions.item(row).setSelected(True)

    page.discard_sessions()

    assert repository.get_session(first.id) is None
    assert repository.get_session(second.id) is None
    assert page.sessions.count() == 0
    for receipt in receipts:
        assert receipt.processed_path in trashed
        assert receipt.original_path in trashed


def test_lists_scroll_by_pixel_so_dragging_is_smooth(qtbot, tmp_path: Path) -> None:
    page, _repository, _session = review_page(qtbot, tmp_path)
    per_pixel = QAbstractItemView.ScrollMode.ScrollPerPixel

    assert page.receipts.verticalScrollMode() == per_pixel
    assert page.sessions.verticalScrollMode() == per_pixel


def test_the_capture_strip_scrolls_by_pixel_too(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, _session = scanning_page(qtbot, tmp_path, monkeypatch)

    assert (
        page.captures.verticalScrollMode()
        == QAbstractItemView.ScrollMode.ScrollPerPixel
    )


def scrollable_area(qtbot) -> QScrollArea:
    area = QScrollArea()
    content = QLabel()
    content.setFixedSize(200, 4000)
    area.setWidget(content)
    area.resize(300, 300)
    qtbot.addWidget(area)
    area.show()
    qtbot.waitExposed(area)
    return area


def wheel(area: QScrollArea, notches: int) -> None:
    event = QWheelEvent(
        QPointF(10, 10),
        QPointF(area.mapToGlobal(QPoint(10, 10))),
        QPoint(0, 0),
        QPoint(0, notches * 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(area.viewport(), event)


def test_wheel_scrolling_glides_to_the_notch_target(qtbot) -> None:
    area = scrollable_area(qtbot)
    install_smooth_scroll(area)
    bar = area.verticalScrollBar()

    wheel(area, -1)

    assert bar.value() < SmoothScroll.WHEEL_PIXELS
    qtbot.waitUntil(lambda: bar.value() == SmoothScroll.WHEEL_PIXELS)


def test_a_second_notch_extends_the_glide_instead_of_restarting_it(qtbot) -> None:
    area = scrollable_area(qtbot)
    install_smooth_scroll(area)
    bar = area.verticalScrollBar()

    wheel(area, -1)
    wheel(area, -1)

    qtbot.waitUntil(lambda: bar.value() == 2 * SmoothScroll.WHEEL_PIXELS)


def test_wheel_scrolling_stops_at_the_end_of_the_range(qtbot) -> None:
    area = scrollable_area(qtbot)
    smooth = install_smooth_scroll(area)
    bar = area.verticalScrollBar()

    smooth.scroll_by(10 * bar.maximum())

    qtbot.waitUntil(lambda: bar.value() == bar.maximum())


def combined_review_page(qtbot, tmp_path: Path, captures: int = 3):
    repository, session, settings = saved_session(tmp_path, captures)
    page = SessionsPage(repository, settings)
    qtbot.addWidget(page)
    page.refresh(session.id)
    return page, repository, session, settings


def select_rows(page, rows: list[int]) -> None:
    page.receipts.clearSelection()
    for row in rows:
        page.receipts.item(row).setSelected(True)
    page.receipts.setCurrentItem(
        page.receipts.item(rows[0]), QItemSelectionModel.SelectionFlag.NoUpdate
    )


def test_review_page_combines_the_selected_receipts_into_one_sheet(
    qtbot, tmp_path: Path
) -> None:
    page, repository, session, _settings = combined_review_page(qtbot, tmp_path)
    sources = repository.list_receipts(session.id)
    select_rows(page, [0, 1, 2])

    qtbot.mouseClick(page.combine_button, Qt.MouseButton.LeftButton)

    listed = repository.list_receipts(session.id)
    assert len(listed) == 1
    assert page.receipts.count() == 1
    assert all(
        repository.get_receipt(item.id).combined_into == listed[0].id
        for item in sources
    )
    assert all(Path(item.processed_path).exists() for item in sources)


def test_review_page_refuses_to_combine_more_than_the_maximum(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session, settings = combined_review_page(qtbot, tmp_path)
    settings.combine_maximum = 2
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, _title, text, *args, **kwargs: warnings.append(text),
    )
    select_rows(page, [0, 1, 2])

    qtbot.mouseClick(page.combine_button, Qt.MouseButton.LeftButton)

    assert len(repository.list_receipts(session.id)) == 3
    assert warnings and "2" in warnings[0]


def test_review_page_uncombines_back_to_the_original_receipts(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("scan_receipts.processing.send2trash", lambda path: None)
    page, repository, session, _settings = combined_review_page(qtbot, tmp_path)
    sources = repository.list_receipts(session.id)
    select_rows(page, [0, 1, 2])
    qtbot.mouseClick(page.combine_button, Qt.MouseButton.LeftButton)

    qtbot.mouseClick(page.uncombine_button, Qt.MouseButton.LeftButton)

    assert [item.id for item in repository.list_receipts(session.id)] == [
        item.id for item in sources
    ]
    assert page.receipts.count() == 3


def test_deleting_a_combined_receipt_also_deletes_what_it_absorbed(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    trashed: list[str] = []
    monkeypatch.setattr(
        "scan_receipts.processing.send2trash", lambda path: trashed.append(str(path))
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )
    page, repository, session, _settings = combined_review_page(qtbot, tmp_path)
    sources = repository.list_receipts(session.id)
    select_rows(page, [0, 1, 2])
    qtbot.mouseClick(page.combine_button, Qt.MouseButton.LeftButton)

    select_rows(page, [0])
    page.remove_selected()

    assert repository.list_receipts(session.id) == []
    assert all(item.processed_path in trashed for item in sources)


def test_capture_strip_combines_the_ticked_captures(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    sources = repository.list_receipts(session.id)
    for receipt in sources:
        page.captures.combine_buttons[receipt.id].setChecked(True)

    qtbot.mouseClick(page.combine_button, Qt.MouseButton.LeftButton)

    listed = repository.list_receipts(session.id)
    assert len(listed) == 1
    assert page.captures.combine_buttons.keys() == {listed[0].id}
    assert page.counter.text() == "Receipts captured: 1"


def test_capture_strip_keeps_combine_ticks_when_a_new_capture_arrives(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    first = repository.list_receipts(session.id)[0]
    page.captures.combine_buttons[first.id].setChecked(True)

    page.refresh_captures()

    assert page.captures.combine_selection() == [first.id]
    assert page.captures.combine_buttons[first.id].isChecked()


def test_the_update_button_sits_on_the_scan_top_bar_with_the_version(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, _session = scanning_page(qtbot, tmp_path, monkeypatch)

    assert page.update_button.text() == "Check for updates"
    assert page.version_label.text() == f"v{APP_VERSION}"
    assert page.update_button.isEnabled()


def test_the_update_button_is_blocked_while_a_session_runs(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, session = scanning_page(qtbot, tmp_path, monkeypatch)

    page.on_started(session)
    assert not page.update_button.isEnabled()

    page.on_finished(session)
    assert page.update_button.isEnabled()


def scan_release(version: str = "0.1.99"):
    from scan_receipts.update import ReleaseInfo

    return ReleaseInfo(
        version=version,
        tag=f"v{version}",
        download_url=(
            "https://github.com/owner/repo/releases/download/"
            f"v{version}/ScanReceipts-Setup.exe"
        ),
        page_url=f"https://github.com/owner/repo/releases/tag/v{version}",
    )


def test_being_up_to_date_tells_the_user_and_downloads_nothing(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, _session = scanning_page(qtbot, tmp_path, monkeypatch)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda *arguments: shown.append(arguments[2])
    )
    requested: list = []
    page.request_download.connect(requested.append)

    page._update_not_needed(APP_VERSION)

    assert shown and APP_VERSION in shown[0]
    assert requested == []
    assert page.update_button.isEnabled()


def test_a_newer_version_is_downloaded_once_the_user_confirms(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, _session = scanning_page(qtbot, tmp_path, monkeypatch)
    monkeypatch.setattr("scan_receipts.ui.is_frozen", lambda: True)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_arguments: QMessageBox.StandardButton.Yes
    )
    requested: list = []
    page.request_download.connect(requested.append)

    page._update_available(scan_release())

    assert requested == [scan_release()]


def test_declining_the_update_downloads_nothing(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, _session = scanning_page(qtbot, tmp_path, monkeypatch)
    monkeypatch.setattr("scan_receipts.ui.is_frozen", lambda: True)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *_arguments: QMessageBox.StandardButton.No
    )
    requested: list = []
    page.request_download.connect(requested.append)

    page._update_available(scan_release())

    assert requested == []
    assert page.update_button.isEnabled()


def test_a_source_checkout_is_never_installed_over(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, _session = scanning_page(qtbot, tmp_path, monkeypatch)
    monkeypatch.setattr("scan_receipts.ui.is_frozen", lambda: False)
    shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda *arguments: shown.append(arguments[2])
    )
    requested: list = []
    page.request_download.connect(requested.append)

    page._update_available(scan_release())

    assert requested == []
    assert shown and "0.1.99" in shown[0]


def test_the_downloaded_installer_is_launched_and_the_app_quits(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, _session = scanning_page(qtbot, tmp_path, monkeypatch)
    installer = tmp_path / "ScanReceipts-Setup.exe"
    installer.write_bytes(b"stub")
    launched: list = []
    quit_calls: list[bool] = []
    monkeypatch.setattr("scan_receipts.ui.run_installer", launched.append)
    # CLAUDE CODE: PySide6 types reject setattr, so replace the module-level name.
    monkeypatch.setattr(
        "scan_receipts.ui.QApplication",
        SimpleNamespace(quit=lambda: quit_calls.append(True)),
    )

    page._installer_ready(installer)

    assert launched == [installer]
    assert quit_calls == [True]


def test_an_update_failure_warns_and_re_enables_the_button(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, _repository, _session = scanning_page(qtbot, tmp_path, monkeypatch)
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "warning", lambda *arguments: warnings.append(arguments[2])
    )

    page._update_failed("Could not reach GitHub: offline")

    assert warnings == ["Could not reach GitHub: offline"]
    assert page.update_button.isEnabled()


def test_settings_shows_the_installed_version_without_a_second_button(
    qtbot, tmp_path: Path
) -> None:
    settings = default_settings()
    settings.receipt_root = str(tmp_path / "Receipts")
    settings.video_root = str(tmp_path / "Videos")
    store = SettingsStore(tmp_path / "settings.json")
    repository = Repository(tmp_path / "history.sqlite3")
    page = SettingsPage(settings, store, repository.path)
    qtbot.addWidget(page)

    labels = [widget.text() for widget in page.findChildren(QLabel) if widget.text()]
    buttons = [
        widget.text() for widget in page.findChildren(QPushButton) if widget.text()
    ]

    assert "Installed version" in labels
    assert APP_VERSION in labels
    assert "Check for updates" not in buttons


def test_duplicate_deck_reports_copies_it_could_not_remove(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    repository, session, _settings = saved_session(tmp_path, 3)
    records = repository.list_receipts(session.id)
    group = records[0].duplicate_group
    assert group
    blocked = records[2]

    def refuse_blocked(path: str) -> None:
        if path == blocked.processed_path:
            raise OSError(5, "The Recycle Bin is unavailable", path)
        os.remove(path)

    monkeypatch.setattr("scan_receipts.processing.send2trash", refuse_blocked)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )
    reported: list[str] = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda _parent, _title, text, *a, **k: reported.append(text)
    )
    dialog = DuplicateDeckDialog(repository, group)
    qtbot.addWidget(dialog)
    dialog.select_card(records[0])

    qtbot.mouseClick(dialog.keep_selected_button, Qt.MouseButton.LeftButton)

    assert reported and blocked.filename in reported[0]
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert [item.id for item in repository.list_receipts(session.id)] == [
        records[0].id,
        blocked.id,
    ]


def test_capture_strip_reports_a_capture_it_could_not_delete(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session = scanning_page(qtbot, tmp_path, monkeypatch)
    newest = repository.list_receipts(session.id)[-1]
    monkeypatch.setattr(
        "scan_receipts.processing.send2trash",
        lambda path: (_ for _ in ()).throw(OSError(5, "Recycle Bin unavailable", path)),
    )
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )
    reported: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, text, *a, **k: reported.append(text),
    )

    qtbot.mouseClick(page.captures.delete_buttons[newest.id], Qt.MouseButton.LeftButton)

    assert reported and newest.filename in reported[0]
    assert newest.id in [item.id for item in repository.list_receipts(session.id)]


def test_review_page_reports_receipt_images_it_could_not_delete(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    page, repository, session, _settings = combined_review_page(qtbot, tmp_path)
    receipts = repository.list_receipts(session.id)
    blocked = receipts[1]

    def refuse_blocked(path: str) -> None:
        if path == blocked.processed_path:
            raise OSError(5, "Recycle Bin unavailable", path)
        os.remove(path)

    monkeypatch.setattr("scan_receipts.processing.send2trash", refuse_blocked)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes
    )
    reported: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, text, *a, **k: reported.append(text),
    )
    select_rows(page, [0, 1])

    page.remove_selected()

    assert reported and blocked.filename in reported[0]
    assert [item.id for item in repository.list_receipts(session.id)] == [
        blocked.id,
        receipts[2].id,
    ]

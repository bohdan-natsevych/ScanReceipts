from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .config import app_data_dir
from .models import AppSettings, ReceiptRecord, SessionRecord, SessionStatus, utc_now

SCHEMA_VERSION = 4


class Repository:
    """Thread-safe SQLite session history and collision-proof file allocation."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or app_data_dir() / "scan_receipts.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version and version != SCHEMA_VERSION:
                shutil.copy2(
                    self.path, self.path.with_suffix(f".v{version}.backup.sqlite3")
                )
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    camera_name TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    receipt_folder TEXT NOT NULL,
                    video_path TEXT,
                    status TEXT NOT NULL,
                    processing_status TEXT NOT NULL DEFAULT 'Idle',
                    receipt_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    captured_at REAL NOT NULL,
                    filename TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    processed_path TEXT NOT NULL,
                    edit_json TEXT NOT NULL DEFAULT '{}',
                    content_hash TEXT NOT NULL DEFAULT '',
                    duplicate_group TEXT,
                    quality_flag TEXT,
                    review_flag INTEGER NOT NULL DEFAULT 0,
                    combined_into INTEGER REFERENCES receipts(id),
                    deleted INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(session_id, sequence),
                    UNIQUE(session_id, filename)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_receipts_session_time
                    ON receipts(session_id, captured_at);
                CREATE INDEX IF NOT EXISTS idx_events_session
                    ON events(session_id, id);
                """
            )
            columns = {
                row[1] for row in db.execute("PRAGMA table_info(receipts)").fetchall()
            }
            if "quality_flag" not in columns:
                db.execute("ALTER TABLE receipts ADD COLUMN quality_flag TEXT")
            if "review_flag" not in columns:
                db.execute(
                    "ALTER TABLE receipts ADD COLUMN review_flag INTEGER NOT NULL DEFAULT 0"
                )
            if "combined_into" not in columns:
                db.execute(
                    "ALTER TABLE receipts ADD COLUMN combined_into INTEGER "
                    "REFERENCES receipts(id)"
                )
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self.recover_interrupted_sessions()
        self._periodic_backup()

    def _periodic_backup(self) -> None:
        backup = self.path.with_name("scan_receipts.daily.sqlite3")
        if (
            backup.exists()
            and datetime.fromtimestamp(backup.stat().st_mtime, UTC).date()
            == datetime.now(UTC).date()
        ):
            return
        with (
            self._lock,
            self._connect() as source,
            closing(sqlite3.connect(backup)) as destination,
        ):
            source.backup(destination)

    def recover_interrupted_sessions(self) -> int:
        with self._lock, self._connect() as db:
            cursor = db.execute(
                """UPDATE sessions
                   SET status=?, processing_status='Interrupted - review required', ended_at=COALESCE(ended_at, ?)
                   WHERE status IN (?, ?)""",
                (
                    SessionStatus.NEEDS_REVIEW.value,
                    utc_now(),
                    SessionStatus.SCANNING.value,
                    SessionStatus.PROCESSING.value,
                ),
            )
            return cursor.rowcount

    @staticmethod
    def _session_folder(root: Path, session_date: str) -> Path:
        day = root / session_date
        day.mkdir(parents=True, exist_ok=True)
        used = []
        for path in day.glob("Session_*"):
            try:
                used.append(int(path.name.rsplit("_", 1)[1]))
            except ValueError:
                continue
        folder = day / f"Session_{max(used, default=0) + 1:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "Originals").mkdir()
        return folder

    def create_session(
        self,
        camera_name: str,
        settings: AppSettings,
        receipt_folder: str | Path | None = None,
    ) -> SessionRecord:
        started = datetime.now().astimezone()
        session_id = str(uuid.uuid4())
        if receipt_folder is None:
            folder = self._session_folder(
                Path(settings.receipt_root), started.date().isoformat()
            )
        else:
            folder = Path(receipt_folder).expanduser().resolve()
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "Originals").mkdir(exist_ok=True)
        video_folder = (
            Path(settings.video_root) / started.date().isoformat() / session_id
        )
        video_path = None
        if settings.recording_mode != "never":
            video_folder.mkdir(parents=True, exist_ok=True)
            video_path = str(video_folder)
        record = SessionRecord(
            id=session_id,
            started_at=started.isoformat(timespec="milliseconds"),
            ended_at=None,
            camera_name=camera_name,
            settings_json=json.dumps(settings.to_dict()),
            receipt_folder=str(folder),
            video_path=video_path,
            status=SessionStatus.SCANNING,
            processing_status="Capturing",
        )
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO sessions
                   (id, started_at, ended_at, camera_name, settings_json, receipt_folder,
                    video_path, status, processing_status, receipt_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    record.started_at,
                    record.ended_at,
                    record.camera_name,
                    record.settings_json,
                    record.receipt_folder,
                    record.video_path,
                    record.status.value,
                    record.processing_status,
                    0,
                ),
            )
        return record

    def update_actual_capture(
        self, session_id: str, width: int, height: int, fps: float
    ) -> None:
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT settings_json FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if not row:
                return
            settings = json.loads(row[0])
            settings["actual_capture"] = {
                "width": width,
                "height": height,
                "fps": fps,
            }
            db.execute(
                "UPDATE sessions SET settings_json=? WHERE id=?",
                (json.dumps(settings), session_id),
            )

    def update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        processing_status: str,
        *,
        ended: bool = False,
    ) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                """UPDATE sessions SET status=?, processing_status=?,
                   ended_at=CASE WHEN ? THEN COALESCE(ended_at, ?) ELSE ended_at END
                   WHERE id=?""",
                (status.value, processing_status, ended, utc_now(), session_id),
            )

    def resume_session(self, session_id: str) -> SessionRecord:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE sessions SET status=?, processing_status='Capturing - resumed' WHERE id=?",
                (SessionStatus.SCANNING.value, session_id),
            )
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def set_video_path(self, session_id: str, video_path: str | None) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE sessions SET video_path=? WHERE id=?", (video_path, session_id)
            )

    def next_receipt_paths(
        self, session_id: str, extension: str
    ) -> tuple[int, Path, Path]:
        session = self.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        date = session.started_at[:10]
        folder = Path(session.receipt_folder)
        with self._lock, self._connect() as db:
            sequence = db.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM receipts WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
        while True:
            filename = f"Receipt_{date}_{sequence:04d}.{extension.lower()}"
            processed = folder / filename
            original = (
                folder / "Originals" / f"Receipt_{date}_{sequence:04d}_original.png"
            )
            if not processed.exists() and not original.exists():
                return sequence, processed, original
            sequence += 1

    def add_receipt(
        self,
        session_id: str,
        sequence: int,
        captured_at: float,
        processed_path: Path,
        original_path: Path,
        content_hash: int,
        edit_json: str = "{}",
        duplicate_group: str | None = None,
        quality_flag: str | None = None,
    ) -> ReceiptRecord:
        with self._lock, self._connect() as db:
            cursor = db.execute(
                """INSERT INTO receipts
                   (session_id, sequence, captured_at, filename, original_path,
                    processed_path, edit_json, content_hash, duplicate_group,
                    quality_flag)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    sequence,
                    captured_at,
                    processed_path.name,
                    str(original_path),
                    str(processed_path),
                    edit_json,
                    f"{content_hash:016x}",
                    duplicate_group,
                    quality_flag,
                ),
            )
            db.execute(
                """UPDATE sessions SET receipt_count=(
                       SELECT COUNT(*) FROM receipts
                       WHERE session_id=? AND deleted=0 AND combined_into IS NULL
                   ) WHERE id=?""",
                (session_id, session_id),
            )
            receipt_id = cursor.lastrowid
        return self.get_receipt(receipt_id)

    def log_event(self, session_id: str, event_type: str, payload: dict) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO events(session_id, created_at, event_type, payload_json) VALUES (?, ?, ?, ?)",
                (session_id, utc_now(), event_type, json.dumps(payload, default=str)),
            )

    @staticmethod
    def _session(row: sqlite3.Row) -> SessionRecord:
        values = dict(row)
        values["status"] = SessionStatus(values["status"])
        return SessionRecord(**values)

    @staticmethod
    def _receipt(row: sqlite3.Row) -> ReceiptRecord:
        values = dict(row)
        values["deleted"] = bool(values["deleted"])
        values["review_flag"] = bool(values["review_flag"])
        return ReceiptRecord(**values)

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        return self._session(row) if row else None

    def list_sessions(self) -> list[SessionRecord]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC"
            ).fetchall()
        return [self._session(row) for row in rows]

    def get_receipt(self, receipt_id: int) -> ReceiptRecord:
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT * FROM receipts WHERE id=?", (receipt_id,)
            ).fetchone()
        if not row:
            raise KeyError(receipt_id)
        return self._receipt(row)

    def list_receipts(
        self, session_id: str, include_deleted: bool = False
    ) -> list[ReceiptRecord]:
        where = (
            "session_id=?"
            if include_deleted
            else "session_id=? AND deleted=0 AND combined_into IS NULL"
        )
        with self._lock, self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM receipts WHERE {where} ORDER BY captured_at, sequence",
                (session_id,),
            ).fetchall()
        return [self._receipt(row) for row in rows]

    def set_combined_into(
        self, receipt_ids: list[int], target_id: int | None
    ) -> None:
        """Absorb receipts into a combined sheet, or release them again."""
        if not receipt_ids:
            return
        placeholders = ",".join("?" for _ in receipt_ids)
        with self._lock, self._connect() as db:
            session_id = db.execute(
                f"SELECT session_id FROM receipts WHERE id IN ({placeholders}) LIMIT 1",
                receipt_ids,
            ).fetchone()[0]
            db.execute(
                f"UPDATE receipts SET combined_into=? WHERE id IN ({placeholders})",
                [target_id, *receipt_ids],
            )
            db.execute(
                """UPDATE sessions SET receipt_count=(
                       SELECT COUNT(*) FROM receipts
                       WHERE session_id=? AND deleted=0 AND combined_into IS NULL
                   ) WHERE id=?""",
                (session_id, session_id),
            )

    def list_combined_members(self, receipt_id: int) -> list[ReceiptRecord]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT * FROM receipts WHERE combined_into=? AND deleted=0 "
                "ORDER BY captured_at, sequence",
                (receipt_id,),
            ).fetchall()
        return [self._receipt(row) for row in rows]

    def set_review_flag(self, receipt_id: int, flagged: bool) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE receipts SET review_flag=? WHERE id=?",
                (1 if flagged else 0, receipt_id),
            )

    def update_receipt_path(self, receipt_id: int, path: Path) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE receipts SET filename=?, processed_path=? WHERE id=?",
                (path.name, str(path), receipt_id),
            )

    def update_receipt_edits(self, receipt_id: int, edit_json: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE receipts SET edit_json=? WHERE id=?", (edit_json, receipt_id)
            )

    def set_duplicate_group(self, receipt_ids: list[int], group: str) -> None:
        if not receipt_ids:
            return
        placeholders = ",".join("?" for _ in receipt_ids)
        with self._lock, self._connect() as db:
            db.execute(
                f"UPDATE receipts SET duplicate_group=? WHERE id IN ({placeholders})",
                (group, *receipt_ids),
            )

    def clear_duplicate_groups(self, session_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE receipts SET duplicate_group=NULL WHERE session_id=?",
                (session_id,),
            )

    def resolve_duplicate_member(self, receipt_id: int) -> None:
        receipt = self.get_receipt(receipt_id)
        group = receipt.duplicate_group
        if not group:
            return
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE receipts SET duplicate_group=NULL WHERE id=?", (receipt_id,)
            )
            remaining = db.execute(
                "SELECT id FROM receipts WHERE duplicate_group=? AND deleted=0",
                (group,),
            ).fetchall()
            if len(remaining) <= 1:
                db.execute(
                    "UPDATE receipts SET duplicate_group=NULL WHERE duplicate_group=?",
                    (group,),
                )

    def mark_receipt_deleted(self, receipt_id: int) -> None:
        receipt = self.get_receipt(receipt_id)
        with self._lock, self._connect() as db:
            db.execute("UPDATE receipts SET deleted=1 WHERE id=?", (receipt_id,))
            db.execute(
                """UPDATE sessions SET receipt_count=(
                       SELECT COUNT(*) FROM receipts
                       WHERE session_id=? AND deleted=0 AND combined_into IS NULL
                   ) WHERE id=?""",
                (receipt.session_id, receipt.session_id),
            )
        if receipt.duplicate_group:
            self.resolve_duplicate_member(receipt_id)

    def remove_session_history(self, session_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def cleanup_successful_history(self, days: int) -> int:
        """Remove old metadata only. Receipt and video files are never touched."""
        if days <= 0:
            return 0
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with self._lock, self._connect() as db:
            cursor = db.execute(
                "DELETE FROM sessions WHERE status=? AND ended_at IS NOT NULL AND ended_at < ?",
                (SessionStatus.SUCCESSFUL.value, cutoff),
            )
            return cursor.rowcount

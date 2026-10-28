"""Database client and query abstraction supporting Supabase PostgREST and in-memory test database."""
from __future__ import annotations

import copy
import fnmatch
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from app.config import (
    ATTENDANCE_ENV,
    DB_SLOW_QUERY_MS,
    ENABLE_DB_TIMING,
    SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_URL,
)

logger = logging.getLogger("attendrix.db")

# PostgREST builder methods that describe the intended operation. Chained
# filters (eq, order, limit, ...) must not overwrite the operation label.
_TIMED_QUERY_ACTIONS = {"select", "insert", "update", "delete", "upsert"}

try:
    from supabase import Client, create_client
except ImportError:
    create_client = None
    Client = Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SingleRowError(Exception):
    """Raised by ``single()`` when a query does not return exactly one row.

    Mirrors PostgREST, which answers ``single()`` with HTTP 406 / ``APIError``
    when zero or multiple rows match.
    """


class MockResponse:
    def __init__(self, data: Any = None, count: Optional[int] = None):
        self.data = data
        self.count = count


class InMemoryTableQuery:
    def __init__(self, table_name: str, db: InMemoryDatabase):
        self.table_name = table_name
        self.db = db
        self.filters: list[tuple[str, str, Any]] = []
        self._select_cols = "*"
        self._count_mode: Optional[str] = None
        self._order_col: Optional[str] = None
        self._order_desc: bool = False
        self._limit_val: Optional[int] = None
        self._single_mode: bool = False
        self._action = "select"
        self._write_data: Any = None

    def select(self, columns: str = "*", count: Optional[str] = None) -> InMemoryTableQuery:
        self._action = "select"
        self._select_cols = columns
        self._count_mode = count
        return self

    def eq(self, column: str, value: Any) -> InMemoryTableQuery:
        self.filters.append((column, "eq", value))
        return self

    def neq(self, column: str, value: Any) -> InMemoryTableQuery:
        self.filters.append((column, "neq", value))
        return self

    def in_(self, column: str, values: list[Any]) -> InMemoryTableQuery:
        self.filters.append((column, "in", set(values)))
        return self

    def order(self, column: str, desc: bool = False) -> InMemoryTableQuery:
        self._order_col = column
        self._order_desc = desc
        return self

    def limit(self, count: int) -> InMemoryTableQuery:
        self._limit_val = count
        return self

    def single(self) -> InMemoryTableQuery:
        self._single_mode = True
        return self

    def insert(self, data: Any) -> InMemoryTableQuery:
        self._action = "insert"
        self._write_data = data
        return self

    def update(self, data: dict[str, Any]) -> InMemoryTableQuery:
        self._action = "update"
        self._write_data = data
        return self

    def delete(self) -> InMemoryTableQuery:
        self._action = "delete"
        return self

    def _matches_filters(self, row: dict[str, Any]) -> bool:
        for col, op, val in self.filters:
            row_val = row.get(col)
            if op == "eq":
                if str(row_val) != str(val):
                    return False
            elif op == "neq":
                if str(row_val) == str(val):
                    return False
            elif op == "in":
                if row_val not in val and str(row_val) not in {str(v) for v in val}:
                    return False
        return True

    def execute(self) -> MockResponse:
        rows = self.db.tables.setdefault(self.table_name, [])

        if self._action == "insert":
            items = self._write_data if isinstance(self._write_data, list) else [self._write_data]
            inserted = []
            for item in items:
                row = copy.deepcopy(item)
                if "id" not in row:
                    row["id"] = str(uuid.uuid4())
                if "created_at" not in row:
                    row["created_at"] = utc_now()
                rows.append(row)
                inserted.append(row)
            return MockResponse(data=inserted if isinstance(self._write_data, list) else inserted)

        if self._action == "update":
            updated = []
            for row in rows:
                if self._matches_filters(row):
                    row.update(self._write_data)
                    updated.append(copy.deepcopy(row))
            return MockResponse(data=updated)

        if self._action == "delete":
            kept = []
            deleted = []
            for row in rows:
                if self._matches_filters(row):
                    deleted.append(row)
                else:
                    kept.append(row)
            self.db.tables[self.table_name] = kept
            return MockResponse(data=deleted)

        # Action: SELECT
        matched = [copy.deepcopy(r) for r in rows if self._matches_filters(r)]
        count = len(matched) if self._count_mode else None

        # Resolve joined fields (e.g. classes(name, code), profiles(name, email))
        if "(" in self._select_cols:
            matched = self._resolve_joins(matched)

        if self._order_col:
            clean_col = self._order_col.split("(")[0]
            matched.sort(
                key=lambda x: str(x.get(clean_col, "")),
                reverse=self._order_desc,
            )

        if self._limit_val is not None:
            matched = matched[:self._limit_val]

        if self._single_mode:
            if len(matched) != 1:
                raise SingleRowError(
                    f"single() expected exactly 1 row from '{self.table_name}', got {len(matched)}"
                )
            return MockResponse(data=matched[0], count=count)

        return MockResponse(data=matched, count=count)

    def _resolve_joins(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Handle profiles join for attendance or enrollments
        for row in rows:
            if "student_id" in row and "profiles" not in row:
                prof = next((p for p in self.db.tables.get("profiles", []) if str(p.get("id")) == str(row.get("student_id"))), None)
                if prof:
                    row["profiles"] = {"id": prof["id"], "name": prof["name"], "email": prof["email"], "role": prof["role"]}
                    row["profiles!attendance_student_id_fkey"] = row["profiles"]
                    row["profiles!enrollments_student_id_fkey"] = row["profiles"]
            if "class_id" in row and "classes" not in row:
                cls = next((c for c in self.db.tables.get("classes", []) if str(c.get("id")) == str(row.get("class_id"))), None)
                if cls:
                    row["classes"] = {"id": cls["id"], "name": cls["name"], "code": cls["code"], "teacher_id": cls.get("teacher_id")}
        return rows


class MockStorageBucket:
    def __init__(self, name: str):
        self.name = name
        self.files: dict[str, bytes] = {}

    def upload(self, path: str, file_data: bytes, file_options: Any = None) -> Any:
        self.files[path] = file_data
        return {"Key": path}

    def create_signed_url(self, path: str, expires_in: int = 3600) -> dict[str, str]:
        return {"signedURL": f"https://mock-storage.local/{self.name}/{path}?expires={expires_in}"}

    def remove(self, paths: list[str]) -> Any:
        for p in paths:
            self.files.pop(p, None)
        return {"message": "deleted"}


class MockStorage:
    def __init__(self):
        self.buckets: dict[str, MockStorageBucket] = {}

    def from_(self, bucket_name: str) -> MockStorageBucket:
        if bucket_name not in self.buckets:
            self.buckets[bucket_name] = MockStorageBucket(bucket_name)
        return self.buckets[bucket_name]


class InMemoryDatabase:
    """In-memory database for testing and offline development."""

    def __init__(self):
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self.storage = MockStorage()
        self.seed_defaults()

    def table(self, name: str) -> InMemoryTableQuery:
        return InMemoryTableQuery(name, self)

    def seed_defaults(self) -> None:
        self.tables.clear()
        teacher_id = "00000000-0000-0000-0000-000000000001"
        student1_id = "00000000-0000-0000-0000-000000000002"
        student2_id = "00000000-0000-0000-0000-000000000003"
        controller_id = "00000000-0000-0000-0000-000000000004"
        class1_id = "11111111-1111-1111-1111-111111111111"
        class2_id = "22222222-2222-2222-2222-222222222222"
        class3_id = "33333333-3333-3333-3333-333333333333"
        class4_id = "44444444-4444-4444-4444-444444444444"

        self.tables["profiles"] = [
            {"id": teacher_id, "name": "Prof. Alan Turing", "email": "teacher@demo.edu", "role": "teacher"},
            {
                "id": student1_id,
                "name": "Ada Lovelace",
                "email": "student1@demo.edu",
                "role": "student",
                "roll_number": "CS-2026-001",
                "student_id": "STU-001",
                "registration_locked": False,
            },
            {
                "id": student2_id,
                "name": "Grace Hopper",
                "email": "student2@demo.edu",
                "role": "student",
                "roll_number": "CS-2026-002",
                "student_id": "STU-002",
                "registration_locked": False,
            },
            {
                "id": controller_id,
                "name": "Academic Controller",
                "email": "controller@demo.edu",
                "role": "controller",
            },
        ]
        self.tables["classes"] = [
            {
                "id": class1_id,
                "name": "CS-101: Introduction to Computing",
                "code": "CS-101",
                "section": "Section A",
                "room_number": "LH-120",
                "capacity": 120,
                "hall_type": "120-Seater Lecture Theater",
                "room_photo": "https://images.unsplash.com/photo-1580582932707-520aed937b7b?w=800",
                "teacher_id": teacher_id,
                "created_at": utc_now(),
            },
            {
                "id": class2_id,
                "name": "CS-204: Data Structures & Algorithms",
                "code": "CS-204",
                "section": "Section B",
                "room_number": "SEM-80",
                "capacity": 80,
                "hall_type": "80-Seater Seminar Hall",
                "room_photo": "https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?w=800",
                "teacher_id": teacher_id,
                "created_at": utc_now(),
            },
            {
                "id": class3_id,
                "name": "CS-301: Operating Systems & Architecture",
                "code": "CS-301",
                "section": "Section C",
                "room_number": "AMP-240",
                "capacity": 240,
                "hall_type": "240-Seater Tiered Amphitheater",
                "room_photo": "https://images.unsplash.com/photo-1524178232363-1fb2b075b655?w=800",
                "teacher_id": teacher_id,
                "created_at": utc_now(),
            },
            {
                "id": class4_id,
                "name": "CS-405: Computer Vision & AI",
                "code": "CS-405",
                "section": "Honors",
                "room_number": "AUD-260",
                "capacity": 260,
                "hall_type": "260-Seater Grand Auditorium",
                "room_photo": "https://images.unsplash.com/photo-1562774053-701939374585?w=800",
                "teacher_id": teacher_id,
                "created_at": utc_now(),
            },
        ]
        self.tables["enrollments"] = [
            {"class_id": class1_id, "student_id": student1_id},
            {"class_id": class1_id, "student_id": student2_id},
            {"class_id": class2_id, "student_id": student1_id},
            {"class_id": class2_id, "student_id": student2_id},
            {"class_id": class3_id, "student_id": student1_id},
            {"class_id": class3_id, "student_id": student2_id},
            {"class_id": class4_id, "student_id": student1_id},
            {"class_id": class4_id, "student_id": student2_id},
        ]
        self.tables["sessions"] = []
        self.tables["attendance"] = []
        self.tables["checkin_attempts"] = []
        self.tables["face_templates"] = []
        self.tables["audit_log"] = []
        self.tables["notifications"] = []
        self.tables["system_settings"] = [
            {
                "id": "setting-reg-window",
                "key": "registration_window",
                "value": "open",
                "is_open": True,
                "updated_at": utc_now(),
                "updated_by": controller_id,
            }
        ]


# Singleton client holder
_shared_client: Optional[Any] = None
_in_memory_db: Optional[InMemoryDatabase] = None


class _TimedQuery:
    """Transparent query-builder proxy that logs slow PostgREST operations.

    It delegates every builder method and only measures ``execute()``. Table
    names and durations are logged; row data and filters are never logged.
    """

    def __init__(self, query: Any, table_name: str, operation: str = "query"):
        self._query = query
        self._table_name = table_name
        self._operation = operation

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._query, name)
        if not callable(attr):
            return attr

        def delegated(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            if hasattr(result, "execute"):
                operation = name if name in _TIMED_QUERY_ACTIONS else self._operation
                return _TimedQuery(result, self._table_name, operation)
            return result

        return delegated

    def execute(self) -> Any:
        started = time.perf_counter()
        failed = False
        try:
            return self._query.execute()
        except Exception:
            failed = True
            logger.warning(
                json.dumps(
                    {
                        "event": "db_query",
                        "table": self._table_name,
                        "operation": self._operation,
                        "status": "error",
                        "duration_ms": round(
                            (time.perf_counter() - started) * 1000.0, 1
                        ),
                        "pid": os.getpid(),
                    },
                    separators=(",", ":"),
                )
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if not failed and elapsed_ms >= DB_SLOW_QUERY_MS:
                logger.warning(
                    json.dumps(
                        {
                            "event": "db_query_slow",
                            "table": self._table_name,
                            "operation": self._operation,
                            "status": "ok",
                            "duration_ms": round(elapsed_ms, 1),
                            "pid": os.getpid(),
                        },
                        separators=(",", ":"),
                    )
                )


class _TimedClient:
    """Client proxy that applies slow-query timing to table operations."""

    def __init__(self, client: Any):
        self._client = client

    def table(self, table_name: str) -> _TimedQuery:
        return _TimedQuery(self._client.table(table_name), table_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _wrap_client(client: Any) -> Any:
    if not ENABLE_DB_TIMING or ATTENDANCE_ENV == "test":
        return client
    return _TimedClient(client)


def get_in_memory_db() -> InMemoryDatabase:
    global _in_memory_db
    if _in_memory_db is None:
        _in_memory_db = InMemoryDatabase()
    return _in_memory_db


def supa() -> Any:
    """Return Supabase client in production or InMemoryDatabase in test."""
    global _shared_client
    if ATTENDANCE_ENV == "test":
        return get_in_memory_db()

    if _shared_client is None:
        if create_client and SUPABASE_URL and not SUPABASE_URL.startswith("https://placeholder"):
            try:
                _shared_client = _wrap_client(
                    create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
                )
            except Exception as exc:
                if ATTENDANCE_ENV == "production":
                    raise RuntimeError(f"Supabase client initialisation failed: {exc}") from exc
                logger.error(
                    "Supabase client initialisation failed (%s); using non-persistent in-memory database.",
                    exc,
                )
                _shared_client = _wrap_client(get_in_memory_db())
        else:
            if ATTENDANCE_ENV == "production":
                raise RuntimeError(
                    "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured in production; "
                    "refusing to store attendance in the non-persistent in-memory database."
                )
            logger.warning("Supabase is not configured; using non-persistent in-memory database.")
            _shared_client = _wrap_client(get_in_memory_db())

    return _shared_client


def is_supabase_backend() -> bool:
    """True when requests are served by a real Supabase project (not the in-memory DB)."""
    client = supa()
    inner = getattr(client, "_client", client)
    return not isinstance(inner, InMemoryDatabase)


def first_row(query: Any) -> Optional[dict[str, Any]]:
    """Execute ``query`` and return its first row, or ``None`` when nothing matches.

    Use instead of ``single()``, which raises on zero rows with PostgREST.
    """
    resp = query.limit(1).execute()
    rows = resp.data or []
    return rows[0] if rows else None


def reset_test_db() -> InMemoryDatabase:
    """Reset the test in-memory database to default seed state."""
    try:
        from app.security import reset_security_caches
        reset_security_caches()
    except Exception:
        pass
    db = get_in_memory_db()
    db.seed_defaults()
    return db

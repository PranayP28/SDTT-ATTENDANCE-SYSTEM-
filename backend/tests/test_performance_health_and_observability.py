"""Performance, health-probe, and observability regression tests.

These tests pin the behavior that keeps Render Free responsive:
- /health is dependency-free and stays fast while CPU-heavy work runs.
- Startup DB connectivity probing is bounded by a timeout.
- Inference capacity exhaustion returns 503 without writing attendance.
- Request/DB timing logs carry operational context but never secrets.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
os.environ["ATTENDANCE_ENV"] = "test"
os.environ["SUPABASE_JWT_SECRET"] = "test_jwt_secret_must_be_at_least_32_chars_long"

import app.main as main
from app.db import MockResponse, reset_test_db, supa
from app.observability import InferenceBusyError
from app.security import create_access_token, encrypt_face_embedding

STUDENT_ID = "00000000-0000-0000-0000-000000000002"
CLASS_ID = "11111111-1111-1111-1111-111111111111"
SESSION_ID = "ssssssss-0001-0001-0001-000000000099"


def _synthetic_jpeg() -> bytes:
    img = np.full((320, 320, 3), 140, dtype=np.uint8)
    cv2.circle(img, (160, 160), 45, (90, 90, 90), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture(autouse=True)
def _fresh_db():
    reset_test_db()
    db = supa()
    db.table("sessions").insert(
        {
            "id": SESSION_ID,
            "class_id": CLASS_ID,
            "status": "open",
            "teacher_id": "00000000-0000-0000-0000-000000000001",
        }
    ).execute()
    yield
    reset_test_db()


def test_health_is_dependency_free_when_db_and_engine_are_unavailable(monkeypatch):
    """/health must not touch the database or face engine."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("dependency unavailable")

    monkeypatch.setattr(main, "supa", _boom)
    monkeypatch.setattr(main, "get_face_engine", _boom)

    with TestClient(main.app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "attendrix-attendance-api",
    }


def test_startup_db_probe_is_bounded(monkeypatch):
    """A hung database must not block application startup indefinitely."""

    class SlowQuery:
        def select(self, *_args: Any, **_kwargs: Any) -> "SlowQuery":
            return self

        def limit(self, *_args: Any, **_kwargs: Any) -> "SlowQuery":
            return self

        def execute(self) -> Any:
            time.sleep(0.4)
            return MockResponse(data=[])

    class SlowDb:
        def table(self, _name: str) -> SlowQuery:
            return SlowQuery()

    monkeypatch.setattr(main, "STARTUP_DB_PROBE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(main, "supa", lambda: SlowDb())
    monkeypatch.setattr(main, "validate_environment_config", lambda: [])

    async def _run_lifespan() -> float:
        started = time.perf_counter()
        async with main.lifespan(main.app):
            return time.perf_counter() - started

    startup_seconds = asyncio.run(_run_lifespan())
    assert startup_seconds < 0.35, (
        f"Startup waited {startup_seconds:.2f}s for a probe bounded to 0.05s"
    )


def test_health_remains_fast_while_inference_is_running(monkeypatch):
    """Sync CV endpoints run in the threadpool; health must stay responsive."""

    class SlowLivenessEngine:
        def verify_active_liveness(self, **_kwargs: Any) -> dict[str, Any]:
            time.sleep(1.2)
            return {
                "liveness_passed": True,
                "frontal_detected": True,
                "left_turn_detected": True,
                "right_turn_detected": True,
                "blink_detected": True,
                "checks": {
                    "frontal": True,
                    "left_turn": True,
                    "right_turn": True,
                    "blink": True,
                },
                "message": "Synthetic slow liveness result.",
            }

    monkeypatch.setattr(main, "get_face_engine", lambda: SlowLivenessEngine())
    token = create_access_token(STUDENT_ID, "student1@demo.edu", "student", "Ada Lovelace")

    async def _scenario() -> tuple[float, int, int]:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            files = {"face_frontal": ("face.jpg", _synthetic_jpeg(), "image/jpeg")}
            inference = asyncio.create_task(
                client.post(
                    f"/api/sessions/{SESSION_ID}/verify-liveness",
                    files=files,
                    headers={"Authorization": f"Bearer {token}"},
                )
            )
            # Let the sync endpoint enter its worker thread and start sleeping.
            await asyncio.sleep(0.15)

            started = time.perf_counter()
            health = await client.get("/api/health")
            health_seconds = time.perf_counter() - started

            inference_response = await inference
            return health_seconds, health.status_code, inference_response.status_code

    health_seconds, health_status, inference_status = asyncio.run(_scenario())
    assert health_status == 200
    assert inference_status == 200
    assert health_seconds < 0.5, (
        f"/api/health took {health_seconds:.2f}s while inference was running"
    )


def test_inference_busy_returns_503_without_writing_attendance(monkeypatch):
    """A saturated inference queue must fail before any attempt is persisted."""

    @contextmanager
    def _busy_slot(_operation: str) -> Iterator[None]:
        raise InferenceBusyError("Verification capacity is saturated.")
        yield  # pragma: no cover

    monkeypatch.setattr(main, "inference_slot", _busy_slot)
    supa().table("face_templates").insert(
        {"user_id": STUDENT_ID, "encrypted_embedding": encrypt_face_embedding(np.ones(512, dtype=np.float32))}
    ).execute()
    token = create_access_token(STUDENT_ID, "student1@demo.edu", "student", "Ada Lovelace")

    with TestClient(main.app) as client:
        response = client.post(
            f"/api/sessions/{SESSION_ID}/checkin",
            headers={"Authorization": f"Bearer {token}"},
            data={"attempt_id": "attempt-busy-1", "metadata": "[]"},
            files={"frames": ("frame.jpg", _synthetic_jpeg(), "image/jpeg")},
        )

    assert response.status_code == 503
    assert response.headers["X-Error-Code"] == "INFERENCE_BUSY"
    assert "INFERENCE_BUSY" in response.json()["detail"]

    db = supa()
    attempts = (
        db.table("checkin_attempts")
        .select("id")
        .eq("session_id", SESSION_ID)
        .execute()
        .data
    )
    attendance = (
        db.table("attendance")
        .select("id")
        .eq("session_id", SESSION_ID)
        .execute()
        .data
    )
    assert attempts == []
    assert attendance == []


def test_request_timing_log_has_context_but_no_query_secrets(caplog):
    with caplog.at_level(logging.INFO, logger="attendrix.requests"):
        with TestClient(main.app) as client:
            response = client.get(
                "/api/time?access_token=super-secret-query-token",
                headers={"CF-Ray": "test-ray-123"},
            )

    assert response.status_code == 200
    assert response.headers["server-timing"].startswith("app;dur=")
    assert response.headers["x-process-time-ms"]

    records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "attendrix.requests"
    ]
    assert records, "expected a structured request timing log"
    entry = records[-1]
    assert entry["event"] == "request_complete"
    assert entry["method"] == "GET"
    assert entry["path"] == "/api/time"
    assert entry["status"] == 200
    assert isinstance(entry["duration_ms"], (int, float))
    assert entry["request_id"] == "test-ray-123"
    assert "super-secret-query-token" not in caplog.text
    assert "access_token" not in caplog.text


def test_slow_query_log_excludes_row_data(caplog, monkeypatch):
    import app.db as db_module

    class FakeQuery:
        def select(self, *_args: Any, **_kwargs: Any) -> "FakeQuery":
            return self

        def eq(self, *_args: Any, **_kwargs: Any) -> "FakeQuery":
            return self

        def execute(self) -> MockResponse:
            time.sleep(0.01)
            return MockResponse(data=[{"email": "private-person@example.com"}])

    class FakeClient:
        def table(self, _name: str) -> FakeQuery:
            return FakeQuery()

    monkeypatch.setattr(db_module, "ENABLE_DB_TIMING", True)
    monkeypatch.setattr(db_module, "ATTENDANCE_ENV", "production")
    monkeypatch.setattr(db_module, "DB_SLOW_QUERY_MS", 0.0)

    timed_client = db_module._wrap_client(FakeClient())
    with caplog.at_level(logging.WARNING, logger="attendrix.db"):
        timed_client.table("profiles").select("*").eq("email", "private-person@example.com").execute()

    records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "attendrix.db"
    ]
    assert records, "expected a slow-query warning"
    entry = records[-1]
    assert entry["event"] == "db_query_slow"
    assert entry["table"] == "profiles"
    assert entry["operation"] == "select"
    assert entry["duration_ms"] >= 0
    assert "private-person@example.com" not in caplog.text

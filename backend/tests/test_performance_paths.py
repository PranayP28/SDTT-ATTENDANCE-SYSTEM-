"""Tests for request metrics, pagination, single-query session counts, and model preload."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ATTENDANCE_ENV"] = "test"
os.environ.setdefault("SUPABASE_JWT_SECRET", "test_jwt_secret_must_be_at_least_32_chars_long")

from app import main  # noqa: E402
from app.db import _TimedClient, reset_test_db, supa  # noqa: E402
from app.observability import _REQUEST_METRICS, new_request_metrics  # noqa: E402
from app.security import create_access_token  # noqa: E402

TEACHER_ID = "00000000-0000-0000-0000-000000000001"
STUDENT_ID = "00000000-0000-0000-0000-000000000002"
CONTROLLER_ID = "00000000-0000-0000-0000-000000000004"
CLASS_ID = "11111111-1111-1111-1111-111111111111"


def _hdr(uid: str, email: str, role: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(uid, email, role, 'User')}"}


TEACHER = _hdr(TEACHER_ID, "teacher@demo.edu", "teacher")
STUDENT = _hdr(STUDENT_ID, "student1@demo.edu", "student")


@pytest.fixture
def client():
    reset_test_db()
    with TestClient(main.app) as c:
        yield c
    reset_test_db()


def test_timed_client_counts_queries_into_request_metrics():
    db = _TimedClient(reset_test_db())
    metrics = new_request_metrics()
    token = _REQUEST_METRICS.set(metrics)
    try:
        db.table("profiles").select("id").execute()
        db.table("classes").select("id").eq("teacher_id", TEACHER_ID).execute()
    finally:
        _REQUEST_METRICS.reset(token)
    assert metrics["db_calls"] == 2
    assert metrics["db_ms"] >= 0.0


def test_timed_client_outside_request_is_noop():
    db = _TimedClient(reset_test_db())
    assert db.table("profiles").select("id").execute().data


def test_response_has_request_id_and_db_server_timing(client, caplog):
    with caplog.at_level(logging.INFO, logger="attendrix.requests"):
        resp = client.get("/api/classes", headers={**TEACHER, "X-Request-ID": "req-abc"})
    assert resp.status_code == 200
    assert resp.headers["x-request-id"] == "req-abc"
    assert 'db;dur=' in resp.headers["server-timing"]
    entry = json.loads([r for r in caplog.records if r.name == "attendrix.requests"][-1].getMessage())
    assert entry["request_id"] == "req-abc"
    assert "db_calls" in entry and "authorization" not in json.dumps(entry).lower()


def test_generated_request_id_when_header_missing(client):
    resp = client.get("/api/time")
    assert len(resp.headers["x-request-id"]) == 32


def _make_sessions(n: int) -> None:
    db = supa()
    for i in range(n):
        db.table("sessions").insert({
            "id": f"sess-{i:03d}", "class_id": CLASS_ID, "status": "closed", "created_by": TEACHER_ID,
            "created_at": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00",
        }).execute()


def test_sessions_list_is_paginated_newest_first(client):
    _make_sessions(7)
    first = client.get("/api/sessions?limit=3", headers=TEACHER).json()
    second = client.get("/api/sessions?limit=3&offset=3", headers=TEACHER).json()
    assert [s["id"] for s in first] == ["sess-006", "sess-005", "sess-004"]
    assert [s["id"] for s in second] == ["sess-003", "sess-002", "sess-001"]


def test_pagination_rejects_invalid_params(client):
    assert client.get("/api/sessions?limit=0", headers=TEACHER).status_code == 422
    assert client.get("/api/notifications?offset=-1", headers=STUDENT).status_code == 422


def test_page_size_is_capped(monkeypatch):
    monkeypatch.setattr(main, "MAX_PAGE_SIZE", 50)
    assert main._page_bounds(10_000, 0) == (0, 49)
    assert main._page_bounds(None, 5) == (5, 5 + main.DEFAULT_PAGE_SIZE - 1)


def test_audit_feed_is_chronological_and_paginated(client):
    for i in range(4):
        supa().table("audit_log").insert({
            "id": f"z{9 - i}", "session_id": None, "action": f"a{i}", "actor_id": TEACHER_ID,
            "timestamp": f"2026-01-01T00:00:0{i}+00:00", "metadata_json": "{}",
        }).execute()
    ctrl = _hdr(CONTROLLER_ID, "controller@demo.edu", "controller")
    rows = client.get("/api/audit?limit=2", headers=ctrl).json()
    assert [r["action"] for r in rows] == ["a3", "a2"]


def test_session_payload_counts_verified_and_flagged(client):
    sess = client.post("/api/sessions", headers=TEACHER, json={"class_id": CLASS_ID, "duration_minutes": 10}).json()
    db = supa()
    for i, state in enumerate(["verified", "flagged", "flagged", "rejected"]):
        db.table("checkin_attempts").insert({
            "attempt_id": f"att-{i}", "session_id": sess["id"], "student_id": STUDENT_ID, "state": state,
        }).execute()
    summary = client.get(f"/api/sessions/{sess['id']}", headers=TEACHER).json()["summary"]
    assert summary["verified_count"] == 1
    assert summary["flagged_count"] == 2


def test_face_engine_preload_runs_once_in_background(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(main, "PRELOAD_FACE_ENGINE", True)
    monkeypatch.setattr(main, "get_face_engine", lambda: calls.append(1))
    with TestClient(main.app) as c:
        assert c.get("/api/health").status_code == 200
    assert calls == [1]


def test_face_engine_preload_failure_does_not_break_startup(monkeypatch):
    def boom():
        raise RuntimeError("model missing")
    monkeypatch.setattr(main, "PRELOAD_FACE_ENGINE", True)
    monkeypatch.setattr(main, "get_face_engine", boom)
    with TestClient(main.app) as c:
        assert c.get("/api/health").status_code == 200

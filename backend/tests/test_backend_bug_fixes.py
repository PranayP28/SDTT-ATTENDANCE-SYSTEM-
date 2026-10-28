"""Regression tests for backend bug fixes (check-in enrollment, review guards, realtime, config)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

os.environ["ATTENDANCE_ENV"] = "test"
os.environ["SUPABASE_JWT_SECRET"] = "test_jwt_secret_must_be_at_least_32_chars_long"
os.environ["ATTENDANCE_SECRET_KEY"] = "test_biometric_encryption_key_32_chars!"

from app import main
from app.db import SingleRowError, first_row, reset_test_db, supa
from app.face_engine import FaceDetection, FakeFaceEngine, set_face_engine
from app.security import create_access_token

TEACHER_ID = "00000000-0000-0000-0000-000000000001"
STUDENT_ID = "00000000-0000-0000-0000-000000000002"
CLASS_ID = "11111111-1111-1111-1111-111111111111"

STUDENT_FACE = FaceDetection(bbox_x=120, bbox_y=80, bbox_w=60, bbox_h=80, confidence=0.98, ipd_px=22.0)


def _frame_bytes() -> bytes:
    import cv2

    img = np.full((300, 400, 3), 120, dtype=np.uint8)
    for i in range(0, 300, 20):
        cv2.line(img, (0, i), (400, i), (255, 255, 255), 2)
    for j in range(0, 400, 20):
        cv2.line(img, (j, 0), (j, 300), (0, 0, 0), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _hdr(user_id: str, email: str, role: str, name: str = "User") -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user_id, email, role, name)}"}


TEACHER_HDR = _hdr(TEACHER_ID, "teacher@demo.edu", "teacher", "Prof. Alan Turing")
STUDENT_HDR = _hdr(STUDENT_ID, "student1@demo.edu", "student", "Ada Lovelace")


@pytest.fixture
def engine():
    fake = FakeFaceEngine()
    set_face_engine(fake)
    yield fake
    set_face_engine(None)


@pytest.fixture
def client(engine):
    reset_test_db()
    with TestClient(main.app) as test_client:
        yield test_client


def _create_session(client: TestClient) -> str:
    resp = client.post("/api/sessions", headers=TEACHER_HDR, json={"class_id": CLASS_ID, "duration_minutes": 10})
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _sweep(attempt_id: str, frames: int = 16) -> tuple[dict[str, str], list]:
    jpg = _frame_bytes()
    meta = [{"index": i, "heading_deg": i * 12.0, "time_offset_s": i * 0.4} for i in range(frames)]
    files = [("frames", (f"f{i}.jpg", jpg, "image/jpeg")) for i in range(frames)]
    return {"attempt_id": attempt_id, "metadata": json.dumps(meta)}, files


def _enroll_student(client: TestClient, engine: FakeFaceEngine) -> None:
    engine.preset_detections = [STUDENT_FACE]
    resp = client.post(
        "/api/auth/enroll",
        headers=STUDENT_HDR,
        data={"consent": "true"},
        files={"photo": ("selfie.jpg", _frame_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200, resp.text


def test_checkin_without_enrolled_reference_is_rejected_before_pipeline(client, engine):
    session_id = _create_session(client)
    engine.preset_detections = [STUDENT_FACE]
    data, files = _sweep("no-enrollment-1")

    resp = client.post(f"/api/sessions/{session_id}/checkin", headers=STUDENT_HDR, data=data, files=files)

    assert resp.status_code == 400
    assert resp.headers["X-Error-Code"] == "MISSING_ENROLLMENT_DATA"
    assert supa().table("checkin_attempts").select("id").eq("session_id", session_id).execute().data == []
    att = supa().table("attendance").select("status").eq("session_id", session_id).eq("student_id", STUDENT_ID).execute().data
    assert att[0]["status"] == "pending"


def test_checkin_succeeds_when_lecturer_has_no_reference_template(client, engine):
    _enroll_student(client, engine)
    session_id = _create_session(client)
    engine.preset_detections = [STUDENT_FACE]
    data, files = _sweep("no-lecturer-template-1")

    resp = client.post(f"/api/sessions/{session_id}/checkin", headers=STUDENT_HDR, data=data, files=files)

    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] in ("verified", "flagged", "rejected")


def test_unreadable_student_template_is_treated_as_not_enrolled(client, engine):
    supa().table("face_templates").insert({"user_id": STUDENT_ID, "encrypted_embedding": "not-a-fernet-token"}).execute()
    session_id = _create_session(client)
    data, files = _sweep("bad-template-1")

    resp = client.post(f"/api/sessions/{session_id}/checkin", headers=STUDENT_HDR, data=data, files=files)

    assert resp.status_code == 400
    assert resp.headers["X-Error-Code"] == "MISSING_ENROLLMENT_DATA"


def _flagged_attempt(client: TestClient, engine: FakeFaceEngine) -> tuple[str, str]:
    _enroll_student(client, engine)
    session_id = _create_session(client)
    engine.preset_detections = [STUDENT_FACE]
    data, files = _sweep("flagged-1")
    resp = client.post(f"/api/sessions/{session_id}/checkin", headers=STUDENT_HDR, data=data, files=files)
    assert resp.status_code == 200 and resp.json()["state"] == "flagged", resp.text
    return session_id, "flagged-1"


def test_review_decision_cannot_be_applied_twice(client, engine):
    session_id, attempt_id = _flagged_attempt(client, engine)
    url = f"/api/sessions/{session_id}/review/{attempt_id}"

    assert client.post(url, headers=TEACHER_HDR, json={"decision": "accept"}).status_code == 200
    second = client.post(url, headers=TEACHER_HDR, json={"decision": "reject"})

    assert second.status_code == 409
    att = supa().table("attendance").select("status").eq("session_id", session_id).eq("student_id", STUDENT_ID).execute().data
    assert att[0]["status"] == "present"


def test_review_decision_rejected_on_finalized_session(client, engine):
    session_id, attempt_id = _flagged_attempt(client, engine)
    fin = client.post(f"/api/sessions/{session_id}/finalize", headers=TEACHER_HDR, json={"mark_pending_absent": True})
    assert fin.status_code == 200

    resp = client.post(f"/api/sessions/{session_id}/review/{attempt_id}", headers=TEACHER_HDR, json={"decision": "accept"})

    assert resp.status_code == 409
    att = supa().table("attendance").select("status").eq("session_id", session_id).eq("student_id", STUDENT_ID).execute().data
    assert att[0]["status"] == "absent"


def test_review_unknown_attempt_returns_404(client):
    session_id = _create_session(client)
    resp = client.post(f"/api/sessions/{session_id}/review/does-not-exist", headers=TEACHER_HDR, json={"decision": "accept"})
    assert resp.status_code == 404


def test_dashboard_ws_ticket_route_and_connected_event_is_json_object(client):
    resp = client.post("/api/auth/ws-ticket", headers=TEACHER_HDR)
    assert resp.status_code == 200
    ticket = resp.json()["ticket"]

    with client.websocket_connect(f"/api/ws?ticket={ticket}") as ws:
        message = ws.receive_json()

    assert isinstance(message, dict)
    assert message["type"] == "connected"
    assert message["data"]["user_id"] == TEACHER_ID


def test_ws_ticket_requires_authentication(client):
    assert client.post("/api/auth/ws-ticket").status_code == 401


def test_admin_role_lists_all_classes(client):
    admin_id = "00000000-0000-0000-0000-0000000000aa"
    supa().table("profiles").insert({"id": admin_id, "name": "Admin", "email": "admin@demo.edu", "role": "admin"}).execute()

    resp = client.get("/api/classes", headers=_hdr(admin_id, "admin@demo.edu", "admin"))

    assert resp.status_code == 200
    assert len(resp.json()) == len(supa().table("classes").select("id").execute().data)


def test_signup_fails_instead_of_issuing_orphan_token_when_auth_provider_fails(client, monkeypatch):
    monkeypatch.setattr(main, "is_supabase_backend", lambda: True)

    resp = client.post("/api/auth/signup", json={"email": "new.student@demo.edu", "password": "Passw0rd!x"})

    assert resp.status_code == 400
    assert resp.headers["X-Error-Code"] == "SIGNUP_FAILED"
    assert "access_token" not in resp.json()
    assert supa().table("profiles").select("id").eq("email", "new.student@demo.edu").execute().data == []


def test_first_row_and_single_mirror_postgrest(client):
    query = supa().table("face_templates").select("*").eq("user_id", "missing")
    assert first_row(query) is None
    with pytest.raises(SingleRowError):
        supa().table("face_templates").select("*").eq("user_id", "missing").single().execute()


def _run_python(code: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if not k.startswith(("ATTENDANCE_", "SUPABASE_"))}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_DIR, env={**base, **env}, capture_output=True, text=True, timeout=60
    )


def test_documented_biometric_key_env_var_is_honoured():
    key = "documented_biometric_key_value_at_least_32_chars"
    proc = _run_python(
        "from app import config; print(config.ATTENDANCE_SECRET_KEY)",
        {"ATTENDANCE_ENV": "test", "ATTENDANCE_BIOMETRIC_ENCRYPTION_KEY": key},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == key


def test_production_refuses_in_memory_database_fallback():
    proc = _run_python(
        "from app.db import supa; supa()",
        {"ATTENDANCE_ENV": "production", "SUPABASE_URL": "https://placeholder.supabase.co"},
    )
    assert proc.returncode != 0
    assert "must be configured in production" in proc.stderr

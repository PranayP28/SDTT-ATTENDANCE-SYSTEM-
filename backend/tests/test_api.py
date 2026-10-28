"""Comprehensive tests for Attendrix API architecture and verification pipeline.

Tests all behaviors, negative security tests, rule verification, idempotency,
session auto-close, role mismatch enforcement, and lecturer review queue.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure backend root is on Python path
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

# Force test environment before importing app
os.environ["ATTENDANCE_ENV"] = "test"
os.environ["SUPABASE_JWT_SECRET"] = "test_jwt_secret_must_be_at_least_32_chars_long"
os.environ["ATTENDANCE_SECRET_KEY"] = "test_biometric_encryption_key_32_chars!"

from app import main
from app.db import reset_test_db, supa
from app.face_engine import FaceDetection, FakeFaceEngine, set_face_engine
from app.security import encrypt_face_embedding


def make_test_frame_bytes(w: int = 400, h: int = 300, color: int = 120) -> bytes:
    """Generate JPEG bytes for a synthetic frame with texture to pass Laplacian blur check."""
    import cv2
    img = np.full((h, w, 3), color, dtype=np.uint8)
    # Add contrast lines/checkerboard so Laplacian variance > 35.0
    for i in range(0, h, 20):
        cv2.line(img, (0, i), (w, i), (255, 255, 255), 2)
    for j in range(0, w, 20):
        cv2.line(img, (j, 0), (j, h), (0, 0, 0), 2)
    success, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes() if success else b"fake_jpeg"


@pytest.fixture
def fake_engine():
    """Inject a deterministic FakeFaceEngine for tests."""
    engine = FakeFaceEngine()
    set_face_engine(engine)
    yield engine
    # Reset back after test
    set_face_engine(None)


@pytest.fixture
def client(fake_engine):
    reset_test_db()
    with main._ws_lock:
        main._ws_tickets.clear()
        main._ws_clients.clear()
        main._ws_client_loops.clear()
        main._ws_client_queues.clear()
    with TestClient(main.app) as test_client:
        yield test_client


def login(client: TestClient, email: str, role: str) -> tuple[str, dict[str, Any]]:
    resp = client.post(
        "/api/auth/login",
        json={"email": email, "password": "any", "selected_role": role},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    return payload["access_token"], payload["user"]


def auth_hdr(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── TEST 1: Role Selection Login & Mismatch Enforcement ───────────────────────
def test_login_and_role_mismatch(client: TestClient):
    # Valid Lecturer login
    t_token, teacher = login(client, "teacher@demo.edu", "teacher")
    assert teacher["role"] == "teacher"
    assert teacher["email"] == "teacher@demo.edu"

    # Valid Student login
    s_token, student = login(client, "student1@demo.edu", "student")
    assert student["role"] == "student"

    # Role Mismatch: Student selects Lecturer role
    mismatch_resp = client.post(
        "/api/auth/login",
        json={"email": "student1@demo.edu", "password": "any", "selected_role": "lecturer"},
    )
    assert mismatch_resp.status_code == 403
    assert "Role mismatch" in mismatch_resp.json()["detail"]

    # Role Mismatch: Lecturer selects Student role
    mismatch_resp2 = client.post(
        "/api/auth/login",
        json={"email": "teacher@demo.edu", "password": "any", "selected_role": "student"},
    )
    assert mismatch_resp2.status_code == 403
    assert "Role mismatch" in mismatch_resp2.json()["detail"]

    # Unknown user
    invalid_resp = client.post(
        "/api/auth/login",
        json={"email": "unknown@demo.edu", "password": "any"},
    )
    assert invalid_resp.status_code == 401


# ── TEST 2: Authorization & Endpoint Guarding Negative Tests ──────────────────
def test_authorization_guards(client: TestClient):
    t_token, _ = login(client, "teacher@demo.edu", "teacher")
    s_token, _ = login(client, "student1@demo.edu", "student")

    # Student cannot access teacher-only routes
    assert client.get("/api/classes/11111111-1111-1111-1111-111111111111/students", headers=auth_hdr(s_token)).status_code == 403
    assert client.post("/api/sessions", headers=auth_hdr(s_token), json={"class_id": "11111111-1111-1111-1111-111111111111"}).status_code == 403

    # Teacher cannot access student-only routes
    assert client.get("/api/students/me/attendance", headers=auth_hdr(t_token)).status_code == 403

    # Unauthenticated requests rejected
    assert client.get("/api/me").status_code == 401
    assert client.get("/api/me", headers={"Authorization": "Bearer invalid_token"}).status_code == 401


# ── TEST 3: Biometric Reference Selfie Enrollment ─────────────────────────────
def test_biometric_reference_selfie_enrollment(client: TestClient, fake_engine: FakeFaceEngine):
    s_token, _ = login(client, "student1@demo.edu", "student")

    # Consent is required
    frame_jpg = make_test_frame_bytes()
    no_consent = client.post(
        "/api/auth/enroll",
        headers=auth_hdr(s_token),
        data={"consent": "false"},
        files={"photo": ("selfie.jpg", frame_jpg, "image/jpeg")},
    )
    assert no_consent.status_code == 400
    assert "consent is required" in no_consent.json()["detail"].lower()

    # Valid enrollment
    valid = client.post(
        "/api/auth/enroll",
        headers=auth_hdr(s_token),
        data={"consent": "true"},
        files={"photo": ("selfie.jpg", frame_jpg, "image/jpeg")},
    )
    assert valid.status_code == 200
    assert valid.json()["status"] == "enrolled"
    assert "consent_timestamp" in valid.json()

    # Status check confirms enrollment without revealing embedding
    status = client.get("/api/auth/enrollment-status", headers=auth_hdr(s_token)).json()
    assert status["enrolled"] is True
    assert "consent_timestamp" in status
    assert "encrypted_embedding" not in status
    assert "embedding" not in status


# ── TEST 4: Session Creation, Duration, Time Limit & Expiry ───────────────────
def test_session_creation_duration_and_expiry(client: TestClient):
    t_token, _ = login(client, "teacher@demo.edu", "teacher")
    s_token, _ = login(client, "student1@demo.edu", "student")

    # Create session with 15 minute duration
    created = client.post(
        "/api/sessions",
        headers=auth_hdr(t_token),
        json={"class_id": "11111111-1111-1111-1111-111111111111", "duration_minutes": 15},
    )
    assert created.status_code == 200
    sess = created.json()
    assert sess["status"] == "open"
    assert sess["duration_minutes"] == 15
    assert sess["starts_at"] is not None
    assert sess["ends_at"] is not None
    assert "server_time" in sess

    # Server time endpoint
    time_resp = client.get("/api/time")
    assert time_resp.status_code == 200
    assert "server_time" in time_resp.json()
    assert "server_timestamp_ms" in time_resp.json()


# ── TEST 5: Session Auto-Close and WebSocket Event Counts ─────────────────────
def test_session_auto_close_and_audit(client: TestClient):
    t_token, teacher = login(client, "teacher@demo.edu", "teacher")

    # Create session
    created = client.post(
        "/api/sessions",
        headers=auth_hdr(t_token),
        json={"class_id": "11111111-1111-1111-1111-111111111111", "duration_minutes": 5},
    ).json()
    session_id = created["id"]

    # Manually backdate ends_at to 10 seconds ago to simulate expiry
    past_iso = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    supa().table("sessions").update({"ends_at": past_iso}).eq("id", session_id).execute()

    # Reading session triggers lazy close
    fetched = client.get(f"/api/sessions/{session_id}", headers=auth_hdr(t_token)).json()
    assert fetched["status"] == "closed"

    # Audit log contains session.closed with counts
    audit = client.get(f"/api/audit?session_id={session_id}", headers=auth_hdr(t_token)).json()
    actions = [a["action"] for a in audit]
    assert "session.closed" in actions
    closed_audit = next(a for a in audit if a["action"] == "session.closed")
    meta = closed_audit["metadata"]
    assert "present" in meta
    assert "needs_review" in meta
    assert "absent" in meta
    assert "total_enrolled" in meta
    assert meta["total_enrolled"] == 2

    # In-app notification created for lecturer
    notifs = client.get("/api/notifications", headers=auth_hdr(t_token)).json()
    assert any(n["type"] == "session.closed" for n in notifs)


# ── TEST 6: Continuous Sweep Verification Rules ───────────────────────────────
def test_sweep_checkin_rules_and_outcomes(client: TestClient, fake_engine: FakeFaceEngine):
    t_token, _ = login(client, "teacher@demo.edu", "teacher")
    s_token, student = login(client, "student1@demo.edu", "student")

    student_face = FaceDetection(bbox_x=120, bbox_y=80, bbox_w=60, bbox_h=80, confidence=0.98, ipd_px=22.0)
    neighbour_face = FaceDetection(bbox_x=220, bbox_y=85, bbox_w=55, bbox_h=75, confidence=0.92, ipd_px=20.0)
    lecturer_face = FaceDetection(bbox_x=150, bbox_y=50, bbox_w=100, bbox_h=120, confidence=0.95, ipd_px=38.0)

    # Enroll student reference selfie first
    fake_engine.preset_detections = [student_face]
    frame_jpg = make_test_frame_bytes()
    client.post(
        "/api/auth/enroll",
        headers=auth_hdr(s_token),
        data={"consent": "true"},
        files={"photo": ("selfie.jpg", frame_jpg, "image/jpeg")},
    )

    # Enroll lecturer reference selfie
    fake_engine.preset_detections = [lecturer_face]
    client.post(
        "/api/auth/enroll",
        headers=auth_hdr(t_token),
        data={"consent": "true"},
        files={"photo": ("lecturer_selfie.jpg", frame_jpg, "image/jpeg")},
    )

    # Create active open session
    created = client.post(
        "/api/sessions",
        headers=auth_hdr(t_token),
        json={"class_id": "11111111-1111-1111-1111-111111111111", "duration_minutes": 10},
    ).json()
    session_id = created["id"]

    # Helper to generate sweep upload payload
    def make_sweep_upload(
        num_frames: int = 15,
        heading_span: float = 180.0,
        max_gap: float = 12.0,
        duration: float = 6.0,
        frame_bytes: bytes = frame_jpg,
    ):
        metadata = []
        files = []
        time_step = duration / max(1, num_frames - 1)
        heading_step = heading_span / max(1, num_frames - 1)
        for i in range(num_frames):
            metadata.append({
                "index": i,
                "heading_deg": round(i * heading_step, 1),
                "time_offset_s": round(i * time_step, 2),
            })
            files.append(("frames", (f"frame_{i}.jpg", frame_bytes, "image/jpeg")))
        return {"attempt_id": f"att_{int(time.time()*1000)}_{num_frames}", "metadata": json.dumps(metadata)}, files

    # 1. Negative Test: Too few frames (< 12)
    data, files = make_sweep_upload(num_frames=8)
    r1 = client.post(f"/api/sessions/{session_id}/checkin", headers=auth_hdr(s_token), data=data, files=files)
    assert r1.status_code == 200
    assert r1.json()["state"] == "rejected"
    assert r1.json()["reason_code"] == "TOO_FEW_FRAMES"

    # 2. Negative Test: Heading span insufficient (< 150°)
    data, files = make_sweep_upload(num_frames=16, heading_span=90.0)
    r2 = client.post(f"/api/sessions/{session_id}/checkin", headers=auth_hdr(s_token), data=data, files=files)
    assert r2.status_code == 200
    assert r2.json()["state"] == "rejected"
    assert r2.json()["reason_code"] == "HEADING_SPAN_INSUFFICIENT"

    # 3. Successful Verification with Neighbours and Lecturer
    # Configure fake engine detections:
    # Student in center, Neighbour at dist 1.5 face widths, Lecturer with IPD >= 30px
    fake_engine.preset_detections = [
        FaceDetection(bbox_x=120, bbox_y=80, bbox_w=60, bbox_h=80, confidence=0.98, ipd_px=22.0),
        FaceDetection(bbox_x=220, bbox_y=85, bbox_w=55, bbox_h=75, confidence=0.92, ipd_px=20.0),  # neighbour
        FaceDetection(bbox_x=150, bbox_y=50, bbox_w=100, bbox_h=120, confidence=0.95, ipd_px=38.0), # lecturer (IPD >= 30)
    ]
    data, files = make_sweep_upload(num_frames=16, heading_span=180.0, duration=7.0)
    att_id = data["attempt_id"]
    r3 = client.post(f"/api/sessions/{session_id}/checkin", headers=auth_hdr(s_token), data=data, files=files)
    assert r3.status_code == 200
    res3 = r3.json()
    assert res3["state"] == "verified"
    assert res3["reason_code"] == "VERIFIED"

    # Attendance row is now present
    sess_status = client.get(f"/api/sessions/{session_id}", headers=auth_hdr(t_token)).json()
    st_att = next(a for a in sess_status["attendance"] if a["student_id"] == student["id"])
    assert st_att["status"] == "present"

    # 4. Duplicate Attempt ID rejected
    data["attempt_id"] = att_id
    dup_resp = client.post(f"/api/sessions/{session_id}/checkin", headers=auth_hdr(s_token), data=data, files=files)
    assert dup_resp.json()["state"] == "rejected"
    assert dup_resp.json()["reason_code"] == "DUPLICATE_ATTEMPT"

    # 5. Second submission after already verified rejected
    data2, files2 = make_sweep_upload(num_frames=16, heading_span=180.0)
    dup_ver = client.post(f"/api/sessions/{session_id}/checkin", headers=auth_hdr(s_token), data=data2, files=files2)
    assert dup_ver.json()["state"] == "rejected"
    assert dup_ver.json()["reason_code"] == "ALREADY_VERIFIED"


# ── TEST 7: Flagged Outcome & Lecturer Review Queue ───────────────────────────
def test_flagged_outcome_and_lecturer_review(client: TestClient, fake_engine: FakeFaceEngine):
    t_token, _ = login(client, "teacher@demo.edu", "teacher")
    s_token, student2 = login(client, "student2@demo.edu", "student")

    # Enroll student reference selfie (check-in requires an enrolled reference)
    fake_engine.preset_detections = [
        FaceDetection(bbox_x=120, bbox_y=80, bbox_w=60, bbox_h=80, confidence=0.98, ipd_px=22.0),
    ]
    enroll = client.post(
        "/api/auth/enroll",
        headers=auth_hdr(s_token),
        data={"consent": "true"},
        files={"photo": ("selfie.jpg", make_test_frame_bytes(), "image/jpeg")},
    )
    assert enroll.status_code == 200, enroll.text

    # Create session
    sess = client.post(
        "/api/sessions",
        headers=auth_hdr(t_token),
        json={"class_id": "11111111-1111-1111-1111-111111111111", "duration_minutes": 10},
    ).json()
    session_id = sess["id"]

    # Configure fake detections with NO neighbours (should cause FLAGGED)
    fake_engine.preset_detections = [
        FaceDetection(bbox_x=120, bbox_y=80, bbox_w=60, bbox_h=80, confidence=0.98, ipd_px=22.0),
    ]

    metadata = [{"index": i, "heading_deg": i * 12.0, "time_offset_s": i * 0.4} for i in range(16)]
    frame_bytes = make_test_frame_bytes()
    files = [("frames", (f"f_{i}.jpg", frame_bytes, "image/jpeg")) for i in range(16)]

    flag_resp = client.post(
        f"/api/sessions/{session_id}/checkin",
        headers=auth_hdr(s_token),
        data={"attempt_id": "att_flagged_1", "metadata": json.dumps(metadata)},
        files=files,
    )
    assert flag_resp.status_code == 200
    outcome = flag_resp.json()
    assert outcome["state"] == "flagged"

    # Check Lecturer Review Queue
    queue = client.get(f"/api/sessions/{session_id}/review-queue", headers=auth_hdr(t_token)).json()
    assert len(queue) >= 1
    item = next(q for q in queue if q["attempt_id"] == "att_flagged_1")
    assert item["state"] == "flagged"
    assert len(item["filmstrip_urls"]) > 0

    # Lecturer Accepts the Flagged Attempt
    decide_resp = client.post(
        f"/api/sessions/{session_id}/review/att_flagged_1",
        headers=auth_hdr(t_token),
        json={"decision": "accept", "note": "Verified manually via filmstrip inspection"},
    )
    assert decide_resp.status_code == 200
    assert decide_resp.json()["attendance_status"] == "present"

    # Verify student attendance is now present
    sess_after = client.get(f"/api/sessions/{session_id}", headers=auth_hdr(t_token)).json()
    st_att = next(a for a in sess_after["attendance"] if a["student_id"] == student2["id"])
    assert st_att["status"] == "present"


# ── TEST 8: Manual Override, Finalization, and Reports ────────────────────────
def test_manual_override_finalize_and_reports(client: TestClient):
    t_token, _ = login(client, "teacher@demo.edu", "teacher")
    sess = client.post(
        "/api/sessions",
        headers=auth_hdr(t_token),
        json={"class_id": "11111111-1111-1111-1111-111111111111", "duration_minutes": 10},
    ).json()
    session_id = sess["id"]

    student_id = "00000000-0000-0000-0000-000000000002"

    # Lecturer manual roll-call override
    over = client.patch(
        f"/api/sessions/{session_id}/attendance/{student_id}",
        headers=auth_hdr(t_token),
        json={"status": "present", "note": "Manual roll call opt-out"},
    )
    assert over.status_code == 200
    assert over.json()["status"] == "present"

    # Finalize session
    fin = client.post(
        f"/api/sessions/{session_id}/finalize",
        headers=auth_hdr(t_token),
        json={"mark_pending_absent": True},
    )
    assert fin.status_code == 200
    assert fin.json()["status"] == "finalized"
    assert fin.json()["summary"]["pending"] == 0

    # Report generation
    report = client.get(f"/api/sessions/{session_id}/report", headers=auth_hdr(t_token)).json()
    assert "csv" in report
    assert "pdf_html" in report
    assert "Ada Lovelace" in report["csv"] or "teacher@demo.edu" in report["csv"] or "CS-101" in report["session"]["class_code"]

    # WebSocket single-use ticket
    ticket = client.post("/api/realtime/ticket", headers=auth_hdr(t_token)).json()
    assert "ticket" in ticket
    assert ticket["expires_in"] == 60

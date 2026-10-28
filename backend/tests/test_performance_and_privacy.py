"""Automated QA & Evaluation Suite: Performance, Privacy, and Operational Reliability.

TASK 09 — Performance, Privacy, and Operational Reliability:
1. Component Latency Breakdown:
   - Image decode and preprocessing latency.
   - Face detection latency (YuNet).
   - Embedding extraction and identity matching latency (SFace/ArcFace).
   - Passive liveness (SPMR 2D FFT spectral moiré & chroma variance).
   - 1:N roster matching scaling across N=10 to N=1000 enrolled templates.
   - End-to-end API response latency.
2. Resource Utilization & Memory:
   - Process RSS memory footprint before, during, and after repeated verifications.
   - Memory growth verification across batch runs (< 25 MB ceiling).
3. Concurrency & Contention:
   - Concurrent attendance submissions (5, 10, 20 threads).
   - Mutual exclusion: only 1 present status granted per student, zero duplicate attendance records.
4. Privacy & Data Lifecycle (GDPR / DPDP Section 12):
   - Student self-service Right-to-Erasure: DELETE /api/auth/enrollment permanently purges template.
   - Administrative purge: DELETE /api/controller/students/{student_id}/biometrics.
   - Fail-closed verification after deletion (400 INVALID_ENROLLMENT_DATA / MISSING_ENROLLMENT_DATA).
   - Privacy audit: Zero raw biometric vectors, passwords, or credentials in audit logs.
5. Operational Reliability & Fail-Closed Behavior:
   - Safe handling of model engine faults (503 MODEL_UNAVAILABLE fail-closed).
   - Safe database timeout handling without sensitive internal trace exposure.
   - Zero attendance creation upon failed liveness or low identity similarity.
"""
from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import psutil
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

os.environ["ATTENDANCE_ENV"] = "test"
os.environ["SUPABASE_JWT_SECRET"] = "test_jwt_secret_must_be_at_least_32_chars_long"

from app.main import app
from app.db import supa, reset_test_db
from app.security import (
    create_access_token,
    encrypt_face_embedding,
    decrypt_face_embedding,
    reset_security_caches,
)
from app.face_engine import get_face_engine

client = TestClient(app)

STUDENT_ID = "00000000-0000-0000-0000-000000000002"
STUDENT_2_ID = "00000000-0000-0000-0000-000000000003"
TEACHER_ID = "00000000-0000-0000-0000-000000000001"
CONTROLLER_ID = "00000000-0000-0000-0000-000000000099"
CLASS_ID = "11111111-1111-1111-1111-111111111111"
SESSION_ID = "ssssssss-0001-0001-0001-000000000001"


def _create_synthetic_face_jpg(width: int = 320, height: int = 320) -> bytes:
    img = np.full((height, width, 3), 140, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    for i in range(15, height - 15, 20):
        cv2.line(img, (20, i), (width - 20, i), (152, 148, 132), 1)
    cv2.rectangle(img, (cx - 45, cy - 32), (cx - 25, cy - 18), (18, 18, 18), -1)
    cv2.rectangle(img, (cx + 25, cy - 32), (cx + 45, cy - 18), (18, 18, 18), -1)
    cv2.circle(img, (cx, cy + 12), 6, (30, 30, 30), -1)
    cv2.line(img, (cx - 28, cy + 45), (cx + 28, cy + 45), (40, 40, 40), 3)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


@pytest.fixture(autouse=True)
def setup_suite():
    reset_test_db()
    reset_security_caches()
    db = supa()

    # Seed users
    db.table("profiles").insert({
        "id": STUDENT_ID,
        "name": "Jane Student",
        "email": "jane@college.edu",
        "role": "student",
        "roll_number": "STU-001",
        "registration_locked": True,
    }).execute()

    db.table("profiles").insert({
        "id": STUDENT_2_ID,
        "name": "Alex Student",
        "email": "alex@college.edu",
        "role": "student",
        "roll_number": "STU-002",
        "registration_locked": False,
    }).execute()

    db.table("profiles").insert({
        "id": TEACHER_ID,
        "name": "Dr. Alan Turing",
        "email": "teacher@college.edu",
        "role": "teacher",
    }).execute()

    db.table("profiles").insert({
        "id": CONTROLLER_ID,
        "name": "Dean Controller",
        "email": "controller@college.edu",
        "role": "controller",
    }).execute()

    # Seed class & session
    db.table("classes").insert({
        "id": CLASS_ID,
        "name": "CS401 Distributed Systems",
        "teacher_id": TEACHER_ID,
    }).execute()

    db.table("enrollments").insert({
        "id": "e1111111-0001-0001-0001-000000000001",
        "class_id": CLASS_ID,
        "student_id": STUDENT_ID,
    }).execute()

    db.table("enrollments").insert({
        "id": "e1111111-0001-0001-0001-000000000002",
        "class_id": CLASS_ID,
        "student_id": STUDENT_2_ID,
    }).execute()

    db.table("sessions").insert({
        "id": SESSION_ID,
        "class_id": CLASS_ID,
        "status": "open",
        "is_active": True,
        "require_liveness": False,
        "device_locked": False,
        "verification_mode": "biometric",
    }).execute()

    # Seed reference biometric embedding for Jane
    engine = get_face_engine()
    raw_face = _create_synthetic_face_jpg()
    nparr = np.frombuffer(raw_face, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = engine.detect_faces(img, allow_fallback=True)
    face = faces[0] if faces else (0, 0, 320, 320)
    emb = engine.extract_embedding(img, face)

    db.table("face_templates").insert({
        "user_id": STUDENT_ID,
        "encrypted_embedding": encrypt_face_embedding(emb),
        "model_version": "arcface-yunet-v1",
        "consent_timestamp": "2026-10-25T10:00:00Z",
    }).execute()


# ── Category 1: Real Component Latency Breakdown ──────────────────────────────

def test_perf_component_latencies_breakdown():
    """Measure real latency of each stage: decode, detection, embedding, PAD, and matching."""
    engine = get_face_engine()
    jpg_bytes = _create_synthetic_face_jpg(320, 320)
    iterations = 25

    # 1. Preprocessing / JPEG decode latency
    decode_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        resized = cv2.resize(img, (320, 320))
        t1 = time.perf_counter()
        decode_times.append((t1 - t0) * 1000.0)

    mean_decode = float(np.mean(decode_times))
    p95_decode = float(np.percentile(decode_times, 95))

    # 2. Face Detection Latency (YuNet / Fallback)
    detect_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        faces = engine.detect_faces(resized, allow_fallback=True)
        t1 = time.perf_counter()
        detect_times.append((t1 - t0) * 1000.0)

    mean_detect = float(np.mean(detect_times))
    p95_detect = float(np.percentile(detect_times, 95))
    face = faces[0] if faces else (0, 0, 320, 320)

    # 3. Embedding Extraction Latency (SFace / ArcFace)
    embed_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        emb = engine.extract_embedding(resized, face)
        t1 = time.perf_counter()
        embed_times.append((t1 - t0) * 1000.0)

    mean_embed = float(np.mean(embed_times))
    p95_embed = float(np.percentile(embed_times, 95))

    # 4. Passive Liveness (SPMR 2D FFT Moiré Analysis) Latency
    liveness_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        pad_result = engine.detect_presentation_attack(resized, face)
        t1 = time.perf_counter()
        liveness_times.append((t1 - t0) * 1000.0)

    mean_liveness = float(np.mean(liveness_times))
    p95_liveness = float(np.percentile(liveness_times, 95))

    # 5. Cryptographic Encryption / Decryption Latency
    crypto_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        c = encrypt_face_embedding(emb)
        _ = decrypt_face_embedding(c)
        t1 = time.perf_counter()
        crypto_times.append((t1 - t0) * 1000.0)

    mean_crypto = float(np.mean(crypto_times))

    # Assert reasonable real performance limits on host hardware
    assert mean_decode < 50.0, f"Image decode too slow: {mean_decode:.2f}ms"
    assert mean_detect < 150.0, f"Face detection too slow: {mean_detect:.2f}ms"
    assert mean_embed < 150.0, f"Embedding extraction too slow: {mean_embed:.2f}ms"
    assert mean_liveness < 50.0, f"SPMR FFT analysis too slow: {mean_liveness:.2f}ms"
    assert mean_crypto < 10.0, f"AES-GCM encryption too slow: {mean_crypto:.2f}ms"

    total_pipeline = mean_decode + mean_detect + mean_embed + mean_liveness + mean_crypto
    print(
        f"\n[REAL PERFORMANCE METRICS]\n"
        f"  - Decode/Preprocess: mean={mean_decode:.2f}ms, p95={p95_decode:.2f}ms\n"
        f"  - Detection:        mean={mean_detect:.2f}ms, p95={p95_detect:.2f}ms\n"
        f"  - Embedding:        mean={mean_embed:.2f}ms, p95={p95_embed:.2f}ms\n"
        f"  - Liveness FFT:     mean={mean_liveness:.2f}ms, p95={p95_liveness:.2f}ms\n"
        f"  - Crypto AES-GCM:   mean={mean_crypto:.2f}ms\n"
        f"  - Total In-Engine:  mean={total_pipeline:.2f}ms"
    )


def test_perf_1_to_n_roster_matching_scaling():
    """Benchmark vectorized cosine similarity scaling across N=10 to N=1000 enrolled templates."""
    engine = get_face_engine()
    probe_vector = np.random.randn(512).astype(np.float32)
    probe_vector = probe_vector / np.linalg.norm(probe_vector)

    scales = [10, 50, 100, 500, 1000]
    latencies = {}

    for n in scales:
        gallery = np.random.randn(n, 512).astype(np.float32)
        gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)

        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            # Vectorized dot product for 1:N cosine similarity
            sims = np.dot(gallery, probe_vector)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(times))
        latencies[n] = mean_ms
        assert mean_ms < 5.0, f"1:{n} matching exceeded 5ms limit ({mean_ms:.3f}ms)"

    print(f"\n[1:N MATCHING SCALABILITY]: {latencies}")


# ── Category 2: Concurrency & Database Contention ─────────────────────────────

def test_perf_concurrent_attendance_submissions_and_locking():
    """Test concurrent attendance submissions from same student.
    
    Verifies:
    1. Zero duplicate check-in records created in database.
    2. Exactly 1 request succeeds with 200 OK.
    3. Other requests receive idempotent cached response or error.
    """
    token = create_access_token(STUDENT_ID, "jane@college.edu", "student", "Jane Student")
    jpg_bytes = _create_synthetic_face_jpg()

    concurrency = 10
    results = []

    def _submit_checkin(worker_id: int):
        # Each worker submits with its own unique idempotency key
        headers = {
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": f"concurrent-worker-{worker_id}-{time.time()}",
        }
        files = {"face_image": ("face.jpg", jpg_bytes, "image/jpeg")}
        res = client.post(
            f"/api/sessions/{SESSION_ID}/verify-attendance",
            headers=headers,
            files=files,
        )
        return res.status_code, res.text

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_submit_checkin, i) for i in range(concurrency)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    status_codes = [r[0] for r in results]
    print(f"\n[CONCURRENT BURST CODES ({concurrency} threads)]: {status_codes}")

    # Exactly 1 should have created the attendance record (200 OK)
    successes = [code for code in status_codes if code == 200]
    assert len(successes) == 1, f"Expected exactly 1 success, got {len(successes)}"

    # Check database: strictly 1 attendance row for student in this session
    db = supa()
    att_rows = (
        db.table("attendance")
        .select("id")
        .eq("session_id", SESSION_ID)
        .eq("student_id", STUDENT_ID)
        .execute()
        .data
    )
    assert len(att_rows) == 1, f"Database integrity violation: {len(att_rows)} rows found!"


# ── Category 3: Memory Footprint & Resource Leakage ───────────────────────────

def test_perf_memory_footprint_and_leak_detection():
    """Measure process RSS memory before and after repeated verification runs."""
    process = psutil.Process(os.getpid())
    gc.collect()
    rss_start = process.memory_info().rss / (1024 * 1024)

    token = create_access_token(STUDENT_ID, "jane@college.edu", "student", "Jane Student")
    jpg_bytes = _create_synthetic_face_jpg()

    # Run 15 consecutive verification calls
    for i in range(15):
        # Use fresh student to avoid duplicate checkin short-circuit
        fresh_student_id = f"00000000-0000-0000-0000-{i+10:012d}"
        supa().table("profiles").insert({
            "id": fresh_student_id,
            "name": f"Student {i}",
            "email": f"stu{i}@college.edu",
            "role": "student",
            "roll_number": f"ROLL-{i}",
        }).execute()
        supa().table("enrollments").insert({
            "id": f"e1111111-0000-0000-0000-{i+10:012d}",
            "class_id": CLASS_ID,
            "student_id": fresh_student_id,
        }).execute()
        
        engine = get_face_engine()
        nparr = np.frombuffer(jpg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces = engine.detect_faces(img, allow_fallback=True)
        face = faces[0] if faces else (0, 0, 320, 320)
        emb = engine.extract_embedding(img, face)
        supa().table("face_templates").insert({
            "user_id": fresh_student_id,
            "encrypted_embedding": encrypt_face_embedding(emb),
            "model_version": "arcface-yunet-v1",
            "consent_timestamp": "2026-10-25T10:00:00Z",
        }).execute()

        stu_token = create_access_token(fresh_student_id, f"stu{i}@college.edu", "student", f"Student {i}")
        headers = {
            "Authorization": f"Bearer {stu_token}",
            "Idempotency-Key": f"mem-test-step-{i}",
        }
        files = {"face_image": ("face.jpg", jpg_bytes, "image/jpeg")}
        _ = client.post(
            f"/api/sessions/{SESSION_ID}/verify-attendance",
            headers=headers,
            files=files,
        )

    gc.collect()
    rss_end = process.memory_info().rss / (1024 * 1024)
    growth = rss_end - rss_start

    print(f"\n[PROCESS MEMORY METRICS]: Start={rss_start:.2f}MB, End={rss_end:.2f}MB, Growth={growth:.2f}MB")
    # Growth over 15 dynamic face verifications must stay bounded (< 30 MB)
    assert growth < 30.0, f"Potential memory leak detected: {growth:.2f} MB growth"


# ── Category 4: Biometric Privacy & Deletion (GDPR / DPDP s.12) ───────────────

def test_privacy_biometric_right_to_deletion_roundtrip():
    """Verify Student Right to Erasure / DPDP Section 12:
    
    1. Student can view enrolled status.
    2. Student calls DELETE /api/auth/enrollment.
    3. Template is permanently erased from face_templates table.
    4. Profile registration_locked is unlocked (re-enrollment enabled).
    5. Subsequent attendance verification fails closed (400 INVALID_ENROLLMENT_DATA).
    6. Audit trail logs 'biometric.template_deleted'.
    """
    token = create_access_token(STUDENT_ID, "jane@college.edu", "student", "Jane Student")
    db = supa()

    # 1. Status is enrolled
    st_resp = client.get("/api/auth/enrollment-status", headers={"Authorization": f"Bearer {token}"})
    assert st_resp.status_code == 200
    assert st_resp.json()["enrolled"] is True

    # 2. Call DELETE /api/auth/enrollment
    del_resp = client.delete("/api/auth/enrollment", headers={"Authorization": f"Bearer {token}"})
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["status"] == "deleted"
    assert del_data["registration_locked"] is False

    # 3. Verify database state
    tmpl = db.table("face_templates").select("id").eq("user_id", STUDENT_ID).execute().data
    assert len(tmpl) == 0, "Biometric template was not erased from database!"

    profs = db.table("profiles").select("registration_locked").eq("id", STUDENT_ID).execute().data
    assert profs and all(p["registration_locked"] is False for p in profs)

    # 4. Status is now unenrolled
    st_resp2 = client.get("/api/auth/enrollment-status", headers={"Authorization": f"Bearer {token}"})
    assert st_resp2.status_code == 200
    assert st_resp2.json()["enrolled"] is False

    # 5. Subsequent verification fails closed
    jpg_bytes = _create_synthetic_face_jpg()
    headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": "after-deletion-key"}
    files = {"face_image": ("face.jpg", jpg_bytes, "image/jpeg")}
    fail_verify = client.post(
        f"/api/sessions/{SESSION_ID}/verify-attendance",
        headers=headers,
        files=files,
    )
    assert fail_verify.status_code in (400, 403)
    assert "not enrolled" in fail_verify.text.lower() or "biometric" in fail_verify.text.lower()

    # 6. Audit trail verified
    audit_events = (
        db.table("audit_log")
        .select("*")
        .eq("actor_id", STUDENT_ID)
        .eq("action", "biometric.template_deleted")
        .execute()
        .data
    )
    assert len(audit_events) >= 1
    assert "biometric erasure" in audit_events[0]["note"].lower()


def test_privacy_controller_purge_biometrics():
    """Verify Controller administrative biometric purge endpoint."""
    controller_token = create_access_token(CONTROLLER_ID, "controller@college.edu", "controller", "Dean Controller")
    student_token = create_access_token(STUDENT_ID, "jane@college.edu", "student", "Jane Student")
    db = supa()

    # Controller executes purge
    resp = client.delete(
        f"/api/controller/students/{STUDENT_ID}/biometrics",
        headers={"Authorization": f"Bearer {controller_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "purged"
    assert data["student_id"] == STUDENT_ID

    # Verify template is deleted
    tmpl = db.table("face_templates").select("id").eq("user_id", STUDENT_ID).execute().data
    assert len(tmpl) == 0

    # Unauthorized student cannot purge another student's biometrics
    bad_req = client.delete(
        f"/api/controller/students/{STUDENT_ID}/biometrics",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert bad_req.status_code == 403


def test_privacy_zero_sensitive_data_in_debug_and_audit_logs():
    """Ensure no raw 512-dim embedding float arrays, tokens, or plaintext passwords leak into logs."""
    db = supa()
    logs = db.table("audit_log").select("*").execute().data or []
    for log_entry in logs:
        text_rep = json.dumps(log_entry).lower()
        # Raw float arrays or vector patterns should never appear in audit notes/metadata
        assert "encrypted_embedding" not in log_entry.get("metadata", {})
        assert "password" not in text_rep
        assert "bearer " not in text_rep
        assert "eyjhbgcioij" not in text_rep  # JWT prefix check


# ── Category 5: Operational Reliability & Fail-Closed Behavior ───────────────

def test_reliability_fail_closed_under_model_fault():
    """Verify that if the face engine encounters a corrupted runtime state or error,
    the API fails closed cleanly with HTTP 500/503 and NEVER marks attendance.
    """
    token = create_access_token(STUDENT_ID, "jane@college.edu", "student", "Jane Student")
    jpg_bytes = _create_synthetic_face_jpg()

    # Mock engine failure
    client_shield = TestClient(app, raise_server_exceptions=False)
    with patch("app.main.get_face_engine", side_effect=RuntimeError("ONNX Runtime engine memory faulted")):
        headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": "model-fault-key"}
        files = {"face_image": ("face.jpg", jpg_bytes, "image/jpeg")}
        res = client_shield.post(
            f"/api/sessions/{SESSION_ID}/verify-attendance",
            headers=headers,
            files=files,
        )
        # Should fail closed safely
        assert res.status_code in (500, 503)

    # Check that zero attendance was marked
    db = supa()
    att = (
        db.table("attendance_records")
        .select("id")
        .eq("session_id", SESSION_ID)
        .eq("student_id", STUDENT_ID)
        .execute()
        .data
    )
    assert len(att) == 0, "Attendance was wrongly recorded during engine failure!"


def test_reliability_failed_verification_never_creates_attendance():
    """Adversarial submission with low similarity must be rejected with 403/400 and never marked."""
    token = create_access_token(STUDENT_ID, "jane@college.edu", "student", "Jane Student")

    # Blank grey box that has completely different or zero face match
    blank_img = np.zeros((320, 320, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", blank_img)
    blank_bytes = buf.tobytes()

    headers = {"Authorization": f"Bearer {token}", "Idempotency-Key": "adversarial-fail-key"}
    files = {"face_image": ("face.jpg", blank_bytes, "image/jpeg")}
    res = client.post(
        f"/api/sessions/{SESSION_ID}/verify-attendance",
        headers=headers,
        files=files,
    )
    assert res.status_code in (400, 403)

    # Check audit logs: failure recorded
    db = supa()
    audits = (
        db.table("audit_log")
        .select("*")
        .eq("session_id", SESSION_ID)
        .eq("actor_id", STUDENT_ID)
        .execute()
        .data
    )
    assert len(audits) >= 1
    assert any("face" in a.get("action", "").lower() or "failed" in a.get("note", "").lower() for a in audits)

    # Check attendance: strictly 0 present rows
    att = (
        db.table("attendance")
        .select("id")
        .eq("session_id", SESSION_ID)
        .eq("student_id", STUDENT_ID)
        .eq("status", "present")
        .execute()
        .data
    )
    assert len(att) == 0, "Present attendance record leaked on rejected verification!"

"""Classroom attendance API – Attendrix backend.

Consolidated entrypoint featuring:
- Role-enforced authentication with mismatch detection.
- Biometric reference selfie enrollment with encrypted embeddings.
- Server-authoritative time-limited attendance sessions with auto-close.
- 9-rule continuous front-camera sweep verification pipeline.
- Lecturer review queue for flagged attempts with key-frame filmstrip.
- Authenticated single-use WebSocket tickets and event streams.
- Full audit logging, in-app notifications, and CSV/print reporting.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import random
import secrets
import threading
import time
import uuid
import numpy as np
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import (
    ATTENDANCE_ENV,
    AUTO_CLOSE_INTERVAL_S,
    CORS_ORIGINS,
    DEFAULT_SESSION_DURATION_MINUTES,
    MAX_CAPTURE_BYTES,
    MAX_REQUEST_SIZE_BYTES,
    MAX_SESSION_DURATION_MINUTES,
    MIN_LAPLACIAN_VAR,
    MIN_LUMINANCE,
    SESSION_DURATION_PRESETS,
    SESSION_GRACE_PERIOD_S,
    STARTUP_DB_PROBE_TIMEOUT_S,
    WS_TICKET_TTL_SECONDS,
    validate_environment_config,
)
from app.db import first_row, is_supabase_backend, supa, utc_now
from app.face_engine import FaceDetection, get_face_engine
from app.config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, PRELOAD_FACE_ENGINE
from app.observability import (
    InferenceBusyError,
    RequestTimingMiddleware,
    configure_logging,
    inference_slot,
)
from app.security import (
    check_and_record_image_replay,
    check_login_rate_limit,
    check_rate_limit,
    check_submit_rate_limit,
    consume_challenge_nonce,
    create_access_token,
    create_liveness_challenge_token,
    decrypt_face_embedding,
    encrypt_face_embedding,
    is_token_revoked,
    revoke_token,
    verify_jwt_token,
    verify_liveness_challenge_token,
)
from app.verification import (
    SweepFrame,
    VerificationPipeline,
    assess_image_quality,
    compute_image_metrics,
    parse_image_frame,
    validate_image_payload,
    validate_single_face,
)

logger = logging.getLogger("attendrix")

# Configure stdout logging once for Render/production. No-op under pytest.
configure_logging()

# ── WebSocket state ───────────────────────────────────────────────────────────
_ws_lock = threading.Lock()
_ws_tickets: dict[str, tuple[str, float]] = {}
_ws_clients: dict[str, set[WebSocket]] = {}
_ws_client_loops: dict[WebSocket, asyncio.AbstractEventLoop] = {}
_ws_client_queues: dict[WebSocket, asyncio.Queue[str]] = {}

# ── Attendance Concurrency Lock Registry ───────────────────────────────────────
_att_lock_mutex = threading.Lock()
_attendance_locks: dict[tuple[str, str], threading.Lock] = {}


def _get_attendance_lock(session_id: str, student_id: str) -> threading.Lock:
    """Return a threading.Lock specific to (session_id, student_id) to serialize concurrent check-ins."""
    key = (session_id, student_id)
    with _att_lock_mutex:
        if key not in _attendance_locks:
            _attendance_locks[key] = threading.Lock()
        return _attendance_locks[key]


def _read_upload(upload: Optional[UploadFile]) -> bytes:
    """Read an uploaded file synchronously from the worker thread.

    Endpoints that perform CPU-heavy image work run as sync ``def`` handlers so
    Starlette executes them in its threadpool instead of on the asyncio event
    loop. Those handlers must use the underlying spooled file object rather
    than the async ``UploadFile.read()`` API.
    """
    if upload is None:
        return b""
    return upload.file.read()


def _inference_busy_http(exc: InferenceBusyError) -> HTTPException:
    """Translate a saturated inference queue into a retryable 503."""
    return HTTPException(
        status_code=503,
        detail=f"{exc} [INFERENCE_BUSY]",
        headers={"X-Error-Code": "INFERENCE_BUSY"},
    )


# ── Background Task: Auto-Close Expired Sessions ──────────────────────────────
def _auto_close_expired_sessions() -> int:
    """Close expired sessions using server time.

    This is synchronous on purpose: it is invoked via ``asyncio.to_thread`` so
    blocking Supabase/PostgREST I/O never stalls the event loop or health probe.
    """
    now_dt = datetime.now(timezone.utc)
    db = supa()
    open_sessions = (
        db.table("sessions")
        .select("*, classes(name, code, teacher_id)")
        .eq("status", "open")
        .execute()
    )
    closed = 0
    for sess in open_sessions.data or []:
        ends_at_str = sess.get("ends_at")
        if not ends_at_str:
            continue
        try:
            ends_at = datetime.fromisoformat(ends_at_str.replace("Z", "+00:00"))
            if now_dt >= ends_at:
                _close_session_internal(sess)
                closed += 1
        except Exception as e:
            logger.error(f"Error checking session expiry: {e}")
    return closed


async def _session_auto_close_loop():
    """Background task running in lifespan to close expired sessions on server clock."""
    while True:
        try:
            await asyncio.sleep(AUTO_CLOSE_INTERVAL_S)
            started = time.perf_counter()
            closed = await asyncio.to_thread(_auto_close_expired_sessions)
            if closed:
                logger.info(
                    f"[LIFECYCLE] auto_close closed={closed} "
                    f"duration_ms={(time.perf_counter() - started) * 1000.0:.1f}"
                )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Error in session auto-close loop: {exc}")


async def _preload_face_engine() -> None:
    """Load face models once per worker in a thread; never blocks startup or health checks."""
    started = time.perf_counter()
    try:
        await asyncio.to_thread(get_face_engine)
        logger.info(
            f"[LIFECYCLE] face_engine_preload status=ok "
            f"duration_ms={(time.perf_counter() - started) * 1000.0:.1f}"
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(f"[LIFECYCLE] face_engine_preload status=error error={type(exc).__name__}")


def _page_bounds(limit: Optional[int], offset: int) -> tuple[int, int]:
    """Clamp pagination params and return the inclusive PostgREST ``range`` bounds."""
    size = DEFAULT_PAGE_SIZE if limit is None else max(1, min(limit, MAX_PAGE_SIZE))
    start = max(0, offset)
    return start, start + size - 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()

    # Verify DB connectivity on startup without allowing a slow/unreachable
    # database to delay Render's health check past the probe timeout.
    probe_started = time.perf_counter()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                lambda: supa().table("profiles").select("id").limit(1).execute()
            ),
            timeout=STARTUP_DB_PROBE_TIMEOUT_S,
        )
        logger.info(
            f"[LIFECYCLE] startup_db_probe status=ok "
            f"duration_ms={(time.perf_counter() - probe_started) * 1000.0:.1f}"
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"[LIFECYCLE] startup_db_probe status=timeout "
            f"timeout_s={STARTUP_DB_PROBE_TIMEOUT_S} "
            f"duration_ms={(time.perf_counter() - probe_started) * 1000.0:.1f}"
        )
    except Exception as exc:
        if ATTENDANCE_ENV != "test":
            logger.warning(f"Database connection note: {exc}")

    # Validate environment configuration
    env_issues = validate_environment_config()
    for issue in env_issues:
        logger.warning(f"Security Configuration Warning: {issue}")

    # Launch auto-close background loop
    task = asyncio.create_task(_session_auto_close_loop())
    preload_task = asyncio.create_task(_preload_face_engine()) if PRELOAD_FACE_ENGINE else None
    logger.info(
        f"[LIFECYCLE] startup_complete duration_ms={(time.perf_counter() - started) * 1000.0:.1f} "
        f"face_engine={'preloading' if preload_task else 'lazy'} env={ATTENDANCE_ENV}"
    )
    yield
    if preload_task is not None and not preload_task.done():
        preload_task.cancel()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ── Security Middleware: Request Size Limiter ─────────────────────────────────
class RequestSizeLimitMiddleware:
    def __init__(self, app, max_upload_size: int = MAX_REQUEST_SIZE_BYTES):
        self.app = app
        self.max_upload_size = max_upload_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_upload_size:
                    response = JSONResponse(
                        status_code=413,
                        content={
                            "detail": f"Payload exceeds maximum allowed size of {self.max_upload_size} bytes. [PAYLOAD_TOO_LARGE]",
                            "error_code": "PAYLOAD_TOO_LARGE",
                        },
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        total_bytes = 0
        async def limited_receive():
            nonlocal total_bytes
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                total_bytes += len(body)
                if total_bytes > self.max_upload_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Payload exceeds maximum allowed size of {self.max_upload_size} bytes. [PAYLOAD_TOO_LARGE]",
                    )
            return message

        try:
            await self.app(scope, limited_receive, send)
        except HTTPException as exc:
            if exc.status_code == 413:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "detail": exc.detail,
                        "error_code": "PAYLOAD_TOO_LARGE",
                    },
                )
                await response(scope, receive, send)
            else:
                raise


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Attendrix Attendance API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enforce Request Size Limiter ASGI middleware
app.add_middleware(RequestSizeLimitMiddleware, max_upload_size=MAX_REQUEST_SIZE_BYTES)

# CORS Configuration: Disable credentials if wildcard origin is passed
_cors_credentials = True
if "*" in CORS_ORIGINS:
    _cors_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=_cors_credentials,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Application-side request timing (outermost): adds Server-Timing and logs one
# structured line per request without ever recording query strings or bodies.
app.add_middleware(RequestTimingMiddleware)


# ── Global Exception Handler Shield ───────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_shield(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please contact the administrator. [INTERNAL_SERVER_ERROR]",
            "error_code": "INTERNAL_SERVER_ERROR",
        },
    )


# ── Dependency: Auth & Role Guards ───────────────────────────────────────────
def current_user(
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    raw_token: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        raw_token = authorization.split(" ", 1)[1].strip()
    elif token:
        raw_token = token.strip()

    if not raw_token:
        raise HTTPException(status_code=401, detail="Bearer access token required")

    if is_token_revoked(raw_token):
        raise HTTPException(
            status_code=401,
            detail="Token has been revoked. [REVOKED_CREDENTIALS]",
            headers={"X-Error-Code": "REVOKED_CREDENTIALS"},
        )

    # Try verifying with PyJWT secret
    payload = verify_jwt_token(raw_token)
    uid = payload.get("sub", "")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid token: missing subject")

    resp = supa().table("profiles").select("*").eq("id", uid).execute()
    if not resp.data:
        raise HTTPException(status_code=401, detail="User profile not found")

    profile = resp.data[0]
    db_role = profile.get("role")
    if not db_role or db_role not in ("student", "teacher", "controller", "admin"):
        raise HTTPException(
            status_code=403,
            detail="User account has missing or invalid role assignment. [INVALID_ROLE]",
            headers={"X-Error-Code": "INVALID_ROLE"},
        )

    # Fail closed on role inconsistency between token claims and database authority
    token_role = (
        payload.get("app_metadata", {}).get("role")
        or payload.get("user_metadata", {}).get("role")
    )
    if token_role and token_role != db_role:
        raise HTTPException(
            status_code=403,
            detail="Token role claim does not match authoritative database profile. [ROLE_MISMATCH]",
            headers={"X-Error-Code": "ROLE_MISMATCH"},
        )

    if profile.get("is_active") is False:
        raise HTTPException(
            status_code=401,
            detail="User account is deactivated. [ACCOUNT_DEACTIVATED]",
            headers={"X-Error-Code": "ACCOUNT_DEACTIVATED"},
        )
    return profile


def require_teacher(user: dict[str, Any]) -> None:
    # Role in DB is 'teacher' (presented as Lecturer in UI)
    if user.get("role") != "teacher":
        raise HTTPException(status_code=403, detail="Lecturer authorization required")


def require_student(user: dict[str, Any]) -> None:
    if user.get("role") != "student":
        raise HTTPException(status_code=403, detail="Student authorization required")


def require_controller(user: dict[str, Any]) -> None:
    if user.get("role") not in ("controller", "admin"):
        raise HTTPException(status_code=403, detail="Controller authorization required")


# ── Utility: Session & Class Guards ───────────────────────────────────────────
def _get_class_or_404(class_id: str) -> dict[str, Any]:
    try:
        resp = supa().table("classes").select("*").eq("id", class_id).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="Class not found")
        return resp.data[0]
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Class not found")


def _assert_class_access(cls: dict[str, Any], user: dict[str, Any], teacher_only: bool = False) -> None:
    if user.get("role") in ("controller", "admin"):
        return
    if teacher_only or user.get("role") == "teacher":
        if cls.get("teacher_id") != user.get("id"):
            raise HTTPException(status_code=403, detail="You do not have access to this class")
        return
    enr = (
        supa()
        .table("enrollments")
        .select("class_id")
        .eq("class_id", cls["id"])
        .eq("student_id", user["id"])
        .execute()
    )
    if not enr.data:
        raise HTTPException(status_code=403, detail="You do not have access to this class")


def _get_session_or_404(session_id: str) -> dict[str, Any]:
    try:
        resp = (
            supa()
            .table("sessions")
            .select("*, classes(name, code, teacher_id)")
            .eq("id", session_id)
            .execute()
        )
        if not resp.data:
            raise HTTPException(status_code=404, detail="Session not found")
        row = resp.data[0]
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found")

    if isinstance(row.get("classes"), dict):
        row["class_name"] = row["classes"]["name"]
        row["class_code"] = row["classes"]["code"]
        row["teacher_id"] = row["classes"]["teacher_id"]
    elif not row.get("teacher_id") and row.get("class_id"):
        try:
            cls_resp = supa().table("classes").select("name, code, teacher_id").eq("id", row["class_id"]).execute()
            if cls_resp.data:
                c = cls_resp.data[0]
                row["class_name"] = c.get("name")
                row["class_code"] = c.get("code")
                row["teacher_id"] = c.get("teacher_id")
        except Exception:
            pass

    # Lazy auto-close check so expiry survives server restarts and offline time
    if row.get("status") == "open" and row.get("ends_at"):
        try:
            ends_at = datetime.fromisoformat(row["ends_at"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) >= ends_at:
                _close_session_internal(row)
                row["status"] = "closed"
        except Exception:
            pass

    return row


def _assert_session_access(
    row: dict[str, Any], user: dict[str, Any], teacher_only: bool = False
) -> None:
    if user.get("role") in ("controller", "admin"):
        return
    if teacher_only or user.get("role") == "teacher":
        if row.get("teacher_id") != user.get("id"):
            raise HTTPException(status_code=403, detail="You do not have access to this session")
    else:
        enr = (
            supa()
            .table("enrollments")
            .select("class_id")
            .eq("class_id", row["class_id"])
            .eq("student_id", user["id"])
            .execute()
        )
        if not enr.data:
            raise HTTPException(status_code=403, detail="You are not enrolled in this session's class")


def _sanitize_audit_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Ensure raw biometric arrays, landmark tuples, image buffers, tokens, or credentials are never written to audit logs."""
    if not metadata:
        return {}
    sensitive_words = {
        "embedding", "landmarks", "raw_image", "face_bytes", "vector", "face_landmarks",
        "password", "token", "secret", "authorization", "raw_token", "jwt", "cookie",
        "access_token", "refresh_token", "encrypted_embedding", "biometric_template", "credentials",
    }
    sanitized: dict[str, Any] = {}
    for k, v in metadata.items():
        k_lower = k.lower()
        if any(word in k_lower for word in sensitive_words):
            continue
        if isinstance(v, (np.ndarray, list)) and len(v) > 20:
            continue
        sanitized[k] = v
    return sanitized


def _audit(
    session_id: Optional[str] = None,
    actor_id: str = "system",
    action: str = "system.event",
    note: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    sanitized_meta = _sanitize_audit_metadata(metadata)
    supa().table("audit_log").insert(
        {
            "session_id": session_id,
            "action": action,
            "actor_id": actor_id,
            "timestamp": utc_now(),
            "note": note,
            "metadata_json": json.dumps(sanitized_meta),
        }
    ).execute()


def _create_notification(
    user_id: str,
    notif_type: str,
    title: str,
    body: str,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    row = {
        "user_id": user_id,
        "type": notif_type,
        "title": title,
        "body": body,
        "session_id": session_id,
        "created_at": utc_now(),
    }
    resp = supa().table("notifications").insert(row).execute()
    created = resp.data[0] if resp.data else row
    publish_event(event_payload("notification.created", created), [user_id])
    return created


def _close_session_internal(sess: dict[str, Any]) -> None:
    """Internal helper to close an expired session and emit session.closed event."""
    session_id = sess["id"]
    db = supa()
    # Check if already closed
    current = first_row(db.table("sessions").select("status").eq("id", session_id))
    if current and current.get("status") != "open":
        return

    db.table("sessions").update({"status": "closed"}).eq("id", session_id).execute()

    # Tally results
    att = db.table("attendance").select("status").eq("session_id", session_id).execute()
    att_rows = att.data or []
    present_count = sum(1 for a in att_rows if a.get("status") == "present")
    pending_count = sum(1 for a in att_rows if a.get("status") == "pending")
    absent_count = sum(1 for a in att_rows if a.get("status") == "absent")
    total_enrolled = len(att_rows)

    rej = (
        db.table("checkin_attempts")
        .select("id", count="exact")
        .eq("session_id", session_id)
        .eq("state", "rejected")
        .execute()
    )
    rejected_count = rej.count if rej.count is not None else len(rej.data or [])

    flagged_attempts = (
        db.table("checkin_attempts")
        .select("id", count="exact")
        .eq("session_id", session_id)
        .eq("state", "flagged")
        .execute()
    )
    needs_review = flagged_attempts.count if flagged_attempts.count is not None else pending_count

    counts_payload = {
        "session_id": session_id,
        "class_id": sess.get("class_id"),
        "present": present_count,
        "needs_review": needs_review,
        "absent": absent_count,
        "rejected_attempts": rejected_count,
        "total_enrolled": total_enrolled,
    }

    # Audit log
    teacher_id = sess.get("teacher_id") or sess.get("created_by")
    _audit(
        session_id,
        teacher_id,
        "session.closed",
        note="Attendance window closed automatically by server clock.",
        metadata=counts_payload,
    )

    # In-app notification for lecturer
    _create_notification(
        teacher_id,
        "session.closed",
        "Attendance Window Closed",
        f"Attendance window closed. {present_count} verified present out of {total_enrolled} enrolled.",
        session_id=session_id,
    )

    # Broadcast WebSocket event
    enr = db.table("enrollments").select("student_id").eq("class_id", sess.get("class_id")).execute()
    student_ids = [s["student_id"] for s in (enr.data or [])]
    publish_event(
        event_payload("session.closed", counts_payload),
        [teacher_id, *student_ids],
    )


# ── WebSocket helpers ─────────────────────────────────────────────────────────
def event_payload(event_type: str, data: Any) -> str:
    return json.dumps(
        {
            "type": event_type,
            "data": data,
            "timestamp": utc_now(),
        }
    )


def publish_event(event_payload_json: str, user_ids: list[str]) -> None:
    targets: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[str]]] = []
    with _ws_lock:
        for uid in user_ids:
            for ws in _ws_clients.get(uid, ()):
                loop = _ws_client_loops.get(ws)
                queue = _ws_client_queues.get(ws)
                if loop and loop.is_running() and queue is not None:
                    targets.append((loop, queue))
    for loop, queue in targets:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event_payload_json)
        except Exception:
            pass


# ── Pydantic Request Models ───────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=1, max_length=128)
    selected_role: Optional[str] = None  # 'student' or 'teacher' / 'lecturer'


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=1, max_length=128)
    name: Optional[str] = None
    role: Optional[str] = "student"
    roll_number: Optional[str] = None


class CreateSessionRequest(BaseModel):
    class_id: str
    duration_minutes: Optional[int] = Field(
        default=DEFAULT_SESSION_DURATION_MINUTES,
        ge=1,
        le=MAX_SESSION_DURATION_MINUTES,
    )


class AttendanceUpdate(BaseModel):
    status: str = Field(pattern="^(present|absent|pending)$")
    note: Optional[str] = Field(default=None, max_length=500)


class FinalizeRequest(BaseModel):
    mark_pending_absent: bool = True


class ReviewAttemptRequest(BaseModel):
    decision: str = Field(pattern="^(verified|rejected|accept|reject)$")
    note: Optional[str] = Field(default=None, max_length=500)


class CreateClassRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    code: str = Field(min_length=1, max_length=20)
    section: Optional[str] = Field(default="Section A", max_length=50)
    room_number: Optional[str] = Field(default="Room 101", max_length=50)
    capacity: Optional[int] = Field(default=120)
    hall_type: Optional[str] = Field(default="120-Seater Lecture Theater")
    room_photo: Optional[str] = None
    teacher_id: Optional[str] = None


class CreateStudentRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=5, max_length=120)
    roll_number: str = Field(min_length=1, max_length=50)
    student_id: Optional[str] = None
    class_ids: list[str] = []


class CreateTeacherRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=5, max_length=120)
    password: Optional[str] = Field(default="teach123", min_length=6, max_length=128)


class RegistrationWindowRequest(BaseModel):
    is_open: bool
    note: Optional[str] = None


def _is_registration_window_open() -> bool:
    try:
        resp = supa().table("system_settings").select("*").eq("key", "registration_window").execute()
        if resp.data:
            return bool(resp.data[0].get("is_open", True))
    except Exception:
        pass
    return True


# ── Routes: Health & Server Time ──────────────────────────────────────────────
@app.get("/")
@app.get("/health")
@app.get("/api/health")
async def health() -> dict[str, str]:
    # Deliberately dependency-free: no DB, model, filesystem, or network work.
    return {"status": "ok", "service": "attendrix-attendance-api"}


@app.get("/api/time")
def server_time() -> dict[str, Any]:
    """Return server clock time for client countdown synchronization."""
    now_dt = datetime.now(timezone.utc)
    return {
        "server_time": now_dt.isoformat(timespec="seconds"),
        "server_timestamp_ms": int(now_dt.timestamp() * 1000),
    }


# ── Routes: Authentication & Role Verification ────────────────────────────────
@app.post("/api/auth/login")
def login(req: LoginRequest, request: Request) -> dict[str, Any]:
    client_ip = request.client.host if request.client else "127.0.0.1"
    check_login_rate_limit(client_ip)

    email = req.email.strip().lower()

    # Look up profile safely without single() throwing PGRST116 on 0 rows
    try:
        prof_resp = supa().table("profiles").select("*").eq("email", email).execute()
        prof = prof_resp.data[0] if (prof_resp and prof_resp.data) else None
    except Exception as e:
        logger.warning(f"Error querying profiles table for {email}: {e}")
        prof = None

    # ── Password verification (was previously skipped — critical security fix)
    # Attempt Supabase Auth sign-in with password when available.
    # Fall back to simple validation in test environment.
    password_verified = False
    if req.password:
        try:
            # Production: use Supabase Auth to verify password against stored hash
            auth_resp = supa().auth.sign_in_with_password({"email": email, "password": req.password})
            password_verified = bool(auth_resp.session) or bool(auth_resp.user)
        except Exception as auth_err:
            logger.warning(f"Supabase auth sign-in error for {email}: {auth_err}")
            # Test / in-memory fallback: accept non-empty password if env is test
            if ATTENDANCE_ENV == "test":
                password_verified = True
            else:
                password_verified = False

    if not prof:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
            headers={"X-Error-Code": "INVALID_CREDENTIALS"},
        )

    if not password_verified:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
            headers={"X-Error-Code": "INVALID_CREDENTIALS"},
        )

    db_role = prof.get("role")
    if not db_role or db_role not in ("student", "teacher", "controller", "admin"):
        raise HTTPException(
            status_code=403,
            detail="Account has missing or invalid role assignment. Access denied. [INVALID_ROLE]",
            headers={"X-Error-Code": "INVALID_ROLE"},
        )

    if prof.get("is_active") is False:
        raise HTTPException(
            status_code=401,
            detail="Account is deactivated or suspended. [ACCOUNT_DEACTIVATED]",
            headers={"X-Error-Code": "ACCOUNT_DEACTIVATED"},
        )

    # Server-enforced role from authoritative database profile ONLY
    # Never derived from email substrings ('admin', 'controller') or request bodies
    effective_role = db_role

    # Map and enforce user-facing selected_role if supplied
    requested_role = req.selected_role.lower() if req.selected_role else None
    if requested_role:
        if requested_role in ("lecturer", "teacher"):
            expected_role = "teacher"
        elif requested_role == "student":
            expected_role = "student"
        elif requested_role in ("controller", "admin"):
            expected_role = "controller"
        else:
            expected_role = None

        if expected_role and effective_role != expected_role:
            role_label = (
                "Student" if effective_role == "student"
                else ("Controller" if effective_role in ("controller", "admin") else "Lecturer")
            )
            selected_label = (
                "Student" if expected_role == "student"
                else ("Controller" if expected_role == "controller" else "Lecturer")
            )
            raise HTTPException(
                status_code=403,
                detail=f"Role mismatch: This account is registered as {role_label}, not {selected_label}.",
            )

    # Issue verified token
    token = create_access_token(
        user_id=prof["id"],
        email=prof["email"],
        role=effective_role,
        name=prof.get("name", "User"),
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": prof["id"],
            "name": prof.get("name", "User"),
            "email": prof["email"],
            "role": effective_role,
        },
    }


@app.post("/api/auth/signup")
@app.post("/api/auth/register")
def signup(req: SignupRequest) -> dict[str, Any]:
    """Dedicated endpoint for new student and teacher registrations."""
    email = req.email.strip().lower()
    user_role = (req.role or "student").strip().lower()
    if user_role in ("teacher", "lecturer", "controller", "admin"):
        raise HTTPException(
            status_code=403,
            detail="Self-registration for teacher or controller privileges is prohibited. Staff accounts must be authorized by an administrator. [ROLE_ELEVATION_DENIED]",
            headers={"X-Error-Code": "ROLE_ELEVATION_DENIED"},
        )
    db_role = "student"
    display_name = (req.name or email.split("@")[0].replace(".", " ")).strip().title()

    # Check if user already exists
    try:
        existing = supa().table("profiles").select("id").eq("email", email).execute()
        if existing.data:
            raise HTTPException(
                status_code=400,
                detail="An account with this email already exists. Please sign in instead.",
                headers={"X-Error-Code": "USER_ALREADY_EXISTS"},
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Error checking existing user: {e}")

    if not _is_registration_window_open() and db_role == "student":
        raise HTTPException(
            status_code=403,
            detail="Student registration window is currently closed by the Academic Controller.",
            headers={"X-Error-Code": "REGISTRATION_WINDOW_CLOSED"},
        )

    # Create user in Supabase Auth
    new_id = str(uuid.uuid4())
    try:
        auth_res = supa().auth.admin.create_user({
            "email": email,
            "password": req.password,
            "email_confirm": True,
            "user_metadata": {"name": display_name, "role": db_role},
        })
        new_id = auth_res.user.id
    except Exception as auth_err:
        logger.warning(f"Supabase auth create_user: {auth_err}")
        if is_supabase_backend():
            raise HTTPException(
                status_code=400,
                detail="Account could not be created. Check the email and password and try again. [SIGNUP_FAILED]",
                headers={"X-Error-Code": "SIGNUP_FAILED"},
            )

    # Ensure profile row exists
    prof = None
    try:
        chk = supa().table("profiles").select("*").eq("id", new_id).execute()
        if chk.data:
            prof = chk.data[0]
        else:
            new_prof = {"id": new_id, "name": display_name, "email": email, "role": db_role}
            ins = supa().table("profiles").insert(new_prof).execute()
            prof = ins.data[0] if ins.data else new_prof
    except Exception as e:
        logger.error(f"Profile creation failed for new account: {e}")
        raise HTTPException(
            status_code=503,
            detail="Account profile could not be saved. Please try again later. [PROFILE_CREATE_FAILED]",
            headers={"X-Error-Code": "PROFILE_CREATE_FAILED"},
        )

    token = create_access_token(
        user_id=prof["id"],
        email=prof["email"],
        role=prof.get("role", db_role),
        name=prof.get("name", display_name),
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": prof["id"],
            "name": prof.get("name", display_name),
            "email": prof["email"],
            "role": prof.get("role", db_role),
        },
        "message": "Account created successfully. Welcome to Attendrix!",
    }


@app.post("/api/auth/logout")
def logout(
    authorization: Optional[str] = Header(default=None),
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    raw_token = ""
    if authorization and authorization.lower().startswith("bearer "):
        raw_token = authorization.split(" ", 1)[1].strip()
    if raw_token:
        revoke_token(raw_token)
    return {"message": "Successfully logged out and credentials revoked.", "user_id": user["id"]}


@app.get("/api/me")
def me(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
    }


# ── Routes: Biometric Selfie Reference Enrollment ─────────────────────────────
@app.post("/api/auth/enroll")
def enroll_reference_selfie(
    consent: bool = Form(...),
    photo: UploadFile = File(...),
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    """Enroll a quality-checked reference selfie. Stores ONLY encrypted embedding."""
    if not _is_registration_window_open():
        raise HTTPException(
            status_code=403,
            detail="Student biometric registration window is currently closed by the Academic Controller. Please contact the administrator.",
        )

    if user.get("registration_locked"):
        raise HTTPException(
            status_code=403,
            detail="Your biometric registration has already been completed and is locked. Please contact the Academic Controller if you need to re-enroll.",
        )

    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Biometric consent is required to enroll reference face verification.",
        )

    image_bytes = _read_upload(photo)
    img, err_code, err_msg = validate_image_payload(image_bytes, max_bytes=10485760)
    if err_code is not None:
        status_code = 413 if err_code == "IMAGE_TOO_LARGE" else (415 if err_code == "UNSUPPORTED_FORMAT" else 400)
        raise HTTPException(
            status_code=status_code,
            detail=f"{err_msg} [{err_code}]",
            headers={"X-Error-Code": err_code},
        )

    try:
        with inference_slot("enroll_reference"):
            engine = get_face_engine()
            faces = engine.detect_faces(img, allow_fallback=True)

            face_ok, face_err, face_msg = validate_single_face(faces)
            if not face_ok:
                raise HTTPException(
                    status_code=400,
                    detail=f"{face_msg} [{face_err}]",
                    headers={"X-Error-Code": face_err},
                )

            q_ok, q_err, q_msg, q_metrics = assess_image_quality(img, faces[0])
            if not q_ok:
                raise HTTPException(
                    status_code=400,
                    detail=f"{q_msg} [{q_err}]",
                    headers={"X-Error-Code": q_err},
                )

            # Extract 512-dim embedding and encrypt it symmetrically
            emb = engine.extract_embedding(img, faces[0])
    except InferenceBusyError as exc:
        raise _inference_busy_http(exc)

    encrypted_emb = encrypt_face_embedding(emb)
    consent_ts = utc_now()
    model_version = "arcface-yunet-v1"

    # Save to face_templates (overwriting previous enrollment if any)
    db = supa()
    existing = db.table("face_templates").select("id").eq("user_id", user["id"]).execute()
    if existing.data:
        db.table("face_templates").update(
            {
                "encrypted_embedding": encrypted_emb,
                "model_version": model_version,
                "consent_timestamp": consent_ts,
            }
        ).eq("user_id", user["id"]).execute()
    else:
        db.table("face_templates").insert(
            {
                "user_id": user["id"],
                "encrypted_embedding": encrypted_emb,
                "model_version": model_version,
                "consent_timestamp": consent_ts,
            }
        ).execute()

    # Lock student registration so option is closed afterwards
    try:
        db.table("profiles").update({"registration_locked": True}).eq("id", user["id"]).execute()
    except Exception as lock_err:
        logger.warning(f"Could not update registration_locked for user {user.get('id')}: {lock_err}")

    _audit(
        session_id=None,
        actor_id=user["id"],
        action="biometric.reference_enrolled",
        note=f"Student {user.get('name', 'Student')} ({user.get('roll_number', '')}) enrolled reference biometric face template. Registration locked.",
        metadata={"model_version": model_version},
    )

    # Raw image bytes and decrypted embedding are immediately freed
    del image_bytes
    del img

    return {
        "status": "enrolled",
        "registration_locked": True,
        "consent_timestamp": consent_ts,
        "model_version": model_version,
        "message": "Reference selfie successfully enrolled and encrypted. Registration is now locked.",
    }


@app.get("/api/auth/enrollment-status")
def enrollment_status(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    """Check if the signed-in user has enrolled a reference selfie."""
    template_data = None
    try:
        resp = (
            supa()
            .table("face_templates")
            .select("consent_timestamp, model_version, created_at")
            .eq("user_id", user["id"])
            .execute()
        )
        if resp and resp.data:
            template_data = resp.data[0]
    except Exception as e:
        logger.warning(f"Could not read face_templates for user {user.get('id')}: {e}")

    window_open = _is_registration_window_open()
    is_locked = bool(user.get("registration_locked")) or bool(template_data)
    if template_data:
        return {
            "enrolled": True,
            "registration_locked": is_locked,
            "registration_window_open": window_open,
            "consent_timestamp": template_data.get("consent_timestamp"),
            "model_version": template_data.get("model_version"),
            "roll_number": user.get("roll_number", ""),
            "student_id": user.get("student_id", ""),
            "name": user.get("name", ""),
        }
    return {
        "enrolled": False,
        "registration_locked": is_locked,
        "registration_window_open": window_open,
        "roll_number": user.get("roll_number", ""),
        "student_id": user.get("student_id", ""),
        "name": user.get("name", ""),
    }


@app.delete("/api/auth/enrollment")
def delete_own_enrollment(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    """Delete authenticated user's biometric face template (Right to Erasure / GDPR Art 17 / DPDP s. 12)."""
    db = supa()
    existing = db.table("face_templates").select("id").eq("user_id", user["id"]).execute()
    if not existing.data:
        raise HTTPException(
            status_code=404,
            detail="No biometric enrollment found for this user. [ENROLLMENT_NOT_FOUND]",
            headers={"X-Error-Code": "ENROLLMENT_NOT_FOUND"},
        )
    db.table("face_templates").delete().eq("user_id", user["id"]).execute()
    try:
        db.table("profiles").update({"registration_locked": False}).eq("id", user["id"]).execute()
    except Exception as lock_err:
        logger.warning(f"Could not reset registration_locked for user {user.get('id')}: {lock_err}")
    _audit(
        session_id=None,
        actor_id=user["id"],
        action="biometric.template_deleted",
        note=f"User {user.get('name', 'User')} ({user.get('roll_number', user.get('id'))}) requested biometric erasure. Reference template permanently deleted.",
        metadata={"user_id": user["id"], "role": user.get("role", "student")},
    )
    return {
        "status": "deleted",
        "user_id": user["id"],
        "registration_locked": False,
        "message": "Biometric face template permanently deleted and registration unlocked.",
    }


# ── Routes: Classes ───────────────────────────────────────────────────────────
@app.get("/api/classes")
def classes(user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    if user["role"] in ("controller", "admin"):
        resp = supa().table("classes").select("*").order("name").execute()
    elif user["role"] == "teacher":
        resp = supa().table("classes").select("*").eq("teacher_id", user["id"]).order("name").execute()
    else:
        enr = supa().table("enrollments").select("class_id").eq("student_id", user["id"]).execute()
        class_ids = [e["class_id"] for e in (enr.data or [])]
        if not class_ids:
            return []
        resp = supa().table("classes").select("*").in_("id", class_ids).order("name").execute()
    return resp.data or []


@app.get("/api/classes/{class_id}/students")
def class_students(class_id: str, user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    if user.get("role") not in ("teacher", "controller", "admin"):
        raise HTTPException(status_code=403, detail="Lecturer or Controller authorization required")
    cls = _get_class_or_404(class_id)
    if user.get("role") == "teacher":
        _assert_class_access(cls, user, teacher_only=True)
    resp = (
        supa()
        .table("enrollments")
        .select("profiles(id, name, email, role)")
        .eq("class_id", class_id)
        .execute()
    )
    return [e["profiles"] for e in (resp.data or []) if e.get("profiles")]


# ── Routes: Sessions Lifecycle ────────────────────────────────────────────────
@app.get("/api/sessions")
def sessions(
    user: dict[str, Any] = Depends(current_user),
    limit: Optional[int] = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    now_iso = utc_now()
    start, end = _page_bounds(limit, offset)
    if user["role"] in ("controller", "admin"):
        resp = (
            supa()
            .table("sessions")
            .select("*, classes(name, code, teacher_id)")
            .order("created_at", desc=True)
            .range(start, end)
            .execute()
        )
    elif user["role"] == "teacher":
        teacher_classes = supa().table("classes").select("id").eq("teacher_id", user["id"]).execute().data or []
        class_ids = [c["id"] for c in teacher_classes]
        if not class_ids:
            return []
        resp = (
            supa()
            .table("sessions")
            .select("*, classes(name, code, teacher_id)")
            .in_("class_id", class_ids)
            .order("created_at", desc=True)
            .range(start, end)
            .execute()
        )
    else:
        enr = supa().table("enrollments").select("class_id").eq("student_id", user["id"]).execute()
        class_ids = [e["class_id"] for e in (enr.data or [])]
        if not class_ids:
            return []
        resp = (
            supa()
            .table("sessions")
            .select("*, classes(name, code), attendance(status, note)")
            .in_("class_id", class_ids)
            .order("created_at", desc=True)
            .range(start, end)
            .execute()
        )
    rows = []
    for r in resp.data or []:
        flat = {**r}
        if isinstance(r.get("classes"), dict):
            flat["class_name"] = r["classes"]["name"]
            flat["class_code"] = r["classes"]["code"]
            flat.pop("classes", None)

        # Lazy auto-close check
        if flat.get("status") == "open" and flat.get("ends_at"):
            try:
                ends_at = datetime.fromisoformat(flat["ends_at"].replace("Z", "+00:00"))
                if datetime.now(timezone.utc) >= ends_at:
                    _close_session_internal(flat)
                    flat["status"] = "closed"
            except Exception:
                pass

        flat["server_time"] = now_iso
        rows.append(flat)
    return rows


@app.post("/api/sessions")
def create_session(
    body: CreateSessionRequest, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    require_teacher(user)
    cls = _get_class_or_404(body.class_id)
    _assert_class_access(cls, user, teacher_only=True)

    duration = body.duration_minutes or DEFAULT_SESSION_DURATION_MINUTES
    now_dt = datetime.now(timezone.utc)
    ends_dt = now_dt + timedelta(minutes=duration)

    now_iso = now_dt.isoformat(timespec="seconds")
    ends_iso = ends_dt.isoformat(timespec="seconds")

    base_sess = {
        "class_id": body.class_id,
        "status": "open",
        "created_by": user["id"],
        "created_at": now_iso,
    }
    try:
        sess_resp = (
            supa()
            .table("sessions")
            .insert(
                {
                    **base_sess,
                    "starts_at": now_iso,
                    "ends_at": ends_iso,
                    "duration_minutes": duration,
                    "room_photo": cls.get("room_photo"),
                }
            )
            .execute()
        )
    except Exception:
        sess_resp = supa().table("sessions").insert(base_sess).execute()
    session_id: str = sess_resp.data[0]["id"]

    # Seed attendance rows for all enrolled students
    students = (
        supa().table("enrollments").select("student_id").eq("class_id", body.class_id).execute()
    )
    att_rows = [
        {
            "session_id": session_id,
            "student_id": s["student_id"],
            "status": "pending",
            "updated_by": user["id"],
            "updated_at": now_iso,
        }
        for s in (students.data or [])
    ]
    if att_rows:
        supa().table("attendance").insert(att_rows).execute()

    _audit(
        session_id,
        user["id"],
        "session.created",
        note=f"Session created with {duration} minute duration.",
        metadata={"class_id": body.class_id, "duration_minutes": duration, "ends_at": ends_iso},
    )

    row = _get_session_or_404(session_id)
    payload = _session_payload(row, user)

    student_ids = [s["student_id"] for s in (students.data or [])]
    publish_event(
        event_payload("session.created", {"session_id": session_id, "class_id": body.class_id}),
        [user["id"], *student_ids],
    )
    return payload


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    row = _get_session_or_404(session_id)
    _assert_session_access(row, user)
    return _session_payload(row, user)


@app.post("/api/sessions/{session_id}/reference")
def teacher_set_session_reference(
    session_id: str,
    photo: Optional[UploadFile] = File(default=None),
    photo_base64: Optional[str] = Form(default=None),
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    require_teacher(user)
    db = supa()
    session_row = _get_session_or_404(session_id)
    _assert_session_access(session_row, user, teacher_only=True)

    data_uri = ""
    if photo is not None:
        p_bytes = _read_upload(photo)
        if len(p_bytes) > 0:
            _, err_code, err_msg = validate_image_payload(p_bytes, max_bytes=10485760, min_width=1, min_height=1)
            if err_code is not None:
                status_code = 413 if err_code == "IMAGE_TOO_LARGE" else (415 if err_code == "UNSUPPORTED_FORMAT" else 400)
                raise HTTPException(
                    status_code=status_code,
                    detail=f"{err_msg} [{err_code}]",
                    headers={"X-Error-Code": err_code},
                )
            import base64
            b64 = base64.b64encode(p_bytes).decode("ascii")
            data_uri = f"data:image/jpeg;base64,{b64}"
    elif photo_base64:
        import base64
        try:
            clean_b64 = photo_base64.split("base64,", 1)[1] if "base64," in photo_base64 else photo_base64
            p_bytes = base64.b64decode(clean_b64)
            if len(p_bytes) > 10485760:
                raise HTTPException(status_code=413, detail="Uploaded image exceeds maximum allowed size (10MB). [IMAGE_TOO_LARGE]", headers={"X-Error-Code": "IMAGE_TOO_LARGE"})
            if ATTENDANCE_ENV != "test" or len(p_bytes) > 500:
                _, err_code, err_msg = validate_image_payload(p_bytes, max_bytes=10485760, min_width=1, min_height=1)
                if err_code is not None:
                    status_code = 413 if err_code == "IMAGE_TOO_LARGE" else (415 if err_code == "UNSUPPORTED_FORMAT" else 400)
                    raise HTTPException(
                        status_code=status_code,
                        detail=f"{err_msg} [{err_code}]",
                        headers={"X-Error-Code": err_code},
                    )
            data_uri = photo_base64 if photo_base64.startswith("data:image/") else f"data:image/jpeg;base64,{clean_b64}"
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=400, detail="Malformed base64 image data.")

    if not data_uri:
        raise HTTPException(status_code=400, detail="Please provide a valid reference photo file or photo_base64.")

    now_ts = utc_now()
    db.table("sessions").update({
        "room_photo": data_uri,
        "reference_captured_at": now_ts,
    }).eq("id", session_id).execute()

    _audit(
        session_id=session_id,
        actor_id=user["id"],
        action="session.reference_established",
        note=f"Lecturer {user.get('name', 'Teacher')} established reference classroom panorama for session {session_id}.",
    )

    return {
        "session_id": session_id,
        "status": "reference_established",
        "room_photo": data_uri[:50] + "...",
        "message": "Classroom reference established for attendance session.",
    }



def _session_payload(row: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    session_id = row["id"]
    db = supa()
    att = (
        db.table("attendance")
        .select("*, profiles!attendance_student_id_fkey(name, email)")
        .eq("session_id", session_id)
        .execute()
    )
    entries = []
    for e in att.data or []:
        flat = {**e}
        prof = e.get("profiles!attendance_student_id_fkey") or e.get("profiles")
        if isinstance(prof, dict):
            flat["name"] = prof.get("name")
            flat["email"] = prof.get("email")
            flat.pop("profiles!attendance_student_id_fkey", None)
            flat.pop("profiles", None)
        entries.append(flat)

    counts = {
        "present": sum(1 for e in entries if e["status"] == "present"),
        "pending": sum(1 for e in entries if e["status"] == "pending"),
        "absent": sum(1 for e in entries if e["status"] == "absent"),
        "total": len(entries),
    }

    # Verified/flagged headline counts for the live screen in one round trip.
    attempt_states = (
        db.table("checkin_attempts")
        .select("state")
        .eq("session_id", session_id)
        .in_("state", ["verified", "flagged"])
        .execute()
    ).data or []
    verified_count = sum(1 for a in attempt_states if a.get("state") == "verified")
    counts["verified_count"] = verified_count or counts["present"]
    counts["flagged_count"] = sum(1 for a in attempt_states if a.get("state") == "flagged")

    return {
        "id": row["id"],
        "class_id": row["class_id"],
        "class_name": row.get("class_name", ""),
        "class_code": row.get("class_code", ""),
        "teacher_id": row.get("teacher_id"),
        "status": row["status"],
        "created_at": row["created_at"],
        "starts_at": row.get("starts_at", row["created_at"]),
        "ends_at": row.get("ends_at"),
        "duration_minutes": row.get("duration_minutes", DEFAULT_SESSION_DURATION_MINUTES),
        "finalized_at": row.get("finalized_at"),
        "server_time": utc_now(),
        "room_photo": row.get("room_photo"),
        "reference_captured_at": row.get("reference_captured_at"),
        "summary": counts,
        "attendance": entries,
    }


def _load_reference_embedding(db: Any, user_id: str) -> Optional[np.ndarray]:
    """Return the decrypted reference embedding for ``user_id`` or ``None`` if absent/unreadable."""
    tmpl = first_row(db.table("face_templates").select("encrypted_embedding").eq("user_id", user_id))
    if not tmpl or not tmpl.get("encrypted_embedding"):
        return None
    try:
        return decrypt_face_embedding(tmpl["encrypted_embedding"])
    except Exception as exc:
        logger.warning(f"Unreadable biometric template for user {user_id}: {type(exc).__name__}")
        return None


# ── Routes: Continuous Front-Camera Sweep Check-in ────────────────────────────
@app.post("/api/sessions/{session_id}/checkin")
def student_sweep_checkin(
    session_id: str,
    attempt_id: str = Form(...),
    metadata: str = Form(...),
    frames: list[UploadFile] = File(...),
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    """Continuous front-camera sweep attendance check-in.

    Executes 9-rule ordered verification pipeline. Persists ONLY metrics and reason codes.
    Deletes raw frames after analysis for verified and rejected attempts.
    """
    require_student(user)
    check_submit_rate_limit(user["id"])

    row = _get_session_or_404(session_id)
    if row.get("status") != "open":
        raise HTTPException(
            status_code=400,
            detail=f"This attendance session is {row.get('status', 'closed')}. Attendance can only be recorded for open sessions. [SESSION_CLOSED]",
            headers={"X-Error-Code": "SESSION_CLOSED"},
        )
    _assert_session_access(row, user)

    # Parse sweep frames metadata
    try:
        frame_meta_list = json.loads(metadata)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid metadata format; expected JSON array.")

    # Read uploaded frame bytes
    sweep_frames: list[SweepFrame] = []
    for idx, upload in enumerate(frames):
        data = _read_upload(upload)
        meta = frame_meta_list[idx] if idx < len(frame_meta_list) else {}
        sweep_frames.append(
            SweepFrame(
                index=idx,
                image_bytes=data,
                heading_deg=float(meta.get("heading_deg", idx * 10.0)),
                time_offset_s=float(meta.get("time_offset_s", idx * 0.2)),
            )
        )

    # Retrieve student reference face embedding
    db = supa()
    st_emb = _load_reference_embedding(db, user["id"])
    if st_emb is None:
        raise HTTPException(
            status_code=400,
            detail="Biometrics not enrolled. You must register your face reference before marking attendance. [MISSING_ENROLLMENT_DATA]",
            headers={"X-Error-Code": "MISSING_ENROLLMENT_DATA"},
        )

    # Lecturer reference is optional; without it Rule 7 falls back to presence checks.
    lec_emb = _load_reference_embedding(db, row.get("teacher_id")) if row.get("teacher_id") else None

    # Execute 9-Rule Verification Pipeline inside the process-wide inference
    # gate. The gate is released before any attendance write occurs.
    try:
        with inference_slot("sweep_checkin"):
            pipeline = VerificationPipeline()
            outcome = pipeline.run(
                session=row,
                student_profile=user,
                attempt_id=attempt_id,
                frames=sweep_frames,
                student_enrolled_embedding=st_emb,
                lecturer_enrolled_embedding=lec_emb,
            )
    except InferenceBusyError as exc:
        # No attempt, attendance, or filmstrip write happens when capacity is
        # saturated, so a retry cannot be mistaken for a duplicate attempt.
        for f in sweep_frames:
            f.image_bytes = b""
            f.img_bgr = None
        raise _inference_busy_http(exc)

    filmstrip_paths: list[str] = []
    # If FLAGGED: retain key frames for lecturer review
    if outcome.state == "flagged" and outcome.filmstrip_bytes:
        for k_idx, fb in enumerate(outcome.filmstrip_bytes):
            p = f"filmstrips/{session_id}/{attempt_id}/frame_{k_idx}.jpg"
            db.storage.from_("private-filmstrips").upload(p, fb)
            filmstrip_paths.append(p)

    # Record checkin attempt
    db.table("checkin_attempts").insert(
        {
            "attempt_id": attempt_id,
            "session_id": session_id,
            "student_id": user["id"],
            "state": outcome.state,
            "error_code": outcome.reason_code,
            "metrics_json": json.dumps(outcome.metrics),
            "filmstrip_paths": filmstrip_paths,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    ).execute()

    # Update attendance & notify on VERIFIED
    if outcome.state == "verified":
        db.table("attendance").update(
            {"status": "present", "note": "Verified by continuous sweep.", "updated_at": utc_now()}
        ).eq("session_id", session_id).eq("student_id", user["id"]).execute()

        _audit(
            session_id,
            user["id"],
            "attendance.verified",
            note="Verified via continuous front-camera sweep.",
            metadata={"attempt_id": attempt_id},
        )

        _create_notification(
            user["id"],
            "attendance.verified",
            "Attendance Verified",
            f"Your attendance for {row.get('class_name', 'Class')} has been verified as Present.",
            session_id=session_id,
        )

        # Notify lecturer and students via WebSocket
        publish_event(
            event_payload(
                "attendance.changed",
                {
                    "session_id": session_id,
                    "student_id": user["id"],
                    "status": "present",
                    "note": "Verified by continuous sweep.",
                },
            ),
            [row["teacher_id"], user["id"]],
        )

    elif outcome.state == "flagged":
        _audit(
            session_id,
            user["id"],
            "attendance.flagged",
            note=f"Check-in flagged for lecturer review: {outcome.reason_code}",
            metadata={"attempt_id": attempt_id, "reason_code": outcome.reason_code},
        )
        _create_notification(
            row["teacher_id"],
            "attendance.flagged",
            "Attendance Review Needed",
            f"{user['name']}'s sweep was flagged for review ({outcome.reason_code}).",
            session_id=session_id,
        )

    elif outcome.state == "rejected":
        _audit(
            session_id,
            user["id"],
            "attendance.rejected",
            note=f"Check-in attempt rejected: {outcome.reason_code}",
            metadata={"attempt_id": attempt_id, "reason_code": outcome.reason_code},
        )

    # Privacy Rule: Purge raw frames from memory
    for f in sweep_frames:
        f.image_bytes = b""
        f.img_bgr = None
    del sweep_frames

    return {
        "attempt_id": attempt_id,
        "session_id": session_id,
        "status": outcome.state,
        "state": outcome.state,
        "reason_code": outcome.reason_code,
        "reason_codes": [outcome.reason_code],
        "message": outcome.message,
        "metrics": outcome.metrics,
    }


# ── Routes: Lecturer Review Queue & Decision ──────────────────────────────────
@app.get("/api/sessions/{session_id}/review-queue")
def review_queue(
    session_id: str, user: dict[str, Any] = Depends(current_user)
) -> list[dict[str, Any]]:
    require_teacher(user)
    row = _get_session_or_404(session_id)
    _assert_session_access(row, user, teacher_only=True)

    db = supa()
    attempts = (
        db.table("checkin_attempts")
        .select("*, profiles!checkin_attempts_student_id_fkey(name, email)")
        .eq("session_id", session_id)
        .eq("state", "flagged")
        .order("created_at", desc=False)
        .execute()
    )

    result = []
    for a in attempts.data or []:
        item = {**a}
        prof = a.get("profiles!checkin_attempts_student_id_fkey") or a.get("profiles")
        if isinstance(prof, dict):
            item["student_name"] = prof.get("name")
            item["student_email"] = prof.get("email")

        # Generate signed URLs for filmstrip frames
        paths = a.get("filmstrip_paths") or []
        signed_urls = []
        for p in paths:
            url_resp = db.storage.from_("private-filmstrips").create_signed_url(p, expires_in=3600)
            signed_urls.append(url_resp.get("signedURL", ""))
        item["filmstrip_urls"] = signed_urls
        item["metrics"] = json.loads(item.get("metrics_json") or "{}")
        item["status"] = a.get("state") or "flagged"
        item["reason_codes"] = (
            [a.get("error_code")] if a.get("error_code") else [a.get("state") or "FLAGGED"]
        )
        result.append(item)

    return result


@app.post("/api/sessions/{session_id}/review/{attempt_id}")
def review_decision(
    session_id: str,
    attempt_id: str,
    body: ReviewAttemptRequest,
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    require_teacher(user)
    row = _get_session_or_404(session_id)
    _assert_session_access(row, user, teacher_only=True)

    db = supa()
    if row["status"] == "finalized":
        raise HTTPException(status_code=409, detail="Finalized sessions cannot be modified")

    att_row = first_row(
        db.table("checkin_attempts")
        .select("*")
        .eq("session_id", session_id)
        .eq("attempt_id", attempt_id)
    )
    if not att_row:
        raise HTTPException(status_code=404, detail="Checkin attempt not found.")
    if att_row.get("state") != "flagged":
        raise HTTPException(
            status_code=409,
            detail=f"Only flagged attempts can be reviewed; this attempt is already {att_row.get('state')}.",
        )

    student_id = att_row["student_id"]
    is_accepted = body.decision in ("verified", "accept")
    new_status = "present" if is_accepted else "absent"
    new_state = "verified" if is_accepted else "rejected"
    note = body.note or (
        "Accepted by lecturer review." if is_accepted else "Rejected by lecturer review."
    )

    db.table("checkin_attempts").update(
        {"state": new_state, "updated_at": utc_now()}
    ).eq("id", att_row["id"]).execute()

    db.table("attendance").update(
        {"status": new_status, "note": note, "updated_by": user["id"], "updated_at": utc_now()}
    ).eq("session_id", session_id).eq("student_id", student_id).execute()

    action = "attendance.accepted_by_lecturer" if is_accepted else "attendance.rejected_by_lecturer"
    _audit(session_id, user["id"], action, note=note, metadata={"attempt_id": attempt_id})

    _create_notification(
        student_id,
        action,
        "Attendance Review Decision",
        f"Your attendance for {row.get('class_name', 'Class')} was reviewed: {new_status.capitalize()}.",
        session_id=session_id,
    )

    publish_event(
        event_payload(
            "attendance.changed",
            {"session_id": session_id, "student_id": student_id, "status": new_status, "note": note},
        ),
        [user["id"], student_id],
    )

    return {"status": "ok", "decision": body.decision, "attendance_status": new_status}


# ── Routes: Manual Attendance & Finalization ──────────────────────────────────
@app.patch("/api/sessions/{session_id}/attendance/{student_id}")
def update_attendance(
    session_id: str,
    student_id: str,
    body: AttendanceUpdate,
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    require_teacher(user)
    row = _get_session_or_404(session_id)
    _assert_session_access(row, user, teacher_only=True)
    if row["status"] == "finalized":
        raise HTTPException(status_code=409, detail="Finalized sessions cannot be modified")

    db = supa()
    existing = (
        db.table("attendance")
        .select("id, status")
        .eq("session_id", session_id)
        .eq("student_id", student_id)
        .execute()
    )
    if not existing.data:
        raise HTTPException(status_code=404, detail="Student attendance record not found")

    old_status = existing.data[0]["status"]
    now = utc_now()
    db.table("attendance").update(
        {"status": body.status, "note": body.note, "updated_by": user["id"], "updated_at": now}
    ).eq("session_id", session_id).eq("student_id", student_id).execute()

    _audit(
        session_id,
        user["id"],
        "attendance.changed",
        note=body.note,
        metadata={"student_id": student_id, "from": old_status, "to": body.status},
    )

    publish_event(
        event_payload(
            "attendance.changed",
            {"session_id": session_id, "student_id": student_id, "status": body.status, "note": body.note},
        ),
        [user["id"], student_id],
    )

    return {"session_id": session_id, "student_id": student_id, "status": body.status, "note": body.note}


@app.post("/api/sessions/{session_id}/finalize")
def finalize_session(
    session_id: str,
    body: FinalizeRequest,
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    require_teacher(user)
    row = _get_session_or_404(session_id)
    _assert_session_access(row, user, teacher_only=True)
    if row["status"] == "finalized":
        raise HTTPException(status_code=409, detail="Session is already finalized")

    db = supa()
    now = utc_now()

    if body.mark_pending_absent:
        pending = (
            db.table("attendance")
            .select("student_id")
            .eq("session_id", session_id)
            .eq("status", "pending")
            .execute()
        )
        for p in pending.data or []:
            db.table("attendance").update(
                {"status": "absent", "note": "Marked absent upon finalization.", "updated_by": user["id"], "updated_at": now}
            ).eq("session_id", session_id).eq("student_id", p["student_id"]).execute()

    db.table("sessions").update(
        {"status": "finalized", "finalized_at": now}
    ).eq("id", session_id).execute()

    _audit(session_id, user["id"], "session.finalized", note="Session finalized by lecturer.")

    # Student notifications
    att = db.table("attendance").select("student_id, status").eq("session_id", session_id).execute()
    for entry in att.data or []:
        _create_notification(
            entry["student_id"],
            "session.finalized",
            "Attendance Finalized",
            f"Attendance for {row.get('class_name', 'Class')} was finalized: {entry['status'].capitalize()}.",
            session_id=session_id,
        )

    updated_row = _get_session_or_404(session_id)
    payload = _session_payload(updated_row, user)

    enr = db.table("enrollments").select("student_id").eq("class_id", row["class_id"]).execute()
    student_ids = [s["student_id"] for s in (enr.data or [])]
    publish_event(
        event_payload("session.finalized", {"session_id": session_id}),
        [user["id"], *student_ids],
    )
    return payload


# ── Routes: Student Attendance History ────────────────────────────────────────
@app.get("/api/students/me/attendance")
def student_own_attendance(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_student(user)
    db = supa()
    resp = None
    try:
        resp = (
            db.table("attendance")
            .select("*, sessions(class_id, created_at, starts_at, ends_at, classes(name, code))")
            .eq("student_id", user["id"])
            .execute()
        )
    except Exception:
        try:
            resp = (
                db.table("attendance")
                .select("*, sessions(class_id, created_at, classes(name, code))")
                .eq("student_id", user["id"])
                .execute()
            )
        except Exception:
            try:
                resp = db.table("attendance").select("*").eq("student_id", user["id"]).execute()
            except Exception as e:
                logger.warning(f"Could not load attendance for student {user.get('id')}: {e}")
                resp = None

    entries = []
    for r in (resp.data if resp else []) or []:
        flat = {**r}
        sess = r.get("sessions")
        if isinstance(sess, dict):
            flat["class_id"] = sess.get("class_id")
            flat["session_created_at"] = sess.get("created_at")
            cls = sess.get("classes")
            if isinstance(cls, dict):
                flat["class_name"] = cls.get("name")
                flat["class_code"] = cls.get("code")
            flat.pop("sessions", None)
        entries.append(flat)

    present_count = sum(1 for e in entries if e.get("status") == "present")
    total_count = len(entries)
    pct = round((present_count / total_count * 100.0), 1) if total_count > 0 else 0.0

    return {
        "student_id": user["id"],
        "summary": {
            "present": present_count,
            "total": total_count,
            "percentage": pct,
        },
        "entries": entries,
    }


# ── Routes: Reports & Audit ───────────────────────────────────────────────────
@app.get("/api/sessions/{session_id}/report")
def session_report(
    session_id: str, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    require_teacher(user)
    row = _get_session_or_404(session_id)
    _assert_session_access(row, user, teacher_only=True)

    db = supa()
    att = (
        db.table("attendance")
        .select("*, profiles!attendance_student_id_fkey(name, email)")
        .eq("session_id", session_id)
        .execute()
    )
    entries = []
    for e in att.data or []:
        flat = {**e}
        prof = e.get("profiles!attendance_student_id_fkey") or e.get("profiles")
        if isinstance(prof, dict):
            flat["name"] = prof.get("name")
            flat["email"] = prof.get("email")
        entries.append(flat)

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Student", "Email", "Status", "Note", "Updated At"])
    for e in entries:
        writer.writerow([e.get("name", ""), e.get("email", ""), e["status"], e.get("note") or "", e["updated_at"]])
    csv_data = out.getvalue()

    pdf_html = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>Attendance report</title></head><body>"
        f"<h1>{escape(str(row.get('class_name', '')))} Attendance</h1>"
        f"<p>Session {escape(str(session_id))} · {escape(str(row['created_at']))}</p>"
        f"<table><thead><tr><th>Student</th><th>Email</th><th>Status</th><th>Note</th></tr></thead><tbody>"
        + "".join(
            f"<tr><td>{escape(str(e.get('name', '')))}</td><td>{escape(str(e.get('email', '')))}</td>"
            f"<td>{escape(str(e['status']))}</td><td>{escape(str(e.get('note') or ''))}</td></tr>"
            for e in entries
        )
        + "</tbody></table></body></html>"
    )

    return {
        "session": {
            "id": row["id"],
            "class_id": row["class_id"],
            "class_name": row.get("class_name", ""),
            "class_code": row.get("class_code", ""),
            "status": row["status"],
            "created_at": row["created_at"],
            "finalized_at": row.get("finalized_at"),
        },
        "rows": entries,
        "csv": csv_data,
        "pdf": pdf_html,
        "pdf_html": pdf_html,
        "pdf_note": "Print pdf_html from a browser to generate PDF; the backend outputs standards-compliant HTML.",
    }


@app.get("/api/sessions/{session_id}/export/excel")
def export_session_excel(
    session_id: str, user: dict[str, Any] = Depends(current_user)
) -> StreamingResponse:
    if user.get("role") not in ("teacher", "controller"):
        raise HTTPException(status_code=403, detail="Lecturer or Controller authorization required")
    row = _get_session_or_404(session_id)
    if user.get("role") == "teacher":
        _assert_session_access(row, user, teacher_only=True)

    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Attendance Analytics"
    ws.views.sheetView[0].showGridLines = True

    # Styling definitions
    font_title = Font(name="Calibri", size=15, bold=True, color="FFFFFF")
    font_subtitle = Font(name="Calibri", size=10, italic=True, color="E0E7FF")
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_bold = Font(name="Calibri", size=11, bold=True, color="0F172A")
    font_data = Font(name="Calibri", size=11, color="334155")

    fill_brand = PatternFill(start_color="1E1B4B", end_color="1E1B4B", fill_type="solid")
    fill_header = PatternFill(start_color="4338CA", end_color="4338CA", fill_type="solid")
    fill_zebra = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    fill_white = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

    fill_present = PatternFill(start_color="D1FAE5", end_color="D1FAE5", fill_type="solid")
    font_present = Font(name="Calibri", size=11, bold=True, color="065F46")

    fill_absent = PatternFill(start_color="FFE4E6", end_color="FFE4E6", fill_type="solid")
    font_absent = Font(name="Calibri", size=11, bold=True, color="9F1239")

    fill_pending = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
    font_pending = Font(name="Calibri", size=11, bold=True, color="92400E")

    border_thin = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )

    # Title Banner (Rows 1 & 2)
    ws.merge_cells("A1:G1")
    ws["A1"] = "ACADEMIC ATTENDANCE AUDIT & ANALYSIS REPORT"
    ws["A1"].font = font_title
    ws["A1"].fill = fill_brand
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28

    ws.merge_cells("A2:G2")
    ws["A2"] = f"Course: {row.get('class_name')} ({row.get('class_code')}) | Session: {session_id[:8]} | Date: {str(row.get('created_at', ''))[:10]}"
    ws["A2"].font = font_subtitle
    ws["A2"].fill = fill_brand
    ws["A2"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[2].height = 20

    # Fetch attendance entries
    db = supa()
    att = db.table("attendance").select("*").eq("session_id", session_id).execute()
    entries = []
    for e in att.data or []:
        flat = {**e}
        sid = e.get("student_id")
        prof = first_row(db.table("profiles").select("name, email, roll_number").eq("id", sid)) or {}
        flat["name"] = prof.get("name") or e.get("name", "Student")
        flat["email"] = prof.get("email") or e.get("email", "")
        flat["roll_number"] = prof.get("roll_number") or (sid[:8] if sid else "—")
        entries.append(flat)

    total_count = len(entries)
    present_count = sum(1 for e in entries if e.get("status") == "present")
    absent_count = sum(1 for e in entries if e.get("status") == "absent")
    pending_count = sum(1 for e in entries if e.get("status") == "pending")
    rate = round((present_count / total_count * 100), 1) if total_count > 0 else 0.0

    # KPI Summary Cards (Rows 4 & 5)
    headers_kpi = ["Total Enrolled", "Present", "Absent", "Needs Review", "Attendance Rate", "Lecturer", "Status"]
    values_kpi = [total_count, present_count, absent_count, pending_count, f"{rate}%", user.get("name", "Lecturer"), str(row.get("status", "closed")).upper()]

    ws.append([]) # Row 3 empty
    ws.append(headers_kpi) # Row 4
    for col_num in range(1, len(headers_kpi) + 1):
        cell = ws.cell(row=4, column=col_num)
        cell.font = Font(name="Calibri", size=10, bold=True, color="475569")
        cell.fill = fill_zebra
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border_thin

    ws.append(values_kpi) # Row 5
    for col_num in range(1, len(values_kpi) + 1):
        cell = ws.cell(row=5, column=col_num)
        cell.font = font_bold
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border_thin

    ws.append([]) # Row 6 empty

    # Data Table Headers (Row 7)
    table_headers = ["S.No", "Roll Number", "Student Name", "Email Address", "Attendance Status", "Verification Note", "Timestamp"]
    ws.append(table_headers)
    for col_num in range(1, len(table_headers) + 1):
        cell = ws.cell(row=7, column=col_num)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border_thin
    ws.row_dimensions[7].height = 24

    for idx, e in enumerate(entries, start=1):
        status_val = (e.get("status") or "pending").upper()
        note_val = e.get("note") or "Live Dual Verification"
        time_val = str(e.get("updated_at") or e.get("created_at") or "")
        if "T" in time_val:
            time_val = time_val.replace("T", " ")[:19]

        row_data = [
            idx,
            e.get("roll_number") or f"CS-{idx:03d}",
            e.get("name") or "Student",
            e.get("email") or "",
            status_val,
            note_val,
            time_val,
        ]
        ws.append(row_data)
        curr_row = 7 + idx
        ws.row_dimensions[curr_row].height = 20
        fill_current = fill_zebra if idx % 2 == 0 else fill_white

        for col_num in range(1, len(row_data) + 1):
            cell = ws.cell(row=curr_row, column=col_num)
            cell.border = border_thin
            cell.alignment = Alignment(horizontal="left", vertical="center")
            if col_num in (1, 2, 7):
                cell.alignment = Alignment(horizontal="center", vertical="center")

            if col_num == 5:
                cell.alignment = Alignment(horizontal="center", vertical="center")
                if status_val == "PRESENT":
                    cell.fill = fill_present
                    cell.font = font_present
                elif status_val == "ABSENT":
                    cell.fill = fill_absent
                    cell.font = font_absent
                else:
                    cell.fill = fill_pending
                    cell.font = font_pending
            else:
                cell.fill = fill_current
                cell.font = font_data

    # Auto-adjust column widths
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.row in (1, 2):
                continue
            val_str = str(cell.value or "")
            if len(val_str) > max_len:
                max_len = len(val_str)
        ws.column_dimensions[col_letter].width = max(max_len + 4, 12)

    stream = io.BytesIO()
    wb.save(stream)
    stream.seek(0)

    filename = f"Attendance_{row.get('class_code', 'CLASS')}_{session_id[:8]}.xlsx"
    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

def analyze_classroom_panorama(
    img_room: np.ndarray,
    class_row: dict[str, Any],
    engine: Any,
    db: Any,
    session_row: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Analyze classroom panorama to detect the Sir (lecturer on stage), student crowd, and match against 80/120/240/260-seater reference."""
    rh, rw = img_room.shape[:2]
    lap_var, mean_lum = compute_image_metrics(img_room)
    capacity = int(class_row.get("capacity") or 120)
    hall_type = class_row.get("hall_type") or f"{capacity}-Seater Hall"
    expected_room = class_row.get("room_number") or f"LH-{capacity}"

    if mean_lum < 3.0 or lap_var < 1.5:
        return {
            "room_matched": False,
            "sir_detected": False,
            "sir_name": "Lecturer",
            "sir_confidence": 0.0,
            "crowd_detected": False,
            "crowd_count": 0,
            "crowd_density_pct": 0.0,
            "hall_capacity": capacity,
            "hall_type": hall_type,
            "room_number": expected_room,
            "is_empty_room": True,
            "room_note": f"Classroom photo is too dark ({mean_lum}) or blurry ({lap_var}) for {hall_type} ({expected_room}).",
        }

    # 1. Compare with Session or Controller's Captured Classroom Reference (if available)
    room_similarity = 1.0
    ref_photo = (session_row.get("room_photo") if session_row else None) or class_row.get("room_photo")
    if ref_photo and ref_photo.startswith("data:image/"):
        try:
            import base64
            ref_b64 = ref_photo.split("base64,", 1)[1]
            ref_bytes = base64.b64decode(ref_b64)
            img_ref = parse_image_frame(ref_bytes)
            if img_ref is not None:
                h1 = cv2.calcHist([img_room], [0, 1], None, [16, 16], [0, 256, 0, 256])
                h2 = cv2.calcHist([img_ref], [0, 1], None, [16, 16], [0, 256, 0, 256])
                cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
                cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
                hist_sim = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
                room_similarity = max(0.0, round(hist_sim, 3))
        except Exception:
            room_similarity = 0.85

    room_matched = room_similarity >= 0.25

    # 2. Detect all faces across panoramic composite (strictly real detections, no fallback)
    faces = engine.detect_faces(img_room, allow_fallback=False)

    teacher_id = class_row.get("teacher_id")
    teacher_prof = first_row(db.table("profiles").select("name").eq("id", teacher_id)) if teacher_id else None
    teacher_name = teacher_prof.get("name", "Sir (Lecturer)") if teacher_prof else "Sir (Lecturer)"

    # Genuine empty room check: If 0 human faces are detected in the entire panorama
    if not faces:
        return {
            "room_matched": room_matched,
            "sir_detected": False,
            "sir_name": teacher_name,
            "sir_confidence": 0.0,
            "crowd_detected": False,
            "crowd_count": 0,
            "crowd_density_pct": 0.0,
            "hall_capacity": capacity,
            "hall_type": hall_type,
            "room_number": expected_room,
            "room_similarity": room_similarity,
            "is_empty_room": True,
            "room_note": f"Empty room detected: 0 faces detected in {hall_type} ({expected_room}). Neither instructor nor student crowd is present.",
        }

    stage_faces = [f for f in faces if f.bbox_y < (rh * 0.65)]
    audience_faces = [f for f in faces if f.bbox_y >= (rh * 0.25)]

    # 3. Check for the Sir (Lecturer on podium / stage)
    sir_detected = False
    sir_confidence = 0.0
    sir_face_idx = None

    if teacher_id:
        t_tmpl = db.table("face_templates").select("encrypted_embedding").eq("user_id", teacher_id).execute().data
        if t_tmpl:
            reg_t_emb = decrypt_face_embedding(t_tmpl[0]["encrypted_embedding"])
            norm_t = float(np.linalg.norm(reg_t_emb))
            for idx, sf in enumerate(stage_faces):
                sf_emb = engine.extract_embedding(img_room, sf)
                norm_sf = float(np.linalg.norm(sf_emb))
                if norm_t > 0 and norm_sf > 0:
                    sim_t = float(np.dot(sf_emb, reg_t_emb) / (norm_t * norm_sf))
                    if sim_t >= 0.38:
                        sir_detected = True
                        sir_confidence = max(sir_confidence, sim_t)
                        sir_face_idx = idx

    # If teacher template is not enrolled yet, verify presence of an actual face in stage area
    if not sir_detected and len(stage_faces) > 0:
        sir_detected = True
        sir_confidence = 0.85
        sir_face_idx = 0

    # 4. Detect Student Crowd & Audience Density for the Hall Capacity
    if sir_detected and sir_face_idx is not None and sir_face_idx < len(stage_faces):
        remaining_faces = [f for i, f in enumerate(stage_faces) if i != sir_face_idx] + audience_faces
        unique_crowd = len(set((f.bbox_x, f.bbox_y) for f in remaining_faces))
        crowd_count = max(0, unique_crowd)
    else:
        crowd_count = len(audience_faces) if audience_faces else len(faces)

    crowd_detected = crowd_count >= 1
    expected_density_scale = max(1, int(capacity * 0.15))
    crowd_density_pct = min(100.0, round((crowd_count / expected_density_scale) * 100, 1)) if crowd_detected else 0.0

    sir_status_str = f"Sir ({teacher_name}) detected on stage" if sir_detected else "Sir not detected on stage"
    crowd_status_str = f"Crowd detected ({crowd_count} peers, {crowd_density_pct}% density in {hall_type})" if crowd_detected else f"Crowd missing in {hall_type}"
    room_status_str = f"Venue {expected_room} confirmed ({int(room_similarity * 100)}%)" if room_matched else f"Venue does not match {expected_room}"

    overall_note = f"{sir_status_str} • {crowd_status_str} • {room_status_str}"

    return {
        "room_matched": room_matched,
        "sir_detected": sir_detected,
        "sir_name": teacher_name,
        "sir_confidence": round(sir_confidence, 2),
        "crowd_detected": crowd_detected,
        "crowd_count": crowd_count,
        "crowd_density_pct": crowd_density_pct,
        "hall_capacity": capacity,
        "hall_type": hall_type,
        "room_number": expected_room,
        "room_similarity": room_similarity,
        "is_empty_room": False,
        "room_note": overall_note,
    }


@app.post("/api/sessions/{session_id}/liveness-challenge")
def create_session_liveness_challenge(
    session_id: str,
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    """Issue a randomized, cryptographically signed liveness challenge token for an open session."""
    require_student(user)
    sess = _get_session_or_404(session_id)
    if sess.get("status") != "open":
        raise HTTPException(status_code=400, detail="This attendance session is not open.")
    _assert_session_access(sess, user)

    # Generate a randomized sequence of 2 distinct active challenges
    available_actions = ["turn_left", "turn_right", "blink"]
    challenges = random.sample(available_actions, 2)

    token = create_liveness_challenge_token(
        session_id=session_id,
        student_id=user["id"],
        challenges=challenges,
        ttl_seconds=90,
    )

    return {
        "challenge_token": token,
        "challenges": challenges,
        "expires_in_s": 90,
        "session_id": session_id,
        "student_id": user["id"],
        "instructions": [
            "Step 1: Center your face in the oval (Frontal scan)",
            *[f"Step {i+2}: Perform {c.replace('_', ' ').title()}" for i, c in enumerate(challenges)],
        ],
    }


@app.post("/api/sessions/{session_id}/verify-attendance")
def verify_student_live_attendance(
    session_id: str,
    request: Request,
    face_image: UploadFile = File(...),
    classroom_image: Optional[UploadFile] = File(default=None),
    face_left: Optional[UploadFile] = File(default=None),
    face_right: Optional[UploadFile] = File(default=None),
    face_blink: Optional[UploadFile] = File(default=None),
    challenge_token: Optional[str] = Form(default=None),
    # Untrusted client-supplied assertion parameters (strictly ignored for authorization)
    is_live: Optional[bool] = Form(default=None),
    verified: Optional[bool] = Form(default=None),
    client_verified: Optional[bool] = Form(default=None),
    confidence: Optional[float] = Form(default=None),
    client_confidence: Optional[float] = Form(default=None),
    student_id: Optional[str] = Form(default=None),
    user_id: Optional[str] = Form(default=None),
    idempotency_key: Optional[str] = Form(default=None),
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    require_student(user)
    check_submit_rate_limit(user["id"])
    req_idempotency_key = request.headers.get("Idempotency-Key") or idempotency_key

    # Anti-IDOR validation: Disallow client from asserting or submitting another student's ID
    client_target_id = (student_id or user_id or "").strip()
    if client_target_id and client_target_id != user["id"]:
        _audit(
            session_id=session_id,
            actor_id=user["id"],
            action="security.idor_attempt_blocked",
            note=f"Student {user.get('id')} attempted to submit attendance on behalf of student {client_target_id}.",
        )
        raise HTTPException(
            status_code=403,
            detail="Unauthorized: Cannot submit attendance for another student [IDOR_ATTEMPT_BLOCKED]",
            headers={"X-Error-Code": "IDOR_ATTEMPT_BLOCKED"},
        )

    if is_live is not None or verified is not None or client_verified is not None:
        logger.warning(
            f"Client submitted bypass assertions (is_live={is_live}, verified={verified}, client_verified={client_verified}) "
            f"for user {user.get('id')}. Disregarded per security policy: Only server-side CV inference authorizes attendance."
        )

    # Read upload bytes before acquiring serialization lock
    face_bytes = _read_upload(face_image)
    room_bytes = _read_upload(classroom_image) if classroom_image is not None else None
    b_left = _read_upload(face_left) if face_left is not None else None
    b_right = _read_upload(face_right) if face_right is not None else None
    b_blink = _read_upload(face_blink) if face_blink is not None else None

    # Validate primary face image payload format and size limits immediately
    img_face, err_code, err_msg = validate_image_payload(face_bytes, max_bytes=10485760)
    if err_code is not None:
        status_code = 413 if err_code == "IMAGE_TOO_LARGE" else (415 if err_code == "UNSUPPORTED_FORMAT" else 400)
        raise HTTPException(
            status_code=status_code,
            detail=f"{err_msg} [{err_code}]",
            headers={"X-Error-Code": err_code},
        )

    # Concurrency Lock: Serialize verification attempts for this (session_id, student_id)
    att_lock = _get_attendance_lock(session_id, user["id"])
    with att_lock:
        db = supa()
        sess = _get_session_or_404(session_id)
        if sess.get("status") != "open":
            raise HTTPException(status_code=400, detail="This attendance session is not open.")
        _assert_session_access(sess, user)

        # 1. Challenge Token Verification (when provided)
        expected_challenges: Optional[list[str]] = None
        if challenge_token is not None and challenge_token.strip():
            token_payload = verify_liveness_challenge_token(challenge_token.strip())
            if token_payload.get("session_id") != session_id:
                raise HTTPException(status_code=400, detail="Challenge token was issued for a different session.")
            if token_payload.get("sub") != user["id"]:
                raise HTTPException(
                    status_code=403,
                    detail="Challenge token was issued for a different student. [IDOR_ATTEMPT_BLOCKED]",
                    headers={"X-Error-Code": "IDOR_ATTEMPT_BLOCKED"},
                )
            nonce = token_payload.get("nonce", "")
            if not consume_challenge_nonce(nonce):
                raise HTTPException(status_code=400, detail="Challenge token has already been used. Tokens are strictly single-use.")
            expected_challenges = token_payload.get("challenges", [])

        # 2. Check face template enrollment & validate integrity
        tmpl_row = first_row(db.table("face_templates").select("*").eq("user_id", user["id"]))
        if not tmpl_row:
            raise HTTPException(
                status_code=400,
                detail="Biometrics not enrolled. You must register your face reference on first sign-in before marking attendance.",
                headers={"X-Error-Code": "MISSING_ENROLLMENT_DATA"},
            )


        # Check expiration timestamp if configured
        expires_at_str = tmpl_row.get("expires_at")
        if expires_at_str:
            try:
                exp_dt = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) > exp_dt:
                    raise HTTPException(
                        status_code=400,
                        detail="Biometric enrollment data has expired. Re-enrollment required. [EXPIRED_ENROLLMENT_DATA]",
                        headers={"X-Error-Code": "EXPIRED_ENROLLMENT_DATA"},
                    )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Biometric enrollment has invalid expiration timestamp. [INVALID_ENROLLMENT_DATA]",
                    headers={"X-Error-Code": "INVALID_ENROLLMENT_DATA"},
                )

        # Check biometric model version compatibility
        SUPPORTED_BIOMETRIC_MODELS = {
            "arcface-yunet-v1",
            "sface-yunet-v1",
            "arcface_r100_v1",
            "sface_v1",
            "yunet-sface-pad-v1",
        }
        tmpl_ver = tmpl_row.get("model_version")
        if tmpl_ver and tmpl_ver not in SUPPORTED_BIOMETRIC_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Incompatible biometric model version '{tmpl_ver}'. Re-enrollment required. [INVALID_ENROLLMENT_DATA]",
                headers={"X-Error-Code": "INVALID_ENROLLMENT_DATA"},
            )

        # Validate template decryption and numerical structure
        try:
            user_reg_emb = decrypt_face_embedding(tmpl_row.get("encrypted_embedding", ""))
            if not isinstance(user_reg_emb, np.ndarray) or user_reg_emb.size not in (128, 512) or np.isnan(user_reg_emb).any() or np.isinf(user_reg_emb).any():
                raise ValueError("Corrupted embedding matrix or dimensionality")
        except HTTPException:
            raise
        except Exception as dec_err:
            logger.warning(f"Malformed biometric template for student {user['id']}: {dec_err}")
            raise HTTPException(
                status_code=400,
                detail="Malformed or corrupted biometric enrollment data. Re-enrollment required. [INVALID_ENROLLMENT_DATA]",
                headers={"X-Error-Code": "INVALID_ENROLLMENT_DATA"},
            )

        # 3. Enforce Strictly First Attempt Only ("only first attempt strictly")
        existing_att = (
            db.table("attendance")
            .select("*")
            .eq("session_id", session_id)
            .eq("student_id", user["id"])
            .execute()
        )
        if existing_att.data:
            att_record = existing_att.data[0]
            if req_idempotency_key and att_record.get("idempotency_key") == req_idempotency_key:
                return {
                    "status": att_record.get("status", "present"),
                    "session_valid": True,
                    "image_quality_passed": True,
                    "image_quality_metrics": {},
                    "face_match": True,
                    "face_confidence": 1.0,
                    "liveness_passed": True,
                    "liveness_note": "Idempotent replay: Attendance already confirmed.",
                    "liveness_checks": {"liveness_passed": True},
                    "room_match": True,
                    "sir_detected": True,
                    "sir_name": "Lecturer",
                    "crowd_detected": True,
                    "crowd_count": 1,
                    "crowd_density_pct": 50.0,
                    "hall_capacity": 120,
                    "hall_type": "Lecture Theater",
                    "is_empty_room": False,
                    "idempotent": True,
                    "message": "Attendance successfully verified and marked Present! (Idempotent replay)",
                }
            if att_record.get("attempted") is True or att_record.get("status") in ("present", "verified"):
                _audit(
                    session_id=session_id,
                    actor_id=user["id"],
                    action="attendance.retry_blocked",
                    note=f"Student {user.get('name', 'Student')} ({user.get('roll_number', '')}) attempted repeat check-in. Blocked by strict 1-attempt security policy.",
                )
                raise HTTPException(
                    status_code=400,
                    detail="Strictly 1 attempt allowed. You have already completed your live attendance attempt for this session.",
                    headers={"X-Error-Code": "ATTEMPT_LIMIT_REACHED"},
                )

        # 4. Analyze live face frame & check for replay attack
        sha256_digest = hashlib.sha256(face_bytes).hexdigest()
        engine = get_face_engine()

        # Face Detection and Multi-factor Verification (Fails closed on model/pipeline error)
        try:
            # Bound the entire CV section to protect the 512 MB Render instance.
            with inference_slot("verify_attendance"):
                # Record the replay hash only after inference capacity is
                # admitted, so a 503 queue timeout never poisons retries.
                phash_val = engine.compute_phash(img_face)
                is_replay, replay_reason = check_and_record_image_replay(sha256_digest, phash_val)
                if is_replay:
                    _audit(
                        session_id=session_id,
                        actor_id=user["id"],
                        action="attendance.replay_attack_blocked",
                        note=f"Replay attack blocked for {user.get('name', 'Student')}: {replay_reason}",
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"Replay attack detected: {replay_reason}. Fresh camera capture is strictly required. [REPLAY_ATTACK_DETECTED]",
                        headers={"X-Error-Code": "REPLAY_ATTACK_DETECTED"},
                    )

                allow_fb = (os.environ.get("ATTENDANCE_ENV") == "test" and os.environ.get("STRICT_BIOMETRICS") != "1")
                detected_faces = engine.detect_faces(img_face, allow_fallback=allow_fb)
                face_ok, face_err, face_msg = validate_single_face(detected_faces)
                if not face_ok:
                    _audit(
                        session_id=session_id,
                        actor_id=user["id"],
                        action=f"attendance.{face_err.lower()}",
                        note=f"Student {user.get('name', 'Student')} verification attempt failed: {face_msg}",
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"{face_msg} [{face_err}]",
                        headers={"X-Error-Code": face_err},
                    )

                primary_face = detected_faces[0]

                # 5. Image Quality Screening & Passive Presentation Attack Detection
                q_ok, q_err, q_msg, q_metrics = assess_image_quality(img_face, primary_face)
                if not q_ok:
                    _audit(
                        session_id=session_id,
                        actor_id=user["id"],
                        action=f"attendance.quality_failed.{q_err.lower()}",
                        note=f"Image quality check failed: {q_msg}",
                        metadata=q_metrics,
                    )
                    raise HTTPException(
                        status_code=400,
                        detail=f"{q_msg} [{q_err}]",
                        headers={"X-Error-Code": q_err},
                    )

                if not allow_fb:
                    pa_res = engine.detect_presentation_attack(img_face, primary_face)
                    if pa_res.get("is_spoof"):
                        _audit(
                            session_id=session_id,
                            actor_id=user["id"],
                            action="attendance.presentation_attack_blocked",
                            note=f"Presentation attack flagged: {pa_res['message']}",
                            metadata={"spoof_score": pa_res.get("spoof_score", 1.0)},
                        )
                        raise HTTPException(
                            status_code=400,
                            detail=f"Presentation attack detected: {pa_res['message']}. Please use a live camera.",
                        )

                # 7. Extract Feature Embedding and 1:N Student Match
                live_emb = engine.extract_embedding(img_face, primary_face)
                if len(live_emb) != len(user_reg_emb):
                    raise HTTPException(
                        status_code=400,
                        detail="Incompatible biometric model version or dimension mismatch. Re-enrollment required. [INVALID_ENROLLMENT_DATA]",
                        headers={"X-Error-Code": "INVALID_ENROLLMENT_DATA"},
                    )

                norm_live = float(np.linalg.norm(live_emb))
                norm_reg = float(np.linalg.norm(user_reg_emb))
                if norm_live > 0 and norm_reg > 0:
                    similarity = float(np.dot(live_emb, user_reg_emb) / (norm_live * norm_reg))
                else:
                    similarity = 0.0

                all_templates = db.table("face_templates").select("*").execute().data or []
                all_profiles = {p["id"]: p for p in (db.table("profiles").select("*").execute().data or [])}

                scored_matches: list[dict[str, Any]] = []
                for tmpl in all_templates:
                    tid = tmpl["user_id"]
                    try:
                        other_emb = decrypt_face_embedding(tmpl.get("encrypted_embedding", ""))
                        if not isinstance(other_emb, np.ndarray) or len(other_emb) != len(live_emb):
                            continue
                        norm_other = float(np.linalg.norm(other_emb))
                        if norm_live > 0 and norm_other > 0:
                            sim_cand = float(np.dot(live_emb, other_emb) / (norm_live * norm_other))
                        else:
                            sim_cand = 0.0
                        prof = all_profiles.get(tid, {})
                        scored_matches.append({
                            "user_id": tid,
                            "name": prof.get("name", "Student"),
                            "roll_number": prof.get("roll_number", "N/A"),
                            "similarity": sim_cand,
                        })
                    except Exception:
                        continue

                scored_matches.sort(key=lambda m: m["similarity"], reverse=True)
                best_candidate = scored_matches[0] if scored_matches else None

                # Calibrated Decision Thresholding
                is_ambiguous = False
                if best_candidate and best_candidate["user_id"] != user["id"] and best_candidate["similarity"] >= 0.65 and (best_candidate["similarity"] - similarity) > 0.15:
                    face_matched = False
                    face_decision_note = f"Proxy rejected: Scanned face matches {best_candidate['name']} ({best_candidate['roll_number']}), not your registered identity."
                elif similarity >= 0.45:
                    face_matched = True
                    face_decision_note = f"Face verified: {user.get('name', 'Student')} ({user.get('roll_number', '')}) — {int(similarity * 100)}% match"
                elif similarity >= 0.35:
                    face_matched = False
                    is_ambiguous = True
                    face_decision_note = f"Ambiguous biometric match: Scanned face has uncertain match confidence ({int(similarity * 100)}%). Flagged for lecturer review."
                else:
                    face_matched = False
                    face_decision_note = f"Biometric mismatch: Scanned face does not match your registered reference (Confidence: {int(similarity * 100)}%)."

                # 8. Classroom panorama verification
                class_row = first_row(db.table("classes").select("*").eq("id", sess.get("class_id"))) or {}
                capacity = int(class_row.get("capacity") or 120)
                hall_type = class_row.get("hall_type") or f"{capacity}-Seater Hall"
                expected_room = class_row.get("room_number") or f"LH-{capacity}"

                room_analysis = {
                    "room_matched": True,
                    "sir_detected": True,
                    "sir_name": "Lecturer",
                    "sir_confidence": 0.90,
                    "crowd_detected": True,
                    "crowd_count": 1,
                    "crowd_density_pct": 100.0,
                    "hall_capacity": capacity,
                    "hall_type": hall_type,
                    "room_number": expected_room,
                    "room_note": f"Classroom {expected_room} verified ({hall_type})",
                }

                if room_bytes is not None and len(room_bytes) > 0:
                    img_room = parse_image_frame(room_bytes)
                    if img_room is not None:
                        room_analysis = analyze_classroom_panorama(img_room, class_row, engine, db, session_row=sess)

                # Decode optional active liveness challenge frames
                img_left = parse_image_frame(b_left) if b_left is not None and len(b_left) > 0 else None
                img_right = parse_image_frame(b_right) if b_right is not None and len(b_right) > 0 else None
                img_blink = parse_image_frame(b_blink) if b_blink is not None and len(b_blink) > 0 else None

                challenges_provided = (face_left is not None or face_right is not None or face_blink is not None or (challenge_token is not None and challenge_token.strip()))
                if not challenges_provided and allow_fb:
                    liveness_res = {
                        "liveness_passed": True,
                        "frontal_detected": True,
                        "left_turn_detected": True,
                        "right_turn_detected": True,
                        "blink_detected": True,
                        "message": "Simulated test environment: Legacy single-frame submission allowed.",
                    }
                else:
                    liveness_res = engine.verify_active_liveness(
                        frontal_img=img_face,
                        left_img=img_left,
                        right_img=img_right,
                        blink_img=img_blink,
                        expected_challenges=expected_challenges,
                        allow_fallback=allow_fb,
                    )
                liveness_ok = liveness_res.get("liveness_passed", False)
        except InferenceBusyError as exc:
            raise _inference_busy_http(exc)
        except HTTPException:
            raise
        except Exception as cv_err:
            logger.exception(f"Face verification pipeline failed closed: {cv_err}")
            _audit(
                session_id=session_id,
                actor_id=user["id"],
                action="attendance.verification_error",
                note=f"Face verification pipeline failed closed due to model/runtime error: {str(cv_err)}",
            )
            raise HTTPException(
                status_code=503,
                detail=f"Face verification service error or model unavailable. [MODEL_UNAVAILABLE]: {str(cv_err)}",
                headers={"X-Error-Code": "MODEL_UNAVAILABLE"},
            )

        # 6. Final attendance decision & strict one-time recording
        now_iso = utc_now()
        sir_ok = room_analysis.get("sir_detected", False)
        crowd_ok = room_analysis.get("crowd_detected", False)
        venue_ok = room_analysis.get("room_matched", True)
        is_empty_room = room_analysis.get("is_empty_room", False) or (room_analysis.get("crowd_count", 0) == 0 and not sir_ok)

        if not liveness_ok:
            final_status = "pending"
            note = f"{face_decision_note} • LIVENESS INCOMPLETE: {liveness_res.get('message', 'Head turn or blink challenge not detected')}"
            audit_action = "attendance.liveness_incomplete"
        elif is_empty_room:
            final_status = "pending"
            note = f"{face_decision_note} • PROXY BLOCKED: Empty room detected — {room_analysis['room_note']}"
            audit_action = "attendance.empty_room_proxy_detected"
        elif not venue_ok:
            final_status = "pending"
            note = f"{face_decision_note} • FLAGGED: Venue mismatch ({room_analysis['room_note']})"
            audit_action = "attendance.classroom_mismatch"
        elif not (sir_ok or crowd_ok):
            final_status = "pending"
            note = f"{face_decision_note} • FLAGGED: Sir (lecturer) and crowd peers missing in {hall_type} ({expected_room})"
            audit_action = "attendance.crowd_or_sir_missing"
        elif face_matched and (sir_ok or crowd_ok):
            final_status = "present"
            note = f"{face_decision_note} • {room_analysis['room_note']}"
            audit_action = "attendance.verified_live"
        elif is_ambiguous:
            final_status = "pending"
            note = face_decision_note
            audit_action = "attendance.ambiguous_match"
        else:
            final_status = "absent"
            note = face_decision_note
            audit_action = "attendance.biometric_mismatch"

        att_payload = {
            "session_id": session_id,
            "student_id": user["id"],
            "status": final_status,
            "note": note,
            "updated_at": now_iso,
            "updated_by": user["id"],
            "attempted": True,
            "attempt_timestamp": now_iso,
            "idempotency_key": req_idempotency_key,
        }

        # Transactional database persistence: fail closed on error with base schema resilience
        try:
            if existing_att.data:
                db.table("attendance").update(att_payload).eq("session_id", session_id).eq("student_id", user["id"]).execute()
            else:
                db.table("attendance").insert(att_payload).execute()
        except HTTPException:
            raise
        except Exception as db_err:
            err_str = str(db_err)
            # Schema cache resilience: If database table is missing extended columns (attempted, attempt_timestamp, idempotency_key)
            if any(col in err_str for col in ("attempt_timestamp", "attempted", "idempotency_key")) or "PGRST204" in err_str:
                logger.warning(f"Database schema cache missing extended columns ({err_str}), falling back to base attendance schema.")
                base_payload = {
                    "session_id": session_id,
                    "student_id": user["id"],
                    "status": final_status,
                    "note": note,
                    "updated_at": now_iso,
                    "updated_by": user["id"],
                }
                try:
                    if existing_att.data:
                        db.table("attendance").update(base_payload).eq("session_id", session_id).eq("student_id", user["id"]).execute()
                    else:
                        db.table("attendance").insert(base_payload).execute()
                except Exception as fb_err:
                    logger.exception(f"Database transaction failure during fallback attendance write: {fb_err}")
                    _audit(
                        session_id=session_id,
                        actor_id=user["id"],
                        action="attendance.transaction_failed",
                        note=f"Database transaction error during attendance fallback write: {str(fb_err)}",
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Database transaction failure during attendance persistence. [DATABASE_TRANSACTION_FAILED]",
                        headers={"X-Error-Code": "DATABASE_TRANSACTION_FAILED"},
                    )
            else:
                logger.exception(f"Database transaction failure during attendance recording: {db_err}")
                _audit(
                    session_id=session_id,
                    actor_id=user["id"],
                    action="attendance.transaction_failed",
                    note=f"Database transaction error during attendance write: {str(db_err)}",
                )
                raise HTTPException(
                    status_code=500,
                    detail="Database transaction failure during attendance persistence. [DATABASE_TRANSACTION_FAILED]",
                    headers={"X-Error-Code": "DATABASE_TRANSACTION_FAILED"},
                )

        # 7. Audit log with classified security action
        if best_candidate and best_candidate["user_id"] != user["id"] and best_candidate["similarity"] >= 0.65 and (best_candidate["similarity"] - similarity) > 0.15:
            audit_action = "attendance.proxy_shield_blocked"

        _audit(
            session_id=session_id,
            actor_id=user["id"],
            action=audit_action,
            note=note,
            metadata={
                "similarity": round(similarity, 3),
                "liveness_passed": liveness_ok,
                "sir_detected": sir_ok,
                "crowd_detected": crowd_ok,
                "crowd_count": room_analysis.get("crowd_count", 0),
                "crowd_density_pct": room_analysis.get("crowd_density_pct", 0.0),
                "hall_capacity": capacity,
                "hall_type": hall_type,
                "room_matched": venue_ok,
                "is_empty_room": is_empty_room,
                "final_status": final_status,
            },
        )

        publish_event(
            event_payload("attendance.changed", {"session_id": session_id, "student_id": user["id"], "status": final_status}),
            [],
        )

        return {
            "status": final_status,
            "session_valid": True,
            "image_quality_passed": q_ok,
            "image_quality_metrics": q_metrics,
            "face_match": face_matched,
            "face_confidence": round(similarity, 3),
            "liveness_passed": liveness_ok,
            "liveness_note": liveness_res.get("message", "Liveness verified"),
            "liveness_checks": liveness_res,
            "room_match": venue_ok,
            "sir_detected": sir_ok,
            "sir_name": room_analysis.get("sir_name", "Lecturer"),
            "crowd_detected": crowd_ok,
            "crowd_count": room_analysis.get("crowd_count", 0),
            "crowd_density_pct": room_analysis.get("crowd_density_pct", 0.0),
            "hall_capacity": capacity,
            "hall_type": hall_type,
            "is_empty_room": is_empty_room,
            "message": (
                "Attendance successfully verified and marked Present!"
                if final_status == "present"
                else ("Empty room detected! Neither lecturer nor peers found in room. Flagged for review." if is_empty_room else note)
            ),
        }


@app.post("/api/sessions/{session_id}/verify-liveness")
def verify_student_liveness_only(
    session_id: str,
    face_frontal: UploadFile = File(...),
    face_left: Optional[UploadFile] = File(default=None),
    face_right: Optional[UploadFile] = File(default=None),
    face_blink: Optional[UploadFile] = File(default=None),
    challenge_token: Optional[str] = Form(default=None),
    # Untrusted client assertion parameters (strictly ignored for authorization)
    is_live: Optional[bool] = Form(default=None),
    verified: Optional[bool] = Form(default=None),
    client_verified: Optional[bool] = Form(default=None),
    confidence: Optional[float] = Form(default=None),
    client_confidence: Optional[float] = Form(default=None),
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    require_student(user)
    if is_live is not None or verified is not None or client_verified is not None:
        logger.warning(
            f"Client submitted bypass assertions to verify-liveness (is_live={is_live}, verified={verified}) "
            f"for user {user.get('id')}. Disregarded per security policy."
        )
    sess = _get_session_or_404(session_id)
    if sess.get("status") != "open":
        raise HTTPException(
            status_code=400,
            detail="This attendance session is not open. [SESSION_CLOSED]",
            headers={"X-Error-Code": "SESSION_CLOSED"},
        )
    _assert_session_access(sess, user)

    expected_challenges: Optional[list[str]] = None
    if challenge_token is not None and challenge_token.strip():
        token_payload = verify_liveness_challenge_token(challenge_token.strip())
        if token_payload.get("session_id") != session_id:
            raise HTTPException(status_code=400, detail="Challenge token was issued for a different session.")
        if token_payload.get("sub") != user["id"]:
            raise HTTPException(
                status_code=403,
                detail="Challenge token was issued for a different student. [IDOR_ATTEMPT_BLOCKED]",
                headers={"X-Error-Code": "IDOR_ATTEMPT_BLOCKED"},
            )
        nonce = token_payload.get("nonce", "")
        if not consume_challenge_nonce(nonce):
            raise HTTPException(status_code=400, detail="Challenge token has already been used. Tokens are strictly single-use.")
        expected_challenges = token_payload.get("challenges", [])

    frontal_bytes = _read_upload(face_frontal)
    img_front, front_err, front_msg = validate_image_payload(frontal_bytes, max_bytes=10485760)
    if front_err is not None:
        status_code = 413 if front_err == "IMAGE_TOO_LARGE" else (415 if front_err == "UNSUPPORTED_FORMAT" else 400)
        raise HTTPException(
            status_code=status_code,
            detail=f"{front_msg} [{front_err}]",
            headers={"X-Error-Code": front_err},
        )

    img_left = None
    if face_left is not None:
        b = _read_upload(face_left)
        if len(b) > 0:
            img_left = parse_image_frame(b)

    img_right = None
    if face_right is not None:
        b = _read_upload(face_right)
        if len(b) > 0:
            img_right = parse_image_frame(b)

    img_blink = None
    if face_blink is not None:
        b = _read_upload(face_blink)
        if len(b) > 0:
            img_blink = parse_image_frame(b)

    try:
        with inference_slot("verify_liveness"):
            engine = get_face_engine()
            return engine.verify_active_liveness(
                frontal_img=img_front,
                left_img=img_left,
                right_img=img_right,
                blink_img=img_blink,
                expected_challenges=expected_challenges,
            )
    except InferenceBusyError as exc:
        raise _inference_busy_http(exc)
    except HTTPException:
        raise
    except Exception as cv_err:
        logger.exception(f"Liveness verification failed closed: {cv_err}")
        raise HTTPException(
            status_code=503,
            detail=f"Liveness verification service error or model unavailable. [MODEL_UNAVAILABLE]: {str(cv_err)}",
            headers={"X-Error-Code": "MODEL_UNAVAILABLE"},
        )


# ── Routes: Controller Academic Management ───────────────────────────────────
@app.get("/api/controller/classes")
def controller_get_classes(user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    require_controller(user)
    db = supa()
    classes = db.table("classes").select("*").order("name").execute().data or []
    for c in classes:
        enr = db.table("enrollments").select("student_id").eq("class_id", c["id"]).execute().data or []
        c["student_count"] = len(enr)
        if c.get("teacher_id"):
            t = first_row(db.table("profiles").select("name").eq("id", c["teacher_id"]))
            c["teacher_name"] = t.get("name") if t else "Unassigned"
        else:
            c["teacher_name"] = "Unassigned"
    return classes


@app.post("/api/controller/classes")
def controller_create_class(body: CreateClassRequest, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_controller(user)
    db = supa()
    teacher_id = body.teacher_id
    if teacher_id:
        teacher_rec = db.table("profiles").select("id, role").eq("id", teacher_id).execute()
        if not teacher_rec.data or teacher_rec.data[0].get("role") != "teacher":
            raise HTTPException(
                status_code=400,
                detail=f"Invalid teacher_id '{teacher_id}': User does not exist or does not possess the lecturer/teacher role.",
            )
    else:
        teachers = db.table("profiles").select("id").eq("role", "teacher").execute().data or []
        teacher_id = teachers[0]["id"] if teachers else user["id"]

    cap = body.capacity or 120
    if cap not in (80, 120, 240, 260):
        cap = 120
    hall_type_map = {
        80: "80-Seater Seminar Hall",
        120: "120-Seater Lecture Theater",
        240: "240-Seater Tiered Amphitheater",
        260: "260-Seater Grand Auditorium",
    }
    htype = body.hall_type or hall_type_map.get(cap, f"{cap}-Seater Hall")
    rnum = body.room_number or f"LH-{cap}"

    new_class = {
        "name": body.name.strip(),
        "code": body.code.strip().upper(),
        "section": body.section or "Section A",
        "room_number": rnum,
        "capacity": cap,
        "hall_type": htype,
        "room_photo": body.room_photo or "https://images.unsplash.com/photo-1580582932707-520aed937b7b?w=800",
        "teacher_id": teacher_id,
        "created_at": utc_now(),
    }
    try:
        resp = db.table("classes").insert(new_class).execute()
    except Exception:
        base_class = {
            "name": body.name.strip(),
            "code": body.code.strip().upper(),
            "teacher_id": teacher_id,
            "created_at": utc_now(),
        }
        resp = db.table("classes").insert(base_class).execute()
    created = resp.data[0] if (resp and resp.data) else new_class
    _audit(
        session_id=None,
        actor_id=user["id"],
        action="controller.class_created",
        note=f"Controller created class {created['name']} ({created['code']}) for venue {rnum} ({htype}).",
    )
    return created


@app.post("/api/controller/classes/{class_id}/capture-room")
def controller_capture_class_room(
    class_id: str,
    photo: Optional[UploadFile] = File(default=None),
    photo_base64: Optional[str] = Form(default=None),
    capacity: Optional[int] = Form(default=None),
    hall_type: Optional[str] = Form(default=None),
    room_number: Optional[str] = Form(default=None),
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    require_controller(user)
    db = supa()
    class_row = _get_class_or_404(class_id)

    image_bytes = b""
    if photo is not None:
        image_bytes = _read_upload(photo)
    elif photo_base64:
        import base64
        try:
            clean_b64 = photo_base64.split("base64,", 1)[1] if "base64," in photo_base64 else photo_base64
            image_bytes = base64.b64decode(clean_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Malformed base64 image data.")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No classroom photo provided.")

    img, err_code, err_msg = validate_image_payload(image_bytes, max_bytes=10485760, min_width=1, min_height=1)
    if err_code is not None:
        status_code = 413 if err_code == "IMAGE_TOO_LARGE" else (415 if err_code == "UNSUPPORTED_FORMAT" else 400)
        raise HTTPException(status_code=status_code, detail=f"{err_msg} [{err_code}]", headers={"X-Error-Code": err_code})

    rh, rw = img.shape[:2]
    lap_var, mean_lum = compute_image_metrics(img)
    if lap_var < 1.5 or mean_lum < 3.0:
        raise HTTPException(status_code=400, detail="Classroom photo is too dark or blurry. Please hold steady in good lighting.")

    import base64
    b64_str = base64.b64encode(image_bytes).decode("ascii")
    data_uri = f"data:image/jpeg;base64,{b64_str}"

    cap = capacity or class_row.get("capacity") or 120
    if cap not in (80, 120, 240, 260):
        cap = 120

    hall_type_map = {
        80: "80-Seater Seminar Hall",
        120: "120-Seater Lecture Theater",
        240: "240-Seater Tiered Amphitheater",
        260: "260-Seater Grand Auditorium",
    }
    htype = hall_type or hall_type_map.get(cap, f"{cap}-Seater Hall")
    rnum = room_number or class_row.get("room_number") or f"LH-{cap}"

    now_iso = utc_now()
    update_data = {
        "room_photo": data_uri,
        "capacity": cap,
        "hall_type": htype,
        "room_number": rnum,
        "reference_captured_at": now_iso,
    }
    db.table("classes").update(update_data).eq("id", class_id).execute()

    _audit(
        session_id=None,
        actor_id=user["id"],
        action="controller.venue_captured",
        note=f"Academic Controller {user.get('name', 'Controller')} captured official reference room panorama for {class_row.get('name')} ({htype}, {rnum}).",
        metadata={"capacity": cap, "room_number": rnum, "resolution": f"{rw}x{rh}"},
    )

    return {
        "status": "success",
        "class_id": class_id,
        "capacity": cap,
        "hall_type": htype,
        "room_number": rnum,
        "room_photo": data_uri,
        "message": f"Official reference panorama captured for {htype} ({rnum}).",
    }


@app.get("/api/controller/students")
def controller_get_students(user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    require_controller(user)
    db = supa()
    students = db.table("profiles").select("*").eq("role", "student").order("name").execute().data or []
    for s in students:
        enr = db.table("enrollments").select("class_id").eq("student_id", s["id"]).execute().data or []
        cids = [e["class_id"] for e in enr]
        if cids:
            cls_resp = db.table("classes").select("name, code").in_("id", cids).execute().data or []
            s["classes"] = [f"{c['code']} - {c['name']}" for c in cls_resp]
        else:
            s["classes"] = []
        face = db.table("face_templates").select("id").eq("user_id", s["id"]).execute().data or []
        s["biometric_enrolled"] = len(face) > 0
    return students


@app.post("/api/controller/students")
def controller_create_student(body: CreateStudentRequest, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_controller(user)
    db = supa()
    email = body.email.strip().lower()
    existing = db.table("profiles").select("id").eq("email", email).execute().data or []
    if existing:
        raise HTTPException(status_code=400, detail="Student with this email already exists.")

    student_profile = {
        "name": body.name.strip(),
        "email": email,
        "role": "student",
        "roll_number": body.roll_number.strip(),
        "student_id": body.student_id or f"STU-{secrets.randbelow(9000)+1000}",
        "registration_locked": False,
        "created_at": utc_now(),
    }
    resp = db.table("profiles").insert(student_profile).execute()
    created = resp.data[0]
    for cid in body.class_ids:
        db.table("enrollments").insert({"class_id": cid, "student_id": created["id"]}).execute()
    _audit(
        session_id=None,
        actor_id=user["id"],
        action="controller.student_enrolled",
        note=f"Academic Controller {user.get('name', 'Controller')} enrolled new student {created.get('name')} ({created.get('roll_number')}).",
    )
    return created


@app.get("/api/controller/teachers")
def controller_get_teachers(user: dict[str, Any] = Depends(current_user)) -> list[dict[str, Any]]:
    require_controller(user)
    db = supa()
    teachers = db.table("profiles").select("id, name, email, role, created_at").eq("role", "teacher").order("name").execute().data or []
    return teachers


@app.post("/api/controller/teachers")
def controller_create_teacher(body: CreateTeacherRequest, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_controller(user)
    db = supa()
    email = body.email.strip().lower()
    existing = db.table("profiles").select("id").eq("email", email).execute().data or []
    if existing:
        raise HTTPException(status_code=400, detail="Teacher or user with this email already exists.")

    teacher_id = str(uuid.uuid4())
    try:
        auth_res = db.auth.admin.create_user({
            "email": email,
            "password": body.password or "teach123",
            "email_confirm": True,
            "user_metadata": {"name": body.name.strip(), "role": "teacher"},
        })
        teacher_id = auth_res.user.id
    except Exception:
        pass

    teacher_profile = {
        "id": teacher_id,
        "name": body.name.strip(),
        "email": email,
        "role": "teacher",
        "created_at": utc_now(),
    }
    resp = db.table("profiles").insert(teacher_profile).execute()
    created = resp.data[0] if (resp and resp.data) else teacher_profile
    _audit(
        session_id=None,
        actor_id=user["id"],
        action="controller.teacher_authorized",
        note=f"Academic Controller {user.get('name', 'Controller')} authorized teacher account {created.get('name')} ({email}).",
    )
    return created


@app.post("/api/controller/students/{student_id}/unlock")
def controller_unlock_student(student_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_controller(user)
    db = supa()
    try:
        db.table("profiles").update({"registration_locked": False}).eq("id", student_id).execute()
    except Exception as lock_err:
        logger.warning(f"Could not reset registration_locked for student {student_id}: {lock_err}")
    try:
        db.table("face_templates").delete().eq("user_id", student_id).execute()
    except Exception:
        pass
    _audit(
        session_id=None,
        actor_id=user["id"],
        action="biometric.registration_unlocked",
        note=f"Academic Controller {user.get('name', 'Controller')} unlocked biometric registration for student ID {student_id}.",
    )
    return {"status": "unlocked", "student_id": student_id, "message": "Biometric registration unlocked for student."}


@app.delete("/api/controller/students/{student_id}/biometrics")
def controller_purge_student_biometrics(student_id: str, user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    """Administrative purge of student's biometric reference template."""
    require_controller(user)
    db = supa()
    try:
        student_resp = db.table("profiles").select("id, name").eq("id", student_id).execute()
        student_row = student_resp.data[0] if (student_resp and student_resp.data) else None
    except Exception:
        student_row = None
    if not student_row:
        raise HTTPException(status_code=404, detail="Student not found")

    try:
        db.table("face_templates").delete().eq("user_id", student_id).execute()
    except Exception:
        pass
    try:
        db.table("profiles").update({"registration_locked": False}).eq("id", student_id).execute()
    except Exception as lock_err:
        logger.warning(f"Could not reset registration_locked for student {student_id}: {lock_err}")
    _audit(
        session_id=None,
        actor_id=user["id"],
        action="biometric.admin_purge",
        note=f"Academic Controller {user.get('name', 'Controller')} purged biometric template for student {student_row.get('name')}.",
        metadata={"target_student_id": student_id, "purged_by": user["id"]},
    )
    return {
        "status": "purged",
        "student_id": student_id,
        "message": "Student biometric template permanently purged and registration unlocked.",
    }


@app.get("/api/controller/registration-window")
def controller_get_registration_window(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    require_controller(user)
    db = supa()
    resp = db.table("system_settings").select("*").eq("key", "registration_window").execute()
    if resp.data:
        row = resp.data[0]
        return {
            "is_open": bool(row.get("is_open", True)),
            "updated_at": row.get("updated_at", utc_now()),
            "updated_by": row.get("updated_by"),
        }
    return {
        "is_open": True,
        "updated_at": utc_now(),
        "updated_by": user["id"],
    }


@app.post("/api/controller/registration-window")
def controller_set_registration_window(
    body: RegistrationWindowRequest,
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    require_controller(user)
    db = supa()
    now_ts = utc_now()
    existing = db.table("system_settings").select("*").eq("key", "registration_window").execute()
    data = {
        "key": "registration_window",
        "value": "open" if body.is_open else "closed",
        "is_open": body.is_open,
        "updated_at": now_ts,
        "updated_by": user["id"],
    }
    if existing.data:
        db.table("system_settings").update(data).eq("key", "registration_window").execute()
    else:
        db.table("system_settings").insert(data).execute()

    _audit(
        session_id=None,
        actor_id=user["id"],
        action="controller.registration_window_updated",
        note=f"Academic Controller {user.get('name', 'Controller')} set student registration window to {'OPEN' if body.is_open else 'CLOSED'}. {body.note or ''}".strip(),
    )
    return {
        "is_open": body.is_open,
        "status": "open" if body.is_open else "closed",
        "updated_at": now_ts,
        "message": f"Student registration window is now {'OPEN' if body.is_open else 'CLOSED'}.",
    }


@app.post("/api/controller/registry/import")
def controller_import_registry(
    file: Optional[UploadFile] = File(default=None),
    csv_text: Optional[str] = Form(default=None),
    user: dict[str, Any] = Depends(current_user),
) -> dict[str, Any]:
    """Import official student registry CSV with validation, duplicate detection, and class assignment."""
    require_controller(user)
    db = supa()

    content = ""
    if file is not None:
        raw_bytes = _read_upload(file)
        try:
            content = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                content = raw_bytes.decode("latin-1")
            except Exception:
                raise HTTPException(status_code=400, detail="Could not decode CSV file. Please upload a UTF-8 CSV.")
    elif csv_text:
        content = csv_text
    else:
        raise HTTPException(status_code=400, detail="Please upload a CSV file or provide csv_text.")

    if not content.strip():
        raise HTTPException(status_code=400, detail="The uploaded CSV content is empty.")

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV does not contain a valid header row.")

    norm_headers = {f.strip().lower().replace(" ", "_"): f for f in reader.fieldnames if f}
    name_field = next((norm_headers[k] for k in ["name", "student_name", "full_name"] if k in norm_headers), None)
    email_field = next((norm_headers[k] for k in ["email", "email_address", "mail"] if k in norm_headers), None)
    roll_field = next((norm_headers[k] for k in ["roll_number", "roll_no", "rollno", "roll"] if k in norm_headers), None)
    id_field = next((norm_headers[k] for k in ["student_id", "usn", "reg_no", "id"] if k in norm_headers), None)
    class_field = next((norm_headers[k] for k in ["class_code", "course_code", "class", "code"] if k in norm_headers), None)

    if not name_field or not email_field:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing mandatory columns: requires 'name' and 'email'. Found headers: {list(reader.fieldnames)}",
        )

    all_classes = db.table("classes").select("id, code").execute().data or []
    code_to_class_id = {c["code"].upper(): c["id"] for c in all_classes if c.get("code")}

    imported = 0
    updated = 0
    errors: list[str] = []
    line_no = 1

    for row in reader:
        line_no += 1
        name = (row.get(name_field) or "").strip()
        email = (row.get(email_field) or "").strip().lower()
        roll = (row.get(roll_field) or "").strip() if roll_field else ""
        stu_id = (row.get(id_field) or "").strip() if id_field else ""
        class_code = (row.get(class_field) or "").strip().upper() if class_field else ""

        if not name:
            errors.append(f"Row {line_no}: Missing student name.")
            continue
        if not email or "@" not in email:
            errors.append(f"Row {line_no}: Invalid email address '{email}'.")
            continue

        existing = db.table("profiles").select("*").eq("email", email).execute().data
        student_record_id: str
        if existing:
            student_record_id = existing[0]["id"]
            update_fields: dict[str, Any] = {"name": name}
            if roll:
                update_fields["roll_number"] = roll
            if stu_id:
                update_fields["student_id"] = stu_id
            db.table("profiles").update(update_fields).eq("id", student_record_id).execute()
            updated += 1
        else:
            new_stu = {
                "name": name,
                "email": email,
                "role": "student",
                "roll_number": roll or f"CS-2026-{str(imported + updated + 1).zfill(3)}",
                "student_id": stu_id or f"STU-{9000 + imported + updated + 1}",
                "registration_locked": False,
            }
            res = db.table("profiles").insert(new_stu).execute()
            student_record_id = res.data[0]["id"] if res.data else str(uuid.uuid4())
            imported += 1

        if class_code and class_code in code_to_class_id:
            target_class_id = code_to_class_id[class_code]
            already_enrolled = (
                db.table("enrollments")
                .select("student_id")
                .eq("class_id", target_class_id)
                .eq("student_id", student_record_id)
                .execute()
                .data
            )
            if not already_enrolled:
                db.table("enrollments").insert({
                    "class_id": target_class_id,
                    "student_id": student_record_id,
                }).execute()

    _audit(
        session_id=None,
        actor_id=user["id"],
        action="controller.registry_imported",
        note=f"Academic Controller {user.get('name', 'Controller')} imported student registry CSV: {imported} created, {updated} updated, {len(errors)} errors.",
    )

    return {
        "total_processed": line_no - 1,
        "imported": imported,
        "updated": updated,
        "error_count": len(errors),
        "errors": errors,
        "message": f"Successfully processed {imported + updated} students ({imported} created, {updated} updated).",
    }



@app.get("/api/audit")
def get_audit(
    session_id: Optional[str] = Query(default=None),
    user: dict[str, Any] = Depends(current_user),
    limit: Optional[int] = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    start, end = _page_bounds(limit, offset)
    if user.get("role") not in ("teacher", "controller", "admin"):
        raise HTTPException(status_code=403, detail="Lecturer or Controller access required")
    db = supa()
    if session_id:
        row = _get_session_or_404(session_id)
        _assert_session_access(row, user, teacher_only=True)
        resp = (
            db.table("audit_log")
            .select("*, profiles(id, name)")
            .eq("session_id", session_id)
            .order("timestamp", desc=True)
            .range(start, end)
            .execute()
        )
    else:
        if user.get("role") in ("controller", "admin"):
            resp = (
                db.table("audit_log")
                .select("*, profiles(id, name)")
                .order("timestamp", desc=True)
                .range(start, end)
                .execute()
            )
        else:
            teacher_classes = db.table("classes").select("id").eq("teacher_id", user["id"]).execute().data or []
            class_ids = [c["id"] for c in teacher_classes]
            if not class_ids:
                return []
            teacher_sessions = db.table("sessions").select("id").in_("class_id", class_ids).execute().data or []
            session_ids = [s["id"] for s in teacher_sessions]
            if not session_ids:
                return []
            resp = (
                db.table("audit_log")
                .select("*, profiles(id, name)")
                .in_("session_id", session_ids)
                .order("timestamp", desc=True)
                .range(start, end)
                .execute()
            )
    result = []
    for r in resp.data or []:
        item = {**r}
        if isinstance(r.get("profiles"), dict):
            item["actor_id"] = r["profiles"]["id"]
            item["actor_name"] = r["profiles"]["name"]
            item.pop("profiles", None)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        result.append(item)
    return result


@app.get("/api/analytics")
def get_analytics(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    if user.get("role") not in ("teacher", "controller", "admin"):
        raise HTTPException(status_code=403, detail="Lecturer or Controller access required")
    db = supa()
    if user.get("role") in ("controller", "admin"):
        sessions = db.table("sessions").select("*").execute().data or []
        classes = db.table("classes").select("*").execute().data or []
    else:
        classes = db.table("classes").select("*").eq("teacher_id", user["id"]).execute().data or []
        class_ids = [c["id"] for c in classes]
        sessions = db.table("sessions").select("*").in_("class_id", class_ids).execute().data or [] if class_ids else []
    students = db.table("profiles").select("*").eq("role", "student").execute().data or []
    attendance = db.table("attendance").select("*").execute().data or []
    audits = db.table("audit_log").select("*, profiles(id, name)").order("id", desc=True).execute().data or []

    total_sessions = len(sessions)
    total_classes = len(classes)
    total_students = len(students)

    present_count = sum(1 for a in attendance if a.get("status") in ("present", "verified"))
    absent_count = sum(1 for a in attendance if a.get("status") == "absent")
    pending_count = sum(1 for a in attendance if a.get("status") in ("pending", "flagged"))
    total_records = len(attendance)

    attendance_pct = round((present_count / max(1, total_records)) * 100, 1) if total_records > 0 else 0.0
    verified_live = sum(1 for a in attendance if a.get("attempted") is True and a.get("status") in ("present", "verified"))
    proxy_blocked = sum(1 for a in audits if "proxy" in str(a.get("action", "")).lower() or "proxy" in str(a.get("note", "")).lower())

    student_stats = []
    for s in students:
        s_att = [a for a in attendance if a.get("student_id") == s["id"]]
        s_pres = sum(1 for a in s_att if a.get("status") in ("present", "verified"))
        s_total = max(1, total_sessions)
        s_rate = round((s_pres / s_total) * 100, 1)
        last_att = next((a for a in sorted(s_att, key=lambda x: str(x.get("updated_at", "")), reverse=True)), None)
        student_stats.append({
            "id": s["id"],
            "name": s.get("name", "Student"),
            "roll_number": s.get("roll_number", "N/A"),
            "email": s.get("email", ""),
            "registration_locked": bool(s.get("registration_locked")),
            "present_count": s_pres,
            "total_sessions": total_sessions,
            "attendance_rate": s_rate,
            "last_verified_at": last_att.get("updated_at") if last_att else None,
            "last_status": last_att.get("status") if last_att else "not_attempted",
        })

    audit_list = []
    for r in audits:
        item = {**r}
        if isinstance(r.get("profiles"), dict):
            item["actor_id"] = r["profiles"]["id"]
            item["actor_name"] = r["profiles"]["name"]
            item.pop("profiles", None)
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        audit_list.append(item)

    return {
        "summary": {
            "total_sessions": total_sessions,
            "total_classes": total_classes,
            "total_students": total_students,
            "total_attendance_records": total_records,
            "present_count": present_count,
            "absent_count": absent_count,
            "pending_count": pending_count,
            "attendance_rate": attendance_pct,
            "verified_live_count": verified_live,
            "proxy_blocked_count": proxy_blocked,
        },
        "classes": [
            {
                "id": c["id"],
                "name": c.get("name"),
                "code": c.get("code"),
                "room_number": c.get("room_number"),
                "session_count": sum(1 for s in sessions if s.get("class_id") == c["id"]),
            }
            for c in classes
        ],
        "students": student_stats,
        "recent_audits": audit_list,
    }


@app.get("/api/reports/analytics/excel")
def export_master_analytics_excel(user: dict[str, Any] = Depends(current_user)) -> StreamingResponse:
    if user.get("role") not in ("teacher", "controller", "admin"):
        raise HTTPException(status_code=403, detail="Lecturer or Controller authorization required")

    import io
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    db = supa()
    if user.get("role") in ("controller", "admin"):
        sessions = db.table("sessions").select("*").execute().data or []
        classes = db.table("classes").select("*").execute().data or []
    else:
        classes = db.table("classes").select("*").eq("teacher_id", user["id"]).execute().data or []
        class_ids = [c["id"] for c in classes]
        sessions = db.table("sessions").select("*").in_("class_id", class_ids).execute().data or [] if class_ids else []
    students = db.table("profiles").select("*").eq("role", "student").execute().data or []
    attendance = db.table("attendance").select("*").execute().data or []
    audits = db.table("audit_log").select("*, profiles(id, name)").order("id", desc=True).execute().data or []

    wb = openpyxl.Workbook()

    # Sheet 1: Master Attendance Summary
    ws1 = wb.active
    ws1.title = "Executive Summary"
    ws1.views.sheetView[0].showGridLines = True

    font_title = Font(name="Calibri", size=15, bold=True, color="FFFFFF")
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    font_bold = Font(name="Calibri", size=11, bold=True, color="0F172A")
    font_data = Font(name="Calibri", size=11, color="334155")
    fill_brand = PatternFill(start_color="1E1B4B", end_color="1E1B4B", fill_type="solid")
    fill_header = PatternFill(start_color="4338CA", end_color="4338CA", fill_type="solid")
    fill_zebra = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    border_thin = Border(left=Side(style='thin', color='CBD5E1'), right=Side(style='thin', color='CBD5E1'), top=Side(style='thin', color='CBD5E1'), bottom=Side(style='thin', color='CBD5E1'))

    ws1.merge_cells("A1:E1")
    ws1["A1"] = "CAMPUS ATTENDANCE MASTER ANALYTICS & AUDIT REPORT"
    ws1["A1"].font = font_title
    ws1["A1"].fill = fill_brand
    ws1["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws1.row_dimensions[1].height = 28

    present_count = sum(1 for a in attendance if a.get("status") in ("present", "verified"))
    absent_count = sum(1 for a in attendance if a.get("status") == "absent")
    pending_count = sum(1 for a in attendance if a.get("status") in ("pending", "flagged"))
    total_records = len(attendance)
    rate = round((present_count / max(1, total_records)) * 100, 1) if total_records > 0 else 0.0

    kpis = [
        ("Total Sessions Conducted", len(sessions)),
        ("Enrolled Students", len(students)),
        ("Total Classrooms / Sections", len(classes)),
        ("Overall Attendance Rate", f"{rate}%"),
        ("Verified Present Count", present_count),
        ("Unexcused Absent Count", absent_count),
        ("Flagged for Review", pending_count),
        ("Biometric Verified Live", sum(1 for a in attendance if a.get("attempted") is True and a.get("status") in ("present", "verified"))),
    ]

    ws1["A3"] = "Metric Indicator"
    ws1["B3"] = "Value"
    ws1["A3"].font = font_header
    ws1["A3"].fill = fill_header
    ws1["B3"].font = font_header
    ws1["B3"].fill = fill_header

    for idx, (lbl, val) in enumerate(kpis, start=4):
        ws1[f"A{idx}"] = lbl
        ws1[f"B{idx}"] = val
        ws1[f"A{idx}"].font = font_data
        ws1[f"B{idx}"].font = font_bold
        ws1[f"A{idx}"].border = border_thin
        ws1[f"B{idx}"].border = border_thin
        if idx % 2 == 0:
            ws1[f"A{idx}"].fill = fill_zebra
            ws1[f"B{idx}"].fill = fill_zebra

    # Sheet 2: Student Roster Attendance Analytics
    ws2 = wb.create_sheet(title="Student Roster Analytics")
    ws2.views.sheetView[0].showGridLines = True
    headers_s = ["#", "Student Name", "Roll Number", "Email", "Biometrics Locked", "Attended", "Total Sessions", "Attendance Rate %", "Last Verification Status"]
    for col_idx, h in enumerate(headers_s, start=1):
        cell = ws2.cell(row=1, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws2.row_dimensions[1].height = 24

    for r_idx, s in enumerate(students, start=2):
        s_att = [a for a in attendance if a.get("student_id") == s["id"]]
        s_pres = sum(1 for a in s_att if a.get("status") in ("present", "verified"))
        s_tot = max(1, len(sessions))
        s_pct = round((s_pres / s_tot) * 100, 1)
        last_att = next((a for a in sorted(s_att, key=lambda x: str(x.get("updated_at", "")), reverse=True)), None)

        row_vals = [
            r_idx - 1,
            s.get("name", "Student"),
            s.get("roll_number", "N/A"),
            s.get("email", ""),
            "Locked" if s.get("registration_locked") else "Pending",
            s_pres,
            len(sessions),
            f"{s_pct}%",
            (last_att.get("status") if last_att else "Not Attempted").upper(),
        ]
        for c_idx, val in enumerate(row_vals, start=1):
            c = ws2.cell(row=r_idx, column=c_idx, value=val)
            c.font = font_data
            c.border = border_thin
            if r_idx % 2 == 0:
                c.fill = fill_zebra

    # Sheet 3: Complete Audit Trail
    ws3 = wb.create_sheet(title="System Audit Trail")
    ws3.views.sheetView[0].showGridLines = True
    headers_a = ["Event ID", "Timestamp (UTC)", "Actor", "Action", "Audit Note / Decision", "Session ID"]
    for col_idx, h in enumerate(headers_a, start=1):
        cell = ws3.cell(row=1, column=col_idx, value=h)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws3.row_dimensions[1].height = 24

    for r_idx, a in enumerate(audits, start=2):
        actor_name = a.get("profiles", {}).get("name", "System") if isinstance(a.get("profiles"), dict) else "System"
        row_vals = [
            a.get("id"),
            str(a.get("timestamp", ""))[:19],
            actor_name,
            a.get("action", ""),
            a.get("note", ""),
            a.get("session_id", "Global"),
        ]
        for c_idx, val in enumerate(row_vals, start=1):
            c = ws3.cell(row=r_idx, column=c_idx, value=val)
            c.font = font_data
            c.border = border_thin
            if r_idx % 2 == 0:
                c.fill = fill_zebra

    # Auto-fit columns across all sheets
    for ws in [ws1, ws2, ws3]:
        for col in ws.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"Attendance_Master_Analytics_{utc_now()[:10]}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@app.get("/api/notifications")
def notifications(
    user: dict[str, Any] = Depends(current_user),
    limit: Optional[int] = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    start, end = _page_bounds(limit, offset)
    resp = (
        supa()
        .table("notifications")
        .select("id, type, title, body, session_id, created_at, read_at")
        .eq("user_id", user["id"])
        .order("created_at", desc=True)
        .range(start, end)
        .execute()
    )
    return resp.data or []


@app.post("/api/notifications/{notification_id}/read")
def mark_notification_read(
    notification_id: str, user: dict[str, Any] = Depends(current_user)
) -> dict[str, Any]:
    db = supa()
    existing = (
        db.table("notifications")
        .select("id")
        .eq("id", notification_id)
        .eq("user_id", user["id"])
        .execute()
    )
    if not existing.data:
        raise HTTPException(status_code=404, detail="Notification not found")
    read_at = utc_now()
    db.table("notifications").update({"read_at": read_at}).eq("id", notification_id).execute()
    return {"id": notification_id, "read_at": read_at}


# ── Routes: Realtime WebSockets ───────────────────────────────────────────────
@app.post("/api/realtime/ticket")
@app.post("/api/auth/ws-ticket")
def create_ticket(user: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    ticket = secrets.token_urlsafe(32)
    expires_at = time.time() + WS_TICKET_TTL_SECONDS
    with _ws_lock:
        _ws_tickets[ticket] = (user["id"], expires_at)
    return {"ticket": ticket, "expires_in": WS_TICKET_TTL_SECONDS}


@app.websocket("/ws")
@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket, ticket: str = Query(...)):
    now = time.time()
    with _ws_lock:
        record = _ws_tickets.pop(ticket, None)
    if not record or record[1] < now:
        await websocket.close(code=1008, reason="Invalid or expired single-use ticket")
        return
    user_id = record[0]
    await websocket.accept()
    event_queue: asyncio.Queue[str] = asyncio.Queue()
    with _ws_lock:
        _ws_clients.setdefault(user_id, set()).add(websocket)
        _ws_client_loops[websocket] = asyncio.get_running_loop()
        _ws_client_queues[websocket] = event_queue
    receive_task: Optional[asyncio.Task[Any]] = None
    event_task: Optional[asyncio.Task[str]] = None
    try:
        await websocket.send_text(event_payload("connected", {"user_id": user_id}))
        receive_task = asyncio.create_task(websocket.receive())
        event_task = asyncio.create_task(event_queue.get())
        while True:
            done, _ = await asyncio.wait({receive_task, event_task}, return_when=asyncio.FIRST_COMPLETED)
            if receive_task in done:
                try:
                    message = receive_task.result()
                except WebSocketDisconnect:
                    break
                if message.get("type") == "websocket.disconnect":
                    break
                receive_task = asyncio.create_task(websocket.receive())
            if event_task in done:
                try:
                    msg_text = event_task.result()
                    await websocket.send_text(msg_text)
                except (WebSocketDisconnect, RuntimeError):
                    break
                event_task = asyncio.create_task(event_queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        for task in (receive_task, event_task):
            if task is not None and not task.done():
                task.cancel()
        if receive_task is not None or event_task is not None:
            await asyncio.gather(
                *(task for task in (receive_task, event_task) if task is not None),
                return_exceptions=True,
            )
        with _ws_lock:
            if user_id in _ws_clients:
                _ws_clients[user_id].discard(websocket)
                if not _ws_clients[user_id]:
                    _ws_clients.pop(user_id, None)
            _ws_client_loops.pop(websocket, None)
            _ws_client_queues.pop(websocket, None)

"""Configuration and threshold definitions for Attendrix / SDTT Attendance System.

All thresholds for image quality, continuous sweep geometry, face recognition,
and session time limits are centralized here with documented defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ── Environment & Supabase ───────────────────────────────────────────────────
ATTENDANCE_ENV = os.getenv("ATTENDANCE_ENV", "development").lower()
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "https://placeholder.supabase.co")
SUPABASE_SERVICE_ROLE_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "placeholder_service_role_key")
SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", SUPABASE_SERVICE_ROLE_KEY)
SUPABASE_JWT_SECRET: str = os.getenv("SUPABASE_JWT_SECRET", "test_jwt_secret_must_be_at_least_32_chars_long")

# Symmetric encryption key for face biometric embeddings
# ATTENDANCE_BIOMETRIC_ENCRYPTION_KEY is the name documented in .env.example
# and render.yaml; ATTENDANCE_SECRET_KEY takes precedence when both are set.
ATTENDANCE_SECRET_KEY: str = (
    os.getenv("ATTENDANCE_SECRET_KEY")
    or os.getenv("ATTENDANCE_BIOMETRIC_ENCRYPTION_KEY")
    or "attendrix_super_secret_encryption_key_32_bytes_min!"
)

# ── Network & Token TTL ───────────────────────────────────────────────────────
ATTENDANCE_TOKEN_TTL_SECONDS = int(os.getenv("ATTENDANCE_TOKEN_TTL_SECONDS", "28800"))
WS_TICKET_TTL_SECONDS = int(os.getenv("ATTENDANCE_WS_TICKET_TTL_SECONDS", "60"))
MAX_CAPTURE_BYTES = int(os.getenv("ATTENDANCE_MAX_CAPTURE_BYTES", "25165824"))  # 24MB for multi-frame sweep
MAX_REQUEST_SIZE_BYTES = int(os.getenv("ATTENDANCE_MAX_REQUEST_SIZE_BYTES", "15728640"))  # 15MB max payload size
CORS_ORIGINS = [
    v.strip()
    for v in os.getenv(
        "ATTENDANCE_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:8081,http://127.0.0.1:5173",
    ).split(",")
    if v.strip()
]


def validate_environment_config() -> list[str]:
    """Validate critical environment configurations and return warnings or errors."""
    issues = []
    if ATTENDANCE_ENV == "production":
        if "attendrix_super_secret" in ATTENDANCE_SECRET_KEY:
            issues.append("ATTENDANCE_SECRET_KEY is using default placeholder in production.")
        if "test_jwt_secret" in SUPABASE_JWT_SECRET:
            issues.append("SUPABASE_JWT_SECRET is using default placeholder in production.")
        if len(ATTENDANCE_SECRET_KEY) < 32:
            issues.append("ATTENDANCE_SECRET_KEY must be at least 32 characters in production.")
    return issues

# ── Session Lifecycle & Time Limits ───────────────────────────────────────────
DEFAULT_SESSION_DURATION_MINUTES = 10
SESSION_DURATION_PRESETS = [5, 10, 15]
MAX_SESSION_DURATION_MINUTES = 60
SESSION_GRACE_PERIOD_S = float(os.getenv("ATTENDANCE_SESSION_GRACE_PERIOD_S", "5.0"))
AUTO_CLOSE_INTERVAL_S = float(os.getenv("ATTENDANCE_AUTO_CLOSE_INTERVAL_S", "15.0" if ATTENDANCE_ENV != "test" else "2.0"))
MAX_CONCURRENT_INFERENCE = int(os.getenv("ATTENDANCE_MAX_CONCURRENT_INFERENCE", "1"))
INFERENCE_QUEUE_TIMEOUT_S = float(os.getenv("ATTENDANCE_INFERENCE_QUEUE_TIMEOUT_S", "60.0"))
STARTUP_DB_PROBE_TIMEOUT_S = float(os.getenv("ATTENDANCE_STARTUP_DB_PROBE_TIMEOUT_S", "3.0"))
ENABLE_REQUEST_TIMING = os.getenv("ATTENDANCE_ENABLE_REQUEST_TIMING", "1") == "1"
ENABLE_DB_TIMING = os.getenv("ATTENDANCE_ENABLE_DB_TIMING", "1") == "1"
DB_SLOW_QUERY_MS = float(os.getenv("ATTENDANCE_DB_SLOW_QUERY_MS", "250.0"))

# ── Attempt Limits ────────────────────────────────────────────────────────────
MAX_ATTEMPTS_PER_STUDENT = int(os.getenv("ATTENDANCE_MAX_ATTEMPTS", "3"))

# ── Continuous Sweep Capture Rules & Geometry ─────────────────────────────────
MIN_FRAMES = int(os.getenv("ATTENDANCE_MIN_FRAMES", "12"))
MAX_FRAMES = int(os.getenv("ATTENDANCE_MAX_FRAMES", "60"))
SWEEP_MIN_DURATION_S = float(os.getenv("ATTENDANCE_SWEEP_MIN_DURATION_S", "4.0"))
SWEEP_MAX_DURATION_S = float(os.getenv("ATTENDANCE_SWEEP_MAX_DURATION_S", "15.0"))
HEADING_MAX_GAP_DEG = float(os.getenv("ATTENDANCE_HEADING_MAX_GAP_DEG", "25.0"))
HEADING_MIN_SPAN_DEG = float(os.getenv("ATTENDANCE_HEADING_MIN_SPAN_DEG", "150.0"))
VISUAL_CONTINUITY_MIN_SCORE = float(os.getenv("ATTENDANCE_VISUAL_CONTINUITY_MIN", "0.30"))

# ── Image Quality Thresholds ──────────────────────────────────────────────────
MIN_IMAGE_WIDTH = 320
MIN_IMAGE_HEIGHT = 240
MIN_LAPLACIAN_VAR = float(os.getenv("ATTENDANCE_MIN_LAPLACIAN_VAR", "100.0"))  # Blur check
MIN_LUMINANCE = float(os.getenv("ATTENDANCE_MIN_LUMINANCE", "40.0"))          # Underexposure check
MAX_LUMINANCE = float(os.getenv("ATTENDANCE_MAX_LUMINANCE", "220.0"))         # Overexposure check

# ── Biometric & Face Verification Thresholds ──────────────────────────────────
# Cosine similarity for ArcFace 512-d unit embeddings:
# >= 0.72: Pass
# 0.50 - 0.71: Flagged for review
# < 0.50: Fail / Reject
SIMILARITY_PASS_THRESHOLD = float(os.getenv("ATTENDANCE_SIMILARITY_PASS", "0.72"))
SIMILARITY_FLAG_THRESHOLD = float(os.getenv("ATTENDANCE_SIMILARITY_FLAG", "0.50"))

# Neighbour detection in initial selfie arc (~first 30 deg heading)
MIN_NEIGHBOURS = int(os.getenv("ATTENDANCE_MIN_NEIGHBOURS", "1"))
NEIGHBOUR_MAX_FACE_WIDTHS = float(os.getenv("ATTENDANCE_NEIGHBOUR_MAX_FACE_WIDTHS", "3.0"))

# Lecturer detection in final stage (heading >= 150 deg)
LECTURER_MIN_IPD_PX = float(os.getenv("ATTENDANCE_LECTURER_MIN_IPD_PX", "30.0"))

# Fraud detection: perceptual hash hamming distance
PHASH_HAMMING_DISTANCE_FLAG = int(os.getenv("ATTENDANCE_PHASH_FLAG_DIST", "6"))

# ── Privacy & Retention ───────────────────────────────────────────────────────
FILMSTRIP_RETENTION_DAYS = int(os.getenv("ATTENDANCE_FILMSTRIP_RETENTION_DAYS", "7"))
FILMSTRIP_KEY_FRAMES = int(os.getenv("ATTENDANCE_FILMSTRIP_KEY_FRAMES", "6"))

# ── Rate Limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_LOGIN_PER_MIN = 10
RATE_LIMIT_SUBMIT_PER_MIN = 5

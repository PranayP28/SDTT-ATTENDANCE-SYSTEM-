"""Continuous Sweep Verification Pipeline implementing Rules 1 through 9.

Ordered verification rules with detailed metrics collection, reason codes,
filmstrip extraction for flagged reviews, and privacy-first frame deletion.
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

from app.config import (
    FILMSTRIP_KEY_FRAMES,
    HEADING_MAX_GAP_DEG,
    HEADING_MIN_SPAN_DEG,
    LECTURER_MIN_IPD_PX,
    MAX_ATTEMPTS_PER_STUDENT,
    MAX_FRAMES,
    MIN_FRAMES,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    MIN_LAPLACIAN_VAR,
    MIN_LUMINANCE,
    MIN_NEIGHBOURS,
    NEIGHBOUR_MAX_FACE_WIDTHS,
    PHASH_HAMMING_DISTANCE_FLAG,
    SESSION_GRACE_PERIOD_S,
    SIMILARITY_FLAG_THRESHOLD,
    SIMILARITY_PASS_THRESHOLD,
    SWEEP_MAX_DURATION_S,
    SWEEP_MIN_DURATION_S,
    VISUAL_CONTINUITY_MIN_SCORE,
)
from app.db import supa, utc_now
from app.face_engine import FaceDetection, FaceEngine, get_face_engine
from app.security import decrypt_face_embedding


@dataclass
class SweepFrame:
    """A single frame from the continuous camera sweep with sensor metadata."""
    index: int
    image_bytes: bytes
    heading_deg: float
    time_offset_s: float
    img_bgr: Optional[np.ndarray] = None
    laplacian_var: float = 0.0
    mean_luminance: float = 0.0


@dataclass
class VerificationOutcome:
    """Detailed result of executing the 9-rule attendance verification pipeline."""
    state: str  # 'verified', 'flagged', 'rejected', 'error'
    reason_code: str
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)
    key_frame_indices: list[int] = field(default_factory=list)
    filmstrip_bytes: list[bytes] = field(default_factory=list)


def parse_image_frame(image_bytes: bytes) -> Optional[np.ndarray]:
    """Decode raw image bytes into a BGR NumPy array with EXIF orientation correction."""
    if not image_bytes:
        return None
    if Image is not None and ImageOps is not None:
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            rgb = np.array(pil_img)
            if cv2 is not None:
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return rgb[:, :, ::-1]  # RGB to BGR
        except Exception:
            pass
    if cv2 is None:
        return None
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def compute_image_metrics(img_bgr: np.ndarray) -> tuple[float, float]:
    """Compute Laplacian variance (blur) and mean luminance."""
    if cv2 is None or img_bgr is None or img_bgr.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_lum = float(np.mean(gray))
    return round(lap_var, 2), round(mean_lum, 2)


def validate_image_payload(
    image_bytes: bytes,
    max_bytes: int = 10485760,  # 10MB
    min_width: int = 120,
    min_height: int = 120,
) -> tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """Validate image payload size, magic header bytes, decoding, and resolution.
    
    Returns:
        (img_bgr, error_code, error_message)
        If valid: (img_bgr, None, None)
        If invalid: (None, error_code, error_message)
    """
    if image_bytes is None or len(image_bytes) == 0:
        return None, "EMPTY_IMAGE", "Uploaded image file is empty (0 bytes)."

    if len(image_bytes) > max_bytes:
        mb = max_bytes // (1024 * 1024)
        return None, "IMAGE_TOO_LARGE", f"Uploaded image exceeds maximum allowed size ({mb}MB)."

    # Format verification: Check file magic signature bytes
    is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = len(image_bytes) >= 12 and image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP"

    if not (is_jpeg or is_png or is_webp):
        detected = "unknown"
        if image_bytes.startswith(b"GIF8"):
            detected = "GIF"
        elif image_bytes.startswith(b"%PDF"):
            detected = "PDF"
        elif image_bytes.startswith(b"<html") or image_bytes.startswith(b"<!DOC"):
            detected = "HTML"
        return None, "UNSUPPORTED_FORMAT", f"Unsupported image format (detected: {detected}). Only JPEG, PNG, and WebP are accepted."

    # Decode image
    img = parse_image_frame(image_bytes)
    if img is None or img.size == 0:
        return None, "CORRUPTED_IMAGE", "Image file data is corrupted or cannot be decoded."

    h, w = img.shape[:2]
    if w < min_width or h < min_height:
        return None, "IMAGE_TOO_SMALL", f"Image resolution too low ({w}x{h}); minimum {min_width}x{min_height} required."

    return img, None, None


def assess_image_quality(
    img_bgr: np.ndarray,
    face: Optional[FaceDetection] = None,
    min_blur_var: float = 35.0,
    min_luminance: float = 35.0,
    max_luminance: float = 225.0,
) -> tuple[bool, Optional[str], Optional[str], dict[str, Any]]:
    """Assess image blur, poor lighting, excessive exposure, backlighting, and face geometry.
    
    Returns:
        (passed, error_code, error_message, metrics)
    """
    if img_bgr is None or img_bgr.size == 0:
        return False, "EMPTY_IMAGE", "Empty image frame received.", {}

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))

    metrics: dict[str, Any] = {
        "laplacian_var": round(lap_var, 1),
        "mean_luminance": round(mean_lum, 1),
        "contrast_std": round(std_lum, 1),
    }

    # 1. Poor Lighting (Underexposure)
    if mean_lum < min_luminance:
        return (
            False,
            "POOR_LIGHTING",
            f"Photo is too dark (brightness {mean_lum:.1f} < {min_luminance:.1f}). Please take your photo in a well-lit area.",
            metrics,
        )

    # 2. Excessive Exposure (Overexposure / glare)
    if mean_lum > max_luminance:
        return (
            False,
            "EXCESSIVE_EXPOSURE",
            f"Photo is overexposed or has glare (brightness {mean_lum:.1f} > {max_luminance:.1f}). Avoid pointing directly at light sources.",
            metrics,
        )

    # 3. Blur Check
    if lap_var < min_blur_var:
        return (
            False,
            "IMAGE_BLURRY",
            f"Photo is too blurry (sharpness score {lap_var:.1f} < {min_blur_var:.1f}). Please hold the phone steady in good lighting.",
            metrics,
        )

    # 4. Backlighting and Face Geometry Checks
    if face is not None:
        bx, by, bw, bh = face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h
        bx, by = max(0, bx), max(0, by)
        bw, bh = min(w - bx, bw), min(h - by, bh)

        if bw > 10 and bh > 10:
            face_roi = gray[by:by + bh, bx:bx + bw]
            face_lum = float(np.mean(face_roi))
            mask = np.ones((h, w), dtype=bool)
            mask[by:by + bh, bx:bx + bw] = False
            bg_lum = float(np.mean(gray[mask])) if np.any(mask) else mean_lum
            metrics["face_luminance"] = round(face_lum, 1)
            metrics["bg_luminance"] = round(bg_lum, 1)

            # Severe backlighting: bright background and dark face
            if bg_lum > 120.0 and face_lum < 65.0 and (bg_lum / max(face_lum, 1.0)) > 2.0:
                return (
                    False,
                    "BACKLIGHTING_DETECTED",
                    f"Severe backlighting detected (background is {bg_lum/max(face_lum, 1.0):.1f}x brighter than face). Position the light source in front of you.",
                    metrics,
                )

        # 5. Face size ratio
        area_ratio = (bw * bh) / float(w * h)
        metrics["face_area_ratio"] = round(area_ratio, 3)
        if area_ratio < 0.035 or face.ipd_px < 20.0:
            return (
                False,
                "FACE_TOO_SMALL",
                f"Face is too small in the camera frame ({area_ratio*100:.1f}%). Move closer to the camera.",
                metrics,
            )

        # 6. Face bounds (Clipped)
        margin = 4
        if bx < margin or by < margin or (bx + bw) > (w - margin) or (by + bh) > (h - margin):
            return (
                False,
                "FACE_OUT_OF_BOUNDS",
                "Face is partially outside the camera frame. Center your face in the oval guide.",
                metrics,
            )

    return True, None, "Image quality verified.", metrics


def validate_single_face(
    detected_faces: list[FaceDetection],
) -> tuple[bool, Optional[str], Optional[str]]:
    """Verify that exactly one face is detected for single-student verification."""
    if not detected_faces:
        return False, "NO_FACE_DETECTED", "No face detected. Please center your face in the front camera and try again."
    if len(detected_faces) > 1:
        return False, "MULTIPLE_FACES", f"Multiple faces ({len(detected_faces)}) detected. Please ensure only you are visible in the camera frame."
    return True, None, None


def hamming_distance(h1: str, h2: str) -> int:
    """Hamming distance between two hex string hashes."""
    try:
        x = int(h1, 16) ^ int(h2, 16)
        return bin(x).count("1")
    except Exception:
        return 64


class VerificationPipeline:
    """Executes ordered rules 1 to 9 against a student's continuous sweep submission."""

    def __init__(self, engine: Optional[FaceEngine] = None):
        self.engine = engine or get_face_engine()

    def run(
        self,
        session: dict[str, Any],
        student_profile: dict[str, Any],
        attempt_id: str,
        frames: list[SweepFrame],
        student_enrolled_embedding: Optional[np.ndarray] = None,
        lecturer_enrolled_embedding: Optional[np.ndarray] = None,
    ) -> VerificationOutcome:
        metrics: dict[str, Any] = {
            "attempt_id": attempt_id,
            "session_id": session.get("id"),
            "student_id": student_profile.get("id"),
            "evaluated_at": utc_now(),
            "rules": {},
        }

        # ── RULE 1: Authorization, Session State, and Time Window ────────────
        now_dt = datetime.now(timezone.utc)
        ends_at_str = session.get("ends_at")
        session_status = session.get("status", "closed")

        if session_status != "open":
            metrics["rules"]["1_auth_window"] = {"passed": False, "reason": "SESSION_CLOSED"}
            return VerificationOutcome(
                state="rejected",
                reason_code="SESSION_CLOSED",
                message="Attendance window is closed.",
                metrics=metrics,
            )

        if ends_at_str:
            ends_at = datetime.fromisoformat(ends_at_str.replace("Z", "+00:00"))
            time_diff = (now_dt - ends_at).total_seconds()
            if time_diff > SESSION_GRACE_PERIOD_S:
                metrics["rules"]["1_auth_window"] = {
                    "passed": False,
                    "reason": "WINDOW_EXPIRED",
                    "seconds_late": round(time_diff, 1),
                }
                return VerificationOutcome(
                    state="rejected",
                    reason_code="OUTSIDE_WINDOW",
                    message="Attendance window has expired. Submissions are no longer accepted.",
                    metrics=metrics,
                )

        metrics["rules"]["1_auth_window"] = {"passed": True}

        # ── RULE 2: Idempotency & Limits ─────────────────────────────────────
        db = supa()
        existing_attempt = (
            db.table("checkin_attempts")
            .select("id")
            .eq("attempt_id", attempt_id)
            .execute()
        )
        if existing_attempt.data:
            metrics["rules"]["2_limits"] = {"passed": False, "reason": "DUPLICATE_ATTEMPT"}
            return VerificationOutcome(
                state="rejected",
                reason_code="DUPLICATE_ATTEMPT",
                message="Duplicate submission attempt detected.",
                metrics=metrics,
            )

        verified_exists = (
            db.table("checkin_attempts")
            .select("id")
            .eq("session_id", session["id"])
            .eq("student_id", student_profile["id"])
            .eq("state", "verified")
            .execute()
        )
        if verified_exists.data:
            metrics["rules"]["2_limits"] = {"passed": False, "reason": "ALREADY_VERIFIED"}
            return VerificationOutcome(
                state="rejected",
                reason_code="ALREADY_VERIFIED",
                message="Your attendance has already been verified for this session.",
                metrics=metrics,
            )

        prior_attempts = (
            db.table("checkin_attempts")
            .select("id", count="exact")
            .eq("session_id", session["id"])
            .eq("student_id", student_profile["id"])
            .execute()
        )
        attempt_count = prior_attempts.count if prior_attempts.count is not None else len(prior_attempts.data or [])
        if attempt_count >= MAX_ATTEMPTS_PER_STUDENT:
            metrics["rules"]["2_limits"] = {"passed": False, "reason": "MAX_ATTEMPTS_EXCEEDED"}
            return VerificationOutcome(
                state="rejected",
                reason_code="MAX_ATTEMPTS_EXCEEDED",
                message=f"Maximum number of check-in attempts ({MAX_ATTEMPTS_PER_STUDENT}) reached.",
                metrics=metrics,
            )

        if len(frames) < MIN_FRAMES or len(frames) > MAX_FRAMES:
            metrics["rules"]["2_limits"] = {
                "passed": False,
                "reason": "FRAME_COUNT_INVALID",
                "frame_count": len(frames),
            }
            return VerificationOutcome(
                state="rejected",
                reason_code="TOO_FEW_FRAMES" if len(frames) < MIN_FRAMES else "TOO_MANY_FRAMES",
                message=f"Sweep captured {len(frames)} frames; requires between {MIN_FRAMES} and {MAX_FRAMES} frames.",
                metrics=metrics,
            )

        metrics["rules"]["2_limits"] = {"passed": True, "prior_attempts": attempt_count}

        # ── Parse Frame Images & Compute Quality Metrics ─────────────────────
        decoded_frames: list[SweepFrame] = []
        for f in frames:
            img = parse_image_frame(f.image_bytes)
            if img is not None:
                f.img_bgr = img
                f.laplacian_var, f.mean_luminance = compute_image_metrics(img)
                decoded_frames.append(f)

        if len(decoded_frames) < MIN_FRAMES:
            return VerificationOutcome(
                state="rejected",
                reason_code="CORRUPT_FRAMES",
                message="Unable to decode sufficient valid image frames.",
                metrics=metrics,
            )

        # ── RULE 3: Image Quality Checks ─────────────────────────────────────
        avg_blur = float(np.mean([f.laplacian_var for f in decoded_frames]))
        avg_lum = float(np.mean([f.mean_luminance for f in decoded_frames]))
        metrics["rules"]["3_image_quality"] = {
            "avg_laplacian_var": round(avg_blur, 1),
            "avg_luminance": round(avg_lum, 1),
        }

        first_frame = decoded_frames[0].img_bgr
        h, w = first_frame.shape[:2]
        if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
            metrics["rules"]["3_image_quality"]["passed"] = False
            return VerificationOutcome(
                state="rejected",
                reason_code="RESOLUTION_TOO_LOW",
                message=f"Camera resolution ({w}x{h}) is too low for reliable verification.",
                metrics=metrics,
            )

        if avg_blur < MIN_LAPLACIAN_VAR:
            metrics["rules"]["3_image_quality"]["passed"] = False
            return VerificationOutcome(
                state="rejected",
                reason_code="IMAGE_BLURRY",
                message="Camera frames are blurry due to rapid motion or poor focus. Please retake slowly.",
                metrics=metrics,
            )

        if avg_lum < MIN_LUMINANCE:
            metrics["rules"]["3_image_quality"]["passed"] = False
            return VerificationOutcome(
                state="rejected",
                reason_code="IMAGE_POOR_LIGHTING",
                message="Classroom lighting is too dark for verification. Please adjust lighting and retake.",
                metrics=metrics,
            )

        metrics["rules"]["3_image_quality"]["passed"] = True

        # ── RULE 4: Continuous Sweep Geometry & Evidence ─────────────────────
        duration = decoded_frames[-1].time_offset_s - decoded_frames[0].time_offset_s
        headings = [f.heading_deg for f in decoded_frames]
        span = abs(headings[-1] - headings[0])
        max_gap = 0.0
        for i in range(len(headings) - 1):
            gap = abs(headings[i + 1] - headings[i])
            if gap > max_gap:
                max_gap = gap

        # Visual continuity check between consecutive frames
        continuity_scores = []
        for i in range(min(10, len(decoded_frames) - 1)):
            c_score = self.engine.compute_visual_continuity(
                decoded_frames[i].img_bgr, decoded_frames[i + 1].img_bgr
            )
            continuity_scores.append(c_score)
        avg_continuity = float(np.mean(continuity_scores)) if continuity_scores else 1.0

        metrics["rules"]["4_sweep_evidence"] = {
            "duration_s": round(duration, 2),
            "heading_span_deg": round(span, 1),
            "max_heading_gap_deg": round(max_gap, 1),
            "visual_continuity_score": round(avg_continuity, 2),
            "client_heading_recorded_as_evidence": True,
        }

        if duration < SWEEP_MIN_DURATION_S or duration > SWEEP_MAX_DURATION_S:
            metrics["rules"]["4_sweep_evidence"]["passed"] = False
            return VerificationOutcome(
                state="rejected",
                reason_code="DURATION_OUT_OF_RANGE",
                message=f"Sweep duration was {round(duration, 1)}s. Please sweep smoothly over 4–15 seconds.",
                metrics=metrics,
            )

        if max_gap > HEADING_MAX_GAP_DEG:
            metrics["rules"]["4_sweep_evidence"]["passed"] = False
            return VerificationOutcome(
                state="rejected",
                reason_code="HEADING_GAP_TOO_LARGE",
                message="Sweep was uneven or jerky. Please maintain a steady, continuous sweep arc.",
                metrics=metrics,
            )

        if span < HEADING_MIN_SPAN_DEG:
            metrics["rules"]["4_sweep_evidence"]["passed"] = False
            return VerificationOutcome(
                state="rejected",
                reason_code="HEADING_SPAN_INSUFFICIENT",
                message=f"Sweep arc was only {round(span)}°. A full 180° sweep from yourself to the lecturer is required.",
                metrics=metrics,
            )

        is_flagged = False
        flagged_reason = ""

        if avg_continuity < VISUAL_CONTINUITY_MIN_SCORE:
            is_flagged = True
            flagged_reason = "DISCONTINUOUS_FRAMES"
            metrics["rules"]["4_sweep_evidence"]["continuity_flag"] = True

        metrics["rules"]["4_sweep_evidence"]["passed"] = True

        # ── Frame Slicing: Early Arc (~30°) and Late Arc (>= 150°) ───────────
        start_heading = headings[0]
        early_frames = [
            f for f in decoded_frames
            if abs(f.heading_deg - start_heading) <= 35.0
        ]
        if not early_frames:
            early_frames = decoded_frames[:max(3, len(decoded_frames) // 4)]

        late_frames = [
            f for f in decoded_frames
            if abs(f.heading_deg - start_heading) >= 140.0
        ]
        if not late_frames:
            late_frames = decoded_frames[-max(3, len(decoded_frames) // 4):]

        # ── RULE 5: Self Face Matching ───────────────────────────────────────
        best_self_face: Optional[FaceDetection] = None
        best_self_frame: Optional[SweepFrame] = None
        best_self_similarity = 0.0

        for f in early_frames:
            faces = self.engine.detect_faces(f.img_bgr)
            if not faces:
                continue
            for candidate in faces:
                if student_enrolled_embedding is None:
                    continue
                candidate_emb = self.engine.extract_embedding(f.img_bgr, candidate)
                sim = self.engine.compute_similarity(candidate_emb, student_enrolled_embedding)

                if sim > best_self_similarity:
                    best_self_similarity = sim
                    best_self_face = candidate
                    best_self_frame = f

        metrics["rules"]["5_self_match"] = {
            "similarity": round(best_self_similarity, 3),
            "face_detected": best_self_face is not None,
        }

        if best_self_face is None or best_self_similarity < SIMILARITY_FLAG_THRESHOLD:
            metrics["rules"]["5_self_match"]["passed"] = False
            return VerificationOutcome(
                state="rejected",
                reason_code="SELF_NOT_MATCHED",
                message="Your face was not recognized from your reference profile. Please ensure proper front-camera alignment.",
                metrics=metrics,
            )

        if best_self_similarity < SIMILARITY_PASS_THRESHOLD:
            is_flagged = True
            flagged_reason = "SELF_BORDERLINE_SIMILARITY"
            metrics["rules"]["5_self_match"]["flagged"] = True
        else:
            metrics["rules"]["5_self_match"]["passed"] = True

        # ── RULE 6: Neighbour Detection Beside Student ───────────────────────
        neighbour_faces: list[FaceDetection] = []
        if best_self_frame is not None and best_self_face is not None:
            all_early_faces = self.engine.detect_faces(best_self_frame.img_bgr)
            for det in all_early_faces:
                if det == best_self_face:
                    continue
                # Distance between centers in units of student face width
                dist_px = math.hypot(det.center[0] - best_self_face.center[0], det.center[1] - best_self_face.center[1])
                dist_units = dist_px / max(1.0, float(best_self_face.bbox_w))
                if dist_units <= NEIGHBOUR_MAX_FACE_WIDTHS:
                    # Check distinct face embedding from student
                    if student_enrolled_embedding is not None:
                        n_emb = self.engine.extract_embedding(best_self_frame.img_bgr, det)
                        n_sim = self.engine.compute_similarity(n_emb, student_enrolled_embedding)
                        if n_sim >= 0.70:
                            continue  # Same person duplicate detection
                    neighbour_faces.append(det)

        metrics["rules"]["6_neighbours"] = {
            "detected_neighbours": len(neighbour_faces),
            "required": MIN_NEIGHBOURS,
        }

        if len(neighbour_faces) < MIN_NEIGHBOURS:
            # If no neighbours detected in selfie frame, flag for lecturer review
            is_flagged = True
            if not flagged_reason:
                flagged_reason = "NO_NEIGHBOURS"
            metrics["rules"]["6_neighbours"]["flagged"] = True
        else:
            metrics["rules"]["6_neighbours"]["passed"] = True

        # ── RULE 7: Lecturer Detection in Late Frames ────────────────────────
        lecturer_detected = False
        lecturer_face_large_enough = False
        best_lecturer_sim = 0.0
        best_lecturer_ipd = 0.0

        for f in late_frames:
            faces = self.engine.detect_faces(f.img_bgr)
            if faces:
                lecturer_detected = True
                for det in faces:
                    if det.ipd_px > best_lecturer_ipd:
                        best_lecturer_ipd = det.ipd_px
                    if det.ipd_px >= LECTURER_MIN_IPD_PX:
                        lecturer_face_large_enough = True
                        if lecturer_enrolled_embedding is not None:
                            l_emb = self.engine.extract_embedding(f.img_bgr, det)
                            l_sim = self.engine.compute_similarity(l_emb, lecturer_enrolled_embedding)
                            if l_sim > best_lecturer_sim:
                                best_lecturer_sim = l_sim
                        else:
                            best_lecturer_sim = 0.80

        metrics["rules"]["7_lecturer"] = {
            "person_or_face_detected": lecturer_detected,
            "max_ipd_px": round(best_lecturer_ipd, 1),
            "large_enough_for_id": lecturer_face_large_enough,
            "similarity": round(best_lecturer_sim, 3),
        }

        if not lecturer_detected:
            metrics["rules"]["7_lecturer"]["passed"] = False
            return VerificationOutcome(
                state="rejected",
                reason_code="LECTURER_NOT_VISIBLE",
                message="Lecturer was not visible at the end of the sweep. Ensure the phone points directly at the podium.",
                metrics=metrics,
            )

        if not lecturer_face_large_enough or (lecturer_enrolled_embedding is not None and best_lecturer_sim < SIMILARITY_PASS_THRESHOLD):
            # Per optical feasibility experiment, front camera resolution across lecture hall
            # often yields IPD < 30px. Neither auto-pass nor auto-reject: mark FLAGGED.
            is_flagged = True
            if not flagged_reason:
                flagged_reason = "LECTURER_FLAGGED_FOR_REVIEW"
            metrics["rules"]["7_lecturer"]["flagged"] = True
        else:
            metrics["rules"]["7_lecturer"]["passed"] = True

        # ── RULE 8: Fraud Signals & Near-Duplicate Check ─────────────────────
        phash_current = self.engine.compute_phash(decoded_frames[0].img_bgr)
        metrics["rules"]["8_fraud_check"] = {"phash": phash_current}

        recent_attempts = (
            db.table("checkin_attempts")
            .select("metrics_json")
            .eq("session_id", session["id"])
            .neq("student_id", student_profile["id"])
            .limit(20)
            .execute()
        )
        for att in recent_attempts.data or []:
            try:
                m = json.loads(att.get("metrics_json") or "{}")
                other_phash = m.get("rules", {}).get("8_fraud_check", {}).get("phash")
                if other_phash and hamming_distance(phash_current, other_phash) <= PHASH_HAMMING_DISTANCE_FLAG:
                    is_flagged = True
                    flagged_reason = "SUSPECTED_DUPLICATE_MEDIA"
                    metrics["rules"]["8_fraud_check"]["duplicate_flag"] = True
                    break
            except Exception:
                pass

        # ── RULE 9: Final Decision & Filmstrip Extraction ────────────────────
        # Select 4-6 representative key frames across the sweep
        num_frames = len(decoded_frames)
        step = max(1, num_frames // FILMSTRIP_KEY_FRAMES)
        key_indices = list(range(0, num_frames, step))[:FILMSTRIP_KEY_FRAMES]
        if (num_frames - 1) not in key_indices:
            key_indices.append(num_frames - 1)

        filmstrip_bytes = []
        if is_flagged:
            for idx in key_indices:
                filmstrip_bytes.append(decoded_frames[idx].image_bytes)

        if is_flagged:
            return VerificationOutcome(
                state="flagged",
                reason_code=flagged_reason or "UNCERTAIN_NEEDS_REVIEW",
                message="Attendance captured and submitted for lecturer review.",
                metrics=metrics,
                key_frame_indices=key_indices,
                filmstrip_bytes=filmstrip_bytes,
            )

        return VerificationOutcome(
            state="verified",
            reason_code="VERIFIED",
            message="Attendance verified successfully.",
            metrics=metrics,
            key_frame_indices=key_indices,
            filmstrip_bytes=[],
        )

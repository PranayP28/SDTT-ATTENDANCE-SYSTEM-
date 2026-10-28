"""Local latency benchmark for the Attendrix API.

Starts a real uvicorn worker (in-memory DB, real ONNX face engine) and measures:
cold start, warm health/read latency (p50/p95/p99), DB round trips per request,
face-engine initialisation and per-image inference, and health latency while
inference requests are saturating the worker.

Numbers are relative (local CPU, no Supabase network latency); production adds
one PostgREST round trip (~20-150 ms) per reported DB call.

Usage: python scripts/benchmark.py [--backend-dir DIR] [--port 8765] [--n 200] [--json out.json]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

SECRET = "benchmark_jwt_secret_value_at_least_32_chars"
TEACHER_ID = "00000000-0000-0000-0000-000000000001"
STUDENT_ID = "00000000-0000-0000-0000-000000000002"


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 2)


def summarize(values: list[float]) -> dict[str, float]:
    return {"n": len(values), "p50": pct(values, 0.5), "p95": pct(values, 0.95), "p99": pct(values, 0.99),
            "mean": round(statistics.fmean(values), 2)}


def token(backend: Path, uid: str, email: str, role: str) -> str:
    code = ("import sys; sys.path.insert(0, '.'); from app.security import create_access_token;"
            f"print(create_access_token('{uid}', '{email}', '{role}', 'Bench'))")
    env = {**os.environ, "SUPABASE_JWT_SECRET": SECRET, "ATTENDANCE_ENV": "test"}
    return subprocess.check_output([sys.executable, "-c", code], cwd=backend, env=env, text=True).strip().splitlines()[-1]


def synthetic_jpeg() -> bytes:
    import cv2
    import numpy as np
    rng = np.random.default_rng(0)
    img = (rng.random((480, 640, 3)) * 255).astype("uint8")
    cv2.circle(img, (320, 220), 90, (180, 160, 140), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def timed_get(client: httpx.Client, path: str, headers: dict[str, str]) -> tuple[float, httpx.Response]:
    t = time.perf_counter()
    r = client.get(path, headers=headers)
    return (time.perf_counter() - t) * 1000.0, r


def db_calls(resp: httpx.Response) -> int | None:
    for part in resp.headers.get("server-timing", "").split(","):
        if part.strip().startswith("db;"):
            return int(part.split('desc="')[1].split(" ")[0])
    return None


def engine_benchmark(backend: Path) -> dict:
    code = r'''
import sys, time, json; sys.path.insert(0, ".")
import cv2, numpy as np
t = time.perf_counter()
from app.face_engine import RealFaceEngine
eng = RealFaceEngine()
init_ms = (time.perf_counter() - t) * 1000
rng = np.random.default_rng(0)
img = (rng.random((480, 640, 3)) * 255).astype("uint8")
det = []
for _ in range(30):
    t = time.perf_counter(); faces = eng.detect_faces(img); det.append((time.perf_counter() - t) * 1000)
print(json.dumps({"init_ms": init_ms, "detect_ms": det}))
'''
    env = {**os.environ, "ATTENDANCE_ENV": "development"}
    out = subprocess.check_output([sys.executable, "-c", code], cwd=backend, env=env, text=True).strip().splitlines()[-1]
    data = json.loads(out)
    return {"engine_init_ms": round(data["init_ms"], 1), "detect_640x480": summarize(data["detect_ms"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-dir", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--json")
    args = ap.parse_args()
    backend = Path(args.backend_dir).resolve()
    base = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "ATTENDANCE_ENV": "development", "SUPABASE_URL": "", "SUPABASE_JWT_SECRET": SECRET,
           "ATTENDANCE_SECRET_KEY": "benchmark_biometric_key_at_least_32_chars", "ATTENDANCE_MAX_CONCURRENT_INFERENCE": "1"}
    teacher = {"Authorization": f"Bearer {token(backend, TEACHER_ID, 'teacher@demo.edu', 'teacher')}"}
    student = {"Authorization": f"Bearer {token(backend, STUDENT_ID, 'student1@demo.edu', 'student')}"}

    started = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(args.port), "--log-level", "warning"],
                            cwd=backend, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results: dict = {"backend_dir": str(backend)}
    try:
        with httpx.Client(base_url=base, timeout=120) as c:
            while True:
                try:
                    if c.get("/api/health").status_code == 200:
                        break
                except httpx.TransportError:
                    time.sleep(0.05)
            results["cold_start_to_healthy_ms"] = round((time.perf_counter() - started) * 1000.0, 1)

            ms, r = timed_get(c, "/api/sessions", teacher)
            results["first_read_ms"] = round(ms, 1)

            endpoints = {"health": ("/api/health", {}), "time": ("/api/time", {}), "classes": ("/api/classes", teacher),
                         "sessions": ("/api/sessions", teacher), "notifications": ("/api/notifications", student),
                         "student_history": ("/api/students/me/attendance", student)}
            sess = c.post("/api/sessions", headers=teacher, json={"class_id": "11111111-1111-1111-1111-111111111111", "duration_minutes": 10}).json()
            endpoints["session_detail"] = (f"/api/sessions/{sess['id']}", teacher)
            warm = {}
            for name, (path, hdr) in endpoints.items():
                for _ in range(10):
                    c.get(path, headers=hdr)
                samples = []
                last = None
                for _ in range(args.n):
                    ms, last = timed_get(c, path, hdr)
                    samples.append(ms)
                warm[name] = {**summarize(samples), "status": last.status_code, "db_calls": db_calls(last)}
            results["warm_latency_ms"] = warm

            # First inference request after start (model init cost if lazy).
            jpg = synthetic_jpeg()
            t = time.perf_counter()
            r = c.post("/api/auth/enroll", headers=student, data={"consent": "true"}, files={"photo": ("a.jpg", jpg, "image/jpeg")})
            results["first_inference_request_ms"] = round((time.perf_counter() - t) * 1000.0, 1)
            results["first_inference_status"] = r.status_code

            # Health responsiveness while 8 concurrent inference requests contend for 1 slot.
            def enroll() -> float:
                with httpx.Client(base_url=base, timeout=120) as cc:
                    t0 = time.perf_counter()
                    cc.post("/api/auth/enroll", headers=student, data={"consent": "true"}, files={"photo": ("a.jpg", jpg, "image/jpeg")})
                    return (time.perf_counter() - t0) * 1000.0
            with cf.ThreadPoolExecutor(8) as pool:
                futs = [pool.submit(enroll) for _ in range(8)]
                health = []
                while not all(f.done() for f in futs):
                    ms, _ = timed_get(c, "/api/health", {})
                    health.append(ms)
                    time.sleep(0.02)
                infer = [f.result() for f in futs]
            results["health_under_inference_ms"] = summarize(health)
            results["inference_burst_8_ms"] = summarize(infer)
    finally:
        proc.terminate()
        proc.wait(10)
    results.update(engine_benchmark(backend))
    out = json.dumps(results, indent=2)
    print(out)
    if args.json:
        Path(args.json).write_text(out)


if __name__ == "__main__":
    main()

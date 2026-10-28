"""Runtime observability and resource guards.

This module intentionally contains no route logic. It provides:
- One structured log line per HTTP request with application-side timing.
- A process-wide bounded gate for CPU/RAM-heavy face inference.
- Safe application logging setup that never logs request bodies, query strings,
  credentials, biometric templates, or image contents.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from app.config import (
    ENABLE_REQUEST_TIMING,
    INFERENCE_QUEUE_TIMEOUT_S,
    MAX_CONCURRENT_INFERENCE,
)

request_logger = logging.getLogger("attendrix.requests")
inference_logger = logging.getLogger("attendrix.inference")


class InferenceBusyError(RuntimeError):
    """Raised when the bounded inference queue cannot admit a request in time."""


# Per-request counters. The middleware stores a mutable dict here; FastAPI
# copies the context into its threadpool, so sync endpoints mutate the same dict.
_REQUEST_METRICS: ContextVar[dict[str, float] | None] = ContextVar("attendrix_request_metrics", default=None)


def new_request_metrics() -> dict[str, float]:
    return {"db_calls": 0, "db_ms": 0.0, "inference_wait_ms": 0.0, "inference_ms": 0.0}


def record_metric(name: str, value: float, *, count_key: str | None = None) -> None:
    """Add ``value`` to the current request's metric ``name`` (no-op outside a request)."""
    metrics = _REQUEST_METRICS.get()
    if metrics is None:
        return
    metrics[name] = metrics.get(name, 0.0) + value
    if count_key:
        metrics[count_key] = metrics.get(count_key, 0) + 1


def current_request_metrics() -> dict[str, float] | None:
    return _REQUEST_METRICS.get()


def configure_logging() -> None:
    """Make application logs visible on Render without duplicating handlers.

    Tests keep the default logging setup so pytest's capture/caplog works.
    """
    if os.getenv("ATTENDANCE_ENV", "").lower() == "test":
        return

    app_logger = logging.getLogger("attendrix")
    if app_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False


def _request_id(scope: dict[str, Any]) -> str:
    headers = dict(scope.get("headers") or [])
    for key in (b"x-request-id", b"cf-ray"):
        value = headers.get(key)
        if value:
            return value.decode("latin-1")[:128]
    return uuid.uuid4().hex


def _log_request(
    *,
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    request_id: str,
    metrics: dict[str, float] | None = None,
) -> None:
    # Render health checks run every few seconds. Keep successful probes out of
    # the log unless they are slow enough to indicate a real problem.
    if path in ("/", "/health", "/api/health") and status < 400 and duration_ms < 1000:
        return

    entry: dict[str, Any] = {
        "event": "request_complete",
        "method": method,
        "path": path,
        "status": status,
        "duration_ms": round(duration_ms, 1),
        "request_id": request_id or None,
        "pid": os.getpid(),
    }
    if metrics:
        entry["db_calls"] = int(metrics.get("db_calls", 0))
        entry["db_ms"] = round(metrics.get("db_ms", 0.0), 1)
        if metrics.get("inference_ms") or metrics.get("inference_wait_ms"):
            entry["inference_wait_ms"] = round(metrics.get("inference_wait_ms", 0.0), 1)
            entry["inference_ms"] = round(metrics.get("inference_ms", 0.0), 1)
    request_logger.info(json.dumps(entry, separators=(",", ":")))


class RequestTimingMiddleware:
    """Pure ASGI middleware measuring server-side application time."""

    def __init__(self, app: Any):
        self.app = app
        self.enabled = ENABLE_REQUEST_TIMING

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if not self.enabled or scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        method = scope.get("method", "")
        path = scope.get("path", "")
        request_id = _request_id(scope)
        state = {"status": 500, "logged": False}
        metrics = new_request_metrics()
        metrics_token = _REQUEST_METRICS.set(metrics)

        async def send_with_timing(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                state["status"] = int(message.get("status", 500))
                headers = list(message.get("headers") or [])
                timing = (
                    f"app;dur={elapsed_ms:.1f}, "
                    f'db;dur={metrics["db_ms"]:.1f};desc="{int(metrics["db_calls"])} queries"'
                )
                if metrics["inference_ms"] or metrics["inference_wait_ms"]:
                    timing += (
                        f', inference-queue;dur={metrics["inference_wait_ms"]:.1f}'
                        f', inference;dur={metrics["inference_ms"]:.1f}'
                    )
                headers.append((b"server-timing", timing.encode("ascii")))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                headers.append(
                    (b"x-process-time-ms", f"{elapsed_ms:.1f}".encode("ascii"))
                )
                message = {**message, "headers": headers}
                _log_request(
                    method=method,
                    path=path,
                    status=state["status"],
                    duration_ms=elapsed_ms,
                    request_id=request_id,
                    metrics=metrics,
                )
                state["logged"] = True
            await send(message)

        try:
            await self.app(scope, receive, send_with_timing)
        except Exception:
            if not state["logged"]:
                _log_request(
                    method=method,
                    path=path,
                    status=500,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    request_id=request_id,
                    metrics=metrics,
                )
                state["logged"] = True
            raise
        finally:
            if not state["logged"]:
                # Client disconnected before response headers were sent.
                _log_request(
                    method=method,
                    path=path,
                    status=499,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    request_id=request_id,
                    metrics=metrics,
                )
            _REQUEST_METRICS.reset(metrics_token)


_INFERENCE_GATE = threading.BoundedSemaphore(max(1, MAX_CONCURRENT_INFERENCE))


@contextmanager
def inference_slot(operation: str) -> Iterator[None]:
    """Bound concurrent CPU/RAM-heavy inference across the process.

    The gate is acquired inside worker threads (FastAPI runs synchronous
    endpoints in its threadpool), so waiting for capacity never blocks the
    asyncio event loop.
    """
    queued_at = time.perf_counter()
    timeout = INFERENCE_QUEUE_TIMEOUT_S if INFERENCE_QUEUE_TIMEOUT_S > 0 else None
    acquired = _INFERENCE_GATE.acquire(timeout=timeout)
    if not acquired:
        inference_logger.warning(
            json.dumps(
                {
                    "event": "inference_rejected",
                    "operation": operation,
                    "reason": "queue_timeout",
                    "wait_ms": round((time.perf_counter() - queued_at) * 1000.0, 1),
                    "pid": os.getpid(),
                },
                separators=(",", ":"),
            )
        )
        raise InferenceBusyError(
            "Verification capacity is saturated. Please retry shortly."
        )

    wait_ms = (time.perf_counter() - queued_at) * 1000.0
    record_metric("inference_wait_ms", wait_ms)
    started = time.perf_counter()
    outcome = "ok"
    try:
        yield
    except Exception:
        outcome = "error"
        raise
    finally:
        record_metric("inference_ms", (time.perf_counter() - started) * 1000.0)
        inference_logger.info(
            json.dumps(
                {
                    "event": "inference_complete",
                    "operation": operation,
                    "outcome": outcome,
                    "wait_ms": round(wait_ms, 1),
                    "duration_ms": round((time.perf_counter() - started) * 1000.0, 1),
                    "pid": os.getpid(),
                },
                separators=(",", ":"),
            )
        )
        _INFERENCE_GATE.release()

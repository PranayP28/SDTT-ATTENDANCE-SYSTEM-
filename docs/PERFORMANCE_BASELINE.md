# Performance baseline (measured)

Measured with `backend/scripts/benchmark.py` on the dev VM (2 vCPU, 8 GB RAM, Python 3.11, in-memory DB, real YuNet/SFace ONNX engine, 1 uvicorn worker, `ATTENDANCE_MAX_CONCURRENT_INFERENCE=1`). Before = `main` at the merge of PR #1, after = this branch. 200 warm samples per endpoint after 10 warm-up calls.

**Read these as relative numbers.** The in-memory DB has no network cost. On Render with Supabase, every DB call listed below is one PostgREST HTTPS round trip, roughly 20–150 ms depending on region. That makes **DB calls per request**, not CPU, the main lever for API latency. Each response now reports it in `Server-Timing: db;dur=…;desc="N queries"`.

| Metric | Before | After |
| --- | --- | --- |
| Cold start → `/api/health` 200 | 568.3 ms | 569.1 ms |
| First inference request after boot | 179.2 ms | 94.8 ms (models preloaded in background) |
| Face engine init (YuNet + SFace) | 110.5 ms | 130.0 ms (now paid once at startup, off the request path) |
| YuNet detection, 640×480, p50 / p95 | 27.8 / 33.78 ms | 34.32 / 40.31 ms |
| `/api/health` p95 while inference requests are queued | 14.28 ms | 4.59 ms |

| Endpoint | Before p50 / p95 / p99 (ms) | After p50 / p95 / p99 (ms) | DB calls (after) |
| --- | --- | --- | --- |
| `health` | 0.43 / 0.73 / 0.9 | 0.61 / 0.75 / 0.83 | 0 |
| `time` | 0.69 / 0.83 / 0.92 | 0.72 / 0.86 / 0.99 | 0 |
| `classes` | 1.11 / 1.48 / 2.81 | 1.2 / 1.73 / 3.84 | 2 |
| `sessions` | 1.15 / 1.75 / 3.4 | 1.28 / 1.75 / 2.04 | 3 |
| `notifications` | 1.28 / 2.04 / 3.53 | 1.31 / 1.86 / 2.57 | 2 |
| `student_history` | 1.24 / 1.68 / 2.3 | 1.29 / 1.87 / 4.65 | 2 |
| `session_detail` | 1.33 / 1.77 / 3.96 | 1.29 / 1.73 / 2.13 | 4 |

`session_detail` went from 5 DB calls to 4: the verified and flagged counts now come from a single query. Every authenticated request still spends 1 call loading the caller's profile. That call is kept on purpose, so role changes and deactivations take effect immediately.

## Known gaps in this measurement
- The burst scenario re-submits `/api/auth/enroll`. After the first enrollment the account is locked and later requests are rejected before inference, so `inference_burst_8_ms` does **not** measure queued inference. The health-under-load figure only shows that the event loop stays responsive. A sweep check-in burst against synthetic identities is still to do.
- Cold-start numbers exclude Render's free-tier wake-up (container spin-up after ~15 min idle), which is typically tens of seconds and outside the app's control.
- The figures in `proof/PERFORMANCE_RESULTS.md` (SQLite/WAL, 48 ms end-to-end) were not reproduced here and do not describe the current Supabase architecture.

Reproduce: `cd backend && . .venv/bin/activate && python scripts/benchmark.py --json /tmp/bench.json`

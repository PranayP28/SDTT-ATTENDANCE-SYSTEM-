-- 003: indexes for the hottest read paths (idempotent, safe to re-run).
-- Each index matches a filter/order used by backend/app/main.py:
--   sessions list / analytics:  sessions.class_id + created_at desc, sessions.status = 'open' (auto-close loop)
--   class access checks:        enrollments.student_id, classes.teacher_id
--   live session counts:        checkin_attempts(session_id, state)
--   audit feed:                 audit_log(timestamp desc)
create index if not exists idx_sessions_class_created on public.sessions(class_id, created_at desc);
create index if not exists idx_sessions_open          on public.sessions(status) where status = 'open';
create index if not exists idx_enrollments_student    on public.enrollments(student_id);
create index if not exists idx_classes_teacher        on public.classes(teacher_id);
create index if not exists idx_checkin_session_state  on public.checkin_attempts(session_id, state);
create index if not exists idx_audit_timestamp        on public.audit_log("timestamp" desc);

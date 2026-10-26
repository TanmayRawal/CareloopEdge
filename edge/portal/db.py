"""Portal database (SQLite, WAL, foreign keys, versioned migrations). Local disk only — never a network or
host-shared filesystem (WAL locking does not survive those; docs/FINDINGS.md §8).

Case lifecycle
  awaiting_review    patient uploaded; AI result attached (or AI could not analyse the image)
  in_review          a doctor opened the case
  reupload_requested doctor asked the patient for a better image (patient can upload to the same case)
  finalized          doctor signed a report (amendments add report versions; nothing is overwritten)

Appointment lifecycle (consultation with the reporting radiologist, booked by the patient after the report)
  booked -> arrived -> completed       marked by the doctor
  booked | arrived -> no_show          marked by the doctor (patient gets a "we missed you" message)
  booked -> cancelled                  by the patient (before the cutoff) or the doctor

Follow-up (CareLoop): a care plan is the next step after a report (consultation / repeat scan / referral).
  proposed -> approved                 the doctor approves every plan (plans made by a doctor start approved)
  approved -> completed | lost | cancelled
The administrator acts as the care coordinator: works the follow-up worklist, logs contacts, sends reminders
(only on approved plans, always triggered by a person), registers walk-in patients and tracks referrals.
Referral: draft -> signed (doctor) -> sent -> appointment_made -> attended -> closed

Audit log: every row carries hash = SHA-256(previous hash, row), so editing or deleting a past row breaks the
chain (verify_audit_chain).
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path

STATUSES = ("awaiting_review", "in_review", "reupload_requested", "finalized")
ROLES = ("patient", "doctor", "admin")
APPT_STATUSES = ("booked", "arrived", "completed", "no_show", "cancelled")
APPT_ACTIVE = ("booked", "arrived")

# Department policy; editable by the admin (Settings page) and stored in the `settings` table.
DEFAULT_SETTINGS = {
    "clinic_name": "Radiology Department",
    "clinic_address": "",                       # letterhead of reports and e-mails (optional)
    "clinic_phone": "",
    "base_url": "https://rpi.local",            # used in e-mail links
    "timezone": "Asia/Kolkata",
    "unread_hours": "4",                        # study not opened -> remind doctors
    "unread_hours_urgent": "1",                 # same, for urgent / attention studies
    "unsigned_hours": "24",                     # opened but not signed -> remind that doctor
    "escalate_hours": "48",                     # still unread -> also e-mail administrators
    "remind_before_hours": "24,2",              # appointment reminders to the patient
    "unmarked_after_min": "60",                 # attendance not marked -> remind the doctor
    "agenda_hour": "7:30",                      # daily agenda e-mail to doctors (local time)
    "booking_lead_hours": "2",
    "booking_horizon_days": "21",
    "cancel_cutoff_hours": "2",
    "followup_due_soon_days": "7",              # follow-up worklist: "due this week"
    "referral_stale_days": "14",                # referral not closed after N days -> admin digest
    "lost_after_days": "60",                    # approved plan overdue this long with no contact -> lost
    "require_staff_2fa": "0",                   # 1 = doctors and admins must set up two-step sign-in
    "upload_device_only": "1",                  # 1 = patients sign up and upload scans only at the device's own screen
}
PLAN_KINDS = ("consultation", "repeat_scan", "referral", "other")
PLAN_STATUSES = ("proposed", "approved", "completed", "lost", "cancelled")
PLAN_OPEN = ("proposed", "approved")
REFERRAL_STATUSES = ("draft", "signed", "sent", "appointment_made", "attended", "closed")
CONTACT_CHANNELS = ("phone", "in_person", "email", "portal")
CONTACT_OUTCOMES = ("reached_confirmed", "reached_declined", "no_answer", "wrong_number", "message_sent")
DOC_KINDS = ("prior_report", "biopsy", "referral_letter", "dicom", "other")
CONSENT_VERSION = "v1"
AUDIT_GENESIS = "0" * 64
DIAGNOSES = ("glioma", "meningioma", "pituitary", "no_tumor", "other", "inconclusive")

MIGRATIONS = [
    # v1
    """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        email TEXT NOT NULL UNIQUE COLLATE NOCASE,
        name TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('patient','doctor','admin')),
        pw_hash TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        failed_logins INTEGER NOT NULL DEFAULT 0,
        locked_until REAL,
        created REAL NOT NULL
    );
    CREATE TABLE patient_profiles (
        user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        dob TEXT, sex TEXT, phone TEXT
    );
    CREATE TABLE cases (
        id TEXT PRIMARY KEY,
        patient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        status TEXT NOT NULL CHECK (status IN ('awaiting_review','in_review','reupload_requested','finalized')),
        created REAL NOT NULL, updated REAL NOT NULL,
        scan_filename TEXT NOT NULL, scan_sha256 TEXT NOT NULL, scan_size INTEGER NOT NULL,
        consent INTEGER NOT NULL CHECK (consent = 1),
        patient_note TEXT,
        doctor_id INTEGER REFERENCES users(id),
        doctor_message TEXT,
        priority INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX cases_patient ON cases(patient_id);
    CREATE INDEX cases_status ON cases(status, priority DESC, created);
    CREATE TABLE ai_results (
        id INTEGER PRIMARY KEY,
        case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
        model_version TEXT NOT NULL,
        created REAL NOT NULL,
        result_json TEXT NOT NULL,
        overlay_kinds TEXT NOT NULL
    );
    CREATE INDEX ai_case ON ai_results(case_id, created);
    CREATE TABLE reports (
        id INTEGER PRIMARY KEY,
        case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
        version INTEGER NOT NULL,
        doctor_id INTEGER NOT NULL REFERENCES users(id),
        final_diagnosis TEXT NOT NULL,
        agrees_with_ai INTEGER,
        findings TEXT NOT NULL,
        recommendation TEXT,
        ai_result_id INTEGER REFERENCES ai_results(id),
        signed_at REAL NOT NULL,
        amendment_reason TEXT,
        snapshot_sha256 TEXT NOT NULL,
        UNIQUE (case_id, version)
    );
    CREATE TABLE audit (
        id INTEGER PRIMARY KEY,
        ts REAL NOT NULL, user_id INTEGER, action TEXT NOT NULL, case_id TEXT, detail TEXT, ip TEXT
    );
    CREATE INDEX audit_ts ON audit(ts);
    """,
    # v2: appointments, doctor availability, notifications, e-mail outbox, settings
    """
    ALTER TABLE users ADD COLUMN notify_email INTEGER NOT NULL DEFAULT 1;
    CREATE TABLE doctor_hours (
        id INTEGER PRIMARY KEY,
        doctor_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        weekday INTEGER NOT NULL CHECK (weekday BETWEEN 0 AND 6),
        start_min INTEGER NOT NULL CHECK (start_min BETWEEN 0 AND 1439),
        end_min INTEGER NOT NULL CHECK (end_min BETWEEN 1 AND 1440),
        slot_min INTEGER NOT NULL DEFAULT 20 CHECK (slot_min BETWEEN 5 AND 120),
        room TEXT,
        CHECK (end_min > start_min)
    );
    CREATE INDEX hours_doctor ON doctor_hours(doctor_id, weekday);
    CREATE TABLE doctor_days_off (
        doctor_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        day TEXT NOT NULL, note TEXT,
        PRIMARY KEY (doctor_id, day)
    );
    CREATE TABLE appointments (
        id INTEGER PRIMARY KEY,
        case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
        patient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        doctor_id INTEGER NOT NULL REFERENCES users(id),
        start_ts REAL NOT NULL, end_ts REAL NOT NULL,
        room TEXT, reason TEXT,
        status TEXT NOT NULL CHECK (status IN ('booked','arrived','completed','no_show','cancelled')),
        created REAL NOT NULL, updated REAL NOT NULL,
        cancelled_by INTEGER REFERENCES users(id), marked_by INTEGER REFERENCES users(id), marked_at REAL
    );
    CREATE UNIQUE INDEX appt_slot ON appointments(doctor_id, start_ts) WHERE status IN ('booked','arrived');
    CREATE INDEX appt_patient ON appointments(patient_id, start_ts);
    CREATE INDEX appt_doctor ON appointments(doctor_id, start_ts);
    CREATE TABLE notifications (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        kind TEXT NOT NULL, title TEXT NOT NULL, body TEXT, link TEXT,
        created REAL NOT NULL, read_at REAL
    );
    CREATE INDEX notif_user ON notifications(user_id, read_at, created);
    CREATE TABLE outbox (
        id INTEGER PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
        to_email TEXT NOT NULL, subject TEXT NOT NULL, text TEXT NOT NULL, html TEXT, ics TEXT,
        status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','sent','failed','disabled')),
        created REAL NOT NULL, next_try REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT, sent_at REAL,
        dedupe_key TEXT UNIQUE
    );
    CREATE INDEX outbox_due ON outbox(status, next_try);
    CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE reminder_log (dedupe_key TEXT PRIMARY KEY, created REAL NOT NULL);
    CREATE INDEX cases_doctor ON cases(doctor_id, status)
    """,
    # v3: CareLoop follow-up (care plans, referrals, contact log, documents, consents, one-time tokens), walk-in
    # patients without a sign-in (email / password optional), staff two-step sign-in, hash-chained audit log
    """
    CREATE TABLE users_v3 (
        id INTEGER PRIMARY KEY,
        email TEXT COLLATE NOCASE,
        name TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('patient','doctor','admin')),
        pw_hash TEXT,
        active INTEGER NOT NULL DEFAULT 1,
        failed_logins INTEGER NOT NULL DEFAULT 0,
        locked_until REAL,
        created REAL NOT NULL,
        notify_email INTEGER NOT NULL DEFAULT 1,
        totp_secret TEXT, totp_on INTEGER NOT NULL DEFAULT 0, totp_last_step INTEGER, recovery_hashes TEXT
    );
    INSERT INTO users_v3 (id,email,name,role,pw_hash,active,failed_logins,locked_until,created,notify_email)
        SELECT id,email,name,role,pw_hash,active,failed_logins,locked_until,created,notify_email FROM users;
    DROP TABLE users;
    ALTER TABLE users_v3 RENAME TO users;
    CREATE UNIQUE INDEX users_email ON users(email) WHERE email IS NOT NULL;
    CREATE INDEX users_role ON users(role, name);
    ALTER TABLE patient_profiles ADD COLUMN address TEXT;
    ALTER TABLE patient_profiles ADD COLUMN district TEXT;
    ALTER TABLE patient_profiles ADD COLUMN distance_km REAL;
    ALTER TABLE patient_profiles ADD COLUMN alt_phone TEXT;
    ALTER TABLE patient_profiles ADD COLUMN contact_status TEXT NOT NULL DEFAULT 'ok';
    ALTER TABLE patient_profiles ADD COLUMN created_by INTEGER;
    CREATE TABLE care_plans (
        id INTEGER PRIMARY KEY,
        patient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        case_id TEXT REFERENCES cases(id) ON DELETE SET NULL,
        kind TEXT NOT NULL CHECK (kind IN ('consultation','repeat_scan','referral','other')),
        due_date TEXT NOT NULL, note TEXT,
        status TEXT NOT NULL CHECK (status IN ('proposed','approved','completed','lost','cancelled')),
        proposed_by INTEGER REFERENCES users(id), proposed_at REAL NOT NULL,
        approved_by INTEGER REFERENCES users(id), approved_at REAL,
        closed_at REAL, close_reason TEXT, updated REAL NOT NULL
    );
    CREATE INDEX plans_patient ON care_plans(patient_id, status);
    CREATE INDEX plans_status ON care_plans(status, due_date);
    CREATE TABLE referrals (
        id INTEGER PRIMARY KEY,
        patient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        case_id TEXT REFERENCES cases(id) ON DELETE SET NULL,
        plan_id INTEGER REFERENCES care_plans(id) ON DELETE SET NULL,
        facility TEXT NOT NULL, specialty TEXT,
        urgency TEXT NOT NULL DEFAULT 'routine' CHECK (urgency IN ('routine','soon','urgent')),
        reason TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('draft','signed','sent','appointment_made','attended','closed')),
        created_by INTEGER REFERENCES users(id), created REAL NOT NULL, updated REAL NOT NULL,
        signed_by INTEGER REFERENCES users(id), signed_at REAL, letter_sha256 TEXT, status_note TEXT
    );
    CREATE INDEX referrals_status ON referrals(status, updated);
    CREATE INDEX referrals_patient ON referrals(patient_id);
    CREATE TABLE contacts (
        id INTEGER PRIMARY KEY,
        patient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        plan_id INTEGER REFERENCES care_plans(id) ON DELETE SET NULL,
        by_user INTEGER REFERENCES users(id),
        channel TEXT NOT NULL CHECK (channel IN ('phone','in_person','email','portal')),
        outcome TEXT NOT NULL CHECK (outcome IN ('reached_confirmed','reached_declined','no_answer','wrong_number','message_sent')),
        note TEXT, next_action_date TEXT, created REAL NOT NULL
    );
    CREATE INDEX contacts_patient ON contacts(patient_id, created);
    CREATE TABLE documents (
        id INTEGER PRIMARY KEY,
        patient_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        uploaded_by INTEGER REFERENCES users(id),
        kind TEXT NOT NULL CHECK (kind IN ('prior_report','biopsy','referral_letter','dicom','other')),
        title TEXT NOT NULL, filename TEXT NOT NULL, mime TEXT NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL,
        created REAL NOT NULL
    );
    CREATE INDEX documents_patient ON documents(patient_id, created);
    CREATE TABLE consents (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        purpose TEXT NOT NULL CHECK (purpose IN ('care','email_reminders')),
        version TEXT NOT NULL, granted_at REAL NOT NULL, withdrawn_at REAL, recorded_by INTEGER REFERENCES users(id)
    );
    CREATE INDEX consents_user ON consents(user_id, purpose);
    CREATE TABLE tokens (
        id INTEGER PRIMARY KEY,
        purpose TEXT NOT NULL CHECK (purpose IN ('invite','reset','appt_yes','appt_no')),
        token_hash TEXT NOT NULL UNIQUE,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        ref_id INTEGER, created REAL NOT NULL, expires REAL NOT NULL, used_at REAL
    );
    CREATE TABLE data_requests (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        kind TEXT NOT NULL CHECK (kind IN ('deletion')),
        note TEXT, created REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','done','rejected')),
        handled_by INTEGER REFERENCES users(id), handled_at REAL
    );
    INSERT INTO consents (user_id, purpose, version, granted_at)
        SELECT id, 'care', 'v1', created FROM users WHERE role = 'patient';
    INSERT INTO consents (user_id, purpose, version, granted_at)
        SELECT id, 'email_reminders', 'v1', created FROM users WHERE role = 'patient' AND notify_email = 1;
    ALTER TABLE audit ADD COLUMN prev_hash TEXT;
    ALTER TABLE audit ADD COLUMN hash TEXT
    """,
    # v5: structured radiology report (indication, technique, comparison, impression) and the reporter's credentials
    """
    ALTER TABLE reports ADD COLUMN indication TEXT;
    ALTER TABLE reports ADD COLUMN technique TEXT;
    ALTER TABLE reports ADD COLUMN comparison TEXT;
    ALTER TABLE reports ADD COLUMN impression TEXT;
    ALTER TABLE users ADD COLUMN credentials TEXT;
    ALTER TABLE users ADD COLUMN reg_no TEXT
    """,
    # v6: signed reports and the audit log are append-only in the database itself (not only by convention); one
    # active appointment per study; e-mail addresses of self-registered patients are confirmed before e-mail is sent
    """
    CREATE TRIGGER reports_immutable_u BEFORE UPDATE ON reports
        BEGIN SELECT RAISE(ABORT, 'signed reports are immutable: sign an amendment instead'); END;
    CREATE TRIGGER reports_immutable_d BEFORE DELETE ON reports
        BEGIN SELECT RAISE(ABORT, 'signed reports are immutable'); END;
    CREATE TRIGGER audit_append_only_u BEFORE UPDATE ON audit WHEN OLD.hash IS NOT NULL
        BEGIN SELECT RAISE(ABORT, 'the audit log is append-only'); END;
    CREATE TRIGGER audit_append_only_d BEFORE DELETE ON audit
        BEGIN SELECT RAISE(ABORT, 'the audit log is append-only'); END;
    UPDATE appointments SET status = 'cancelled', updated = updated WHERE status IN ('booked','arrived') AND id NOT IN
        (SELECT MIN(id) FROM appointments WHERE status IN ('booked','arrived') GROUP BY case_id);
    CREATE UNIQUE INDEX appt_case_active ON appointments(case_id) WHERE status IN ('booked','arrived');
    ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 1;
    CREATE TABLE tokens_v6 (
        id INTEGER PRIMARY KEY,
        purpose TEXT NOT NULL CHECK (purpose IN ('invite','reset','verify','appt_yes','appt_no')),
        token_hash TEXT NOT NULL UNIQUE,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        ref_id INTEGER, created REAL NOT NULL, expires REAL NOT NULL, used_at REAL
    );
    INSERT INTO tokens_v6 SELECT id, purpose, token_hash, user_id, ref_id, created, expires, used_at FROM tokens;
    DROP TABLE tokens;
    ALTER TABLE tokens_v6 RENAME TO tokens
    """,
]


def _statements(sql: str) -> list[str]:
    """Split a migration into statements; a trigger body (BEGIN ...; END) stays one statement."""
    out, buf = [], ""
    for part in sql.split(";"):
        buf += part + ";"
        if buf.strip(" \n;") and sqlite3.complete_statement(buf):
            out.append(buf.strip())
            buf = ""
    return out


def _audit_hash(prev: str, row) -> str:
    """Chain hash of one audit row (id, ts, user, action, case, detail, ip) on top of the previous row's hash."""
    payload = json.dumps([prev, *list(row)[:7]], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=FULL")      # durability over speed: these are medical records
        self._migrate()

    # --------------------------------------------------------------------------- plumbing
    def _migrate(self) -> None:
        with self._lock:
            self.conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self.conn.execute("SELECT version FROM schema_version").fetchone()
            v = row[0] if row else 0
            if v < len(MIGRATIONS):
                # table rebuilds (v3 users) must not cascade-delete child rows; FKs are re-checked afterwards
                self.conn.execute("PRAGMA foreign_keys=OFF")
            for i, sql in enumerate(MIGRATIONS[v:], start=v + 1):
                self.conn.execute("BEGIN")
                try:
                    for stmt in _statements(sql):                       # executescript would auto-commit
                        self.conn.execute(stmt)
                    self.conn.execute("DELETE FROM schema_version")
                    self.conn.execute("INSERT INTO schema_version VALUES (?)", (i,))
                    self.conn.execute("COMMIT")
                except Exception:
                    self.conn.execute("ROLLBACK")
                    self.conn.execute("PRAGMA foreign_keys=ON")
                    raise
            self.conn.execute("PRAGMA foreign_keys=ON")
            bad = self.conn.execute("PRAGMA foreign_key_check").fetchall()
            if bad:
                raise RuntimeError(f"foreign key check failed after migration: {[tuple(b) for b in bad[:3]]}")
            self._backfill_audit_chain()

    def _backfill_audit_chain(self) -> None:
        """Rows written before v3 (or by an older process) get their chain hash, in id order."""
        if self.conn.execute("SELECT 1 FROM audit WHERE hash IS NULL LIMIT 1").fetchone() is None:
            return
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            prev = AUDIT_GENESIS
            for r in self.conn.execute("SELECT id,ts,user_id,action,case_id,detail,ip,hash FROM audit ORDER BY id").fetchall():
                h = r[7] or _audit_hash(prev, r)
                if r[7] is None:
                    self.conn.execute("UPDATE audit SET prev_hash = ?, hash = ? WHERE id = ?", (prev, h, r[0]))
                prev = h
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, args).fetchall()

    def one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        r = self.q(sql, args)
        return r[0] if r else None

    def tx(self):
        """Context manager for a write transaction."""
        db = self

        class _Tx:
            def __enter__(self):
                db._lock.acquire()
                db.conn.execute("BEGIN IMMEDIATE")
                return db.conn

            def __exit__(self, et, *_):
                try:
                    db.conn.execute("ROLLBACK" if et else "COMMIT")
                finally:
                    db._lock.release()
        return _Tx()

    def audit(self, user_id: int | None, action: str, case_id: str | None = None, detail: dict | None = None,
              ip: str | None = None) -> None:
        with self._lock:
            own = not self.conn.in_transaction
            if own:
                self.conn.execute("BEGIN IMMEDIATE")
            try:
                last = self.conn.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
                prev = (last[0] if last else None) or AUDIT_GENESIS
                row = [None, time.time(), user_id, action, case_id, json.dumps(detail or {}), ip]
                row[0] = self.conn.execute("INSERT INTO audit (ts,user_id,action,case_id,detail,ip,prev_hash) "
                                           "VALUES (?,?,?,?,?,?,?)", (*row[1:], prev)).lastrowid
                self.conn.execute("UPDATE audit SET hash = ? WHERE id = ?", (_audit_hash(prev, row), row[0]))
                if own:
                    self.conn.execute("COMMIT")
            except Exception:
                if own:
                    self.conn.execute("ROLLBACK")
                raise

    def verify_audit_chain(self) -> dict:
        """Recompute the chain. ok=False names the first row that was edited, or that follows a deleted row."""
        prev, n = AUDIT_GENESIS, 0
        for r in self.q("SELECT id,ts,user_id,action,case_id,detail,ip,prev_hash,hash FROM audit ORDER BY id"):
            if r["prev_hash"] != prev or r["hash"] != _audit_hash(prev, tuple(r)):
                return {"ok": False, "n": n, "bad_id": r["id"]}
            prev, n = r["hash"], n + 1
        return {"ok": True, "n": n, "bad_id": None}

    def backup(self, dest: Path) -> None:
        """Consistent online snapshot (SQLite backup API) — safe while the portal is running."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        out = sqlite3.connect(dest)
        with self._lock:
            self.conn.backup(out)
        out.close()

    # --------------------------------------------------------------------------- settings
    def settings(self) -> dict:
        out = dict(DEFAULT_SETTINGS)
        out.update({r["key"]: r["value"] for r in self.q("SELECT key, value FROM settings")})
        return out

    def setting(self, key: str) -> str:
        r = self.one("SELECT value FROM settings WHERE key = ?", (key,))
        return r["value"] if r else DEFAULT_SETTINGS[key]

    def set_setting(self, key: str, value: str) -> None:
        if key not in DEFAULT_SETTINGS:
            raise ValueError(f"unknown setting {key}")
        with self.tx() as c:
            c.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                      (key, str(value)))

    def once(self, dedupe_key: str, now: float | None = None) -> bool:
        """True the first time a key is seen (reminder rules fire exactly once per key)."""
        with self.tx() as c:
            cur = c.execute("INSERT OR IGNORE INTO reminder_log (dedupe_key, created) VALUES (?, ?)",
                            (dedupe_key, now or time.time()))
            return cur.rowcount == 1

    # ------------------------------------------------------------------------------ users
    def create_user(self, email: str | None, name: str, role: str, pw_hash: str | None, profile: dict | None = None,
                    created_by: int | None = None, email_consent: bool = True) -> int:
        """Patients also get a profile and consent records. email / pw_hash may be None for a walk-in patient
        registered at the desk (no sign-in until they accept an invite)."""
        if role not in ROLES:
            raise ValueError(role)
        email = (email or "").strip().lower() or None
        now = time.time()
        with self.tx() as c:
            cur = c.execute("INSERT INTO users (email,name,role,pw_hash,created,notify_email) VALUES (?,?,?,?,?,?)",
                            (email, name.strip(), role, pw_hash, now, int(email_consent)))
            uid = cur.lastrowid
            if role == "patient":
                p = profile or {}
                dist = p.get("distance_km")
                c.execute("INSERT INTO patient_profiles (user_id,dob,sex,phone,address,district,distance_km,alt_phone,"
                          "created_by) VALUES (?,?,?,?,?,?,?,?,?)",
                          (uid, p.get("dob"), p.get("sex"), p.get("phone"), p.get("address"), p.get("district"),
                           float(dist) if dist not in (None, "") else None, p.get("alt_phone"), created_by))
                c.execute("INSERT INTO consents (user_id,purpose,version,granted_at,recorded_by) VALUES (?,?,?,?,?)",
                          (uid, "care", CONSENT_VERSION, now, created_by))
                if email_consent:
                    c.execute("INSERT INTO consents (user_id,purpose,version,granted_at,recorded_by) VALUES (?,?,?,?,?)",
                              (uid, "email_reminders", CONSENT_VERSION, now, created_by))
        return int(uid)

    def user_by_email(self, email: str | None):
        email = (email or "").strip().lower()
        return self.one("SELECT * FROM users WHERE email = ?", (email,)) if email else None

    def profile(self, uid: int):
        return self.one("SELECT * FROM patient_profiles WHERE user_id = ?", (uid,))

    def user(self, uid: int):
        return self.one("SELECT * FROM users WHERE id = ?", (uid,))

    # ------------------------------------------------------------------------------ cases
    def create_case(self, patient_id: int, filename: str, sha256: str, size: int, note: str | None) -> str:
        cid = secrets.token_hex(12)
        now = time.time()
        with self.tx() as c:
            c.execute("INSERT INTO cases (id,patient_id,status,created,updated,scan_filename,scan_sha256,scan_size,"
                      "consent,patient_note) VALUES (?,?,?,?,?,?,?,?,1,?)",
                      (cid, patient_id, "awaiting_review", now, now, filename[:200], sha256, size, (note or "")[:1000]))
        return cid

    def set_case(self, cid: str, **fields) -> None:
        allowed = {"status", "doctor_id", "doctor_message", "priority", "scan_filename", "scan_sha256", "scan_size",
                   "patient_note"}
        if not fields or set(fields) - allowed:
            raise ValueError(f"bad fields {set(fields) - allowed}")
        if "status" in fields and fields["status"] not in STATUSES:
            raise ValueError(fields["status"])
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.tx() as c:
            c.execute(f"UPDATE cases SET {cols}, updated = ? WHERE id = ?", (*fields.values(), time.time(), cid))

    def case(self, cid: str):
        return self.one("SELECT * FROM cases WHERE id = ?", (cid,))

    def add_ai_result(self, cid: str, model_version: str, result: dict, overlay_kinds: list[str]) -> int:
        with self.tx() as c:
            cur = c.execute("INSERT INTO ai_results (case_id,model_version,created,result_json,overlay_kinds) "
                            "VALUES (?,?,?,?,?)", (cid, model_version, time.time(), json.dumps(result),
                                                   json.dumps(overlay_kinds)))
        return int(cur.lastrowid)

    def latest_ai(self, cid: str) -> dict | None:
        r = self.one("SELECT * FROM ai_results WHERE case_id = ? ORDER BY created DESC, id DESC LIMIT 1", (cid,))
        if r is None:
            return None
        return {"id": r["id"], "model_version": r["model_version"], "created": r["created"],
                "result": json.loads(r["result_json"]), "overlays": json.loads(r["overlay_kinds"])}

    # ---------------------------------------------------------------------------- reports
    def add_report(self, cid: str, doctor_id: int, diagnosis: str, agrees: bool | None, findings: str,
                   recommendation: str, ai_result_id: int | None, snapshot_sha256: str,
                   amendment_reason: str | None = None, sections: dict | None = None) -> int:
        """sections: indication, technique, comparison, impression (free text, optional)."""
        sec = {k: ((sections or {}).get(k) or "").strip()[:2000] or None
               for k in ("indication", "technique", "comparison", "impression")}
        if diagnosis not in DIAGNOSES:
            raise ValueError(diagnosis)
        with self.tx() as c:
            v = c.execute("SELECT COALESCE(MAX(version),0) FROM reports WHERE case_id = ?", (cid,)).fetchone()[0] + 1
            if v > 1 and not (amendment_reason or "").strip():
                raise ValueError("an amendment needs a reason")
            c.execute("INSERT INTO reports (case_id,version,doctor_id,final_diagnosis,agrees_with_ai,findings,"
                      "recommendation,ai_result_id,signed_at,amendment_reason,snapshot_sha256,"
                      "indication,technique,comparison,impression) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (cid, v, doctor_id, diagnosis, None if agrees is None else int(agrees), findings[:5000],
                       (recommendation or "")[:2000], ai_result_id, time.time(), amendment_reason, snapshot_sha256,
                       sec["indication"], sec["technique"], sec["comparison"], sec["impression"]))
            c.execute("UPDATE cases SET status = 'finalized', updated = ? WHERE id = ?", (time.time(), cid))
        return v

    def reports(self, cid: str) -> list:
        return self.q("SELECT r.*, u.name AS doctor_name, u.credentials AS doctor_credentials, u.reg_no AS doctor_reg_no "
                      "FROM reports r JOIN users u ON u.id = r.doctor_id "
                      "WHERE r.case_id = ? ORDER BY r.version DESC", (cid,))

    def next_report_version(self, cid: str) -> int:
        return self.one("SELECT COALESCE(MAX(version),0) + 1 AS v FROM reports WHERE case_id = ?", (cid,))["v"]

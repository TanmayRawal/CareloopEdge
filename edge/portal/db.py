"""Portal database (SQLite, WAL, foreign keys, versioned migrations). Local disk only — never a network or
host-shared filesystem (WAL locking does not survive those; docs/FINDINGS.md §8).

Case lifecycle
  awaiting_review    patient uploaded; AI result attached (or AI could not analyse the image)
  in_review          a doctor opened the case
  reupload_requested doctor asked the patient for a better image (patient can upload to the same case)
  finalized          doctor signed a report (amendments add report versions; nothing is overwritten)
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path

STATUSES = ("awaiting_review", "in_review", "reupload_requested", "finalized")
ROLES = ("patient", "doctor", "admin")
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
]


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
            for i, sql in enumerate(MIGRATIONS[v:], start=v + 1):
                self.conn.execute("BEGIN")
                try:
                    for stmt in [s for s in sql.split(";") if s.strip()]:   # executescript would auto-commit
                        self.conn.execute(stmt)
                    self.conn.execute("DELETE FROM schema_version")
                    self.conn.execute("INSERT INTO schema_version VALUES (?)", (i,))
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
            self.conn.execute("INSERT INTO audit (ts,user_id,action,case_id,detail,ip) VALUES (?,?,?,?,?,?)",
                              (time.time(), user_id, action, case_id, json.dumps(detail or {}), ip))

    def backup(self, dest: Path) -> None:
        """Consistent online snapshot (SQLite backup API) — safe while the portal is running."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        out = sqlite3.connect(dest)
        with self._lock:
            self.conn.backup(out)
        out.close()

    # ------------------------------------------------------------------------------ users
    def create_user(self, email: str, name: str, role: str, pw_hash: str, profile: dict | None = None) -> int:
        if role not in ROLES:
            raise ValueError(role)
        with self.tx() as c:
            cur = c.execute("INSERT INTO users (email,name,role,pw_hash,created) VALUES (?,?,?,?,?)",
                            (email.strip().lower(), name.strip(), role, pw_hash, time.time()))
            uid = cur.lastrowid
            if role == "patient":
                p = profile or {}
                c.execute("INSERT INTO patient_profiles (user_id,dob,sex,phone) VALUES (?,?,?,?)",
                          (uid, p.get("dob"), p.get("sex"), p.get("phone")))
        return int(uid)

    def user_by_email(self, email: str):
        return self.one("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))

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
                   amendment_reason: str | None = None) -> int:
        if diagnosis not in DIAGNOSES:
            raise ValueError(diagnosis)
        with self.tx() as c:
            v = c.execute("SELECT COALESCE(MAX(version),0) FROM reports WHERE case_id = ?", (cid,)).fetchone()[0] + 1
            if v > 1 and not (amendment_reason or "").strip():
                raise ValueError("an amendment needs a reason")
            c.execute("INSERT INTO reports (case_id,version,doctor_id,final_diagnosis,agrees_with_ai,findings,"
                      "recommendation,ai_result_id,signed_at,amendment_reason,snapshot_sha256) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (cid, v, doctor_id, diagnosis, None if agrees is None else int(agrees), findings[:5000],
                       (recommendation or "")[:2000], ai_result_id, time.time(), amendment_reason, snapshot_sha256))
            c.execute("UPDATE cases SET status = 'finalized', updated = ? WHERE id = ?", (time.time(), cid))
        return v

    def reports(self, cid: str) -> list:
        return self.q("SELECT r.*, u.name AS doctor_name FROM reports r JOIN users u ON u.id = r.doctor_id "
                      "WHERE r.case_id = ? ORDER BY r.version DESC", (cid,))

    def next_report_version(self, cid: str) -> int:
        return self.one("SELECT COALESCE(MAX(version),0) + 1 AS v FROM reports WHERE case_id = ?", (cid,))["v"]

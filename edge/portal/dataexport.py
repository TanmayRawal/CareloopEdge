"""Patient data export ("Download my data", right of access under the DPDP Act).

A zip with everything the clinic holds about the patient that the patient is entitled to see in the portal:
profile, consents, studies, signed reports (structured + the archived report documents), appointments, approved
follow-up plans, referrals and letters, contact history (date, channel, outcome), uploaded scans and documents.
Like the portal itself, it does not include AI output (the patient's record is the doctor's signed report).
Decrypted only for the signed-in owner; the request is rate-limited and audited.
"""

from __future__ import annotations

import io
import json
import zipfile

from edge.portal import careplan

README = """This archive contains your records from {clinic}.

  record.json        your details, consents, studies, reports, appointments, follow-up plans, referrals, contacts
  reports/           signed radiology reports (open in a web browser; print to PDF if needed)
  referrals/         signed referral letters
  scans/             the images you or the care team uploaded
  documents/         other documents in your record

Keep this file safe: it contains medical information.
"""


def _rows(rows, drop=()) -> list[dict]:
    return [{k: r[k] for k in r.keys() if k not in drop} for r in rows]


def build(db, store, uid: int) -> bytes:
    user = db.user(uid)
    record = {
        "patient": {"id": uid, "name": user["name"], "email": user["email"], "registered": user["created"],
                    "email_reminders": bool(user["notify_email"])},
        "profile": _rows([db.profile(uid)], drop=("user_id", "created_by"))[0] if db.profile(uid) else {},
        "consents": _rows(db.q("SELECT purpose, version, granted_at, withdrawn_at FROM consents WHERE user_id = ?", (uid,))),
        "studies": _rows(db.q("SELECT id, status, created, scan_filename, scan_sha256, patient_note FROM cases "
                              "WHERE patient_id = ? ORDER BY created", (uid,))),
        "reports": _rows(db.q("SELECT r.case_id, r.version, r.final_diagnosis, r.findings, r.recommendation, r.signed_at, "
                              "r.amendment_reason, u.name AS doctor FROM reports r JOIN cases c ON c.id = r.case_id "
                              "JOIN users u ON u.id = r.doctor_id WHERE c.patient_id = ? ORDER BY r.signed_at", (uid,))),
        "appointments": _rows(db.q("SELECT a.start_ts, a.end_ts, a.room, a.reason, a.status, d.name AS doctor FROM appointments a "
                                   "JOIN users d ON d.id = a.doctor_id WHERE a.patient_id = ? ORDER BY a.start_ts", (uid,))),
        "follow_up_plans": _rows(db.q("SELECT kind, due_date, note, status, approved_at, closed_at, close_reason FROM care_plans "
                                      "WHERE patient_id = ? AND status != 'proposed' ORDER BY proposed_at", (uid,))),
        "referrals": _rows(db.q("SELECT facility, specialty, urgency, reason, status, signed_at FROM referrals "
                                "WHERE patient_id = ? AND status != 'draft'", (uid,))),
        "contacts": _rows(db.q("SELECT created, channel, outcome FROM contacts WHERE patient_id = ? ORDER BY created", (uid,))),
        "documents": _rows(db.q("SELECT id, kind, title, filename, size, sha256, created FROM documents WHERE patient_id = ?", (uid,))),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", README.format(clinic=db.setting("clinic_name")))
        z.writestr("record.json", json.dumps(record, indent=2, default=str))
        for c in record["studies"]:
            n = 1
            while store.exists(c["id"], f"scan_{n}"):
                ext = c["scan_filename"].rsplit(".", 1)[-1].lower() if "." in c["scan_filename"] else "img"
                z.writestr(f"scans/{c['id'][:8]}_{n}.{ext}", store.get(c["id"], f"scan_{n}"))
                n += 1
        for r in record["reports"]:
            kind = f"report_v{r['version']}"
            if store.exists(r["case_id"], kind):
                z.writestr(f"reports/report_{r['case_id'][:8]}_v{r['version']}.html", store.get(r["case_id"], kind))
        for ref in db.q("SELECT id FROM referrals WHERE patient_id = ? AND status != 'draft'", (uid,)):
            if store.exists(*careplan.letter_kind(ref["id"])):
                z.writestr(f"referrals/referral_{ref['id']}.html", store.get(*careplan.letter_kind(ref["id"])))
        for doc in record["documents"]:
            safe = "".join(ch for ch in doc["filename"] if ch.isalnum() or ch in "._-")[:80] or "document"
            z.writestr(f"documents/{doc['id']}_{safe}", store.get("documents", f"doc_{doc['id']}"))
    return buf.getvalue()

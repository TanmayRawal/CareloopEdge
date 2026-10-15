"""Follow-up risk score: who is most likely to be lost between visits, and why.

Assistive and explainable by design: a weighted sum of OPERATIONAL signals only (how overdue the next step is,
missed appointments, unanswered calls, contact details, open referrals, travel distance). It never looks at the
diagnosis, the report or any AI output — this module does not query those tables (a test enforces it). Each
score comes with the factors that produced it, so the coordinator can see why a patient is at the top of the
list. It ranks the worklist; it does not decide anything — people act on it.
"""

from __future__ import annotations

import datetime as dt
import time

from edge.portal import scheduling

# points per signal (named so the model card and the UI can show them)
W_OVERDUE_PER_DAY, W_OVERDUE_MAX = 1.5, 35
W_DUE_SOON = 6                     # due within the "due soon" window and nothing booked yet
W_NO_SHOW, W_NO_SHOW_MAX = 15, 30
W_FAILED_CONTACT, W_FAILED_MAX = 8, 24
W_UNREACHABLE = 15                 # contact status: unreachable / wrong number
W_REFERRAL_STALE = 15
W_FAR, W_VERY_FAR = 5, 10          # > 20 km, > 50 km
W_NEVER_CONTACTED = 5              # overdue and nobody has tried to reach the patient
W_BOOKED = -20                     # an appointment is booked: much less likely to be lost
BANDS = ((60, "high", "High"), (30, "medium", "Medium"), (0, "low", "Low"))
FAILED = ("no_answer", "wrong_number")


def score(sig: dict) -> dict:
    """sig -> {score 0..100, band, band_text, factors: [(text, points)]}."""
    f: list[tuple[str, float]] = []
    od = sig.get("overdue_days") or 0
    if od > 0:
        f.append((f"Overdue by {od} day{'s' if od != 1 else ''}", min(W_OVERDUE_MAX, od * W_OVERDUE_PER_DAY)))
    elif sig.get("due_soon") and not sig.get("booked"):
        f.append(("Due soon, not booked", W_DUE_SOON))
    if sig.get("no_shows"):
        n = sig["no_shows"]
        f.append((f"{n} missed appointment{'s' if n != 1 else ''}", min(W_NO_SHOW_MAX, n * W_NO_SHOW)))
    if sig.get("failed_contacts"):
        n = sig["failed_contacts"]
        f.append((f"{n} unanswered contact attempt{'s' if n != 1 else ''}", min(W_FAILED_MAX, n * W_FAILED_CONTACT)))
    if sig.get("contact_status") in ("unreachable", "wrong_number"):
        f.append(("Contact details not working", W_UNREACHABLE))
    if sig.get("referral_stale"):
        f.append(("Referral not completed", W_REFERRAL_STALE))
    km = sig.get("distance_km") or 0
    if km > 50:
        f.append((f"Lives {km:.0f} km away", W_VERY_FAR))
    elif km > 20:
        f.append((f"Lives {km:.0f} km away", W_FAR))
    if od > 0 and not sig.get("contacted"):
        f.append(("Not contacted yet", W_NEVER_CONTACTED))
    if sig.get("booked"):
        f.append(("Appointment booked", W_BOOKED))
    total = max(0, min(100, round(sum(p for _, p in f))))
    band, text = next((b, t) for lim, b, t in BANDS if total >= lim)
    return {"score": total, "band": band, "band_text": text, "factors": [(t, round(p)) for t, p in f]}


def signals(db, patient_ids: list[int] | None = None, now: float | None = None) -> dict[int, dict]:
    """Operational signals for patients with an open plan or an open referral (or the given ids), set-based."""
    now = time.time() if now is None else now
    today = scheduling.local(now, scheduling.tz(db)).date()
    soon = int(db.setting("followup_due_soon_days"))
    stale_s = float(db.setting("referral_stale_days")) * 86400
    if patient_ids is None:
        ids = [r["patient_id"] for r in db.q(
            "SELECT patient_id FROM care_plans WHERE status IN ('proposed','approved') "
            "UNION SELECT patient_id FROM referrals WHERE status IN ('signed','sent','appointment_made')")]
    else:
        ids = list(patient_ids)
    if not ids:
        return {}
    ph = ",".join("?" * len(ids))
    out = {i: {"overdue_days": 0, "due_soon": False, "no_shows": 0, "failed_contacts": 0, "contact_status": "ok",
               "referral_stale": False, "distance_km": None, "contacted": False, "booked": False} for i in ids}
    for r in db.q(f"SELECT patient_id, MIN(due_date) AS due FROM care_plans WHERE status = 'approved' "
                  f"AND patient_id IN ({ph}) GROUP BY patient_id", ids):
        days = (dt.date.fromisoformat(r["due"]) - today).days
        out[r["patient_id"]]["overdue_days"] = max(0, -days)
        out[r["patient_id"]]["due_soon"] = 0 <= days <= soon
    for r in db.q(f"SELECT patient_id, COUNT(*) AS n FROM appointments WHERE status = 'no_show' AND patient_id IN ({ph}) "
                  f"GROUP BY patient_id", ids):
        out[r["patient_id"]]["no_shows"] = r["n"]
    for r in db.q(f"SELECT DISTINCT patient_id FROM appointments WHERE status IN ('booked','arrived') AND end_ts > ? "
                  f"AND patient_id IN ({ph})", (now, *ids)):
        out[r["patient_id"]]["booked"] = True
    for r in db.q(f"SELECT user_id, contact_status, distance_km FROM patient_profiles WHERE user_id IN ({ph})", ids):
        out[r["user_id"]]["contact_status"] = r["contact_status"] or "ok"
        out[r["user_id"]]["distance_km"] = r["distance_km"]
    # consecutive failed attempts since the last successful contact (newest first)
    seen_ok: set[int] = set()
    for r in db.q(f"SELECT patient_id, outcome FROM contacts WHERE patient_id IN ({ph}) ORDER BY created DESC, id DESC", ids):
        s = out[r["patient_id"]]
        s["contacted"] = True
        if r["patient_id"] in seen_ok:
            continue
        if r["outcome"] in FAILED:
            s["failed_contacts"] += 1
        else:
            seen_ok.add(r["patient_id"])
    for r in db.q(f"SELECT DISTINCT patient_id FROM referrals WHERE status IN ('signed','sent','appointment_made') "
                  f"AND COALESCE(signed_at, created) < ? AND patient_id IN ({ph})", (now - stale_s, *ids)):
        out[r["patient_id"]]["referral_stale"] = True
    return out


def scores(db, patient_ids: list[int] | None = None, now: float | None = None) -> dict[int, dict]:
    return {pid: score(sig) | {"signals": sig} for pid, sig in signals(db, patient_ids, now).items()}

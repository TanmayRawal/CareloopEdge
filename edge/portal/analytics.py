"""Department analytics: is every patient's care loop being closed, and how fast and how well is reading done?

compute(db, days) returns plain numbers for the analytics page (views/doctor.py: stats). Everything is counted from
recorded events; nothing is estimated. Rates come with their denominators so a small clinic is not misled by a
percentage of three. Two kinds of figures:
  * period figures (the chosen window, by the date the event happened): received, reported, plans closed, contacts,
    appointments, referrals, e-mails;
  * snapshot figures (right now): reading backlog, overdue plans, risk bands, unreachable patients.

Care-loop definitions (CareLoop):
  loop closed       a follow-up plan completed;   lost to follow-up = plan closed as lost
  closure rate      completed / (completed + lost), over plans closed in the window
  on time           completed on or before its due date
  reached           a contact with outcome "reached" (confirmed or declined), or a booked consultation
Risk bands are those of the coordinator's worklist (edge/portal/risk.py: operational signals only, never the
diagnosis or any AI output).
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
import time

from edge.portal import risk, scheduling

PERIODS = {"30": 30, "90": 90, "365": 365, "all": None}
DISTANCE_BANDS = ((0, 10, "< 10 km"), (10, 30, "10–30 km"), (30, 60, "30–60 km"), (60, 100, "60–100 km"),
                  (100, 1e9, "≥ 100 km"))
PLAN_KINDS = ("consultation", "repeat_scan", "referral", "other")
REFERRAL_STAGES = ("signed", "sent", "appointment_made", "attended", "closed")
RISK_FACTORS = (("Overdue", "Overdue"), ("Due soon", "Due soon, not booked"), ("missed appointment", "Missed appointments"),
                ("unanswered", "Unanswered contact attempts"), ("Contact details", "Contact details not working"),
                ("Referral not completed", "Referral not completed"), ("Lives", "Lives > 20 km away"),
                ("Not contacted", "Overdue, never contacted"))
CLASSES = ("glioma", "meningioma", "pituitary", "no_tumor")


def _rate(num: int, den: int) -> dict:
    return {"n": num, "d": den, "p": (num / den) if den else None}


def _median(xs):
    return statistics.median(xs) if xs else None


def _pct(xs, q: float):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def _band(km):
    if km is None:
        return "Not recorded"
    return next(t for lo, hi, t in DISTANCE_BANDS if lo <= km < hi)


def compute(db, days: int | None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    t0 = now - days * 86400 if days else 0.0
    zone = scheduling.tz(db)
    today = scheduling.local(now, zone).date()
    out: dict = {"days": days}

    # ---------------------------------------------------------------- reading (radiology)
    sla_h = float(db.setting("unsigned_hours"))
    received = db.one("SELECT COUNT(*) AS n FROM cases WHERE created >= ?", (t0,))["n"]
    first = db.q("SELECT r.case_id, r.signed_at, c.created, c.priority FROM reports r JOIN cases c ON c.id = r.case_id "
                 "WHERE r.version = 1 AND r.signed_at >= ?", (t0,))
    tat = [(r["signed_at"] - r["created"]) / 3600 for r in first]
    tat_urgent = [(r["signed_at"] - r["created"]) / 3600 for r in first if r["priority"] >= 2]
    backlog = db.q("SELECT created, priority FROM cases WHERE status IN ('awaiting_review','in_review')")
    ages = [(now - r["created"]) / 3600 for r in backlog]
    out["reading"] = {
        "received": received, "reported": len(first),
        "tat_median_h": _median(tat), "tat_p90_h": _pct(tat, .9), "tat_urgent_median_h": _median(tat_urgent),
        "within_sla": _rate(sum(1 for h in tat if h <= sla_h), len(tat)), "sla_h": sla_h,
        "amended": db.one("SELECT COUNT(*) AS n FROM reports WHERE version > 1 AND signed_at >= ?", (t0,))["n"],
        "reupload": db.one("SELECT COUNT(*) AS n FROM audit WHERE action = 'reupload_requested' AND ts >= ?", (t0,))["n"],
        "backlog": len(backlog), "backlog_urgent": sum(1 for r in backlog if r["priority"] >= 2),
        "backlog_ages": [("< 24 h", sum(1 for a in ages if a < 24)), ("1–3 days", sum(1 for a in ages if 24 <= a < 72)),
                         ("3–7 days", sum(1 for a in ages if 72 <= a < 168)), ("> 7 days", sum(1 for a in ages if a >= 168))],
    }
    days_axis = [today - dt.timedelta(days=i) for i in range(13, -1, -1)]
    c_days = [scheduling.local(r["created"], zone).date() for r in db.q("SELECT created FROM cases WHERE created >= ?",
                                                                       (now - 15 * 86400,))]
    s_days = [scheduling.local(r["signed_at"], zone).date() for r in db.q(
        "SELECT signed_at FROM reports WHERE version = 1 AND signed_at >= ?", (now - 15 * 86400,))]
    daily = [{"label": x.strftime("%d %b"), "received": c_days.count(x), "signed": s_days.count(x)} for x in days_axis]
    out["daily"], out["daily_max"] = daily, max([1] + [max(x["received"], x["signed"]) for x in daily])

    # ---------------------------------------------------------------- AI quality (against the signed report)
    rows = db.q("SELECT r.final_diagnosis, r.agrees_with_ai, ai.result_json FROM reports r "
                "LEFT JOIN ai_results ai ON ai.id = (SELECT id FROM ai_results WHERE case_id = r.case_id "
                "ORDER BY created DESC, id DESC LIMIT 1) "
                "WHERE r.version = (SELECT MAX(version) FROM reports WHERE case_id = r.case_id) AND r.signed_at >= ?", (t0,))
    pairs, said = [], {"yes": 0, "no": 0, "na": 0}
    for r in rows:
        res = json.loads(r["result_json"]) if r["result_json"] else {}
        if res.get("status") == "ok" and r["final_diagnosis"] in CLASSES:
            pairs.append((res["prediction"], r["final_diagnosis"]))
        said["na" if r["agrees_with_ai"] is None else "yes" if r["agrees_with_ai"] else "no"] += 1
    analysed = [json.loads(r["result_json"]) for r in db.q(
        "SELECT result_json FROM ai_results WHERE created >= ?", (t0,))]
    t_study = [sum(v for k, v in a.get("timings_ms", {}).items() if k != "per_slice_ms") / 1000
               for a in analysed if a.get("series")]
    t_image = [sum(a.get("timings_ms", {}).values()) / 1000 for a in analysed if not a.get("series")]
    out["ai"] = {
        "agreement": _rate(sum(1 for a, x in pairs if a == x), len(pairs)),
        "confusion": {a: {x: sum(1 for p in pairs if p == (a, x)) for x in CLASSES} for a in CLASSES},
        "doctor_said": said,
        "analysed": len(analysed), "rejected": _rate(sum(1 for a in analysed if a.get("status") != "ok"), len(analysed)),
        "flagged": _rate(sum(1 for a in analysed if a.get("needs_expert_review")), len(analysed)),
        "whole_study": _rate(sum(1 for a in analysed if a.get("series")), len(analysed)),
        "t_study_s": _median(t_study), "t_image_s": _median(t_image),
    }

    # ---------------------------------------------------------------- care loop
    plans = db.q("SELECT p.id, p.patient_id, p.case_id, p.kind, p.status, p.due_date, p.proposed_at, p.approved_at, "
                 "p.closed_at, pp.distance_km, pp.district FROM care_plans p "
                 "LEFT JOIN patient_profiles pp ON pp.user_id = p.patient_id")
    closed = [p for p in plans if p["status"] in ("completed", "lost") and (p["closed_at"] or 0) >= t0]
    done = [p for p in closed if p["status"] == "completed"]
    on_time = [p for p in done if scheduling.local(p["closed_at"], zone).date() <= dt.date.fromisoformat(p["due_date"])]
    open_ = [p for p in plans if p["status"] in ("proposed", "approved")]
    overdue = [p for p in open_ if p["status"] == "approved" and dt.date.fromisoformat(p["due_date"]) < today]
    soon = int(db.setting("followup_due_soon_days"))
    unreachable = db.one("SELECT COUNT(*) AS n FROM patient_profiles WHERE contact_status IN ('unreachable','wrong_number')")["n"]
    appts = db.q("SELECT patient_id, status, start_ts, created FROM appointments WHERE start_ts >= ? AND start_ts <= ?",
                 (t0, now))
    att = sum(1 for a in appts if a["status"] in ("completed", "arrived"))
    noshow = [a for a in appts if a["status"] == "no_show"]
    out["loop"] = {
        "closure": _rate(len(done), len(closed)), "lost": _rate(len(closed) - len(done), len(closed)),
        "on_time": _rate(len(on_time), len(done)),
        "days_to_close": _median([(p["closed_at"] - p["proposed_at"]) / 86400 for p in done]),
        "open": len(open_), "awaiting_approval": sum(1 for p in open_ if p["status"] == "proposed"),
        "overdue": len(overdue), "due_soon": sum(1 for p in open_ if p["status"] == "approved"
                                                 and 0 <= (dt.date.fromisoformat(p["due_date"]) - today).days <= soon),
        "unreachable": unreachable, "no_show": _rate(len(noshow), att + len(noshow)),
    }

    # funnel: signed reports in the window -> follow-up planned -> approved -> patient reached -> completed
    rep_cases = {r["case_id"] for r in first}
    pl = [p for p in plans if p["case_id"] in rep_cases]
    pl_patients = {p["patient_id"] for p in pl}
    reached = {r["patient_id"] for r in db.q(
        "SELECT patient_id FROM contacts WHERE outcome IN ('reached_confirmed','reached_declined') AND created >= ? "
        "UNION SELECT patient_id FROM appointments WHERE status != 'cancelled' AND created >= ?", (t0, t0))}
    out["funnel"] = [("Reports signed", len(rep_cases)),
                     ("Follow-up planned", len({p["case_id"] for p in pl})),
                     ("Approved by a doctor", len({p["case_id"] for p in pl if p["approved_at"]})),
                     ("Patient reached", len({p["case_id"] for p in pl if p["patient_id"] in reached})),
                     ("Loop closed", len({p["case_id"] for p in pl if p["status"] == "completed"}))]
    out["funnel_patients"] = len(pl_patients)

    # by next-step kind
    out["by_kind"] = []
    for k in PLAN_KINDS:
        ks = [p for p in plans if p["kind"] == k and p["proposed_at"] >= t0]
        if not ks:
            continue
        kc = [p for p in ks if p["status"] == "completed"]
        kl = [p for p in ks if p["status"] == "lost"]
        kt = [p for p in kc if scheduling.local(p["closed_at"], zone).date() <= dt.date.fromisoformat(p["due_date"])]
        out["by_kind"].append({"kind": k, "planned": len(ks), "completed": len(kc), "lost": len(kl),
                               "open": sum(1 for p in ks if p["status"] in ("proposed", "approved")),
                               "on_time": _rate(len(kt), len(kc)), "closure": _rate(len(kc), len(kc) + len(kl))})

    # distance to care (closed plans in the window) and districts
    bands = {}
    for p in closed + open_:
        b = bands.setdefault(_band(p["distance_km"]), {"completed": 0, "lost": 0, "open": 0})
        b["open" if p["status"] in ("proposed", "approved") else p["status"]] += 1
    order = [t for _, _, t in DISTANCE_BANDS] + ["Not recorded"]
    out["distance"] = [{"band": t, **bands[t], "closure": _rate(bands[t]["completed"], bands[t]["completed"] + bands[t]["lost"])}
                       for t in order if t in bands]
    districts = {}
    for p in closed + open_:
        d = districts.setdefault((p["district"] or "Not recorded").strip() or "Not recorded",
                                 {"patients": set(), "completed": 0, "lost": 0, "open": 0, "overdue": 0})
        d["patients"].add(p["patient_id"])
        d["open" if p["status"] in ("proposed", "approved") else p["status"]] += 1
        d["overdue"] += p in overdue
    out["districts"] = sorted(({"district": k, "patients": len(v["patients"]), "completed": v["completed"],
                                "lost": v["lost"], "open": v["open"], "overdue": v["overdue"],
                                "closure": _rate(v["completed"], v["completed"] + v["lost"])}
                               for k, v in districts.items()), key=lambda x: (-x["patients"], x["district"]))[:10]

    # risk now (the coordinator's worklist)
    sc = risk.scores(db, now=now)
    out["risk"] = {b: sum(1 for s in sc.values() if s["band"] == b) for b in ("high", "medium", "low")}
    out["risk_total"] = len(sc)
    factors = []
    for key, label in RISK_FACTORS:
        n = sum(1 for s in sc.values() if any(key in t and pts > 0 for t, pts in s["factors"]))
        factors.append((label, n))
    out["risk_factors"] = sorted(factors, key=lambda x: -x[1])

    # outreach and reminders
    contacts = db.q("SELECT outcome, channel FROM contacts WHERE created >= ?", (t0,))
    oc = {o: sum(1 for c in contacts if c["outcome"] == o)
          for o in ("reached_confirmed", "reached_declined", "no_answer", "wrong_number", "message_sent")}
    attempts = oc["reached_confirmed"] + oc["reached_declined"] + oc["no_answer"] + oc["wrong_number"]
    mail = {r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) AS n FROM outbox WHERE created >= ? GROUP BY status", (t0,))}
    out["outreach"] = {"outcomes": oc, "outcomes_max": max([1] + list(oc.values())), "contacts": len(contacts),
                       "reach": _rate(oc["reached_confirmed"] + oc["reached_declined"], attempts),
                       "confirmed": _rate(oc["reached_confirmed"], oc["reached_confirmed"] + oc["reached_declined"]),
                       "mail_sent": mail.get("sent", 0), "mail_failed": mail.get("failed", 0),
                       "mail_queued": mail.get("queued", 0),
                       "mail_delivery": _rate(mail.get("sent", 0), mail.get("sent", 0) + mail.get("failed", 0))}

    # consultations
    rebooked = 0
    for a in noshow:
        rebooked += bool(db.one("SELECT 1 AS x FROM appointments WHERE patient_id = ? AND created > ? AND status != 'cancelled'",
                                (a["patient_id"], a["start_ts"])))
    lead = [(a["start_ts"] - a["created"]) / 86400 for a in appts if a["status"] != "cancelled"]
    out["consult"] = {"held": len(appts), "attended": att, "no_show": len(noshow),
                      "unmarked": sum(1 for a in appts if a["status"] == "booked"),     # past, attendance not recorded
                      "cancelled": sum(1 for a in appts if a["status"] == "cancelled"),
                      "attendance": _rate(att, att + len(noshow)), "rebooked": _rate(rebooked, len(noshow)),
                      "lead_days": _median(lead),
                      "upcoming": db.one("SELECT COUNT(*) AS n FROM appointments WHERE status = 'booked' AND start_ts > ?",
                                         (now,))["n"]}

    # referrals to other centres
    refs = db.q("SELECT status, signed_at, created, updated FROM referrals WHERE status != 'draft' AND created >= ?", (t0,))
    stale = float(db.setting("referral_stale_days"))
    fin = [r for r in refs if r["status"] in ("attended", "closed")]
    out["referrals"] = {"stages": [(s, sum(1 for r in refs if r["status"] == s)) for s in REFERRAL_STAGES],
                        "total": len(refs), "completed": _rate(len(fin), len(refs)),
                        "days_to_complete": _median([(r["updated"] - (r["signed_at"] or r["created"])) / 86400 for r in fin]),
                        "stale": sum(1 for r in refs if r["status"] in ("signed", "sent", "appointment_made")
                                     and now - (r["signed_at"] or r["created"]) > stale * 86400),
                        "stale_days": stale}
    return out

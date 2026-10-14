"""Human-readable audit log: event names, categories and one-line descriptions of the stored JSON detail."""

from __future__ import annotations

import json

from edge.portal.common import CLASS_TEXT, ROLE_TEXT

IMAGE_KIND = {"original": "original scan", "cam": "Grad-CAM", "lesion": "lesion outline", "cam_lesion": "Grad-CAM + lesion"}

# action -> (label, category); category drives the badge colour and the filter
EVENTS = {
    "login": ("Signed in", "access"), "logout": ("Signed out", "access"), "login_failed": ("Failed sign-in", "alert"),
    "login_locked": ("Account locked", "alert"), "registered": ("Registered", "account"),
    "password_changed": ("Password changed", "account"), "preferences_updated": ("Preferences updated", "account"),
    "user_created": ("User created", "account"), "user_created_cli": ("User created (console)", "account"),
    "user_disabled": ("User disabled", "alert"), "user_enabled": ("User enabled", "account"),
    "case_submitted": ("Scan uploaded", "clinical"), "case_resubmitted": ("New image uploaded", "clinical"),
    "review_started": ("Review started", "clinical"), "case_viewed": ("Study opened", "clinical"),
    "image_viewed": ("Image viewed", "view"), "report_viewed": ("Report viewed", "view"),
    "report_signed": ("Report signed", "clinical"), "report_amended": ("Report amended", "clinical"),
    "reupload_requested": ("New image requested", "clinical"), "assistant_narrative": ("Assistant summary", "clinical"),
    "appointment_booked": ("Consultation booked", "booking"), "appointment_rescheduled": ("Consultation rescheduled", "booking"),
    "appointment_cancelled": ("Consultation cancelled", "booking"), "appointment_arrived": ("Patient arrived", "booking"),
    "appointment_completed": ("Consultation completed", "booking"), "appointment_no_show": ("Patient did not attend", "booking"),
    "hours_updated": ("Hours updated", "account"), "settings_updated": ("Settings changed", "account"),
    "email_test": ("Test e-mail", "account"), "email_retry": ("E-mail retried", "account"),
}
CATEGORIES = {"access": "Sign-ins", "clinical": "Clinical", "booking": "Consultations", "account": "Accounts & settings",
              "alert": "Security alerts", "view": "Image and report views"}


def label(action: str) -> str:
    return EVENTS.get(action, (action.replace("_", " ").capitalize(), ""))[0]


def category(action: str) -> str:
    return EVENTS.get(action, ("", "account"))[1]


def describe(action: str, detail: str | None) -> str:
    try:
        d = json.loads(detail or "{}")
    except ValueError:
        return ""
    if not isinstance(d, dict) or not d:
        return ""
    if action == "image_viewed" and "kind" in d:
        base, _, attempt = d["kind"].rpartition("_")
        return IMAGE_KIND.get(base, base).capitalize() + (f" (upload {attempt})" if attempt not in ("", "1") else "")
    if action in ("report_signed", "report_amended"):
        parts = [f"Version {d.get('version')}", CLASS_TEXT.get(d.get("diagnosis"), d.get("diagnosis") or "")]
        if d.get("agrees_with_ai") is not None:
            parts.append("agrees with AI" if d["agrees_with_ai"] else "differs from AI")
        return " · ".join(p for p in parts if p)
    if action == "report_viewed":
        return f"Version {d.get('version')}"
    if action in ("case_submitted", "case_resubmitted"):
        s = "AI analysis complete" if d.get("ai_status") == "ok" else "AI could not analyse the image"
        return s + (f" · upload {d['attempt']}" if d.get("attempt", 1) != 1 else "")
    if action == "assistant_narrative":
        return "Local language model" if d.get("llm_used") else "Standard summary"
    if action in ("user_created", "user_created_cli"):
        return ROLE_TEXT.get(d.get("role"), "") + (f" · account #{d['user']}" if d.get("user") else "")
    if action in ("user_disabled", "user_enabled"):
        return f"Account #{d.get('user')}"
    if action == "settings_updated":
        return ", ".join(k.replace("_", " ") for k in d.get("changed", [])) or "No changes"
    if action == "hours_updated":
        return f"{d.get('blocks', 0)} time blocks"
    if action == "appointment_rescheduled":
        return f"Appointment #{d.get('from')} → #{d.get('to')}"
    if action.startswith("appointment_") and d.get("appointment"):
        return f"Appointment #{d['appointment']}" + (" · by doctor" if d.get("by") == "doctor" else "")
    if action == "email_test":
        return f"{d.get('sent', 0)} sent" if "sent" in d else ""
    return ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in d.items())

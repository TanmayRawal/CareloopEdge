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
    # follow-up (CareLoop)
    "plan_created": ("Follow-up planned", "followup"), "plan_proposed": ("Follow-up proposed", "followup"),
    "plan_approved": ("Follow-up approved", "followup"), "plan_rejected": ("Follow-up declined", "followup"),
    "plan_completed": ("Follow-up completed", "followup"), "plan_lost": ("Lost to follow-up", "alert"),
    "plan_cancelled": ("Follow-up closed", "followup"),
    "contact_logged": ("Contact recorded", "followup"), "followup_reminder_sent": ("Reminder sent", "followup"),
    "referral_signed": ("Referral signed", "followup"), "referral_updated": ("Referral updated", "followup"),
    "referral_viewed": ("Referral letter viewed", "view"),
    "patient_registered": ("Patient registered at desk", "account"), "patient_updated": ("Patient details updated", "account"),
    "patient_record_viewed": ("Patient record opened", "view"), "scan_uploaded_for_patient": ("Scan uploaded for patient", "clinical"),
    "document_added": ("Document added", "clinical"), "document_viewed": ("Document opened", "view"),
    "invite_sent": ("Portal invitation sent", "account"), "invite_accepted": ("Invitation accepted", "account"),
    "password_reset_requested": ("Password reset requested", "access"), "password_reset": ("Password reset", "access"),
    "appointment_reply_yes": ("Patient confirmed by e-mail", "booking"), "appointment_reply_no": ("Patient declined by e-mail", "booking"),
    "two_step_enabled": ("Two-step sign-in on", "account"), "two_step_disabled": ("Two-step sign-in off", "alert"),
    "login_code_failed": ("Wrong sign-in code", "alert"), "login_recovery_code": ("Recovery code used", "alert"),
    "data_exported": ("Data downloaded by patient", "account"), "deletion_requested": ("Deletion requested", "account"),
    "deletion_request_done": ("Account closed on request", "account"), "deletion_request_rejected": ("Deletion declined", "account"),
}
CATEGORIES = {"access": "Sign-ins", "clinical": "Clinical", "followup": "Follow-up", "booking": "Consultations", "account": "Accounts & settings",
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
    if action in ("plan_created", "plan_proposed") and d.get("kind"):
        from edge.portal.careplan import PLAN_TEXT
        return PLAN_TEXT.get(d["kind"], d["kind"]) + (f" · due {d['due']}" if d.get("due") else "")
    if action == "contact_logged" and d.get("outcome"):
        from edge.portal.outreach import OUTCOME_TEXT
        return OUTCOME_TEXT.get(d["outcome"], d["outcome"])
    if action == "referral_updated" and d.get("status"):
        from edge.portal.careplan import REFERRAL_TEXT
        return REFERRAL_TEXT.get(d["status"], d["status"])
    if action == "followup_reminder_sent":
        return "E-mail and portal" if d.get("email") else "Portal only"
    if d.get("patient") and action in ("patient_registered", "patient_record_viewed", "patient_updated", "invite_sent",
                                       "document_added", "plan_approved", "plan_rejected", "plan_completed", "plan_lost",
                                       "plan_cancelled", "referral_signed"):
        return f"Patient NE-{int(d['patient']):05d}"
    if action == "email_test":
        return f"{d.get('sent', 0)} sent" if "sent" in d else ""
    return ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in d.items())

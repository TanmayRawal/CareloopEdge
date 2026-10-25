"""Display vocabulary and small helpers shared by the portal views and templates."""

from __future__ import annotations

import datetime as dt
import ipaddress
import socket
import time

BRAND = "CareLoop-Edge"
DISCLAIMER = "CareLoop-Edge research prototype"
CLASS_TEXT = {"glioma": "Glioma", "meningioma": "Meningioma", "pituitary": "Pituitary tumor", "no_tumor": "No tumor",
              "other": "Other finding", "inconclusive": "Inconclusive — further imaging needed"}
STATUS_TEXT = {"awaiting_review": "Awaiting doctor review", "in_review": "In review",
               "reupload_requested": "New image requested", "finalized": "Report ready"}
APPT_TEXT = {"booked": "Booked", "arrived": "Arrived", "completed": "Completed", "no_show": "Did not attend",
             "cancelled": "Cancelled"}
ROLE_TEXT = {"doctor": "Radiologist", "admin": "Care coordinator", "patient": "Patient"}
FLAG_TEXT = {
    "out_of_distribution": "Image differs from the scans the model was trained on.",
    "unusual_image": "Image is less typical than 95% of calibration scans.",
    "color_image": "Looks like a colour photo, not a grayscale scan.",
    "mostly_empty": "Image is almost empty.",
    "low_resolution": "Very low resolution (< 128 px).",
    "blurry_or_flat": "Very little image detail.",
    "uncertain_prediction_set": "More than one class is plausible at the 95% level.",
    "empty_prediction_set": "No class reaches the calibrated 95% level.",
    "explanation_outside_lesion": "The model's evidence lies almost entirely outside the detected lesion.",
    "classifier_segmenter_disagree": "Classifier and lesion detector disagree on whether a tumor is present.",
}
FLAG_SHORT = {"out_of_distribution": "Atypical image", "unusual_image": "Less typical", "color_image": "Colour image",
              "mostly_empty": "Mostly empty", "low_resolution": "Low resolution", "blurry_or_flat": "Low detail",
              "uncertain_prediction_set": "Two+ classes plausible", "empty_prediction_set": "Low confidence",
              "explanation_outside_lesion": "Evidence off lesion", "classifier_segmenter_disagree": "Detector disagrees"}
PAGE_SIZE = 50


def mrn(uid) -> str:
    """Display patient identifier (stable, derived from the account id)."""
    return f"NE-{int(uid):05d}"


def age_from_dob(dob: str | None) -> int | None:
    try:
        b = dt.date.fromisoformat(dob)
    except (TypeError, ValueError):
        return None
    t = dt.date.today()
    return t.year - b.year - ((t.month, t.day) < (b.month, b.day))


def fmt_iso(day: str | None) -> str:
    """'2026-10-29' -> '29 Oct 2026' (dates stored as ISO text, e.g. follow-up due dates)."""
    try:
        return dt.date.fromisoformat(day).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return day or "—"


def initials(name: str | None) -> str:
    """'Meera Joshi' -> 'M. J.': how patient names appear on the device's shared screen (privacy)."""
    parts = [p for p in (name or "").replace(".", " ").split() if p]
    return " ".join(p[0].upper() + "." for p in parts[:3]) or "—"


def ago(ts, now: float | None = None) -> str:
    if not ts:
        return "—"
    m = max(0, int(((time.time() if now is None else now) - ts) // 60))
    if m < 60:
        return f"{m} min"
    if m < 48 * 60:
        return f"{m // 60} h"
    return f"{m // 1440} d"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def tls_name_allowed(name: str) -> bool:
    """Names Caddy may issue an on-demand certificate for: this device's mDNS name, localhost, or a private /
    link-local / loopback IP (how browsers on the clinic LAN reach the Pi). Anything else is refused, so a
    stranger cannot make the Pi mint certificates for arbitrary names."""
    name = name.strip().lower().rstrip(".")
    if name in {"localhost", f"{socket.gethostname().lower()}.local"}:
        return True
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        return False
    return ip.is_private or ip.is_link_local or ip.is_loopback


def safe_next(nxt: str) -> str | None:
    """Only same-site paths: '/x' yes; '//evil', '/\\evil' (browsers read '\\' as '/'), 'https://evil' and
    control characters no."""
    if not nxt.startswith("/") or nxt.startswith(("//", "/\\")) or "\\" in nxt:
        return None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in nxt):
        return None
    return nxt


def page_args(request) -> tuple[int, int]:
    try:
        p = max(1, int(request.args.get("page", 1)))
    except ValueError:
        p = 1
    return p, (p - 1) * PAGE_SIZE

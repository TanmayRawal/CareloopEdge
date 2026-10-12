"""Clinical workflow (no HTTP here, so it is testable directly).

Upload -> AI analysis -> doctor worklist -> doctor review -> signed report -> patient sees the report.
Every scan attempt n of a case is stored as blobs  scan_n, original_n, cam_n, lesion_n, cam_lesion_n
and one ai_results row; earlier attempts are kept (medical records are append-only).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from flask import render_template

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"}
MAX_UPLOAD = 20 * 1024 * 1024
REVIEW_FLAGS = {"unusual_image", "uncertain_prediction_set", "empty_prediction_set",
                "explanation_outside_lesion", "classifier_segmenter_disagree"}


class UploadError(ValueError):
    pass


@dataclass
class Ctx:
    db: object
    store: object
    engine: object
    model_version: str


def priority(result: dict) -> int:
    """Worklist order (higher first): AI could not analyse > AI asks for review > routine."""
    if result.get("status") != "ok":
        return 3
    if set(result.get("flags", [])) & REVIEW_FLAGS:
        return 2
    return 1


def _validate(filename: str, data: bytes) -> str:
    name = (filename or "").strip()
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if ext not in ALLOWED_EXT:
        raise UploadError(f"Unsupported file type '{ext or '?'}'. Use PNG, JPG or DICOM.")
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > MAX_UPLOAD:
        raise UploadError("File too large (max 20 MB).")
    return name


def _analyse_into(ctx: Ctx, cid: str, attempt: int, data: bytes, filename: str) -> dict:
    """Run the AI on an attempt, store the scan + overlays encrypted, record the result."""
    try:
        res = ctx.engine.analyze(data, filename=filename)
    except ValueError as e:                          # undecodable / unsupported content
        raise UploadError(f"Could not read this file as an image: {e}") from e
    overlays = res.pop("overlays")
    res.pop("_state", None)
    kinds = []
    for kind, png in overlays.items():
        ctx.store.put(cid, f"{kind}_{attempt}", png)
        kinds.append(f"{kind}_{attempt}")
    res["attempt"] = attempt
    ctx.db.add_ai_result(cid, ctx.model_version, res, kinds)
    ctx.db.set_case(cid, priority=priority(res))
    return res


def submit_scan(ctx: Ctx, patient, data: bytes, filename: str, note: str, consent: bool, ip: str | None) -> str:
    if not consent:
        raise UploadError("Please confirm the consent statement to submit a scan.")
    name = _validate(filename, data)
    sha = hashlib.sha256(data).hexdigest()
    cid = ctx.db.create_case(patient["id"], name, sha, len(data), note)
    try:
        ctx.store.put(cid, "scan_1", data)
        res = _analyse_into(ctx, cid, 1, data, name)
    except UploadError:
        with ctx.db.tx() as c:                       # nothing usable was submitted: remove the empty case
            c.execute("DELETE FROM cases WHERE id = ?", (cid,))
        ctx.store.delete_case(cid)
        raise
    ctx.db.audit(patient["id"], "case_submitted", cid, {"ai_status": res["status"], "model": ctx.model_version}, ip)
    return cid


def resubmit_scan(ctx: Ctx, patient, case, data: bytes, filename: str, ip: str | None) -> None:
    if case["status"] != "reupload_requested":
        raise UploadError("This case is not waiting for a new image.")
    name = _validate(filename, data)
    attempt = (ctx.db.latest_ai(case["id"]) or {}).get("result", {}).get("attempt", 1) + 1
    ctx.store.put(case["id"], f"scan_{attempt}", data)
    res = _analyse_into(ctx, case["id"], attempt, data, name)
    ctx.db.set_case(case["id"], status="awaiting_review", scan_filename=name,
                    scan_sha256=hashlib.sha256(data).hexdigest(), scan_size=len(data))
    ctx.db.audit(patient["id"], "case_resubmitted", case["id"], {"attempt": attempt, "ai_status": res["status"]}, ip)


def open_for_review(ctx: Ctx, case, doctor, ip: str | None) -> None:
    if case["status"] == "awaiting_review":
        ctx.db.set_case(case["id"], status="in_review", doctor_id=doctor["id"])
        ctx.db.audit(doctor["id"], "review_started", case["id"], ip=ip)


def request_reupload(ctx: Ctx, case, doctor, message: str, ip: str | None) -> None:
    if case["status"] == "finalized":
        raise ValueError("finalized cases cannot be sent back")
    ctx.db.set_case(case["id"], status="reupload_requested", doctor_id=doctor["id"], doctor_message=message[:1000])
    ctx.db.audit(doctor["id"], "reupload_requested", case["id"], ip=ip)


def sign_report(ctx: Ctx, case, doctor, diagnosis: str, agrees: bool | None, findings: str, recommendation: str,
                amendment_reason: str | None, ip: str | None) -> int:
    """Render the immutable report snapshot (no AI content: it is the patient's document), store it encrypted,
    and record the structured report with the snapshot's SHA-256."""
    if not findings.strip():
        raise ValueError("Findings are required.")
    patient = ctx.db.user(case["patient_id"])
    ai = ctx.db.latest_ai(case["id"])
    version = ctx.db.next_report_version(case["id"])
    html = render_template("portal/report_snapshot.html", case=case, patient=patient, doctor=doctor, version=version,
                           diagnosis=diagnosis, findings=findings, recommendation=recommendation,
                           amendment_reason=amendment_reason, signed=time.strftime("%Y-%m-%d %H:%M"))
    sha = ctx.store.put(case["id"], f"report_v{version}", html.encode("utf-8"))
    v = ctx.db.add_report(case["id"], doctor["id"], diagnosis, agrees, findings, recommendation,
                          ai["id"] if ai else None, sha, amendment_reason)
    ctx.db.audit(doctor["id"], "report_signed" if v == 1 else "report_amended", case["id"],
                 {"version": v, "diagnosis": diagnosis, "agrees_with_ai": agrees}, ip)
    return v

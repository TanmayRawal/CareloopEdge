"""Clinical workflow (no HTTP here, so it is testable directly).

Upload -> AI analysis -> doctor worklist -> doctor review -> signed report -> patient sees the report.
Every scan attempt n of a case is stored as blobs  scan_n, original_n, cam_n, lesion_n, cam_lesion_n
and one ai_results row; earlier attempts are kept (medical records are append-only).
A whole series (several DICOM files, a .zip, or NIfTI) is analysed as one study (Engine.analyze_series); it adds
s{i}_original_n for every analysed slice and s{i}_{cam,lesion,cam_lesion}_n for the key slices.
"""

from __future__ import annotations

import hashlib
import io
import threading
import time
import zipfile
from dataclasses import dataclass

from flask import render_template

from edge.portal import common, notify, scheduling

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"}
SERIES_EXT = (".zip", ".nii", ".nii.gz")
MAX_UPLOAD = 20 * 1024 * 1024
MAX_SERIES = 200 * 1024 * 1024
MAX_SERIES_FILES = 2000
# One whole-study analysis at a time: a wide field-of-view volume needs ~1 GB at peak, and the Pi also runs the
# LLM. A second series upload waits for the first (gunicorn timeout 180 s); single images are not queued.
SERIES_LOCK = threading.Lock()
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


def is_series_name(name: str) -> bool:
    return (name or "").strip().lower().endswith(SERIES_EXT)


def _validate(filename: str, data: bytes) -> str:
    name = (filename or "").strip()
    if is_series_name(name):
        if not data:
            raise UploadError("The file is empty.")
        if len(data) > MAX_SERIES:
            raise UploadError("Series too large (max 200 MB).")
        return name
    ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if ext not in ALLOWED_EXT:
        raise UploadError(f"Unsupported file type '{ext or '?'}'. Use PNG, JPG, DICOM, or a series (.zip, .nii.gz).")
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > MAX_UPLOAD:
        raise UploadError("File too large (max 20 MB).")
    return name


def combine_files(files: list[tuple[str, bytes]]) -> tuple[str, bytes]:
    """Upload form -> (filename, bytes). Several files are the slices of one DICOM series: they are packed into one
    .zip (stored, not compressed), which is what is archived and analysed."""
    files = [(n or "", d) for n, d in files if n or d]
    if len(files) <= 1:
        return files[0] if files else ("", b"")
    if len(files) > MAX_SERIES_FILES:
        raise UploadError(f"Too many files (max {MAX_SERIES_FILES} slices).")
    if any(is_series_name(n) for n, _ in files):
        raise UploadError("Upload one series at a time: either one .zip / NIfTI file, or the DICOM files of one series.")
    if sum(len(d) for _, d in files) > MAX_SERIES:
        raise UploadError("Series too large (max 200 MB).")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for i, (_, d) in enumerate(files):
            z.writestr(f"IM{i:05d}.dcm", d)          # original names are not kept (they can carry identity)
    return f"series_{len(files)}_files.zip", buf.getvalue()


def _analyse_into(ctx: Ctx, cid: str, attempt: int, data: bytes, filename: str) -> dict:
    """Run the AI on an attempt, store the scan + overlays encrypted, record the result."""
    try:
        if is_series_name(filename):
            with SERIES_LOCK:
                res = ctx.engine.analyze_series(data, filename=filename)
        else:
            res = ctx.engine.analyze(data, filename=filename)
    except ValueError as e:                          # undecodable / unsupported content
        what = "a scan series" if is_series_name(filename) else "an image"
        raise UploadError(f"Could not read this file as {what}: {e}") from e
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
    from edge.portal import careplan
    if careplan.complete_matching(ctx.db, patient["id"], ("repeat_scan",), "New scan received"):
        ctx.db.audit(patient["id"], "plan_completed", cid, {"kind": "repeat_scan"}, ip)
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
    notify.notify(ctx.db, ctx.db.user(case["patient_id"]), "reupload", {"message": message[:300]},
                  f"/patient/case/{case['id']}", dedupe=f"reupload:{case['id']}:{int(time.time())}")


# ------------------------------------------------------------------ radiology report
# Section order follows the ACR practice parameter for communication of diagnostic imaging findings:
# indication, technique, comparison, findings, impression, recommendation.
SECTIONS = ("indication", "technique", "comparison", "impression")
NORMAL_TEMPLATE = ("Brain parenchyma: Normal grey-white matter differentiation. No focal lesion.\n"
                   "Ventricles and CSF spaces: Normal in size and configuration.\n"
                   "Midline: Central. No mass effect.\n"
                   "Extra-axial spaces: No collection.\n"
                   "Sella and pituitary: Normal.\n"
                   "Posterior fossa: Normal.\n"
                   "Orbits, paranasal sinuses and mastoids: Unremarkable within the field of view.")
LESION_TEMPLATE = ("Lesion: [location], measuring [AP x TR x CC] mm. [Signal and enhancement characteristics].\n"
                   "Perilesional oedema: [none / mild / moderate / marked].\n"
                   "Mass effect: [sulcal effacement / ventricular compression / midline shift of _ mm].\n"
                   "Ventricles and CSF spaces: \n"
                   "Rest of the brain parenchyma: \n"
                   "Sella and pituitary: \n"
                   "Posterior fossa: \n"
                   "Orbits, paranasal sinuses and mastoids: ")


def technique_text(ai: dict | None) -> str:
    """Default 'Technique' from what was actually received (the doctor can edit it)."""
    s = (ai or {}).get("result", {}).get("series")
    if not s:
        return "Single two-dimensional MR image of the brain submitted for review."
    what = "DICOM series" if s.get("source") == "dicom" else "NIfTI volume"
    sp = s.get("spacing_mm") or []
    spacing = f", voxel spacing {' x '.join(f'{v:g}' for v in sp)} mm" if len(sp) == 3 else ""
    return (f"MRI of the brain received as a {what}{spacing}. Reviewed in axial, coronal and sagittal planes "
            f"({s.get('n_slices', 0)} representative sections).")


def comparison_text(db, case) -> str:
    """Default 'Comparison': earlier reported MRI studies of this patient at this clinic."""
    prior = db.q("SELECT id, created FROM cases WHERE patient_id = ? AND id != ? AND created < ? AND status = 'finalized' "
                 "ORDER BY created DESC LIMIT 3", (case["patient_id"], case["id"], case["created"]))
    if not prior:
        return "None available."
    zone = scheduling.tz(db)
    return "; ".join(f"MRI brain of {scheduling.local(p['created'], zone):%d %b %Y} (accession {p['id'][:8].upper()})"
                     for p in prior) + "."


def report_doc(db, case, rep, doctor, version: int, signed_at: float, ai_version: str | None) -> dict:
    """What the report document shows. The archived snapshot and the patient's portal view use the same one."""
    patient = db.user(case["patient_id"])
    profile = db.one("SELECT * FROM patient_profiles WHERE user_id = ?", (case["patient_id"],))
    dob = (profile["dob"] if profile else None) or ""
    age = common.age_from_dob(dob or None)
    sex = (profile["sex"] if profile else None) or ""
    zone = scheduling.tz(db)
    agrees = rep["agrees_with_ai"]
    return {
        "clinic": db.setting("clinic_name"), "address": db.setting("clinic_address"), "phone": db.setting("clinic_phone"),
        "patient": patient["name"], "mrn": common.mrn(patient["id"]),
        "age_sex": " / ".join(x for x in (f"{age} y" if age is not None else "", sex) if x) or "—", "dob": dob or "—",
        "accession": case["id"][:8].upper(), "exam": "MRI Brain",
        "received": f"{scheduling.local(case['created'], zone):%d %b %Y}",
        "reported": f"{scheduling.local(signed_at, zone):%d %b %Y, %H:%M}",
        "status": "Amended" if version > 1 else "Final", "version": version, "amendment_reason": rep["amendment_reason"],
        "indication": rep["indication"] or case["patient_note"] or "Not provided.",
        "technique": rep["technique"] or "Not recorded.", "comparison": rep["comparison"] or "None available.",
        "findings": rep["findings"], "diagnosis": common.CLASS_TEXT.get(rep["final_diagnosis"], rep["final_diagnosis"]),
        "impression": rep["impression"] or "", "recommendation": rep["recommendation"] or "",
        "doctor": doctor["name"], "credentials": doctor["credentials"] or "Radiologist", "reg_no": doctor["reg_no"] or "",
        # a disclosure only: the AI output itself is never part of the patient's document
        "ai_note": (f"Computer-aided image analysis (CareLoop-Edge model {ai_version}) was available to the reporting "
                    "radiologist as decision support. This report is the radiologist's own interpretation.")
                   if agrees is not None and ai_version else "",
        "sha": (case["scan_sha256"] or "")[:16],
    }


def sign_report(ctx: Ctx, case, doctor, diagnosis: str, agrees: bool | None, findings: str, recommendation: str,
                amendment_reason: str | None, ip: str | None, sections: dict | None = None) -> int:
    """Render the immutable report snapshot (no AI output: it is the patient's document), store it encrypted,
    and record the structured report with the snapshot's SHA-256. sections: indication, technique, comparison,
    impression."""
    if not findings.strip():
        raise ValueError("Findings are required.")
    patient = ctx.db.user(case["patient_id"])
    ai = ctx.db.latest_ai(case["id"])
    version = ctx.db.next_report_version(case["id"])
    sections = {k: ((sections or {}).get(k) or "").strip() for k in SECTIONS}
    rep = {"final_diagnosis": diagnosis, "agrees_with_ai": agrees, "findings": findings.strip(),
           "recommendation": (recommendation or "").strip(), "amendment_reason": amendment_reason, **sections}
    doc = report_doc(ctx.db, case, rep, doctor, version, time.time(), ai["model_version"] if ai else None)
    html = render_template("portal/report_snapshot.html", doc=doc)
    sha = ctx.store.put(case["id"], f"report_v{version}", html.encode("utf-8"))
    v = ctx.db.add_report(case["id"], doctor["id"], diagnosis, agrees, findings, recommendation,
                          ai["id"] if ai else None, sha, amendment_reason, sections)
    ctx.db.audit(doctor["id"], "report_signed" if v == 1 else "report_amended", case["id"],
                 {"version": v, "diagnosis": diagnosis, "agrees_with_ai": agrees}, ip)
    if v == 1:                                          # amendments are visible in the portal; no repeat e-mail
        notify.notify(ctx.db, patient, "report_ready",
                      {"exam": "MRI Brain", "accession": doc["accession"], "doctor": doctor["name"],
                       "received": doc["received"]},
                      f"/patient/case/{case['id']}", dedupe=f"report:{case['id']}")
    return v

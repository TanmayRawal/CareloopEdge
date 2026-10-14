"""Encrypted case files shared by patients and staff: scan / overlay images and archived report snapshots."""

from __future__ import annotations

from flask import Blueprint, abort, g

from edge.portal import auth


def blueprint(d) -> Blueprint:
    bp = Blueprint("case", __name__, url_prefix="/case")
    db, store = d.db, d.store

    @bp.get("/<cid>/img/<kind>.png")
    @auth.require("patient", "doctor", "admin")
    def image(cid, kind):
        case = d.get_case_or_404(cid)
        base = kind.rsplit("_", 1)[0]
        if g.user["role"] == "patient" and base != "original":      # patients never receive AI overlays
            abort(404)
        if base not in ("original", "cam", "lesion", "cam_lesion") or not store.exists(case["id"], kind):
            abort(404)
        if g.user["role"] != "patient":
            db.audit(g.user["id"], "image_viewed", cid, {"kind": kind}, d.ip())
        return d.app.response_class(store.get(case["id"], kind), mimetype="image/png")

    @bp.get("/<cid>/report/<int:version>")
    @auth.require("patient", "doctor", "admin")
    def report_snapshot(cid, version):
        case = d.get_case_or_404(cid)
        kind = f"report_v{version}"
        if not store.exists(case["id"], kind):
            abort(404)
        db.audit(g.user["id"], "report_viewed", cid, {"version": version}, d.ip())
        resp = d.app.response_class(store.get(case["id"], kind), mimetype="text/html")
        # the archived snapshot is self-contained (embedded CSS); it may never run script
        resp.headers["Content-Security-Policy"] = ("default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; "
                                                   "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        return resp

    return bp

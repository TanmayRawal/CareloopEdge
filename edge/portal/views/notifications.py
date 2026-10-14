"""In-app notification centre (all roles)."""

from __future__ import annotations

from flask import Blueprint, g, redirect, render_template, request, url_for

from edge.portal import auth, notify
from edge.portal.common import PAGE_SIZE, page_args, safe_next


def blueprint(d) -> Blueprint:
    bp = Blueprint("notifications", __name__, url_prefix="/notifications")
    db = d.db

    @bp.get("/")
    @auth.require("patient", "doctor", "admin")
    def index():
        page, offset = page_args(request)
        rows = db.q("SELECT * FROM notifications WHERE user_id = ? ORDER BY created DESC LIMIT ? OFFSET ?",
                    (g.user["id"], PAGE_SIZE + 1, offset))
        return render_template("portal/notifications.html", rows=rows[:PAGE_SIZE], more=len(rows) > PAGE_SIZE, page=page)

    @bp.post("/read")
    @auth.require("patient", "doctor", "admin")
    def read():
        nid = request.form.get("id")
        notify.mark_read(db, g.user["id"], int(nid) if nid and nid.isdigit() else None)
        return redirect(safe_next(request.form.get("next", "")) or url_for("notifications.index"))

    @bp.get("/<int:nid>")
    @auth.require("patient", "doctor", "admin")
    def open(nid):
        n = db.one("SELECT * FROM notifications WHERE id = ? AND user_id = ?", (nid, g.user["id"]))
        if n is None:
            return redirect(url_for("notifications.index"))
        notify.mark_read(db, g.user["id"], nid)
        return redirect(safe_next(n["link"] or "") or url_for("notifications.index"))

    return bp

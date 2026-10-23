"""Device home: the page the Pi's own screen shows (kiosk) and anyone on the network may open.

It carries NO patient data, only the state of the device: model version and signature check, engine status,
temperature / load / disk, the addresses to reach the portal (with a QR code for phones) and the two entry
points. /device?kiosk=1 marks this browser as the device's kiosk (cookie): every portal page then shows a
"device home" button and returns here after a few idle minutes (static/portal.js).
"""

from __future__ import annotations

import os
import shutil
import socket
import time
from pathlib import Path

from flask import Blueprint, make_response, render_template, request

from edge.portal.common import ago, human_bytes
from edge.portal.views.admin import STARTED

KIOSK_COOKIE = "ne_kiosk"


def soc_temp_c() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except (OSError, ValueError):
        return None


def lan_address() -> str | None:
    """The address other devices on the LAN use for this one (no packet is sent: UDP connect only picks a route)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None


def qr_svg(url: str) -> str | None:
    try:
        import segno
    except ImportError:                          # optional dependency: the page works without the QR code
        return None
    return segno.make(url, error="m").svg_inline(scale=4, border=1, dark="#0b1f2e", light="#ffffff", omitsize=True)


def status(d) -> dict:
    info = d.model_info
    temp = soc_temp_c()
    disk = shutil.disk_usage(d.data_dir)
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    last = d.db.one("SELECT MAX(created) AS t FROM ai_results")["t"]
    base = (d.db.setting("base_url") or "").rstrip("/")
    lan = lan_address()
    urls = [u for u in dict.fromkeys([base, f"https://{lan}" if lan else ""]) if u]
    return {
        "model_version": info.get("version", "—"), "verified": bool(info.get("verified")),
        "whole_study": bool(info.get("whole_study")), "precision": info.get("precision") or "—",
        "uptime": ago(STARTED), "last_analysis": ago(last) if last else None,
        "temp_c": temp, "temp_level": None if temp is None else "err" if temp >= 80 else "warn" if temp >= 70 else "ok",
        "load": None if load is None else round(load, 2), "cpus": os.cpu_count() or 1,
        "disk_free": human_bytes(disk.free), "disk_pct_used": round(100 * (disk.total - disk.free) / disk.total),
        "host": socket.gethostname(), "urls": urls, "qr": qr_svg(urls[0]) if urls else None,
    }


def blueprint(d) -> Blueprint:
    bp = Blueprint("device", __name__)

    @bp.get("/device")
    def home():
        resp = make_response(render_template("portal/device.html", s=status(d), now=time.time()))
        kiosk = request.args.get("kiosk")
        if kiosk in ("0", "1"):
            secure = d.app.config.get("SESSION_COOKIE_SECURE", False)
            if kiosk == "1":
                resp.set_cookie(KIOSK_COOKIE, "1", max_age=10 * 365 * 86400, secure=secure, httponly=True,
                                samesite="Strict")
            else:
                resp.delete_cookie(KIOSK_COOKIE, secure=secure, httponly=True, samesite="Strict")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    return bp

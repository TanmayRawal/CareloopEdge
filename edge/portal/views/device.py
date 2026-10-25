"""Device home: the page the Pi's own screen shows (kiosk) and anyone on the network may open.

It carries NO patient data, only the state of the device: model version and signature check, engine status,
temperature / load / disk, the addresses to reach the portal (with a QR code for phones) and the two entry
points. /device?kiosk=1 marks this browser as the device's kiosk (cookie): every portal page then shows a
"device home" button and returns here after a few idle minutes (static/portal.js).
"""

from __future__ import annotations

import re
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
    """What the device screen shows: ready or not, plus the specific problems, if any, in plain words."""
    info = d.model_info
    temp = soc_temp_c()
    disk = shutil.disk_usage(d.data_dir)
    # low = little room in absolute terms (a whole study and its slice images take ~5-100 MB), not a percentage
    low = disk.free < 2 * 1024 ** 3 or (disk.total - disk.free) / disk.total > 0.97
    last = d.db.one("SELECT MAX(created) AS t FROM ai_results")["t"]
    base = (d.db.setting("base_url") or "").rstrip("/")
    lan = lan_address()
    # the address a phone on the clinic network can open: the device's IP address. The configured base URL only if it
    # is not this machine alone (localhost) and not an mDNS ".local" name, which Android phones usually cannot resolve.
    local_only = not base or any(x in base for x in ("localhost", "127.0.0.1", ".local"))
    url = f"https://{lan}" if lan and local_only else (base or (f"https://{lan}" if lan else ""))
    version = str(info.get("version") or "")
    released = bool(re.fullmatch(r"\d+\.\d+\.\d+", version))
    issues = []                 # only problems someone at the device can act on; model provenance is on Admin > System
    if temp is not None and temp >= 80:
        issues.append(f"The device is hot ({temp:.0f} °C). Analysis slows down; check the cooler and airflow.")
    if low:
        issues.append(f"Storage is nearly full ({human_bytes(disk.free)} left). Back up and archive old studies.")
    return {
        "ready": not issues, "issues": issues,
        "model": f"Model {version}" if released else None,
        "verified": bool(info.get("verified")), "whole_study": bool(info.get("whole_study")),
        "last_analysis": ago(last) if last else None, "uptime": ago(STARTED),
        "temp_c": temp, "hot": temp is not None and temp >= 70,
        "disk_free": human_bytes(disk.free), "disk_low": low,
        "lan": lan, "url": url, "qr": qr_svg(url) if url else None, "host": socket.gethostname(),
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

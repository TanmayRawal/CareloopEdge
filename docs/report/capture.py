"""Screenshots for the project report: a scripted walk-through of the real portal, captured with headless Chrome.

    python docs/report/capture.py --base http://127.0.0.1:8090 --data C:/Users/Tanmay/neuroedge_data/portal_report

Runs against a demo portal started with `edge.serve --init-dev --portal-data <data>` (fresh database, test accounts in
<data>/DEV_ACCOUNTS.txt, e-mails saved as .eml files in <data>/mail). A patient registers and uploads a whole MRI study,
the radiologist reviews and signs it with a follow-up, the patient books a consultation, the coordinator works the
follow-up list; every screen and every e-mail along the way is saved as a PNG in docs/report/figures/.
Chrome is driven through the DevTools protocol with a minimal websocket client (no extra packages).
"""

from __future__ import annotations

import argparse
import base64
import email
import functools
import http.server
import json
import os
import random
import socket
import sqlite3
import string
import struct
import subprocess
import tempfile
import threading
import time
import urllib.request
from email import policy
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
STUDY = r"C:\Users\Tanmay\neuroedge_data\test_uploads\meningioma_1.nii.gz"


# ---------------------------------------------------------------------------------------------- minimal websocket
class WS:
    def __init__(self, url: str):
        assert url.startswith("ws://")
        host, _, path = url[5:].partition("/")
        h, _, p = host.partition(":")
        self.s = socket.create_connection((h, int(p)), timeout=120)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET /{path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += self.s.recv(4096)
        assert b" 101 " in resp.split(b"\r\n")[0], resp[:200]
        self.buf = resp.split(b"\r\n\r\n", 1)[1]

    def _read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.s.recv(1 << 20)
            if not chunk:
                raise ConnectionError("websocket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, text: str) -> None:
        data = text.encode()
        head = bytearray([0x81])
        n = len(data)
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126); head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127); head += struct.pack(">Q", n)
        mask = os.urandom(4)
        self.s.sendall(bytes(head) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self) -> str:
        parts = []
        while True:
            b1, b2 = self._read(2)
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(n)
            op = b1 & 0x0F
            if op == 9:                       # ping
                continue
            parts.append(payload)
            if b1 & 0x80:
                return b"".join(parts).decode("utf-8", "replace")


class Chrome:
    def __init__(self, port: int = 9333):
        self.profile = tempfile.mkdtemp(prefix="report-chrome-")
        self.proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}",
                                      f"--user-data-dir={self.profile}", "--hide-scrollbars", "--no-first-run",
                                      "--force-device-scale-factor=1", "--window-size=1440,900", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        self.ws = WS(page["webSocketDebuggerUrl"])
        self.n = 0
        for m in ("Page.enable", "Runtime.enable", "DOM.enable"):
            self.call(m)
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)

    def call(self, method: str, **params):
        self.n += 1
        mid = self.n
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS error: {r['exceptionDetails'].get('exception', {}).get('description', r)}")
        return r.get("result", {}).get("value")

    def settle(self, extra: float = 0.6) -> None:
        for _ in range(200):
            if self.js("document.readyState") == "complete":
                break
            time.sleep(0.1)
        self.js("Promise.all([document.fonts ? document.fonts.ready : 0, ...[...document.images].map(i => i.complete ? 0 :"
                " new Promise(r => { i.onload = i.onerror = r; }))]).then(() => true)")
        time.sleep(extra)

    def size(self, w: int, h: int) -> None:
        self.call("Emulation.setDeviceMetricsOverride", width=w, height=h, deviceScaleFactor=1, mobile=False)

    def go(self, url: str, extra: float = 0.6) -> None:
        self.call("Page.navigate", url=url)
        time.sleep(0.3)
        self.settle(extra)

    def submit(self, js_fill_and_submit: str, extra: float = 0.8) -> None:
        before = self.js("performance.timeOrigin")
        self.js(js_fill_and_submit)
        for _ in range(600):                      # wait for the next document (a whole study takes a few seconds)
            time.sleep(0.1)
            try:
                if self.js("performance.timeOrigin") != before:
                    break
            except Exception:
                pass
        self.settle(extra)

    def shot(self, name: str, w: int = 1440, h: int = 900, full: bool = False, max_h: int = 2600) -> Path:
        self.size(w, h)
        time.sleep(0.25)
        if full:
            hh = min(max_h, int(self.js("Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)")))
            self.size(w, max(h, hh))
            time.sleep(0.35)
        data = self.call("Page.captureScreenshot", format="png", captureBeyondViewport=full)["data"]
        out = FIG / f"{name}.png"
        out.write_bytes(base64.b64decode(data))
        self.size(w, h)
        print(f"  {out.name}")
        return out

    def close(self) -> None:
        self.proc.terminate()


# ---------------------------------------------------------------------------------------------- helpers
def q(s: str) -> str:
    return json.dumps(s)


def fill(fields: dict) -> str:
    """JS that sets form fields by name (inputs, selects, textareas, radios, checkboxes)."""
    parts = []
    for name, val in fields.items():
        parts.append(f"""(() => {{ const els = document.querySelectorAll('[name={q(name)}]');
          for (const e of els) {{
            if (e.type === 'radio') {{ if (e.value === {q(str(val))}) e.checked = true; }}
            else if (e.type === 'checkbox') e.checked = {str(bool(val)).lower()};
            else if (e.tagName === 'SELECT' && ![...e.options].some(o => o.value === {q(str(val))}))
              e.value = ([...e.options].find(o => o.text.startsWith({q(str(val))})) || {{}}).value;
            else e.value = {q(str(val))};
            e.dispatchEvent(new Event('change', {{ bubbles: true }}));
          }} }})();""")
    return "\n".join(parts)


def login(c: Chrome, base: str, email_: str, pw: str, staff: bool) -> None:
    c.go(f"{base}/login" + ("?as=staff" if staff else ""))
    c.submit(fill({"email": email_, "password": pw}) + "document.querySelector('form.form, form').requestSubmit();")


def logout(c: Chrome) -> None:
    c.submit("document.querySelector('form[action$=\"/logout\"]').requestSubmit();")


def accounts(data: Path) -> dict:
    out = {}
    for line in (data / "DEV_ACCOUNTS.txt").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) == 3 and "@" in p[1]:
            out[p[0]] = (p[1], p[2])
    return out


def serve_dir(d: Path, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(d))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


def mails(data: Path) -> list[tuple[float, str, str, str]]:
    out = []
    for p in sorted((data / "mail").glob("*.eml")):
        m = email.message_from_bytes(p.read_bytes(), policy=policy.default)
        body = m.get_body(preferencelist=("html", "plain"))
        out.append((p.stat().st_mtime, str(m["Subject"]), str(m["To"]),
                    body.get_content() if body else ""))
    return out


# ---------------------------------------------------------------------------------------------- walk-through
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--data", type=Path, default=Path("C:/Users/Tanmay/neuroedge_data/portal_report"))
    ap.add_argument("--emails-only", action="store_true", help="only re-render the e-mails already sent")
    a = ap.parse_args()
    base, data = a.base, a.data
    FIG.mkdir(parents=True, exist_ok=True)
    acc = accounts(data)
    doc_email, doc_pw = acc["doctor"]
    adm_email, adm_pw = acc["admin"]
    pat_email = "meera.joshi@example.test"
    pat_pw = "".join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(16))
    c = Chrome()
    try:
        if not a.emails_only:
          print("device home and sign-in")
          c.go(f"{base}/device", 1.5)
          c.shot("01_device_home")
          c.shot("01b_device_home_7inch", 800, 480)
          c.go(f"{base}/login")
          c.shot("02_login")

          print("patient: register, upload a whole study")
          c.go(f"{base}/register")
          c.js(fill({"name": "Meera Joshi", "email": pat_email, "dob": "1979-05-14", "sex": "F", "phone": "98290 41177",
                     "password": pat_pw, "password2": pat_pw, "consent": True}))
          c.shot("03_register")
          c.submit("document.querySelector('form').requestSubmit();")
          c.shot("04_patient_dashboard_new")
          c.go(f"{base}/patient/upload")
          doc = c.call("DOM.getDocument")["root"]["nodeId"]
          node = c.call("DOM.querySelector", nodeId=doc, selector="#scan")["nodeId"]
          c.call("DOM.setFileInputFiles", files=[STUDY], nodeId=node)
          c.js("document.querySelector('#scan').dispatchEvent(new Event('change', {bubbles: true}));" +
               fill({"note": "Headaches for three months; MRI done at the district hospital.", "consent": True}))
          c.shot("05_patient_upload")
          c.submit("document.querySelector('#upload-form').requestSubmit();", 1.0)
          c.shot("06_patient_case_received", full=True)
          cid = c.js("location.pathname.split('/').pop()")
          logout(c)

          print("radiologist: worklist, case review, sign with follow-up")
          login(c, base, doc_email, doc_pw, staff=True)
          c.shot("07_doctor_worklist")
          c.go(f"{base}/doctor/case/{cid}", 1.5)
          c.shot("08_doctor_case_review")
          c.js("document.querySelector('.v-modes button[data-kind^=\"cam_lesion\"]')?.click()")
          time.sleep(0.8)
          c.shot("09_doctor_case_combined")
          c.js("const b=[...document.querySelectorAll('#slice-bars span')].find(s=>!s.classList.contains('key')); b && b.click()")
          time.sleep(0.8)
          c.shot("10_doctor_case_other_slice")
          c.js("document.querySelector('.v-modes button')?.click(); document.querySelector('.vs-keys button')?.click()")
          c.js(fill({"diagnosis": "meningioma", "agrees": "yes",
                     "findings": "Extra-axial, dural-based enhancing mass over the left frontal convexity with a dural tail, "
                                 "about 3 cm. Mild mass effect, no midline shift. Appearances typical of a meningioma.",
                     "recommendation": "Neurosurgical consultation; contrast-enhanced MRI in 6 months.",
                     "fu_kind": "consultation", "fu_days": "14", "password": doc_pw}))
          c.js("document.querySelector('form[action$=\"/sign\"]').scrollIntoView({block: 'start'})")
          time.sleep(0.4)
          c.shot("11_doctor_sign_report", full=True)
          c.submit("document.querySelector('form[action$=\"/sign\"]').requestSubmit();")
          c.shot("12_doctor_signed")
          # consultation hours for the demo radiologist (Mon-Sat 10:00-13:00, 14:00-17:00), set before the booking step
          doc_id = sqlite3.connect(data / "portal.db").execute("SELECT id FROM users WHERE email = ?", (doc_email,)).fetchone()[0]
          with sqlite3.connect(data / "portal.db") as db:
              if not db.execute("SELECT 1 FROM doctor_hours WHERE doctor_id = ?", (doc_id,)).fetchone():
                  db.executemany("INSERT INTO doctor_hours (doctor_id, weekday, start_min, end_min, slot_min, room) "
                                 "VALUES (?,?,?,?,?,?)", [(doc_id, d, s, e, 20, "OPD 3")
                                                          for d in range(6) for s, e in ((600, 780), (840, 1020))])
          c.go(f"{base}/doctor/availability")
          c.shot("13_doctor_availability", full=True)
          logout(c)

          print("patient: report, next step, booking")
          login(c, base, pat_email, pat_pw, staff=False)
          c.shot("14_patient_dashboard_next_step", full=True)
          c.go(f"{base}/patient/case/{cid}")
          c.shot("15_patient_report", full=True)
          c.go(f"{base}/patient/book/{cid}")
          c.js("document.querySelectorAll('input[name=start_ts]')[2].click();" + fill({"reason": "Discuss the report and the next steps"}))
          time.sleep(0.4)
          c.shot("16_patient_booking", full=True)
          c.submit("document.querySelector('#book-btn').click();")
          c.shot("17_patient_booked")
          c.go(f"{base}/patient/appointments")
          c.shot("18_patient_appointments")
          logout(c)

          print("radiologist: schedule and analytics")
          login(c, base, doc_email, doc_pw, staff=True)
          c.go(f"{base}/doctor/schedule")
          c.shot("19_doctor_schedule")
          c.go(f"{base}/doctor/stats?period=all", 1.0)
          c.shot("20_analytics", full=True, max_h=3600)
          logout(c)

          print("care coordinator / administrator")
          login(c, base, adm_email, adm_pw, staff=True)
          c.go(f"{base}/care/")
          c.shot("21_followup_worklist", full=True)
          pid = sqlite3.connect(data / "portal.db").execute(
              "SELECT patient_id FROM care_plans ORDER BY id LIMIT 1").fetchone()[0]
          c.go(f"{base}/care/patients/{pid}")
          c.shot("22_patient_record", full=True)
          c.go(f"{base}/care/patients/new")
          c.shot("23_walkin_registration", full=True)
          c.go(f"{base}/care/referrals")
          c.shot("24_referrals")
          for path, name in (("/admin/users", "25_admin_users"), ("/admin/audit", "26_audit_log"),
                             ("/admin/system", "27_admin_system"), ("/admin/outbox", "28_email_outbox"),
                             ("/admin/settings", "29_admin_settings")):
              c.go(base + path)
              c.shot(name, full=True)
          logout(c)

          print("7-inch kiosk views")
          c.go(f"{base}/device?kiosk=1", 1.2)
          login(c, base, doc_email, doc_pw, staff=True)
          c.size(800, 480)
          c.go(f"{base}/doctor/case/{cid}", 1.5)
          c.shot("30_kiosk_case_review", 800, 480)
          c.go(f"{base}/doctor/", 0.8)
          c.shot("31_kiosk_worklist", 800, 480)
          logout(c)
          c.go(f"{base}/login?as=staff")
          c.js("document.querySelector('input[type=email]').focus();")
          time.sleep(0.5)
          c.shot("32_kiosk_keyboard", 800, 480)

        print("e-mails")
        md = Path(tempfile.mkdtemp(prefix="report-mail-"))
        serve_dir(md, 8091)
        wanted = [m for m in mails(data) if m[2].lower() == pat_email]
        for i, (_, subj, _, body) in enumerate(wanted[:6], 1):
            html = body if "<html" in body.lower() else f"<pre style='font:15px/1.5 sans-serif;white-space:pre-wrap'>{body}</pre>"
            frame = (f"<html><head><meta charset='utf-8'></head><body style='margin:0;background:#eceef1;font-family:Segoe UI,Arial,sans-serif'>"
                     f"<div style='max-width:760px;margin:24px auto;background:#fff;border:1px solid #d6dade'>"
                     f"<div style='padding:14px 20px;border-bottom:1px solid #e4e7ea;font-size:14px;color:#3c434b'>"
                     f"<b>From:</b> {acc_label(data)}<br><b>To:</b> {pat_email}<br><b>Subject:</b> {subj}</div>"
                     f"<div>{html}</div></div></body></html>")
            (md / f"m{i}.html").write_text(frame, encoding="utf-8")
            c.go(f"http://127.0.0.1:8091/m{i}.html", 0.5)
            c.shot(f"40_email_{i}", 860, 700, full=True)
        (FIG / "emails.json").write_text(json.dumps([s for _, s, _, _ in wanted[:6]], indent=1), encoding="utf-8")
    finally:
        c.close()


def acc_label(data: Path) -> str:
    return "Radiology Department &lt;noreply@clinic&gt;"


if __name__ == "__main__":
    main()

# Deploying NeuroEdge on a Raspberry Pi 5 (8 GB)

What runs on the Pi, all offline after setup:

| Service | What | Port |
|---|---|---|
| `caddy` | HTTPS for the portal (Caddy local CA, certificates issued on demand for the Pi's IP / `<hostname>.local`); HTTP only redirects and serves the CA certificate | `:443`, `:80` (LAN) |
| `neuroedge-web` | Clinic portal (Flask under gunicorn): patient upload → INT8 ONNX classifier + segmentation, exact Grad-CAM, lesion outline, conformal set, OOD check → doctor worklist → signed report; doctor assistant | `127.0.0.1:8080` (loopback only) |
| `neuroedge-firewall` | nftables table `inet neuroedge`: only SSH, HTTP(S), mDNS, DHCP and ICMP inbound | — |
| `neuroedge-llm` | llama.cpp server with Qwen3-1.7B Q4_K_M (assistant narratives and Q&A) | `127.0.0.1:8081` (loopback only) |
| `neuroedge-sync.timer` | daily check for a new signed + encrypted model; a new version is trial-decrypted and verified before it is activated | — |
| `neuroedge-web-restart.path` | restarts the web app when a newly verified model version is activated | — |
| `neuroedge-reminders.timer` | every 5 min: reminder rules (unread/unsigned studies, appointment reminders, daily agenda, attendance, no-shows) and e-mail delivery from the outbox | outbound SMTP only |

The model is stored only as ciphertext on the Pi. It is decrypted and its signature verified in RAM at startup.
Patient data (portal database, encrypted scans, overlays and signed reports) lives in `/var/lib/neuroedge/portal`;
its key is `/etc/neuroedge/portal.key`. Memory budget: web app ≈ 0.3–1 GB + LLM ≈ 1.6 GB (1.1 GB weights + 0.47 GB
KV cache for 4096 tokens), so under 3 GB of the 8 GB.

## 1. Hardware and OS
* Raspberry Pi 5, 8 GB, with the **Active Cooler** (sustained inference throttles without it), the official
  27 W USB-C supply, and a microSD card of 32 GB or more (an NVMe HAT is faster but optional).
* Flash **Raspberry Pi OS Lite (64-bit)** with Raspberry Pi Imager. In the Imager settings, set a hostname
  (e.g. `neuroedge`), a user, Wi-Fi or Ethernet, and enable SSH.
* Boot, then: `sudo apt update && sudo apt full-upgrade -y && sudo reboot`.

## 2. Copy the code (and optionally the LLM file) to the Pi
From the PC (PowerShell or Git Bash), in the project folder:
```bash
ssh <user>@neuroedge.local "mkdir -p ~/neuroedge"
```
```bash
scp -r neuroedge edge requirements-edge.txt pyproject.toml <user>@neuroedge.local:~/neuroedge/
```
To avoid downloading the 1.1 GB LLM again, copy the copy that is already verified on the PC:
```bash
scp C:/Users/Tanmay/neuroedge_data/llm/Qwen3-1.7B-Q4_K_M.gguf <user>@neuroedge.local:~/
```
For benchmarking, also copy 40 held-out test scans (made by `python -m training.export_bench_samples`):
```bash
scp -r C:/Users/Tanmay/neuroedge_data/bench_samples <user>@neuroedge.local:~/
```

## 3. Install
On the Pi:
```bash
cd ~/neuroedge
sudo bash edge/install_pi.sh --gguf ~/Qwen3-1.7B-Q4_K_M.gguf
```
* Building llama.cpp takes about 10–15 minutes on the Pi 5. It is pinned to release `b11320`, the build the
  assistant was evaluated with.
* The GGUF file is checked against its pinned SHA-256 and refused if it differs.
* It also installs Caddy (HTTPS), the firewall, the portal key and database, and prints the portal CA fingerprint.
  PyTorch is not needed on the Pi (`--with-deep-explain` exists only for research tools).
* Re-running the script is safe; use it to upgrade the code.

Check: `systemctl status neuroedge-llm` should be active. The web app waits until a model is installed.

## 4. Get the model onto the Pi
**Normal (secure) path.** The registry server runs on your PC or a VPS. One-time server setup, from the repo root
(in Git Bash, prefix commands with `MSYS_NO_PATHCONV=1` so `/data` is not rewritten into a Windows path):
```bash
printf 'DOMAIN=<pc-hostname-or-LAN-IP>
TLS=internal
HTTP_PORT=8088
HTTPS_PORT=8443
' > server/.env
```
```bash
docker compose -f server/docker-compose.yml --env-file server/.env run --rm api python -m server.manage init --data /data
```
(prints the admin token once; store it safely)
```bash
docker compose -f server/docker-compose.yml --env-file server/.env up -d --build
```
```bash
docker compose -f server/docker-compose.yml cp caddy:/data/caddy/pki/authorities/local/root.crt keys/caddy_root.crt
```
Keep the private signing key and the admin token **outside any cloud-synced folder** (OneDrive, Dropbox, Google
Drive): here `~/neuroedge_secrets/` with access for your account only. If a secret was ever synced, rotate it:
a new signing key (`manage signing-key`), re-publish, update the devices' pinned `signing_pub.pem`, and a new admin
token (`manage init` inside the container).

Publish a model from the training PC (the signing key never leaves it):
```bash
NE_ADMIN_TOKEN=<token> python -m server.publish --bundle <bundle_dir> --name mri --version 0.2.0 --server https://<host>:8443 --signing-key ~/neuroedge_secrets/signing/signing_priv.pem
```
Verify the deployment end to end (publish, enroll, pull, attacker checks):
```bash
NE_ADMIN_TOKEN=<token> python -m server.e2e_check --server https://<host>:8443 --ca keys/caddy_root.crt --bundle <bundle_dir> --signing-pub keys/signing_pub.pem --version 0.2.0 --skip-publish
```
Then on the Pi:
1. Copy the publisher's *public* signing key `signing_pub.pem` to the Pi. Never copy the private key.
2. If the server uses Caddy's internal certificate on a LAN, give the installer its root CA:
   `sudo bash edge/install_pi.sh --ca-cert caddy_root.crt` (re-run; it is idempotent).
3. On the server machine, create a one-time enrollment token:
   `curl -X POST -H "Authorization: Bearer $NE_ADMIN_TOKEN" https://<registry>:8443/v1/admin/enrollment-tokens`
4. On the Pi:
   ```bash
   sudo install -m 640 -g neuroedge signing_pub.pem /etc/neuroedge/signing_pub.pem
   sudo neuroedge-sync enroll --server https://<registry> --token <token> --name pi5-01 --signing-pub /etc/neuroedge/signing_pub.pem
   sudo neuroedge-sync pull --model mri
   sudo neuroedge-sync status        # "mri: version ... verified OK"
   ```
   The web app starts automatically once the pull succeeds.

**Development shortcut (no server).** This serves a plaintext bundle directory. The model then sits
unencrypted on the SD card, so use it only for a first smoke test:
```bash
scp -r C:/Users/Tanmay/neuroedge_data/bundles/mri_v1_0.2.0_int8 <user>@neuroedge.local:~/
sudo bash edge/install_pi.sh --dev-bundle ~/mri_v1_0.2.0_int8
```

## 5. Use it
1. **Trust the portal's certificate once per device.** Download `http://<pi-ip>/neuroedge-ca.crt`. Compare its
   SHA-256 fingerprint with `sudo neuroedge-ca` on the Pi, then install it as a trusted root (Windows: Install
   Certificate → Current User → Trusted Root Certification Authorities; Android/iOS: install as CA certificate).
2. **Create the first administrator** (password typed interactively, never on the command line):
   `sudo neuroedge-portal create-user --role admin --email you@clinic.org --name "Your Name"`.
   Add doctors from the admin **Users** page. Patients register themselves.
3. Open `https://<pi-ip>/` or `https://<hostname>.local/` from any device on the same network.

### E-mail reminders (optional but recommended)
Every reminder is always shown inside the portal (notification bell). To also send e-mails:
```bash
sudo neuroedge-portal smtp-config           # host, port, security, user, password (hidden), from address
sudo neuroedge-portal smtp-test --to you@hospital.org
```
* **Hospital mail server:** use its SMTP host and port (usually 587 + STARTTLS).
* **Gmail:**
  * host `smtp.gmail.com`, port `587`, `starttls`;
  * the password must be a Google **App Password** (Google Account → Security → 2-Step Verification →
    App passwords), never the account password.
* **Where the credentials live:** `/etc/neuroedge/smtp.env` (root:neuroedge, mode 640). They are never in the
  database or the web UI.
* **Offline:** messages wait in the outbox (Admin → E-mail) and are retried with backoff.
* **Content:** e-mails contain no diagnoses or findings, only "your report is ready", appointment times and
  links.
* **Clock:** reminders depend on the Pi's clock. The Pi has no battery-backed clock, so it needs network time
  (NTP, on by default). Check it with `timedatectl`.
* **Department policy** (reporting targets, reminder times, booking window, cancellation cutoff, time zone,
  portal address used in links) is set under **Admin → Settings**.

### Consultations
* Each radiologist sets weekly hours, slot length, room and days off under **Availability**.
* After a report is signed, the patient can book a follow-up consultation with the reporting radiologist.
  They get a confirmation with a calendar invite, plus reminders (24 h and 2 h before by default).
* The radiologist marks **Arrived / Completed / No-show** on the **Schedule**. A no-show sends the patient a
  rebooking invitation.

### Follow-up (CareLoop)
* While signing a report, the radiologist chooses the **next step**: consultation, repeat MRI or referral, and
  when it is due. A referral letter is signed with the same password and shared with the patient.
* The **administrator is the care coordinator**. **Follow-up** lists every patient with an open plan or
  referral, ranked by the risk score, with the reasons shown. The coordinator can:
  * log calls;
  * send reminders (only for plans a doctor approved);
  * book consultations for the patient;
  * track referrals.
* Plans the coordinator proposes wait under the doctor's **Approvals** until a doctor approves them.
* Plans close automatically when the consultation is completed, a new scan arrives or the referral is attended.
  Approved plans overdue for `lost_after_days` with no contact are marked **lost to follow-up**.
* **Walk-in patients:** Patients → Register patient. An e-mail is optional. With an e-mail, the patient gets an
  invitation link to set a password.
* Reminder e-mails carry **I'll be there / I can't make it** links (single use). "Can't make it" frees the slot
  (if still before the cutoff) and alerts the coordinator.

### Staff sign-in and audit
* Each doctor or administrator can turn on **two-step sign-in** under Account, using any authenticator app.
  To make it mandatory, set **Admin → Settings → Require two-step sign-in = 1**.
* If a staff member loses their phone and their recovery codes, reset it from the console:
  `sudo neuroedge-portal reset-2fa --email <their address>`.
* The audit log is hash-chained. **Admin → Audit log** and **System** show whether it is intact. From the console:
  `sudo neuroedge-portal verify-audit` (exits with an error if any entry was altered or removed).
* Patients can download their data (Account → Download my data) and request account deletion. The request
  appears on **Admin → Users**. Closing the account stops sign-in and messages; clinical records are kept as
  the law requires.

## 5b. Showcase kiosk (the Pi's own 7" screen)
The screen shows a **device home** page (`/device`) with:
* the model version and signature check, and engine / assistant status;
* temperature and storage;
* the portal address with a QR code for phones;
* two touch buttons, Clinician and Patient.

It shows no patient data. Every portal page gets a "device home" button, and the screen returns home after 3 idle
minutes. The portal adapts to 800×480, 1024×600 and 1280×720 (tabs on the case page, touch-sized controls).

**Hardware:**
* a case that fits a Pi 5 **with the Active Cooler**;
* the 27 W supply (the screen draws from the Pi).

| Screen | Cable on a Pi 5 | Installer option |
|---|---|---|
| Original 7" Touch Display (800×480) | Raspberry Pi 5 display cable (22-pin to 15-pin) | `--display td1` |
| Touch Display 2 (720×1280 portrait panel) | included | `--display td2 --rotate 90` |
| 7" HDMI (1024×600) | micro-HDMI to HDMI | `--display hdmi` |

```bash
sudo bash edge/install_pi.sh --kiosk --display td1
```
```bash
sudo reboot
```
* After the reboot, the screen logs in as the locked-down user `kiosk` (no password, no sudo, no SSH) and opens
  `https://localhost/device?kiosk=1` full-screen. The portal CA is trusted for that user, so there is no warning.
* The on-screen keyboard (squeekboard) appears when a text field is tapped; a USB keyboard also works.
* Settings (URL, rotation) are in `/etc/neuroedge/kiosk.env`; reboot after editing. On a desktop image the kiosk logs in through LightDM; on Lite, through the console on tty1.
* Pi 5 + original 7" display: the kit's 15-pin ribbon does not fit the Pi 5; use the Raspberry Pi 5 display cable (22-pin to 15-pin). Power the adapter board from the Pi's 5 V / GND pins (jumper wires in the kit).
* If the panel stays black, check `kmsprint` and the cable orientation, and that `/boot/firmware/config.txt` has the
  overlay line.

## 5c. Upgrading to `mri/0.3.0` (whole-study upload)
1. **Back up first:** the portal data, and the installed code for a rollback.
   ```bash
   sudo neuroedge-portal backup --dest ~/backup-pre-0.3.0
   ```
   ```bash
   sudo tar -czf ~/backup-pre-0.3.0/code.tgz -C /opt/neuroedge src
   ```
   Then copy the backup folder to another machine (the portal backup is encrypted).
2. Copy the new code (section 2) and re-run the installer, adding `--kiosk` if wanted. This also installs `nibabel`
   (NIfTI upload) and `segno` (QR code).
3. Pull and verify the new model:
   ```bash
   sudo neuroedge-sync pull --model mri
   ```
   ```bash
   sudo neuroedge-sync status
   ```
   Expect `mri: version 0.3.0 verified OK`; the web app restarts by itself.
4. **Benchmark whole studies** (target under 25 s per study):
   ```bash
   cd /opt/neuroedge/src && sudo -u neuroedge env NEUROEDGE_HOME=/var/lib/neuroedge /opt/neuroedge/venv/bin/python -m edge.bench --series ~/test_uploads --llm 0
   ```
5. **Rollback:** restore the code with `sudo tar -xzf ~/backup-pre-0.3.0/code.tgz -C /opt/neuroedge`, pull `0.2.0`
   again with `neuroedge-sync`, and run `sudo systemctl restart neuroedge-web`.

## 6. Benchmark (results go into docs/FINDINGS.md)
```bash
cd /opt/neuroedge/src
sudo -u neuroedge env NEUROEDGE_HOME=/var/lib/neuroedge /opt/neuroedge/venv/bin/python -m edge.bench \
     --images /tmp/bench_samples --n 50 --llm 3 --out /tmp/bench_pi.json
```
For a dev-bundle install, add `--bundle-dir /var/lib/neuroedge/dev_bundle`.
The images must be readable by the `neuroedge` user (home directories are not), so copy them to `/tmp` first.
It reports the median and p95 analysis time per stage, LLM tokens/s, peak RAM, SoC temperature and
`vcgencmd get_throttled` (it should stay `0x0`). v1 (`mri/0.2.0`) on the Pi: 238 ms median, 250 ms p95
(model 120 ms), peak RSS 261 MB.

## 7. Operations
| Task | Command |
|---|---|
| Logs | `journalctl -u neuroedge-web -f`, `journalctl -u neuroedge-llm -f` |
| Model status / update now | `sudo neuroedge-sync status`, `sudo systemctl start neuroedge-sync` |
| Restart | `sudo systemctl restart neuroedge-web neuroedge-llm` |
| Temperature / throttling | `vcgencmd measure_temp`, `vcgencmd get_throttled` |
| Portal users | `sudo neuroedge-portal create-user --role doctor --email ... --name ...` (or the admin Users page) |
| Reminders / e-mail | `systemctl list-timers neuroedge-reminders.timer`, `journalctl -u neuroedge-reminders`, `sudo neuroedge-portal smtp-test --to ...` |
| Back up patient data | `sudo neuroedge-portal backup --dest /media/usb/neuroedge-backups` (database + encrypted files; **back up `/etc/neuroedge/portal.key` separately**) |
| Restore | stop `neuroedge-web`, then `sudo neuroedge-portal restore --src <backup dir>` |
| Audit log integrity | `sudo neuroedge-portal verify-audit` |
| Staff lost their authenticator | `sudo neuroedge-portal reset-2fa --email ...` |
| Portal CA fingerprint | `sudo neuroedge-ca` |
| Firewall rules | `sudo nft list table inet neuroedge` |

## Security notes
* **Network:** the portal is reachable only over HTTPS through Caddy. The Flask app and the LLM listen on loopback,
  and the firewall drops everything except SSH, HTTP(S) and mDNS (it also closes VNC and rpcbind).
  Caddy's `/_tls_ask` hook lets it issue certificates only for this device's names and private IPs.
* **Patient data:** scans, overlays and signed reports are encrypted at rest (AES-256-GCM, per-file nonce,
  AAD = case and kind) and integrity-checked on every read. Reports are versioned; nothing is overwritten.
* **Accounts:** argon2id hashes, lockout after 5 failures, rate limits on login and registration, 30-minute idle
  timeout, HTTPS-only `__Host-` session cookie, CSRF on every form, strict CSP, and an audit log of every access.
  Patients can never see AI output.
* Services run as the unprivileged `neuroedge` user with a read-only filesystem view (`ProtectSystem=strict`).
* Threat model: encryption protects the model and patient data in transit and at rest and blocks unenrolled
  downloads. It cannot stop someone with root on an enrolled Pi from reading the model from RAM or the portal key
  from disk: the Pi 5 has no hardware key store. Keep SSH key-only.

# CareLoop-Edge

On-device brain-MRI AI for rural health centres, with a care loop that follows every patient from scan to
specialist. It runs on a **Raspberry Pi 4 clinic kiosk** with a 7" touch screen. (Repository and Python package
name: NeuroEdge.)

**Health-a-thon 2026 · team CareLoop-Edge (Y5XV55) · Dept. of ECE, The LNM Institute of Information Technology
(LNMIIT), Jaipur.** Doctor partner: Dr. Madhur Saxena, Radiology, Bhagwan Mahaveer Cancer Hospital & Research Centre
(BMCHRC), Jaipur.

**Live evaluation copy:** <https://careloop-eval.3-111-140-240.sslip.io>. The same software on a small cloud
server (AWS), with fictional demo data; evaluator logins are shared privately, not in this repository.

> Research prototype on public data, not a medical device. A doctor reviews and signs every result before a patient
> sees anything; patients never see AI output. See the [model card](docs/MODEL_CARD.md).

## The problem
Rural and district centres can get a brain MRI done but rarely have a radiologist to read it. Studies travel to a
city, reports take days to weeks, and after the report nobody makes sure the patient reaches treatment: 28–34% of
cancer patients are lost to follow-up in Indian hospital cohorts.

## What CareLoop-Edge does
1. **Kiosk intake.** The patient signs up on the clinic kiosk's touch screen and uploads the whole MRI study from a
   USB drive: a DICOM series folder or a NIfTI file. The study is checked to be one patient and one series.
2. **AI pre-read on the device.** In 43–51 s on the Pi 4: 36 slices in 3 planes, one calibrated result per patient,
   a 95% prediction set, a quality gate that rejects non-MRI or unusual images, and Grad-CAM with the lesion outline
   on the 3 key slices. Studies go onto a triaged worklist.
3. **The doctor signs.** A remote radiologist reviews every slice, records whether they agree with the AI, and
   signs a structured report and a formal referral letter with one signature, setting the next step and its due
   date.
4. **The patient is informed.** An e-mail without any clinical content brings the patient to the portal (on a
   phone): signed report, next step, consultation booking, reminders 24 h and 2 h before.
5. **The admin (care coordinator) tracks every patient.** A follow-up list ranked by an explainable risk score,
   a call log, referral tracking until the specialist visit is attended, escalation of unread studies.
6. **The loop closes.** When the step has happened the plan closes; an approved plan 60 days overdue with no
   contact marks the patient as lost to follow-up. Analytics show loop closure, losses, distance to care and
   AI-doctor agreement.

```
 Kiosk (Pi 4 + 7" touch screen, Chromium kiosk) ──USB study──┐
                                                             ▼
 Patient phone ─┐            ┌──────────── Raspberry Pi 4 (4 GB) ─────────────────────────────────┐
 Doctor laptop ─┼─HTTPS──►   │ Caddy TLS → Flask portal (gunicorn) · SQLite · AES-256-GCM files   │
 Admin laptop  ─┘ (Tailscale │ INT8 ResNet-50 + lesion attention + segmentation (ONNX Runtime)    │
                   Funnel)   │ whole study: 36 slices → one result · Grad-CAM · 95% set · OOD gate│
                             │ worklist → signed report + referral → care loop, reminders, audit │
                             └────────────────────────────────────────────────────────────────────┘
                                                             ▲  signed (OpenSSF Model Signing, ECDSA P-256)
                                                             │  + encrypted (AES-256-GCM, key wrapped per device, X25519)
 Training PC ──publish──► Model registry: FastAPI + Caddy TLS in Docker (enrollment, revocation, audit log)
```

## Hardware (prototype kit, prices actually paid)
| Item | ₹ |
|---|---|
| Raspberry Pi 4 Model B, 4 GB | 12,000 |
| Official Raspberry Pi 7" touch display | 7,000 |
| SanDisk microSD | 4,000 |
| Official power adapter | 1,000 |
| Pen drive (scan transfer) | 800 |
| **Total per centre** | **24,800** |

Running cost: about ₹0 per study (no cloud AI or per-scan API), about ₹40 a month of electricity, free public
access (Tailscale) and e-mail (Gmail SMTP).

## The AI model
* **Architecture:** ResNet-50 backbone (ImageNet-pretrained, output stride 8) with an FPN segmentation head and a
  **lesion-attention pooling** classification head: the predicted lesion mask weights the pooling, so the exact
  Grad-CAM stays on the lesion. 4 classes: glioma, meningioma, pituitary tumour, no tumour.
* **Whole-study mode (`mri/0.3.0`):** 12 slices per plane (axial, coronal, sagittal) → per-slice INT8 analysis →
  mean of the 5 slices most suggestive of tumour → one answer per patient.
* **Uncertainty and safety:** temperature-calibrated probabilities, a 95% conformal prediction set, a
  Mahalanobis out-of-distribution check, and reject flags for non-MRI or unusual input.
* **On the device:** static INT8 ONNX (the attention path and classifier stay FP32 so the CAM identity holds),
  96.6 → 24.7 MB, CPU only.
* **Data:** BraTS 2023 (glioma, meningioma), NHNN pituitary, IXI healthy volunteers (never used for training),
  BRISC 2025 (audited for leakage). Splits are by patient. Details: [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## Results: v2, `mri/0.3.0`, whole-study model (current)
v1 read one image. Our doctor partner pointed out that no radiologist confirms a tumour from a single slice, so v2
reads the whole study. Held-out patients, targets fixed before testing, INT8 model as deployed:

| | v1 (one image) | **v2 (whole study)** | Target |
|---|---|---|---|
| Tumour-type macro-F1 | 0.71 | **0.98** | ≥ 0.85 |
| Slice tumour AUROC | 0.72 | **0.95** | ≥ 0.90 |
| False alarms on healthy brains (IXI) | 23% | **5%** | ≤ 10% |
| Glioma / meningioma / pituitary / healthy | | 76/76 · 62/68 · 15/15 · 361/380 patients | |
| Raspberry Pi 4, whole study (load + 36 slices + explanations) | | **43–51 s** | |

* **Leakage audit:** 566 of the 1,000 official BRISC test images are copies or mirrors of training images; all
  splits were rebuilt by patient.
* **INT8 vs FP32:** within 1 point (McNemar p = 1.0) on the 2D test sets.
* **Limits:** one training seed; test patients come from the same datasets as training; the external whole-study
  test (BraTS-Africa) is pending; the 95% set is calibrated on slices, not patients.

Full details: [model card](docs/MODEL_CARD.md) and [findings](docs/FINDINGS.md).

### Earlier: v1, `mri/0.2.0`, single image ([findings](docs/FINDINGS.md) §10)
| | v0 | **v1** (3-seed mean ± sd) |
|---|---|---|
| Test macro-F1 | 0.935 | **0.944** (0.938 ± 0.007) |
| External BDNeuro macro-F1 | 0.729 | **0.836** (0.827 ± 0.012) |
| Grad-CAM evidence inside the tumour | 0.15 | **0.73** (98% of the 28×28 ceiling) |
| Pointing game (hotspot on the tumour) | 0.53 | **0.99** |
| 95% prediction-set coverage | 0.87 | **0.91** (target 0.93 not met) |
| Raspberry Pi full analysis, median | 178 ms | **238 ms** |
| Registry end-to-end security check | 16/16 | **16/16** |

A 1.7B local language model could not be trusted to interpret statistics, so it was removed from the portal (the
experiment stays in `edge/agent`, installed only with `install_pi.sh --with-llm`).

## Portal features
* **Patients:**
  * sign up and upload a study at the kiosk; follow its review; read and print the signed report;
  * see their care journey and next step (consultation, repeat MRI or referral);
  * book, reschedule or cancel a consultation, with calendar invites; confirm or decline from the reminder e-mail;
  * view referral letters; add earlier reports or biopsy results;
  * consent records, download all their data, request account deletion, reset a forgotten password.
* **Radiologists:**
  * a prioritised worklist with SLA timers and a slice viewer (Grad-CAM, lesion outline, zoom and pan);
  * the AI findings panel (classification, 95% set, lesion, quality flags) on the key slices;
  * a structured report and a referral letter released with one signature; amendments keep every version;
  * the next step chosen while signing; approval of follow-up plans proposed by the coordinator;
  * the consultation schedule with attendance marking; availability (hours and days off);
  * analytics: reporting turnaround, loop closure, lost to follow-up, distance to care, AI agreement.
* **Care coordinator (the administrator role):**
  * the follow-up worklist (overdue, due soon, awaiting approval, referral open, not reached, missed visits),
    ranked by a risk score with the reasons shown;
  * a contact log, reminders for doctor-approved plans, booking on a patient's behalf;
  * walk-in registration (e-mail optional, portal invitation); referral tracking (sent, appointment, attended);
  * users, settings, the e-mail outbox, the tamper-evident audit log, deletion requests and system status.
* **Kiosk:** a device home page on the 7" screen (model version and signature check, temperature, storage, the
  portal address with a QR code for phones), touch-sized pages and an on-screen keyboard; it shows no patient data
  and returns home when idle.
* **Follow-up risk score:** a transparent weighted sum of days overdue, missed appointments, unanswered calls,
  contact details not working, an open referral and travel distance. It never reads the diagnosis, the report or
  any AI output; it only ranks the list, and people decide.
* **Reminders** (in the portal and by e-mail, never with clinical content): unread or unsigned studies with
  escalation after 48 h, booking confirmations, appointment reminders, the daily agenda, unrecorded attendance,
  rebooking after a no-show, the coordinator's daily follow-up summary and doctors' pending approvals.

## Security and privacy
* **The model:** published signed (OpenSSF Model Signing, ECDSA P-256) and encrypted (AES-256-GCM, the data key
  wrapped for each enrolled device with X25519). The device verifies the signature before any model runs and keeps
  it only as ciphertext on disk.
* **Patient data:** scans, overlays and reports encrypted at rest (AES-256-GCM); signed reports are append-only and
  versioned; a hash-chained audit log of every access.
* **Accounts:** argon2id passwords, lockout and rate limits, two-step sign-in for staff, HTTPS-only session
  cookies, CSRF protection and a strict content security policy. Sign-up and upload only at the clinic kiosk.
* **Network:** only Caddy (HTTPS) is exposed; the app listens on loopback; an nftables firewall admits only SSH,
  HTTP(S) and mDNS; services run as an unprivileged user.
* **Consent:** recorded at sign-up and with each upload; data export and deletion requests.

## Deployment
* **Raspberry Pi 4 kiosk:** [docs/PI_DEPLOY.md](docs/PI_DEPLOY.md), sections 1–7: install, model
  registry, kiosk on the 7" screen, upgrading to `mri/0.3.0`, internet access with Tailscale Funnel.
* **Cloud evaluation copy on AWS EC2:** [docs/PI_DEPLOY.md](docs/PI_DEPLOY.md) section 8: t4g.small (ARM),
  Debian 13, the same installer, the registry over an SSH tunnel, direct HTTPS with Let's Encrypt.

The short version:
1. Registry: `docker compose -f server/docker-compose.yml run --rm api python -m server.manage init --data /data`,
   then `docker compose -f server/docker-compose.yml up -d --build`.
2. Publish from the training PC: `python -m server.publish ...` (the signing key never leaves the PC).
3. On the Pi: `sudo bash edge/install_pi.sh --kiosk --display td1`, then `sudo neuroedge-sync enroll ...` and
   `sudo neuroedge-sync pull`, then `sudo neuroedge-portal create-user --role admin ...`.

## Repository
| Path | What |
|---|---|
| `neuroedge/` | Shared core for PC, server and Pi: preprocessing, whole-study volume handling, model, CAM maths, uncertainty, metrics, crypto, bundles |
| `training/` | Data manifests and leakage audit, training, calibration, export, INT8 quantization, XAI evaluation, acceptance reports |
| `server/` | Model registry (FastAPI), Docker Compose + Caddy, publish tool, end-to-end security check |
| `edge/` | Device runtime: inference engine, model sync, clinic portal (`edge/portal`), kiosk, installer, Caddy, firewall, systemd units, benchmark; language-model experiment (`edge/agent`, not used by the portal) |
| `docs/` | [Deployment guide](docs/PI_DEPLOY.md), [model card](docs/MODEL_CARD.md), [findings](docs/FINDINGS.md), [data sources](docs/DATA_SOURCES.md) |
| `tests/` | 179 automated tests: care loop and risk score, referrals, reminders, mailer, appointments, two-step sign-in, audit chain, portal access control and security, kiosk, whole-study input checks, CAM exactness, crypto and tamper detection, leakage dedupe, deploy files |
| `reference/` | The original notebook (reference only) |

## Quick start (PC)
Tests:
```bash
python -m pytest -q
```
Run the portal from a plaintext development bundle (`--init-dev` creates test accounts in
`<portal-data>/DEV_ACCOUNTS.txt`):
```bash
python -m edge.serve --bundle-dir <DATA_ROOT>/bundles/<bundle> --portal-data <DATA_ROOT>/portal_dev --init-dev --port 8080
```
Training data lives outside the repo in `DATA_ROOT` (env `NEUROEDGE_DATA`, default `~/neuroedge_data`). Private
keys and admin tokens belong in a non-synced folder (e.g. `~/neuroedge_secrets/`), never in the repo or a
cloud-synced directory. Model files, scans and keys are excluded by `.gitignore`.

## Status and next steps
v2 runs end to end on the Raspberry Pi 4 kiosk and on the AWS evaluation copy: kiosk intake from USB, whole-study
model `mri/0.3.0` verified on the device, signed reports and referral letters, the care loop with reminders and
referral tracking.

Next:
* External whole-study test on BraTS-Africa; two more training seeds; a prospective test with BMCHRC.
* A pilot at a partner rural centre.
* Indian languages and voice for patients; SMS / WhatsApp reminders.
* Calibration of the 95% set on patients rather than slices.
* PACS / DICOM and ABDM integration.

## Data licences
BraTS data is CC BY-NC (research and non-commercial use only); IXI is CC BY-SA 3.0. No patient data, scans or
model weights are included in this repository.

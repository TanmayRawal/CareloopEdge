# NeuroEdge

Secure, explainable brain-tumor MRI review that runs offline on a **Raspberry Pi 5**.

Patients upload a scan to a clinic portal on the Pi. The AI result (calibrated probabilities, a 95% prediction set,
an **exact, lesion-focused Grad-CAM** and the lesion outline) goes to the doctor's worklist. The doctor signs the
final report, and the patient sees only that report. The model is trained on a PC and reaches the Pi as a **signed
and encrypted** bundle through a self-hosted registry.

> Research prototype on public data. See the [model card](docs/MODEL_CARD.md).

```
 Patient ──HTTPS──► Pi 5 clinic portal (Caddy TLS → Flask, SQLite, encrypted scan/report store)
                    ├─ INT8 ResNet-50, lesion-attention head + segmentation (ONNX Runtime)   120 ms
                    ├─ exact Grad-CAM (28×28) · lesion outline · 95% conformal set · OOD reject/warn
                    ├─ doctor worklist → review → signed, versioned report → patient
                    ├─ consultation booking · reminders (in-app + e-mail) · attendance / no-shows
                    └─ doctor assistant: Qwen3-1.7B (llama.cpp) + cited KB + deterministic verifier
                              ▲
                              │  signed (OpenSSF Model Signing) + AES-256-GCM, key wrapped per device (X25519)
 Training PC ──publish──►  Registry: FastAPI + Caddy TLS in Docker (enrollment, revocation, audit log)
```

## Portal features (radiology department)
* **Patients:**
  * register, upload a scan, follow its review and read the signed report (print / PDF);
  * book, reschedule or cancel a consultation with the reporting radiologist, with calendar invites.
* **Radiologists:**
  * a prioritised worklist with SLA timers and a PACS-style viewer (Grad-CAM, lesion outline, zoom and pan);
  * the AI findings panel and the doctor assistant;
  * signed and amended reports;
  * the consultation schedule with attendance marking;
  * availability editing (hours and days off);
  * analytics.
* **Administrators:** users, department settings (reporting targets, reminder and booking rules), the e-mail
  outbox, the audit log and system status.
* **Reminders**, in-app and by e-mail through SMTP, carry no clinical content:
  * studies left unread or unsigned (with escalation);
  * booking confirmations and appointment reminders;
  * the daily agenda;
  * unrecorded attendance;
  * rebooking after a no-show.

## Results: v1, `mri/0.2.0` (details in [docs/FINDINGS.md](docs/FINDINGS.md) §10)
All numbers are for the deployed INT8 model on the leakage-free held-out test set, compared with v0.

| | v0 | **v1** (3-seed mean ± sd) |
|---|---|---|
| Test macro-F1 | 0.935 | **0.944** (0.938 ± 0.007) |
| External BDNeuro macro-F1 | 0.729 | **0.836** (0.827 ± 0.012) |
| Grad-CAM evidence inside the tumor | 0.15 | **0.73** (98% of the 28×28 ceiling) |
| Pointing game (hotspot on the tumor) | 0.53 | **0.99** |
| ROAD faithfulness | 0.29 | **0.32** |
| 95% prediction-set coverage | 0.87 | **0.91** (target 0.93 not met) |
| Pi 5 full analysis, median | 178 ms | **238 ms** |
| Pi vs PC parity | 40/40 | **40/40**, CAM r ≥ 0.997 |
| Registry end-to-end security check | 16/16 | **16/16** |

Key findings:
* The public benchmarks leak heavily: the official BRISC test set is 57% copies of training images.
* Lesion-attention pooling makes Grad-CAM lesion-focused while keeping it exact. Explanation-guided training had
  failed (a negative result).
* v0's INT8 model quietly lost 11 F1 points on external data, which v1 avoids.
* A 1.7B local LLM cannot be trusted to interpret statistics, so those answers are deterministic.

## Repository
| Path | What |
|---|---|
| `neuroedge/` | Shared core for PC, server and Pi: preprocessing, model, CAM maths, uncertainty, metrics, crypto, bundles |
| `training/` | Data manifests and leakage audit, training, ablation, calibration, export, INT8 quantization, XAI evaluation, acceptance report |
| `server/` | Model registry (FastAPI), Docker Compose + Caddy, publish tool, live end-to-end check |
| `edge/` | Pi runtime: engine, sync, clinic portal (`edge/portal`), assistant, benchmark, parity check, installer, Caddy, firewall, systemd units |
| `docs/` | [Findings](docs/FINDINGS.md), [model card](docs/MODEL_CARD.md), [Pi deployment guide](docs/PI_DEPLOY.md) |
| `tests/` | 108 tests: appointments, reminders, mailer, CAM exactness, crypto and tamper detection, leakage dedupe, portal access control and security, verifier, deploy files |
| `reference/` | The original notebook (reference only) |

## Quick start
**Tests (PC):**
```bash
python -m pytest -q
```

**Run the portal on the PC from a plaintext development bundle** (`--init-dev` creates test accounts in
`<portal-data>/DEV_ACCOUNTS.txt`):
```bash
python -m edge.serve --bundle-dir <DATA_ROOT>/bundles/mri_v1_0.2.0_int8 --portal-data <DATA_ROOT>/portal_dev --init-dev --port 8080
```

**Registry, publishing and the Raspberry Pi:** follow [docs/PI_DEPLOY.md](docs/PI_DEPLOY.md). The short version:
1. `docker compose -f server/docker-compose.yml run --rm api python -m server.manage init --data /data`
2. `docker compose -f server/docker-compose.yml up -d --build`
3. `python -m server.publish ...`
4. On the Pi:
   * `sudo bash edge/install_pi.sh ...`
   * `sudo neuroedge-sync enroll ...`, then `sudo neuroedge-sync pull`
   * `sudo neuroedge-portal create-user --role admin ...`

Training data lives outside the repo in `DATA_ROOT` (env `NEUROEDGE_DATA`, default `~/neuroedge_data`).
Private keys and the admin token belong in a non-synced folder (e.g. `~/neuroedge_secrets/`), never in the repo
or a cloud-synced directory.

## Status and next steps
v1 runs end to end on the Pi: HTTPS portal, firewall, encrypted patient store, model `mri/0.2.0`. Known
limitations are listed in the [model card](docs/MODEL_CARD.md).

Next:
* Calibration per data source, to reach 95% coverage on the test release.
* BraTS-based fixes for the sequence shortcut.
* The 3D MRI track.
* Fit the Raspberry Pi **Active Cooler**.

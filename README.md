# NeuroEdge

Secure, explainable brain-tumor MRI classification that runs offline on a **Raspberry Pi 5**.

The model is trained on a PC and published as a **signed and encrypted** bundle to a self-hosted registry. An
enrolled Pi pulls the bundle, verifies and decrypts it in RAM, and serves a Flask web app. The app shows the
tumor class with calibrated uncertainty, a **faithful Grad-CAM**, the lesion outline, and an **offline AI
assistant** whose answers are checked against the model output.

> **Research and education only. Not a medical device.** See the [model card](docs/MODEL_CARD.md).

```
 MRI slice ──► Pi 5 web app (Flask) ─────────────────────────────────────────────────────────┐
               ├─ INT8 ResNet-50 + segmentation head (ONNX Runtime)   47 ms                   │
               ├─ exact Grad-CAM · Finer-CAM · LayerCAM ("Deep Explain", FP32)                │
               ├─ temperature scaling · 95% conformal set · Mahalanobis OOD (reject / warn)   │
               └─ offline assistant: Qwen3-1.7B (llama.cpp) + cited KB + deterministic verifier
                         ▲
                         │  signed (OpenSSF Model Signing) + AES-256-GCM, key wrapped per device (X25519)
 Training PC ──publish──►  Registry: FastAPI + Caddy TLS in Docker (enrollment, revocation, audit log)
```

## Results (details in [docs/FINDINGS.md](docs/FINDINGS.md))
| | |
|---|---|
| Held-out test (patient-grouped, leakage-free) | accuracy 0.949, macro-F1 0.942 |
| External (BDNeuro, deduplicated) | accuracy 0.864, macro-F1 0.843 |
| INT8 vs FP32 | macro-F1 0.939 → 0.935, McNemar p = 0.38; 24.7 MB |
| Pi 5: full analysis / Deep Explain / LLM | 178 ms / 0.78 s / 8–9 tok/s |
| Pi vs PC parity | 40/40 identical predictions |
| Registry end-to-end security check | 16/16 checks pass |

Key findings: the public benchmarks leak heavily (the official BRISC test set is 57% copies of training
images); explanation-guided training is a negative result (it conflicts with an exact CAM head); and a 1.7B
local LLM cannot be trusted to interpret statistics, so those answers are deterministic.

## Repository
| Path | What |
|---|---|
| `neuroedge/` | Shared core for PC, server and Pi: preprocessing, model, CAM maths, uncertainty, metrics, crypto, bundles |
| `training/` | Data manifests and leakage audit, training, calibration, export, INT8 quantization, XAI and assistant evaluation |
| `server/` | Model registry (FastAPI), Docker Compose + Caddy, publish tool, live end-to-end check |
| `edge/` | Pi runtime: engine, sync, Flask web app, assistant, benchmark, parity check, installer, systemd units |
| `docs/` | [Findings](docs/FINDINGS.md), [model card](docs/MODEL_CARD.md), [Pi deployment guide](docs/PI_DEPLOY.md) |
| `tests/` | 75 tests: CAM exactness, crypto and tamper detection, leakage dedupe, web security, verifier, deploy files |
| `reference/` | The original notebook (reference only; its flaws are listed in the plan) |

## Quick start
**Tests (PC):**
```bash
python -m pytest -q
```

**Run the web app on the PC from a plaintext development bundle:**
```bash
python -m edge.serve --bundle-dir <DATA_ROOT>/bundles/mri_v0_int8 --port 8080
```

**Registry, publishing and the Raspberry Pi:** follow [docs/PI_DEPLOY.md](docs/PI_DEPLOY.md). The short version:
1. `docker compose -f server/docker-compose.yml run --rm api python -m server.manage init --data /data`
2. `docker compose -f server/docker-compose.yml up -d --build`
3. `python -m server.publish ...`
4. On the Pi: `sudo bash edge/install_pi.sh ...`, then `sudo neuroedge-sync enroll ...` and `sudo neuroedge-sync pull`

Training data lives outside the repo in `DATA_ROOT` (env `NEUROEDGE_DATA`, default `~/neuroedge_data`).
Private keys and the admin token belong in a non-synced folder (e.g. `~/neuroedge_secrets/`), never in the repo
or a cloud-synced directory.

## Status and next steps
v0 runs end to end on the Pi. Known limitations are listed in the [model card](docs/MODEL_CARD.md).

Next:
* BraTS-based fixes for the sequence shortcut and a separate calibration split
* a 3-seed retrain
* a segmentation-weighted pooling head (lesion-focused, still exact CAM)
* the 3D MRI track

Fit the Raspberry Pi **Active Cooler**: without it the SoC reaches its soft temperature limit under load.

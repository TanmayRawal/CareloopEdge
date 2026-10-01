# Deploying NeuroEdge on a Raspberry Pi 5 (8 GB)

What runs on the Pi, all offline after setup:

| Service | What | Port |
|---|---|---|
| `neuroedge-web` | Flask app under gunicorn: upload → INT8 ONNX classifier + segmentation, exact Grad-CAM / Finer-CAM, conformal set, OOD check, report, assistant | `0.0.0.0:8080` (LAN) |
| `neuroedge-llm` | llama.cpp server with Qwen3-1.7B Q4_K_M (assistant narratives and Q&A) | `127.0.0.1:8081` (loopback only) |
| `neuroedge-sync.timer` | daily check for a new signed + encrypted model; a new version is trial-decrypted and verified before it is activated | — |
| `neuroedge-web-restart.path` | restarts the web app when a newly verified model version is activated | — |

The model is stored only as ciphertext on the Pi. It is decrypted and its signature verified in RAM at startup.
Memory budget: web app ≈ 1 GB (≈ 1.5 GB with Deep Explain) + LLM ≈ 1.6 GB (1.1 GB weights + 0.47 GB KV cache
for 4096 tokens), so about 3 GB of the 8 GB.

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
sudo bash edge/install_pi.sh --gguf ~/Qwen3-1.7B-Q4_K_M.gguf --with-deep-explain
```
* Building llama.cpp takes about 10–15 minutes on the Pi 5. It is pinned to release `b11320`, the build the
  assistant was evaluated with.
* The GGUF file is checked against its pinned SHA-256 and refused if it differs.
* Without `--with-deep-explain`, PyTorch is not installed (saves about 1 GB) and the Deep Explain tab is hidden.
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
NE_ADMIN_TOKEN=<token> python -m server.publish --bundle <bundle_dir> --name mri --version 0.1.0 --server https://<host>:8443 --signing-key ~/neuroedge_secrets/signing/signing_priv.pem
```
Verify the deployment end to end (publish, enroll, pull, attacker checks):
```bash
NE_ADMIN_TOKEN=<token> python -m server.e2e_check --server https://<host>:8443 --ca keys/caddy_root.crt --bundle <bundle_dir> --signing-pub keys/signing_pub.pem --version 0.1.0 --skip-publish
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
scp -r C:/Users/Tanmay/neuroedge_data/bundles/mri_v0_int8 <user>@neuroedge.local:~/
sudo bash edge/install_pi.sh --dev-bundle ~/mri_v0_int8
```

## 5. Use it
Open `http://neuroedge.local:8080` (or `http://<pi-ip>:8080`) from any device on the same network.

## 6. Benchmark (results go into docs/FINDINGS.md)
```bash
cd /opt/neuroedge/src
sudo -u neuroedge env NEUROEDGE_HOME=/var/lib/neuroedge /opt/neuroedge/venv/bin/python -m edge.bench \
     --images ~/bench_samples --n 50 --deep 3 --llm 3 --out /tmp/bench_pi.json
```
For a dev-bundle install, add `--bundle-dir /var/lib/neuroedge/dev_bundle`.
It reports the median and p95 analysis time per stage, Finer-CAM re-render time, Deep Explain (cold and warm),
LLM tokens/s, peak RAM, SoC temperature and `vcgencmd get_throttled` (it should stay `0x0`).
For comparison, the PC numbers (i7, 4 threads) are: analysis 97 ms (model 12 ms, overlay rendering 48 ms),
Finer-CAM re-render 20 ms, Deep Explain 175 ms warm and 3.6 s cold, LLM 24 tokens/s.

## 7. Operations
| Task | Command |
|---|---|
| Logs | `journalctl -u neuroedge-web -f`, `journalctl -u neuroedge-llm -f` |
| Model status / update now | `sudo neuroedge-sync status`, `sudo systemctl start neuroedge-sync` |
| Restart | `sudo systemctl restart neuroedge-web neuroedge-llm` |
| Temperature / throttling | `vcgencmd measure_temp`, `vcgencmd get_throttled` |

## Security notes
* The LLM listens on loopback only. The web app is plain HTTP on the LAN; put it behind a TLS reverse proxy
  before any use beyond a trusted network.
* Uploaded images live only in RAM and expire after an hour. Nothing patient-related is written to disk.
* Services run as the unprivileged `neuroedge` user with a read-only filesystem view (`ProtectSystem=strict`).
* Threat model: encryption protects the model in transit and at rest and blocks unenrolled downloads. It
  cannot stop someone with root on an enrolled Pi from extracting the model from RAM.

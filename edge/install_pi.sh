#!/usr/bin/env bash
# NeuroEdge installer for Raspberry Pi 5 (Raspberry Pi OS 64-bit, Bookworm or newer). Run from the repo root:
#
#   sudo bash edge/install_pi.sh [--with-deep-explain] [--gguf FILE] [--ca-cert FILE] [--dev-bundle DIR]
#
#   --with-deep-explain  also install CPU PyTorch + timm (research tools only; the v1 portal shows Grad-CAM + lesion)
#   --gguf FILE          use a local copy of Qwen3-1.7B-Q4_K_M.gguf instead of downloading it (fully offline install)
#   --ca-cert FILE       trust this CA for the model registry (e.g. Caddy's internal root for a LAN server)
#   --dev-bundle DIR     DEVELOPMENT ONLY: serve a plaintext bundle directory (no server, no encryption at rest)
#
# Layout: code /opt/neuroedge/src, venv /opt/neuroedge/venv, llama.cpp /opt/neuroedge/bin, LLM /opt/neuroedge/llm,
# device state (keys, encrypted bundles) /var/lib/neuroedge, config /etc/neuroedge. Services run as user "neuroedge".
# Re-running the script is safe (idempotent); it upgrades the code in place.
set -euo pipefail

APP=/opt/neuroedge
STATE=/var/lib/neuroedge
ETC=/etc/neuroedge
LLAMA_TAG=b11320                       # the llama.cpp build the assistant was evaluated with (docs/FINDINGS.md §7)
GGUF_NAME=Qwen3-1.7B-Q4_K_M.gguf
GGUF_URL=https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/$GGUF_NAME
GGUF_SHA256=b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897

DEEP=0; GGUF_SRC=""; CA_CERT=""; DEV_BUNDLE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-deep-explain) DEEP=1; shift ;;
    --gguf) GGUF_SRC="$2"; shift 2 ;;
    --ca-cert) CA_CERT="$2"; shift 2 ;;
    --dev-bundle) DEV_BUNDLE="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
[[ "$(uname -m)" == "aarch64" ]] || { echo "needs 64-bit Raspberry Pi OS (aarch64), found $(uname -m)" >&2; exit 1; }
[[ -f edge/serve.py && -d neuroedge ]] || { echo "run from the NeuroEdge repo root" >&2; exit 1; }
SRC_DIR="$(pwd)"

say "System packages"
apt-get update -qq
apt-get install -y -qq python3-venv python3-dev git cmake build-essential curl ca-certificates rsync openssl

say "Service user and directories"
id neuroedge &>/dev/null || useradd --system --home-dir "$STATE" --shell /usr/sbin/nologin neuroedge
install -d -m 755 "$APP" "$APP/bin" "$APP/llm"
install -d -m 750 -o neuroedge -g neuroedge "$STATE"
install -d -m 750 -o root -g neuroedge "$ETC"

say "Application code -> $APP/src"
rsync -a --delete --exclude '__pycache__' --exclude '.pytest_cache' \
  "$SRC_DIR/neuroedge" "$SRC_DIR/edge" "$SRC_DIR/requirements-edge.txt" "$SRC_DIR/pyproject.toml" "$APP/src/"

say "Python environment"
[[ -x "$APP/venv/bin/python" ]] || python3 -m venv "$APP/venv"
"$APP/venv/bin/pip" install -q --upgrade pip wheel
# Exact versions verified on the Pi (cross-platform parity checked) are used as constraints when present.
LOCK=(); [[ -f "$APP/src/edge/requirements-pi.lock" ]] && LOCK=(-c "$APP/src/edge/requirements-pi.lock")
"$APP/venv/bin/pip" install -q -r "$APP/src/requirements-edge.txt" "${LOCK[@]}"
if [[ $DEEP -eq 1 ]]; then
  # CPU-only aarch64 wheels (no CUDA payload)
  "$APP/venv/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu "${LOCK[@]}"
  "$APP/venv/bin/pip" install -q timm "${LOCK[@]}"
fi

say "llama.cpp $LLAMA_TAG (native build for the Cortex-A76: NEON + dotprod)"
# A shallow clone reports "build 1" in --version, so the installed tag is recorded in a stamp file instead.
if [[ ! -x "$APP/bin/llama-server" || "$(cat "$APP/bin/llama-server.tag" 2>/dev/null)" != "$LLAMA_TAG" ]]; then
  rm -rf "$APP/llama.cpp-src"
  git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$APP/llama.cpp-src"
  # Static: one self-contained binary in $APP/bin (a shared build would depend on libs left in the build dir).
  # No prebuilt web UI: it is downloaded unpinned at configure time, and NeuroEdge only uses the HTTP API.
  cmake -S "$APP/llama.cpp-src" -B "$APP/llama.cpp-src/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
        -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON >/dev/null
  cmake --build "$APP/llama.cpp-src/build" --target llama-server -j"$(nproc)"
  install -m 755 "$APP/llama.cpp-src/build/bin/llama-server" "$APP/bin/llama-server"
  echo "$LLAMA_TAG" > "$APP/bin/llama-server.tag"
fi

say "LLM weights ($GGUF_NAME, SHA-256 pinned)"
GGUF="$APP/llm/$GGUF_NAME"
if [[ -n "$GGUF_SRC" ]]; then install -m 644 "$GGUF_SRC" "$GGUF";
elif [[ ! -f "$GGUF" ]]; then curl -L --fail --retry 5 -C - -o "$GGUF.part" "$GGUF_URL" && mv "$GGUF.part" "$GGUF"; fi
echo "$GGUF_SHA256  $GGUF" | sha256sum -c - || { echo "GGUF checksum mismatch: refusing to use it" >&2; rm -f "$GGUF"; exit 1; }

if [[ -n "$CA_CERT" ]]; then
  say "Trusting registry CA $CA_CERT"
  install -m 644 "$CA_CERT" /usr/local/share/ca-certificates/neuroedge-registry.crt
  update-ca-certificates >/dev/null
fi

say "Configuration $ETC/web.env"
if [[ ! -f "$ETC/web.env" ]]; then
  umask 027
  cat > "$ETC/web.env" <<EOF
NEUROEDGE_SECRET=$(openssl rand -hex 32)
NEUROEDGE_HOME=$STATE
NEUROEDGE_MODEL=mri
NEUROEDGE_LLM_URL=http://127.0.0.1:8081
NEUROEDGE_PORTAL_DATA=$STATE/portal
NEUROEDGE_PORTAL_KEY=$ETC/portal.key
EOF
  chgrp neuroedge "$ETC/web.env"
fi
grep -q '^NEUROEDGE_PORTAL_DATA=' "$ETC/web.env" || printf 'NEUROEDGE_PORTAL_DATA=%s/portal
NEUROEDGE_PORTAL_KEY=%s/portal.key
' "$STATE" "$ETC" >> "$ETC/web.env"

say "Clinical portal: encryption key + database"
# The key encrypts every stored scan / overlay / report. Created once, never overwritten; back it up separately.
if [[ ! -f "$ETC/portal.key" ]]; then
  "$APP/venv/bin/python" -c "import os,sys; open(sys.argv[1],'wb').write(os.urandom(32))" "$ETC/portal.key"
fi
chown root:neuroedge "$ETC/portal.key"; chmod 640 "$ETC/portal.key"
install -d -m 750 -o neuroedge -g neuroedge "$STATE/portal"
(cd "$APP/src" && runuser -u neuroedge -- "$APP/venv/bin/python" -m edge.portal.manage init --data "$STATE/portal" --key "$ETC/portal.key")
if [[ -n "$DEV_BUNDLE" ]]; then
  echo "WARNING: dev bundle = plaintext model on disk; use enrollment + sync for real deployments." >&2
  rsync -a --delete "$DEV_BUNDLE/" "$STATE/dev_bundle/"
  chown -R neuroedge:neuroedge "$STATE/dev_bundle"
  grep -q '^NEUROEDGE_BUNDLE_DIR=' "$ETC/web.env" || echo "NEUROEDGE_BUNDLE_DIR=$STATE/dev_bundle" >> "$ETC/web.env"
fi

say "Helper commands /usr/local/bin/neuroedge-sync, neuroedge-portal"
cat > /usr/local/bin/neuroedge-portal <<EOF
#!/bin/sh
# Portal administration as the service user, e.g.
#   sudo neuroedge-portal create-user --role admin --email you@clinic.org --name "Your Name"
#   sudo neuroedge-portal backup --dest /media/usb/neuroedge-backups
cmd="\$1"; shift
cd $APP/src && exec runuser -u neuroedge -- $APP/venv/bin/python -m edge.portal.manage "\$cmd" --data $STATE/portal "\$@"
EOF
chmod 755 /usr/local/bin/neuroedge-portal
cat > /usr/local/bin/neuroedge-sync <<EOF
#!/bin/sh
# Run edge.sync as the service user, e.g.  sudo neuroedge-sync status
cd $APP/src && exec runuser -u neuroedge -- env NEUROEDGE_HOME=$STATE $APP/venv/bin/python -m edge.sync "\$@"
EOF
chmod 755 /usr/local/bin/neuroedge-sync

say "systemd units"
install -m 644 "$APP/src/edge/systemd/"neuroedge-*.{service,timer,path} /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now neuroedge-llm.service neuroedge-sync.timer neuroedge-web-restart.path
systemctl enable neuroedge-web.service
systemctl restart neuroedge-web.service || true    # skipped (condition) until a model is installed

IP="$(hostname -I | awk '{print $1}')"
cat <<EOF

NeuroEdge installed.
  LLM:  systemctl status neuroedge-llm      (loopback :8081)
  Web:  http://$IP:8080                       (starts once a model is installed)

Create the first portal administrator (then add doctors from the Users page):
  sudo neuroedge-portal create-user --role admin --email you@clinic.org --name "Your Name"

Next, enroll this Pi with the model registry. On the server machine, create a one-time token:
  curl -X POST -H "Authorization: Bearer \$NE_ADMIN_TOKEN" https://<registry>/v1/admin/enrollment-tokens
Then on the Pi:
  sudo install -m 640 -g neuroedge signing_pub.pem $ETC/signing_pub.pem
  sudo neuroedge-sync enroll --server https://<registry> --token <one-time-token> --name pi5-01 --signing-pub $ETC/signing_pub.pem
  sudo neuroedge-sync pull --model mri       # downloads, trial-decrypts and verifies; the web app then starts itself
  sudo neuroedge-sync status
Benchmark:  cd $APP/src && sudo -u neuroedge env NEUROEDGE_HOME=$STATE $APP/venv/bin/python -m edge.bench --images <dir>
EOF

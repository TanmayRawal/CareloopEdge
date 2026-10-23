#!/usr/bin/env bash
# NeuroEdge installer for Raspberry Pi 5 (Raspberry Pi OS 64-bit, Bookworm or newer). Run from the repo root:
#
#   sudo bash edge/install_pi.sh [--with-deep-explain] [--gguf FILE] [--ca-cert FILE] [--dev-bundle DIR]
#                                [--kiosk] [--display td1|td2|hdmi] [--rotate 0|90|180|270] [--with-llm]
#
#   --with-deep-explain  also install CPU PyTorch + timm (research tools only; the v1 portal shows Grad-CAM + lesion)
#   --gguf FILE          with --with-llm: use a local copy of Qwen3-1.7B-Q4_K_M.gguf instead of downloading it
#   --ca-cert FILE       trust this CA for the model registry (e.g. Caddy's internal root for a LAN server)
#   --dev-bundle DIR     DEVELOPMENT ONLY: serve a plaintext bundle directory (no server, no encryption at rest)
#   --kiosk              the Pi's own screen shows the portal full-screen: device home (/device) -> portal, touch
#                        friendly, on-screen keyboard. Runs as a locked-down user "kiosk" (no password, no sudo).
#   --display MODEL      enable the panel driver in /boot/firmware/config.txt (reboot needed):
#                        td1 = original 7" Touch Display (800x480; Pi 5 needs the 22-to-15-pin display cable),
#                        td2 = Touch Display 2 (720x1280 portrait panel; use --rotate 90 for landscape), hdmi = none
#   --rotate DEG         rotate the kiosk screen (e.g. 90 for Touch Display 2 in landscape)
#   --with-llm           RESEARCH ONLY: also build llama.cpp and run the local language model (Qwen3-1.7B, ~1.6 GB RAM)
#                        used by edge/agent experiments. The portal does not use it; a normal install leaves it out and
#                        stops / masks the service if an earlier install had it.
#
# Layout: code /opt/neuroedge/src, venv /opt/neuroedge/venv (llama.cpp /opt/neuroedge/bin + /opt/neuroedge/llm only with --with-llm),
# device state (keys, encrypted bundles) /var/lib/neuroedge, config /etc/neuroedge. Services run as user "neuroedge".
# Network exposure: Caddy serves the portal over HTTPS (:443, local CA) and redirects :80; the Flask app and the
# optional LLM listen on loopback only; an nftables table admits only SSH, HTTP(S) and mDNS.
# Re-running the script is safe (idempotent); it upgrades the code in place.
set -euo pipefail

APP=/opt/neuroedge
STATE=/var/lib/neuroedge
ETC=/etc/neuroedge
LLAMA_TAG=b11320                       # the llama.cpp build the assistant was evaluated with (docs/FINDINGS.md §7)
GGUF_NAME=Qwen3-1.7B-Q4_K_M.gguf
GGUF_URL=https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/$GGUF_NAME
GGUF_SHA256=b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897

DEEP=0; GGUF_SRC=""; CA_CERT=""; DEV_BUNDLE=""; KIOSK=0; DISPLAY_MODEL=""; ROTATE=0; LLM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-deep-explain) DEEP=1; shift ;;
    --gguf) GGUF_SRC="$2"; shift 2 ;;
    --ca-cert) CA_CERT="$2"; shift 2 ;;
    --dev-bundle) DEV_BUNDLE="$2"; shift 2 ;;
    --kiosk) KIOSK=1; shift ;;
    --display) DISPLAY_MODEL="$2"; shift 2 ;;
    --rotate) ROTATE="$2"; shift 2 ;;
    --with-llm) LLM=1; shift ;;
    --no-llm) LLM=0; shift ;;                  # the default; kept for older instructions
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
apt-get install -y -qq python3-venv python3-dev git cmake build-essential curl ca-certificates rsync openssl \
  caddy nftables avahi-daemon

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

if [[ $LLM -eq 1 ]]; then
say "llama.cpp $LLAMA_TAG (native build for this CPU: NEON + dotprod where available)"
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
else
  say "Local language model: not installed (portal does not use it; --with-llm for research)"
fi

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
NEUROEDGE_BEHIND_PROXY=1
EOF
  chgrp neuroedge "$ETC/web.env"
fi
grep -q '^NEUROEDGE_BEHIND_PROXY=' "$ETC/web.env" || echo "NEUROEDGE_BEHIND_PROXY=1" >> "$ETC/web.env"
sed -i '/^NEUROEDGE_LLM=/d' "$ETC/web.env"
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
# Links in e-mails point at this device (admins can change it under Settings).
(cd "$APP/src" && runuser -u neuroedge -- "$APP/venv/bin/python" -m edge.portal.manage set-setting --data "$STATE/portal" base_url "https://$(hostname).local" >/dev/null)
# Outgoing e-mail credentials: empty until `sudo neuroedge-portal smtp-config`; never in the database.
[[ -f "$ETC/smtp.env" ]] || install -m 640 -o root -g neuroedge /dev/null "$ETC/smtp.env"
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
#   sudo neuroedge-portal smtp-config            (outgoing e-mail; writes $ETC/smtp.env as root)
#   sudo neuroedge-portal smtp-test --to you@clinic.org
cmd="\$1"; shift
if [ "\$cmd" = "smtp-config" ]; then
  cd $APP/src && $APP/venv/bin/python -m edge.portal.manage smtp-config --file $ETC/smtp.env "\$@" && chown root:neuroedge $ETC/smtp.env && chmod 640 $ETC/smtp.env
  exit
fi
cd $APP/src && exec runuser -u neuroedge -- env NEUROEDGE_SMTP_ENV=$ETC/smtp.env $APP/venv/bin/python -m edge.portal.manage "\$cmd" --data $STATE/portal "\$@"
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
if [[ $LLM -eq 1 ]]; then
  systemctl unmask neuroedge-llm.service >/dev/null 2>&1 || true
  systemctl enable --now neuroedge-llm.service
else
  systemctl disable --now neuroedge-llm.service >/dev/null 2>&1 || true
  systemctl mask neuroedge-llm.service >/dev/null       # the web unit only "Wants" it; masked = never started
fi
systemctl enable --now neuroedge-sync.timer neuroedge-web-restart.path neuroedge-reminders.timer
systemctl enable neuroedge-web.service
systemctl restart neuroedge-web.service || true    # skipped (condition) until a model is installed

say "Firewall (nftables table inet neuroedge: SSH, HTTP, HTTPS, mDNS in; everything else dropped)"
systemctl enable neuroedge-firewall.service
systemctl restart neuroedge-firewall.service

say "HTTPS: Caddy with its local CA in front of the portal"
# If Tailscale Funnel is used for internet access, tailscaled also listens on :443 (its own addresses);
# Caddy binds all addresses, so it must start first or it fails at boot.
mkdir -p /etc/systemd/system/caddy.service.d
printf '[Unit]
Before=tailscaled.service
' > /etc/systemd/system/caddy.service.d/before-tailscale.conf
# No RTC on the Pi: tailscaled must wait for NTP, or Funnel's TLS breaks when the clock jumps after boot.
mkdir -p /etc/systemd/system/tailscaled.service.d
printf '[Unit]
Wants=time-sync.target
After=time-sync.target
' > /etc/systemd/system/tailscaled.service.d/after-time-sync.conf
systemctl enable systemd-time-wait-sync.service >/dev/null 2>&1 || true
systemctl daemon-reload
install -m 644 "$APP/src/edge/caddy/Caddyfile" /etc/caddy/Caddyfile
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null
systemctl enable caddy
systemctl restart caddy                       # (restart, not reload: the admin API is off)
CA_ROOT=/var/lib/caddy/.local/share/caddy/pki/authorities/local/root.crt
for _ in $(seq 30); do
  [[ -f "$CA_ROOT" ]] && break
  curl -sk -o /dev/null https://127.0.0.1/ || true   # first handshake makes Caddy create its CA
  sleep 1
done
install -d -m 755 /srv/neuroedge-public       # world-readable: Caddy (user caddy) serves the CA cert from here
if [[ -f "$CA_ROOT" ]]; then install -m 644 "$CA_ROOT" /srv/neuroedge-public/neuroedge-ca.crt; fi
cat > /usr/local/bin/neuroedge-ca <<EOF
#!/bin/sh
# Show the portal CA certificate's fingerprint (compare it on each device that installs neuroedge-ca.crt).
exec openssl x509 -in /srv/neuroedge-public/neuroedge-ca.crt -noout -subject -fingerprint -sha256
EOF
chmod 755 /usr/local/bin/neuroedge-ca

if [[ -n "$DISPLAY_MODEL" ]]; then
  say "Display driver: $DISPLAY_MODEL"
  CFG=/boot/firmware/config.txt
  case "$DISPLAY_MODEL" in
    td1) OVL="dtoverlay=vc4-kms-dsi-7inch" ;;
    td2) OVL="dtoverlay=vc4-kms-dsi-ili9881-7inch" ;;
    hdmi) OVL="" ;;
    *) echo "unknown --display $DISPLAY_MODEL (td1, td2, hdmi)" >&2; exit 2 ;;
  esac
  # Raspberry Pi OS detects the official panels by itself (display_auto_detect=1); a second overlay can conflict.
  if [[ -n "$OVL" ]] && grep -q '^display_auto_detect=1' "$CFG"; then
    echo "display_auto_detect=1 is set: the official panel is detected automatically, no overlay added."
  elif [[ -n "$OVL" ]] && ! grep -q "^$OVL" "$CFG"; then
    printf '
# NeuroEdge kiosk display
%s
' "$OVL" >> "$CFG"
    echo "Added '$OVL' to $CFG: reboot to activate the panel."
  fi
fi

if [[ $KIOSK -eq 1 ]]; then
  say "Kiosk: device screen (labwc + Chromium + on-screen keyboard, user 'kiosk')"
  # Minimal Wayland session, no desktop. Chromium is 'chromium' on Debian 13 / Pi OS Trixie, 'chromium-browser' before.
  CHROMIUM_PKG=chromium; apt-cache show chromium >/dev/null 2>&1 || CHROMIUM_PKG=chromium-browser
  apt-get install -y -qq labwc "$CHROMIUM_PKG" squeekboard wlr-randr libnss3-tools fonts-dejavu-core
  CHROMIUM_BIN="$(command -v chromium || command -v chromium-browser)"
  # A user with no password, no sudo and no shell access over SSH; only the physical screen logs it in.
  id kiosk &>/dev/null || useradd --create-home --shell /bin/bash --groups video,render,input kiosk
  passwd -l kiosk >/dev/null
  KH=/home/kiosk
  install -d -o kiosk -g kiosk -m 700 "$KH/.config" "$KH/.config/labwc" "$KH/.pki" "$KH/.pki/nssdb"
  # Trust the portal's CA so https://localhost has no warning (cookies stay Secure / __Host-, through Caddy).
  if [[ -f "$CA_ROOT" ]]; then
    runuser -u kiosk -- certutil -d "sql:$KH/.pki/nssdb" -N --empty-password 2>/dev/null || true
    runuser -u kiosk -- certutil -d "sql:$KH/.pki/nssdb" -D -n neuroedge-portal 2>/dev/null || true
    runuser -u kiosk -- certutil -d "sql:$KH/.pki/nssdb" -A -t "C,," -n neuroedge-portal -i "$CA_ROOT"
  fi
  [[ -f "$ETC/kiosk.env" ]] || cat > "$ETC/kiosk.env" <<EOF
# NeuroEdge kiosk (the Pi's own screen). Edit, then reboot (or restart lightdm / getty@tty1).
KIOSK_URL=https://localhost/device?kiosk=1
KIOSK_ROTATE=$ROTATE
EOF
  chmod 644 "$ETC/kiosk.env"
  # labwc: no key bindings (no terminal / menu escape from the kiosk), no window decorations
  cat > "$KH/.config/labwc/rc.xml" <<'EOF'
<?xml version="1.0"?>
<labwc_config>
  <core><decoration>client</decoration></core>
  <keyboard></keyboard>
  <mouse><default/></mouse>
</labwc_config>
EOF
  printf '<?xml version="1.0"?>
<openbox_menu></openbox_menu>
' > "$KH/.config/labwc/menu.xml"
  cat > "$KH/.config/labwc/autostart" <<'EOF'
. /etc/neuroedge/kiosk.env
if [ "${KIOSK_ROTATE:-0}" != "0" ]; then
  for o in $(wlr-randr | awk '/^[^ ]/ {print $1}'); do wlr-randr --output "$o" --transform "$KIOSK_ROTATE"; done
fi
squeekboard &                       # on-screen keyboard: appears when a text field is focused (text-input-v3)
/usr/local/bin/neuroedge-kiosk-browser &
EOF
  cat > /usr/local/bin/neuroedge-kiosk-browser <<EOF
#!/bin/sh
# Full-screen browser for the device screen; restarted if it ever exits. Waits for the portal first.
. $ETC/kiosk.env
until curl -sk -o /dev/null https://localhost/device; do sleep 2; done
while true; do
  $CHROMIUM_BIN --kiosk --incognito --noerrdialogs --disable-translate --no-first-run --disable-features=Translate \
    --overscroll-history-navigation=0 --ozone-platform=wayland \
    --enable-wayland-ime --wayland-text-input-version=3 --password-store=basic "\$KIOSK_URL"
  sleep 2
done
EOF
  chmod 755 /usr/local/bin/neuroedge-kiosk-browser
  # Start labwc on tty1 after autologin
  cat > "$KH/.bash_profile" <<'EOF'
if [ "$(tty)" = "/dev/tty1" ] && [ -z "$WAYLAND_DISPLAY" ]; then exec labwc >/tmp/labwc.log 2>&1; fi
EOF
  chown -R kiosk:kiosk "$KH/.config" "$KH/.bash_profile"
  if systemctl is-enabled lightdm >/dev/null 2>&1; then
    # Desktop image (LightDM): log the kiosk user straight into a plain labwc session (our ~/.config/labwc).
    mkdir -p /etc/lightdm/lightdm.conf.d
    printf '[Seat:*]
autologin-user=kiosk
autologin-user-timeout=0
autologin-session=labwc
'       > /etc/lightdm/lightdm.conf.d/50-neuroedge-kiosk.conf
    KIOSK_RESTART="systemctl restart lightdm"
  else
    # Lite image: autologin on tty1, ~/.bash_profile starts labwc
    mkdir -p /etc/systemd/system/getty@tty1.service.d
    printf '[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin kiosk --noclear %%I $TERM
'       > /etc/systemd/system/getty@tty1.service.d/autologin.conf
    KIOSK_RESTART="systemctl restart getty@tty1"
  fi
  # No SSH for the kiosk account
  if [[ -d /etc/ssh/sshd_config.d ]]; then
    printf 'DenyUsers kiosk
' > /etc/ssh/sshd_config.d/neuroedge-kiosk.conf
    systemctl reload ssh 2>/dev/null || systemctl reload sshd 2>/dev/null || true
  fi
  systemctl daemon-reload
  $KIOSK_RESTART
fi

IP="$(hostname -I | awk '{print $1}')"
cat <<EOF

NeuroEdge installed.
  Web:  https://$IP/  or  https://$(hostname).local/      (starts once a model is installed)
  CA:   install http://$IP/neuroedge-ca.crt on each laptop/phone; check its fingerprint with: sudo neuroedge-ca

Create the first portal administrator (then add doctors from the Users page):
  sudo neuroedge-portal create-user --role admin --email you@clinic.org --name "Your Name"
Reminder e-mails (hospital SMTP server, or Gmail with an App Password):
  sudo neuroedge-portal smtp-config && sudo neuroedge-portal smtp-test --to you@clinic.org

Next, enroll this Pi with the model registry. On the server machine, create a one-time token:
  curl -X POST -H "Authorization: Bearer \$NE_ADMIN_TOKEN" https://<registry>/v1/admin/enrollment-tokens
Then on the Pi:
  sudo install -m 640 -g neuroedge signing_pub.pem $ETC/signing_pub.pem
  sudo neuroedge-sync enroll --server https://<registry> --token <one-time-token> --name pi5-01 --signing-pub $ETC/signing_pub.pem
  sudo neuroedge-sync pull --model mri       # downloads, trial-decrypts and verifies; the web app then starts itself
  sudo neuroedge-sync status
Benchmark:  cd $APP/src && sudo -u neuroedge env NEUROEDGE_HOME=$STATE $APP/venv/bin/python -m edge.bench --images <dir>
Kiosk:      sudo bash edge/install_pi.sh --kiosk [--display td1|td2] [--rotate 90]   (device screen shows /device)
EOF

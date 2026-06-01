#!/usr/bin/env bash
# setup_vps.sh — Bootstrap ClothBot on a fresh Ubuntu 22.04 VPS
# Usage: sudo bash setup_vps.sh
# -----------------------------------------------------------------------
set -euo pipefail

APP_DIR="/opt/clothing-classifier"
SERVICE_USER="www-data"
PYTHON_BIN="python3.12"

echo "=== [1/8] System packages ==="
apt-get update -qq
apt-get install -y --no-install-recommends \
    software-properties-common curl gnupg \
    python3.12 python3.12-venv python3.12-dev \
    build-essential libgl1 libglib2.0-0 \
    nginx mosquitto mosquitto-clients \
    ufw git

echo "=== [2/8] Python pip bootstrap ==="
# Ubuntu 24.04 uses externally-managed-environment; install pip via apt
apt-get install -y --no-install-recommends python3-pip python3-full

echo "=== [3/8] Application directory ==="
mkdir -p "$APP_DIR"
chown "$SERVICE_USER":"$SERVICE_USER" "$APP_DIR"

echo "=== [4/8] Python virtual environment ==="
"$PYTHON_BIN" -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip wheel

echo "=== [5/8] Install Python dependencies ==="
if [[ -f "$APP_DIR/requirements.txt" ]]; then
    "$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"
else
    echo "WARNING: $APP_DIR/requirements.txt not found — skipping pip install."
    echo "         Upload the app files first and re-run:  sudo $APP_DIR/venv/bin/pip install -r $APP_DIR/requirements.txt"
fi

echo "=== [6/8] Systemd service ==="
if [[ -f "$APP_DIR/clothing-classifier.service" ]]; then
    cp "$APP_DIR/clothing-classifier.service" /etc/systemd/system/clothing-classifier.service
    systemctl daemon-reload
    systemctl enable clothing-classifier
    systemctl restart clothing-classifier
    echo "    Service enabled and started."
else
    echo "WARNING: clothing-classifier.service not found in $APP_DIR — skip."
fi

echo "=== [7/8] Nginx reverse proxy ==="
if [[ -f "$APP_DIR/nginx.conf" ]]; then
    cp "$APP_DIR/nginx.conf" /etc/nginx/sites-available/clothbot
    ln -sf /etc/nginx/sites-available/clothbot /etc/nginx/sites-enabled/clothbot
    rm -f /etc/nginx/sites-enabled/default
    nginx -t && systemctl restart nginx
    echo "    Nginx configured."
else
    echo "WARNING: nginx.conf not found in $APP_DIR — skip."
fi

echo "=== [8/8] Firewall ==="
ufw allow OpenSSH
ufw allow 80/tcp      # HTTP (nginx)
ufw allow 1883/tcp    # MQTT (Mosquitto)
ufw --force enable
echo "    UFW enabled: SSH + 80 + 1883."

echo ""
echo "=== Setup complete ==="
echo "    App dir : $APP_DIR"
echo "    Service : systemctl status clothing-classifier"
echo "    Nginx   : systemctl status nginx"
echo "    Mosquitto: systemctl status mosquitto"
echo ""
echo "    Next: upload app files with deploy.sh, then restart the service."

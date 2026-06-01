#!/usr/bin/env bash
# ============================================================
# deploy.sh — Upload model artifacts & restart service on VPS
# Uses the 'clothbot-vps' alias from ~/.ssh/config (see ssh_config).
#
# Quick setup:
#   1. Append gce/ssh_config to ~/.ssh/config and fill in your IP/user/key
#   2. Run:  bash deploy.sh
#
# Usage:
#   bash deploy.sh [SSH_ALIAS]        default alias: clothbot-vps
#
# Or override with env var:
#   SSH_ALIAS=my-server bash deploy.sh
# ============================================================

set -euo pipefail

SSH_ALIAS="${SSH_ALIAS:-${1:-clothbot-vps}}"
DEPLOY_DIR="/opt/clothing-classifier"
SERVICE_NAME="clothing-classifier"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  ClothBot — VPS Deploy"
echo "  SSH alias : $SSH_ALIAS  (configure in ~/.ssh/config)"
echo "  Source    : $SCRIPT_DIR"
echo "  Remote    : ${DEPLOY_DIR}"
echo "============================================================"

# ---- Step 1: Validate local artifacts ----
echo ""
echo "[1/5] Validating local artifacts..."

REQUIRED_FILES=(
    "best.onnx"
    "labels.txt"
    "app.py"
    "mqtt_publisher.py"
    "servo_map.py"
    "requirements.txt"
    "nginx.conf"
    "clothing-classifier.service"
)
for f in "${REQUIRED_FILES[@]}"; do
    fpath="$SCRIPT_DIR/$f"
    if [[ ! -f "$fpath" ]]; then
        echo "[ERROR] Required file not found: $fpath"
        exit 1
    fi
    size=$(du -sh "$fpath" | cut -f1)
    echo "  [OK] $f  ($size)"
done

# ---- Step 2: Upload files ----
echo ""
echo "[2/5] Uploading files to ${SSH_ALIAS}:${DEPLOY_DIR} ..."

scp \
    "$SCRIPT_DIR/best.onnx" \
    "$SCRIPT_DIR/labels.txt" \
    "$SCRIPT_DIR/app.py" \
    "$SCRIPT_DIR/mqtt_publisher.py" \
    "$SCRIPT_DIR/servo_map.py" \
    "$SCRIPT_DIR/requirements.txt" \
    "$SCRIPT_DIR/nginx.conf" \
    "$SCRIPT_DIR/clothing-classifier.service" \
    "${SSH_ALIAS}:${DEPLOY_DIR}/"

# Upload model/ folder (fold pipeline)
if [[ -d "$SCRIPT_DIR/model" ]]; then
    echo "  Uploading model/ folder..."
    ssh "$SSH_ALIAS" "mkdir -p ${DEPLOY_DIR}/model"
    scp -r "$SCRIPT_DIR/model/." "${SSH_ALIAS}:${DEPLOY_DIR}/model/"
fi

echo "[OK] Files uploaded."

# ---- Step 3: Install deps & restart service ----
echo ""
echo "[3/5] Installing requirements and restarting service..."

# shellcheck disable=SC2087
ssh "$SSH_ALIAS" bash << REMOTE
set -e

echo '[VM] Updating Python requirements...'
${DEPLOY_DIR}/venv/bin/pip install -q --no-cache-dir -r ${DEPLOY_DIR}/requirements.txt

echo '[VM] Updating systemd service...'
sudo cp ${DEPLOY_DIR}/${SERVICE_NAME}.service /etc/systemd/system/${SERVICE_NAME}.service
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}

echo '[VM] Restarting service...'
sudo systemctl restart ${SERVICE_NAME}

echo '[VM] Service status:'
sudo systemctl status ${SERVICE_NAME} --no-pager || true
REMOTE

# ---- Step 4: Health check ----
echo ""
echo "[4/5] Waiting 5 s for service to come up..."
sleep 5

VPS_IP=$(ssh "$SSH_ALIAS" "curl -s4 ifconfig.me 2>/dev/null || hostname -I | awk '{print \$1}'" 2>/dev/null || echo "UNKNOWN")

echo "VPS IP: $VPS_IP"
echo "Running health check via SSH tunnel..."

if ssh "$SSH_ALIAS" "curl -sf --max-time 10 http://localhost/health" | python3 -m json.tool; then
    echo ""
    echo "[OK] Service is healthy!"
else
    echo ""
    echo "[WARN] Health check failed. Check logs with:"
    echo "  ssh ${SSH_ALIAS} 'journalctl -u ${SERVICE_NAME} -n 50 --no-pager'"
    exit 1
fi

# ---- Step 5: Print endpoint info ----
echo ""
echo "============================================================"
echo "  DEPLOY COMPLETE"
echo "============================================================"
echo ""
echo "  Predict  : http://${VPS_IP}/predict"
echo "  Health   : http://${VPS_IP}/health"
echo "  Metrics  : http://${VPS_IP}/metrics"
echo "  Web UI   : http://${VPS_IP}/ui"
echo "  API docs : http://${VPS_IP}/docs"
echo ""
echo "  ESP32 config:"
echo "    URL    : http://${VPS_IP}/predict"
echo "    Method : POST multipart/form-data  field: image (JPEG)"
echo ""
echo "  Monitor logs:"
echo "    ssh ${SSH_ALIAS} 'journalctl -u ${SERVICE_NAME} -f'"
echo "============================================================"


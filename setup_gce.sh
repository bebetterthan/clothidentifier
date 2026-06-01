#!/usr/bin/env bash
# ============================================================
# setup_gce.sh — Setup GCE VM dari nol untuk Clothing Classifier
# Jalankan sebagai root/sudo pada Ubuntu 22.04 LTS baru
# ============================================================

set -e   # exit immediately on any error

DEPLOY_DIR="/opt/clothing-classifier"
SERVICE_NAME="clothing-classifier"
NGINX_SITE="clothing-classifier"

echo "============================================================"
echo "  Clothing Classifier — GCE VM Setup"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---- Step 1: Update system packages ----
echo ""
echo "[1/12] Updating system packages..."
apt-get update -y
apt-get upgrade -y

# ---- Step 2: Install system dependencies ----
echo ""
echo "[2/12] Installing system dependencies..."
apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    nginx \
    libgomp1 \
    curl \
    git
echo "[OK] System dependencies installed."

# ---- Step 3: Create deploy directory ----
echo ""
echo "[3/12] Creating deploy directory: $DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"
chown www-data:www-data "$DEPLOY_DIR"
chmod 755 "$DEPLOY_DIR"
echo "[OK] Directory created."

# ---- Step 4: Create Python 3.11 virtual environment ----
echo ""
echo "[4/12] Creating Python 3.11 virtual environment..."
python3.11 -m venv "$DEPLOY_DIR/venv"
"$DEPLOY_DIR/venv/bin/pip" install --upgrade pip --quiet
echo "[OK] venv created at $DEPLOY_DIR/venv"

# ---- Step 5: Install Python requirements ----
echo ""
echo "[5/12] Installing Python requirements..."
if [[ -f "$DEPLOY_DIR/requirements.txt" ]]; then
    "$DEPLOY_DIR/venv/bin/pip" install \
        -r "$DEPLOY_DIR/requirements.txt" \
        --quiet --no-cache-dir
    echo "[OK] Python packages installed."
else
    echo "[WARN] requirements.txt not found in $DEPLOY_DIR."
    echo "       Run deploy.sh first to upload files, then re-run this step:"
    echo "       /opt/clothing-classifier/venv/bin/pip install -r /opt/clothing-classifier/requirements.txt"
fi

# ---- Step 6: Configure Nginx ----
echo ""
echo "[6/12] Configuring Nginx reverse proxy..."
if [[ -f "$DEPLOY_DIR/nginx.conf" ]]; then
    cp "$DEPLOY_DIR/nginx.conf" "/etc/nginx/sites-available/$NGINX_SITE"
    echo "[OK] Nginx config copied to sites-available."
else
    echo "[WARN] nginx.conf not found in $DEPLOY_DIR. Skipping Nginx config."
fi

# ---- Step 7: Enable site, remove default ----
echo ""
echo "[7/12] Enabling Nginx site..."
if [[ -f "/etc/nginx/sites-available/$NGINX_SITE" ]]; then
    ln -sf \
        "/etc/nginx/sites-available/$NGINX_SITE" \
        "/etc/nginx/sites-enabled/$NGINX_SITE"
    rm -f /etc/nginx/sites-enabled/default
    echo "[OK] Site enabled."
else
    echo "[WARN] Nginx site config not found; skipping symlink."
fi

# ---- Step 8: Test Nginx config ----
echo ""
echo "[8/12] Validating Nginx configuration..."
nginx -t
echo "[OK] Nginx config is valid."

# ---- Step 9: Install systemd service file ----
echo ""
echo "[9/12] Installing systemd unit file..."
if [[ -f "$DEPLOY_DIR/$SERVICE_NAME.service" ]]; then
    cp "$DEPLOY_DIR/$SERVICE_NAME.service" \
       "/etc/systemd/system/$SERVICE_NAME.service"
    chmod 644 "/etc/systemd/system/$SERVICE_NAME.service"
    echo "[OK] Service file installed."
else
    echo "[WARN] $SERVICE_NAME.service not found in $DEPLOY_DIR. Skipping."
fi

# ---- Step 10: Reload systemd daemon ----
echo ""
echo "[10/12] Reloading systemd daemon..."
systemctl daemon-reload
systemctl enable "$SERVICE_NAME" 2>/dev/null || true
echo "[OK] Systemd reloaded. Service enabled."

# ---- Step 11: Start / restart Nginx ----
echo ""
echo "[11/12] Starting Nginx..."
systemctl enable nginx
systemctl restart nginx
echo "[OK] Nginx started."

# ---- Step 12: Print next steps ----
echo ""
echo "[12/12] Setup complete!"
echo ""
echo "============================================================"
echo "  NEXT STEPS"
echo "============================================================"
echo ""
echo "  1. Upload model artifacts (dari laptop/Colab):"
echo "     bash deploy.sh"
echo ""
echo "  2. Start classifier service:"
echo "     sudo systemctl start $SERVICE_NAME"
echo "     sudo systemctl status $SERVICE_NAME"
echo ""
echo "  3. Monitor logs:"
echo "     journalctl -u $SERVICE_NAME -f"
echo ""
echo "  4. Test health:"
echo "     curl http://localhost/health"
echo ""
echo "  5. Cek port terbuka:"
echo "     sudo ss -tlnp | grep -E '80|8000'"
echo "============================================================"

#!/usr/bin/env bash
# ============================================================
# create_vm.sh — Buat GCE VM untuk Clothing Classifier
#
# Usage:
#   bash create_vm.sh [GCP_PROJECT_ID]
#   GCP_PROJECT_ID=my-project bash create_vm.sh
#
# Requirements:
#   gcloud CLI terinstall & authenticated (gcloud auth login)
#   Billing aktif pada project GCP
# ============================================================

set -e

# ---- Configuration ----
GCP_PROJECT_ID="${GCP_PROJECT_ID:-${1:-}}"
VM_NAME="${VM_NAME:-clothing-classifier-vm}"
ZONE="${ZONE:-asia-southeast2-a}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-small}"
DISK_SIZE="${DISK_SIZE:-20GB}"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
FIREWALL_TAG="clothing-classifier"
BOOT_DISK_TYPE="pd-standard"

PROJECT_FLAG=""
if [[ -n "$GCP_PROJECT_ID" ]]; then
    PROJECT_FLAG="--project=${GCP_PROJECT_ID}"
fi

echo "============================================================"
echo "  Clothing Classifier — GCE VM Creation"
echo "============================================================"
echo "  VM Name      : $VM_NAME"
echo "  Zone         : $ZONE"
echo "  Machine      : $MACHINE_TYPE"
echo "  OS           : Ubuntu 22.04 LTS"
echo "  Disk         : $DISK_SIZE  ($BOOT_DISK_TYPE)"
echo "  Project      : ${GCP_PROJECT_ID:-'(default gcloud project)'}"
echo "============================================================"
echo ""

# ---- Confirm ----
read -r -p "Lanjutkan? Ini akan membuat VM baru dan memakan biaya. [y/N] " confirm
if [[ "${confirm,,}" != "y" ]]; then
    echo "Dibatalkan."
    exit 0
fi

# ---- Step 1: Create VM instance ----
echo ""
echo "[1/4] Membuat VM instance: $VM_NAME ..."
gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$DISK_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --tags="$FIREWALL_TAG,http-server" \
    --metadata=enable-oslogin=TRUE \
    $PROJECT_FLAG

echo "[OK] VM created."

# ---- Step 2: Create firewall rule for HTTP (port 80) ----
echo ""
echo "[2/4] Membuat firewall rule untuk port 80 (HTTP) ..."

RULE_NAME="allow-http-${FIREWALL_TAG}"
# Check if rule already exists
if gcloud compute firewall-rules describe "$RULE_NAME" \
        $PROJECT_FLAG &>/dev/null; then
    echo "[SKIP] Firewall rule '$RULE_NAME' sudah ada."
else
    gcloud compute firewall-rules create "$RULE_NAME" \
        --direction=INGRESS \
        --priority=1000 \
        --network=default \
        --action=ALLOW \
        --rules=tcp:80 \
        --source-ranges=0.0.0.0/0 \
        --target-tags="$FIREWALL_TAG" \
        $PROJECT_FLAG
    echo "[OK] Firewall rule created: port 80 → all sources."
fi

# ---- Step 3: Wait for VM to be ready ----
echo ""
echo "[3/4] Menunggu VM siap untuk SSH (30 detik) ..."
sleep 30

# Verify SSH connectivity
echo "Mencoba SSH ke VM..."
for i in {1..5}; do
    if gcloud compute ssh "$VM_NAME" \
            --zone="$ZONE" \
            $PROJECT_FLAG \
            --command="echo '[VM] SSH OK'" \
            --ssh-flag="-o ConnectTimeout=10" 2>/dev/null; then
        echo "[OK] SSH berhasil."
        break
    fi
    echo "  Percobaan $i/5 gagal — tunggu 10 detik..."
    sleep 10
done

# ---- Step 4: Get external IP ----
echo ""
echo "[4/4] Mendapatkan external IP..."
EXTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
    --zone="$ZONE" \
    $PROJECT_FLAG \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo "[OK] External IP: $EXTERNAL_IP"

# ---- Summary ----
echo ""
echo "============================================================"
echo "  VM CREATED SUCCESSFULLY"
echo "============================================================"
echo ""
echo "  VM Name    : $VM_NAME"
echo "  Zone       : $ZONE"
echo "  Machine    : $MACHINE_TYPE"
echo "  External IP: $EXTERNAL_IP"
echo ""
echo "  NEXT STEPS:"
echo ""
echo "  1. SSH ke VM:"
echo "     gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "  2. Upload artifacts dari laptop:"
echo "     bash deploy.sh ${GCP_PROJECT_ID:-<YOUR_PROJECT_ID>} $VM_NAME $ZONE"
echo ""
echo "  3. Setup environment di VM (jalankan setelah upload):"
echo "     gcloud compute ssh $VM_NAME --zone=$ZONE \\"
echo "         -- sudo bash /opt/clothing-classifier/setup_gce.sh"
echo ""
echo "  4. Cek service:"
echo "     gcloud compute ssh $VM_NAME --zone=$ZONE \\"
echo "         -- sudo systemctl status clothing-classifier"
echo ""
echo "  5. Test endpoint:"
echo "     python test_endpoint.py --url http://$EXTERNAL_IP --image path/to/test.jpg"
echo ""
echo "  Biaya estimasi e2-small jakarta:"
echo "  ~\$13-15 USD/bulan (running 24/7)"
echo "  Matikan saat tidak dipakai: gcloud compute instances stop $VM_NAME --zone=$ZONE"
echo "============================================================"

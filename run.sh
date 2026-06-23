#!/usr/bin/env bash
# ============================================================
# run.sh — Start ClothBot API server for local development
#
# Usage:
#   ./run.sh               # default: hot-reload on 127.0.0.1:8000
#   ./run.sh --no-reload   # production-like (no --reload flag)
#   PORT=9000 ./run.sh     # custom port
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load .env if present ──────────────────────────────────────
if [ -f ".env" ]; then
  # Export non-comment, non-empty lines
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "[run.sh] Loaded .env"
else
  echo "[run.sh] No .env found — using defaults (MQTT disabled, size estimation on)"
  export MQTT_ENABLED=false
  export FOLD_ENABLED=true
  export SIZE_ESTIMATION_ENABLED=true
  export SIZE_DEBUG_MODE=false
fi

# ── Resolve Python / venv ─────────────────────────────────────
if [ -f ".venv/bin/uvicorn" ]; then
  UVICORN=".venv/bin/uvicorn"
elif command -v uvicorn &>/dev/null; then
  UVICORN="uvicorn"
else
  echo "[run.sh] ERROR: uvicorn not found."
  echo "         Run: pip install -r requirements.txt"
  exit 1
fi

# ── Configuration ─────────────────────────────────────────────
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "[run.sh] Starting ClothBot API"
echo "         Host  : http://${HOST}:${PORT}"
echo "         UI    : http://${HOST}:${PORT}/ui"
echo "         Docs  : http://${HOST}:${PORT}/docs"
echo "         MQTT  : ${MQTT_ENABLED:-false}"
echo "         Fold  : ${FOLD_ENABLED:-true}"
echo "         Size  : ${SIZE_ESTIMATION_ENABLED:-true}"
echo ""

# ── Calibration reminder ──────────────────────────────────────
if [ ! -f "config.json" ] && [ "${SIZE_ESTIMATION_ENABLED:-true}" = "true" ]; then
  echo "[run.sh] WARN: config.json not found — size estimation will be inactive."
  echo "         Opsi kalibrasi:"
  echo "           1. Kamera live (klik manual): python calibration.py --camera"
  echo "           2. Dari foto:                python calibration.py --image <foto_pelipat.jpg>"
  echo "           3. Upload via API:           POST http://${HOST}:${PORT}/calibrate"
  echo ""
fi

# ── Parse --no-reload flag ────────────────────────────────────
RELOAD_FLAG="--reload"
for arg in "$@"; do
  if [ "$arg" = "--no-reload" ]; then
    RELOAD_FLAG=""
    break
  fi
done

# ── Run ───────────────────────────────────────────────────────
exec "$UVICORN" app:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  $RELOAD_FLAG

"""
app.py — ClothBot Inference API v2.0
Pipeline: Image → YOLOv8l-cls (ONNX) → Folding Planner v3.2 (Canny)
"""

# ── Thread limits — MUST be set before any heavy native library is imported ─────────
import multiprocessing as _mp
import os as _os

_cpu = _mp.cpu_count() or 2
_threads = str(min(_cpu, 4))
_os.environ.setdefault("OMP_NUM_THREADS", _threads)
_os.environ.setdefault("OPENBLAS_NUM_THREADS", _threads)
_os.environ.setdefault("MKL_NUM_THREADS", _threads)
del _mp, _os, _threads

import base64
import io
import json
import logging
import multiprocessing
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

# Use the modern Resampling enum when available (Pillow ≥ 9.1); fall back to
# the legacy constant for older installations.
_PIL_BILINEAR = (
    Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
)

# Optional: opencv — required for fold pipeline
try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# Optional: MQTT publisher + servo map
try:
    from mqtt_publisher import ClothbotMQTT
    from servo_map import SERVO_MAP

    _MQTT_MODULE_AVAILABLE = True
except ImportError:
    _MQTT_MODULE_AVAILABLE = False
    SERVO_MAP = {}

# Optional: size estimation modules (require opencv-python)
try:
    from calibration import detect_folder_edges
    from size_estimator import (
        FOLDER_WIDTH_CM,
        estimate_size,
        is_size_enabled,
        load_folder_config,
    )
    from size_estimator import (
        get_warp_matrix as _get_warp_matrix,
    )
    from size_estimator import (
        load_config as _size_load_config,
    )
    from size_estimator import (
        run_size_estimation as _size_module_run,
    )
    from visualization import (
        annotate_size,
        draw_reference_lines,
        draw_size_result,
    )

    _SIZE_MODULE_AVAILABLE = True
except ImportError:
    _SIZE_MODULE_AVAILABLE = False
    FOLDER_WIDTH_CM = 90.0  # fallback constant (mirrors size_estimator.FOLDER_WIDTH_CM)

# ── Thread limits — set before heavy imports so OpenMP picks them up ──────────
# (values already applied above at module top; block kept for documentation)

# ----------------------------------------------------------------
# Logging
# ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("clothbot")

# ----------------------------------------------------------------
# Configuration — read from environment variables
# ----------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "./best.onnx")
LABELS_PATH = os.getenv("LABELS_PATH", "./labels.txt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.75"))
FOLD_ENABLED = os.getenv("FOLD_ENABLED", "true").lower() == "true"

# 0 = auto-detect: min(cpu_count, 4). Override for production VMs.
INTRA_OP_NUM_THREADS = int(os.getenv("INTRA_OP_NUM_THREADS", "0"))

SIZE_ESTIMATION_ENABLED = os.getenv("SIZE_ESTIMATION_ENABLED", "true").lower() == "true"
SIZE_DEBUG_MODE = os.getenv("SIZE_DEBUG_MODE", "false").lower() == "true"
CONFIG_PATH = os.getenv("CONFIG_PATH", "./config.json")

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
INPUT_SIZE = 224

MQTT_ENABLED = os.getenv("MQTT_ENABLED", "true").lower() == "true"
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER", "clothbot")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "clothbot/servo/command")


# ----------------------------------------------------------------
# Application state
# ----------------------------------------------------------------


@dataclass
class AppState:
    session: Optional[ort.InferenceSession] = None
    input_name: str = ""
    labels: list = field(default_factory=list)
    conf_threshold: float = CONF_THRESHOLD
    start_time: float = field(default_factory=time.time)
    fold_pipeline: Optional[dict] = None
    fold_enabled: bool = False
    mqtt: Optional[object] = None  # ClothbotMQTT
    folder_width_px: Optional[float] = None
    folder_x1: Optional[int] = None
    folder_x2: Optional[int] = None
    calib_image_width: Optional[int] = None  # resolution of calibration image
    scale_cm_per_px: Optional[float] = None  # cm per pixel at calibration resolution
    size_enabled: bool = False
    size_config: Optional[dict] = None  # raw config dict (includes perspective fields)

    total_requests: int = 0
    success_requests: int = 0
    error_requests: int = 0
    inference_count: int = 0
    _latencies: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, latency_ms: float, success: bool) -> None:
        with self._lock:
            self.total_requests += 1
            if success:
                self.success_requests += 1
                self.inference_count += 1
                self._latencies.append(latency_ms)
                if len(self._latencies) > 1000:
                    self._latencies = self._latencies[-1000:]
            else:
                self.error_requests += 1

    def stats_snapshot(self) -> dict:
        with self._lock:
            lats = self._latencies.copy()
        if lats:
            return {
                "avg_latency_ms": round(float(np.mean(lats)), 3),
                "min_latency_ms": round(float(np.min(lats)), 3),
                "max_latency_ms": round(float(np.max(lats)), 3),
            }
        return {"avg_latency_ms": 0.0, "min_latency_ms": 0.0, "max_latency_ms": 0.0}


_state = AppState()


# ----------------------------------------------------------------
# YOLO inference pipeline
# ----------------------------------------------------------------


def _find_garment_bbox(
    img_rgb: np.ndarray,
) -> "tuple[int, int, int, int] | None":
    """
    Deteksi bounding box garmen menggunakan CLAHE + adaptive Canny.

    Dipakai oleh preprocess() untuk crop ke area baju sebelum center-crop,
    sehingga lengan tidak terpotong meski foto diambil dari jauh.

    Parameters
    ----------
    img_rgb : RGB numpy array (H, W, 3)

    Returns
    -------
    (x, y, w, h) bounding rect kontur terlebar, atau None jika gagal.
    """
    h, w = img_rgb.shape[:2]

    # CLAHE pada channel L (LAB) — sama dengan _decode_and_canny
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    gray = cv2.cvtColor(cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Canny via Otsu
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo = max(10.0, otsu * 0.33)
    hi = max(20.0, otsu * 0.66)
    edges = cv2.Canny(blur, lo, hi)

    # Fallback: morphological gradient
    if np.count_nonzero(edges) < 50:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        _, edges = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter noise — sama dengan run_size_estimation
    min_area = h * w * 0.05
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid:
        return None

    best = max(valid, key=lambda c: cv2.boundingRect(c)[2])
    return cv2.boundingRect(best)  # (x, y, w, h)


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB")
    w, h = img.size

    # Garment-aware crop: gunakan Canny untuk mendeteksi area baju,
    # lalu crop ke sana sebelum center-crop dijalankan.
    # Ini memastikan lengan baju tidak terpotong tanpa perlu retrain model —
    # setelah crop, baju memenuhi sebagian besar frame sehingga center-crop
    # menangkap seluruh garmen termasuk ujung lengan.
    if _CV2_AVAILABLE:
        try:
            arr_rgb = np.array(img)
            bbox = _find_garment_bbox(arr_rgb)
            if bbox is not None:
                bx, by, bw, bh = bbox
                # Padding 10% dari sisi terpanjang agar tidak terlalu mepet
                pad = int(max(bw, bh) * 0.10)
                x1 = max(0, bx - pad)
                y1 = max(0, by - pad)
                x2 = min(w, bx + bw + pad)
                y2 = min(h, by + bh + pad)
                # Hanya crop kalau hasilnya cukup besar (bukan deteksi noise)
                if (x2 - x1) > INPUT_SIZE // 2 and (y2 - y1) > INPUT_SIZE // 2:
                    img = img.crop((x1, y1, x2, y2))
                    w, h = img.size
        except Exception as exc:
            logger.debug("[PREPROC] garment crop failed, using full frame: %s", exc)

    # YOLOv8-cls standard: resize shortest side ke 256 lalu center-crop 224×224.
    # Konsisten dengan preprocessing training.
    scale = 256 / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), _PIL_BILINEAR)
    left = (new_w - INPUT_SIZE) // 2
    top = (new_h - INPUT_SIZE) // 2
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0).astype(np.float32)


def run_inference(tensor: np.ndarray) -> np.ndarray:
    return _state.session.run(None, {_state.input_name: tensor})[0]


def postprocess(raw_output: np.ndarray) -> dict:
    probs = raw_output[0]
    idx = int(np.argmax(probs))
    conf = round(float(probs[idx]), 5)
    return {
        "predicted_class": _state.labels[idx],
        "confidence": conf,
        "is_confident": conf >= _state.conf_threshold,
        "threshold": round(_state.conf_threshold, 5),
        "probabilities": {
            _state.labels[i]: round(float(p), 5) for i, p in enumerate(probs)
        },
    }


# ----------------------------------------------------------------
# Folding pipeline — professional fold line drawing system
# ----------------------------------------------------------------

_FOLD_SKIP_LABELS = {"null"}

FOLD_CONFIG = {
    "baju kaos": {
        "v_left": 0.33,
        "v_right": 0.67,
        "h_fold": 0.70,
        "labels": ["Lipat kiri", "Lipat kanan", "Lipat bawah"],
    },
    "kemeja": {
        "v_left": 0.30,
        "v_right": 0.70,
        "h_fold": 0.65,
        "labels": ["Lipat kiri", "Lipat kanan", "Lipat bawah"],
    },
    "celana": {
        "v_left": 0.50,
        "v_right": None,
        "h_fold": 0.50,
        "labels": ["Lipat tengah", None, "Lipat bawah"],
    },
    "jaket": {
        "v_left": 0.28,
        "v_right": 0.72,
        "h_fold": 0.60,
        "labels": ["Lipat kiri", "Lipat kanan", "Lipat bawah"],
    },
}

# Map YOLO class labels → FOLD_CONFIG keys
_LABEL_MAP: dict = {
    "baju_lengan_pendek": "baju kaos",
    "baju_lengan_panjang": "kemeja",
    "celana_panjang": "celana",
    "celana_pendek": "celana",
}

DEFAULT_CONFIG = FOLD_CONFIG["baju kaos"]

# Sizes that require an extra (outer) fold line on each SIDE.
DOUBLE_FOLD_SIZES: frozenset = frozenset({"L", "XL", "XXL"})

# Clothing types whose BOTTOM fold is always doubled, regardless of size.
# baju_lengan_panjang (kemeja) membutuhkan 2x lipatan bawah karena lebih panjang.
ALWAYS_DOUBLE_BOTTOM_TYPES: frozenset = frozenset({"kemeja"})

# Clothing types whose SIDE folds are always doubled, regardless of size.
# baju_lengan_panjang (kemeja) selalu butuh 2x lipat kiri & kanan.
ALWAYS_DOUBLE_SIDES_TYPES: frozenset = frozenset({"kemeja"})

# How far the outer fold line sits from the main fold line,
# expressed as a fraction of the image dimension.
DOUBLE_FOLD_OUTER_DELTA: float = 0.08


def _dashed_line(img, pt1, pt2, color, thickness, dash=12, gap=7):
    """Draw a dashed line from pt1 to pt2 directly onto img."""
    x1, y1 = pt1
    x2, y2 = pt2
    total = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if total == 0:
        return
    drawn = 0
    while drawn < total:
        start = drawn / total
        end = min((drawn + dash) / total, 1.0)
        sx = int(x1 + start * (x2 - x1))
        sy = int(y1 + start * (y2 - y1))
        ex = int(x1 + end * (x2 - x1))
        ey = int(y1 + end * (y2 - y1))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        drawn += dash + gap


def _pill_label(img, text, cx, cy, scale=0.38, alpha=0.55):
    """
    Draw a pill-style label centred at (cx, cy).
    Background: semi-transparent black rectangle.
    Foreground: white text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    px, py = 5, 3  # horizontal / vertical padding
    x0 = cx - tw // 2 - px
    y0 = cy - th - py
    x1 = cx + tw // 2 + px
    y1 = cy + bl + py

    # blend dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    # draw white text on blended background
    tx = cx - tw // 2
    ty = cy
    cv2.putText(img, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_fold_lines(img: np.ndarray, cloth_type: str, size: str | None = None) -> tuple:
    """
    Draw clean fold guide lines on *img* using the FOLD_CONFIG system.

    Lines drawn:
      • LINE 1 — solid white vertical at v_left  (left fold)
      • LINE 2 — solid white vertical at v_right (right fold, optional)
      • LINE 3 — dashed white horizontal at h_fold (bottom fold)

    All elements use alpha blending:
      lines   → alpha 0.85
      arrows  → alpha 0.70
      labels  → alpha 0.55

    Returns
    -------
    (annotated_img, fold_lines)
        annotated_img : np.ndarray  — copy of img with guides drawn
        fold_lines    : list[dict]  — serialisable list for JSON response
    """
    # ── resolve config ─────────────────────────────────────────────────────
    ct = _LABEL_MAP.get(cloth_type.lower().strip(), cloth_type.lower().strip())
    config = FOLD_CONFIG.get(ct, DEFAULT_CONFIG)

    h, w = img.shape[:2]
    WHITE = (255, 255, 255)

    x1_v = int(w * config["v_left"])
    x2_v = int(w * config["v_right"]) if config["v_right"] is not None else None
    y3 = int(h * config["h_fold"])
    labels = config["labels"]

    # Double-fold: pisah kontrol untuk sisi (kiri/kanan) dan bawah.
    #   is_double_sides  → extra outer vertical fold line (L/XL/XXL saja)
    #   is_double_bottom → extra outer bottom fold line
    #                      (L/XL/XXL ATAU tipe baju yang selalu 2x lipat bawah,
    #                       mis. kemeja = baju_lengan_panjang)
    is_double_sides = size in DOUBLE_FOLD_SIZES or (ct in ALWAYS_DOUBLE_SIDES_TYPES)
    is_double_bottom = is_double_sides or (ct in ALWAYS_DOUBLE_BOTTOM_TYPES)

    if is_double_sides:
        x1_outer = max(
            int(w * 0.05), int(w * (config["v_left"] - DOUBLE_FOLD_OUTER_DELTA))
        )
        x2_outer = (
            min(int(w * 0.95), int(w * (config["v_right"] + DOUBLE_FOLD_OUTER_DELTA)))
            if config["v_right"] is not None
            else None
        )
    if is_double_bottom:
        y3_outer = min(
            int(h * 0.92), int(h * (config["h_fold"] + DOUBLE_FOLD_OUTER_DELTA))
        )

    y_top = int(h * 0.10)
    y_bot = int(h * 0.90)
    x_left = int(w * 0.05)
    x_right = int(w * 0.95)
    mid_y = (y_top + y_bot) // 2
    cx_img = w // 2

    result = img.copy()
    fold_lines: list = []

    # ── LINE 1 & LINE 2 — solid vertical lines (alpha 0.85) ────────────────
    ov_lines = result.copy()

    # For double-fold sizes, draw the outer SIDE fold lines first (dashed)
    if is_double_sides:
        _dashed_line(ov_lines, (x1_outer, y_top), (x1_outer, y_bot), WHITE, 2)
        fold_lines.append(
            {
                "from": [x1_outer, y_top],
                "to": [x1_outer, y_bot],
                "color": [255, 255, 255],
                "label": "Lipat kiri 1",
            }
        )
        if x2_outer is not None:
            _dashed_line(ov_lines, (x2_outer, y_top), (x2_outer, y_bot), WHITE, 2)
            fold_lines.append(
                {
                    "from": [x2_outer, y_top],
                    "to": [x2_outer, y_bot],
                    "color": [255, 255, 255],
                    "label": "Lipat kanan 1",
                }
            )

    # Main (inner) fold lines — always drawn
    label_left = (
        ("Lipat kiri 2" if is_double_sides else None) or labels[0] or "Lipat kiri"
    )
    label_right = (
        ("Lipat kanan 2" if is_double_sides else None) or labels[1] or "Lipat kanan"
    )

    cv2.line(ov_lines, (x1_v, y_top), (x1_v, y_bot), WHITE, 2, cv2.LINE_AA)
    fold_lines.append(
        {
            "from": [x1_v, y_top],
            "to": [x1_v, y_bot],
            "color": [255, 255, 255],
            "label": label_left,
        }
    )

    if x2_v is not None:
        cv2.line(ov_lines, (x2_v, y_top), (x2_v, y_bot), WHITE, 2, cv2.LINE_AA)
        fold_lines.append(
            {
                "from": [x2_v, y_top],
                "to": [x2_v, y_bot],
                "color": [255, 255, 255],
                "label": label_right,
            }
        )

    cv2.addWeighted(ov_lines, 0.85, result, 0.15, 0, result)

    # ── LINE 3 — dashed horizontal line(s) (alpha 0.85) ────────────────────
    ov_dash = result.copy()

    # For double-fold BOTTOM: kemeja selalu 2x lipat bawah, L/XL/XXL juga
    if is_double_bottom:
        _dashed_line(ov_dash, (x_left, y3_outer), (x_right, y3_outer), WHITE, 2)
        fold_lines.append(
            {
                "from": [x_left, y3_outer],
                "to": [x_right, y3_outer],
                "color": [255, 255, 255],
                "label": "Lipat bawah 1",
            }
        )

    label_bottom = (
        ("Lipat bawah 2" if is_double_bottom else None) or labels[2] or "Lipat bawah"
    )
    _dashed_line(ov_dash, (x_left, y3), (x_right, y3), WHITE, 2)
    cv2.addWeighted(ov_dash, 0.85, result, 0.15, 0, result)
    fold_lines.append(
        {
            "from": [x_left, y3],
            "to": [x_right, y3],
            "color": [255, 255, 255],
            "label": label_bottom,
        }
    )

    # ── Arrows (alpha 0.70) ─────────────────────────────────────────────────
    ov_arr = result.copy()

    # LINE 1 arrow — dari KIRI ke KANAN (menuju tengah/center)
    # Lipat kiri = kain dari sisi kiri dilipat ke arah tengah → panah ke kanan
    cv2.arrowedLine(
        ov_arr, (x1_v - 30, mid_y), (x1_v + 5, mid_y), WHITE, 1, cv2.LINE_AA, 0, 0.3
    )
    if is_double_sides:
        cv2.arrowedLine(
            ov_arr,
            (x1_outer - 30, mid_y),
            (x1_outer + 5, mid_y),
            WHITE,
            1,
            cv2.LINE_AA,
            0,
            0.3,
        )

    if x2_v is not None:
        # LINE 2 arrow — dari KANAN ke KIRI (menuju tengah/center)
        # Lipat kanan = kain dari sisi kanan dilipat ke arah tengah → panah ke kiri
        cv2.arrowedLine(
            ov_arr, (x2_v + 30, mid_y), (x2_v - 5, mid_y), WHITE, 1, cv2.LINE_AA, 0, 0.3
        )
    if is_double_sides and x2_outer is not None:
        cv2.arrowedLine(
            ov_arr,
            (x2_outer + 30, mid_y),
            (x2_outer - 5, mid_y),
            WHITE,
            1,
            cv2.LINE_AA,
            0,
            0.3,
        )

    # LINE 3 arrow — pointing up
    cv2.arrowedLine(
        ov_arr, (cx_img, y3 + 30), (cx_img, y3 + 5), WHITE, 1, cv2.LINE_AA, 0, 0.3
    )
    if is_double_bottom:
        cv2.arrowedLine(
            ov_arr,
            (cx_img, y3_outer + 30),
            (cx_img, y3_outer + 5),
            WHITE,
            1,
            cv2.LINE_AA,
            0,
            0.3,
        )

    cv2.addWeighted(ov_arr, 0.70, result, 0.30, 0, result)

    # ── Labels (pill style, alpha 0.55 background) ──────────────────────────
    label_y = int(h * 0.35)
    label_y_outer = int(h * 0.25)

    if is_double_sides:
        _pill_label(result, "Lipat kiri 1", x1_outer, label_y_outer)
        if x2_outer is not None:
            _pill_label(result, "Lipat kanan 1", x2_outer, label_y_outer)
    if is_double_bottom:
        _pill_label(result, "Lipat bawah 1", cx_img, y3_outer - 10)

    _pill_label(result, label_left, x1_v, label_y)
    if x2_v is not None:
        _pill_label(result, label_right, x2_v, label_y)
    _pill_label(result, label_bottom, cx_img, y3 - 10)

    return result, fold_lines


# ── JPEG quality for all server-side image outputs (visualisations, not originals)
_JPEG_QUALITY = [cv2.IMWRITE_JPEG_QUALITY, 85] if _CV2_AVAILABLE else []

# ── Size label → BGR colour map for annotation overlays
COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "S": (255, 200, 0),
    "M": (50, 205, 50),
    "L": (0, 165, 255),
    "XL": (0, 80, 255),
    "XXL": (0, 0, 220),
}


def _decode_and_canny(
    raw_bytes: bytes,
) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """
    Decode raw image bytes once and compute Canny edges once.

    Both outputs are shared between run_fold and run_size_estimation so the
    decode + Canny pass only happens a single time per request.

    Returns
    -------
    (img, edges) on success, (None, None) on any failure.
    """
    try:
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Fix 1: CLAHE pada channel L (LAB) untuk normalisasi pencahayaan.
        # Tanpa ini, baju putih di background terang membuat otsu_thresh
        # sangat rendah sehingga Canny hanya mendeteksi noise.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        gray_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        gray_eq = cv2.cvtColor(gray_eq, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

        # Adaptive Canny: derive thresholds from Otsu so the detector works
        # across all contrast levels (high-contrast shirts, light-on-light,
        # dark-on-dark, etc.) instead of relying on hardcoded 50/150.
        otsu_thresh, _ = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        lo = max(10.0, otsu_thresh * 0.33)
        hi = max(20.0, otsu_thresh * 0.66)
        edges = cv2.Canny(blur, lo, hi)

        # Fallback: morphological gradient when Canny finds almost nothing
        # (e.g. near-uniform images or very low-contrast clothing).
        if np.count_nonzero(edges) < 50:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
            _, edges = cv2.threshold(
                gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return img, edges
    except Exception as exc:
        logger.warning("[CV2] decode/Canny failed: %s", exc)
        return None, None


def run_size_estimation(img: np.ndarray, edges: np.ndarray, label: str) -> dict:
    """
    Estimate clothing size from a decoded frame and (optionally) pre-computed Canny edges.

    Pipeline split:
    - Perspective mode (has_perspective=True in config): frame is warped first,
      then Canny runs on the warped frame. Pre-computed edges are NOT used.
      Advantage: geometrically correct measurements regardless of camera tilt.
    - Fallback mode: pre-computed edges from _decode_and_canny() are reused
      with ROI masking. Efficient — no redundant Canny computation.
    """
    if not _state.size_enabled:
        return {"available": False, "reason": "not_configured"}
    if not _SIZE_MODULE_AVAILABLE:
        return {"available": False, "reason": "module_unavailable"}
    if not is_size_enabled(label):
        return {"available": False, "reason": "label_not_supported"}

    # ── Perspective mode: delegate entirely to size_estimator module ──────────
    has_perspective = bool(
        _state.size_config
        and _state.size_config.get("has_perspective")
        and _state.size_config.get("src_points")
    )
    if has_perspective:
        try:
            result = _size_module_run(img, _state.size_config)
            if result["bbox"] is None:
                return {"available": False, "reason": "no_contours_detected"}

            size = result["size"]
            lebar_cm = result["lebar_cm"]
            panjang_cm = result.get("panjang_cm", 0.0)
            bbox = result["bbox"]
            ratio = round(lebar_cm / FOLDER_WIDTH_CM, 3)
            cw = bbox[3]  # height in warped = chest measurement dimension

            # === Display on ORIGINAL frame (no crop/warp) ===
            # Gunakan measured_bbox (dimensi terukur: chest_px x length_px)
            # bukan raw detected bbox agar yang ditampilkan sesuai hasil ukur.
            # measured_bbox dicentrasi di atas detected shirt di warped space.
            display_bbox = result.get("measured_bbox") or bbox

            annotated = img.copy()
            try:
                M_disp, _ = _get_warp_matrix(_state.size_config)
                color = COLOR_MAP.get(size, (255, 255, 255))
                if M_disp is not None:
                    M_inv = np.linalg.inv(M_disp)
                    bx, by, bw_b, bh_b = display_bbox
                    corners_w = np.array(
                        [
                            [bx, by],
                            [bx + bw_b, by],
                            [bx + bw_b, by + bh_b],
                            [bx, by + bh_b],
                        ],
                        dtype=np.float32,
                    ).reshape(-1, 1, 2)
                    corners_o = cv2.perspectiveTransform(corners_w, M_inv)
                    corners_o = corners_o.reshape(-1, 2).astype(np.float64)

                    # === Expand bbox ke arah dada (horizontal di frame asli) ===
                    # Papan pelipat (src_points) lebih sempit dari lebar baju
                    # yang terlihat di kamera, sehingga bbox yang diproyeksikan
                    # hanya mencakup area board. Expansion menambah padding
                    # horizontal agar bbox menutupi tubuh baju secara menyeluruh.
                    expand_pct = float(_state.size_config.get("bbox_chest_expand", 0.4))
                    x_min = corners_o[:, 0].min()
                    x_max = corners_o[:, 0].max()
                    y_min = corners_o[:, 1].min()
                    y_max = corners_o[:, 1].max()
                    w_rect = x_max - x_min
                    pad = w_rect * expand_pct / 2
                    x1 = max(0, int(x_min - pad))
                    x2 = min(img.shape[1] - 1, int(x_max + pad))
                    y1 = max(0, int(y_min))
                    y2 = min(img.shape[0] - 1, int(y_max))

                    # Gambar rectangle di frame asli (bukan polylines)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

                    # Label di atas bbox
                    lx = x1
                    ly = max(15, y1 - 10)
                else:
                    lx, ly = 12, 30

                # Build label text
                text = f"{label} \u2014 {size}"
                if SIZE_DEBUG_MODE:
                    text += f"  (d:{lebar_cm:.1f}cm | p:{panjang_cm:.1f}cm)"
                else:
                    text += f"  ({lebar_cm:.1f}cm)"
                cv2.putText(
                    annotated,
                    text,
                    (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            except Exception as disp_exc:
                logger.debug("[SIZE/DISP] Display annotation error: %s", disp_exc)

            _, buf = cv2.imencode(".jpg", annotated, _JPEG_QUALITY)
            size_b64 = base64.b64encode(buf).decode("utf-8")

            logger.debug(
                "[SIZE/WARP] size=%s  chest=%.1f  length=%.1f  bbox=%s",
                size,
                lebar_cm,
                panjang_cm,
                bbox,
            )
            return {
                "available": True,
                "size": size,
                "lebar_cm": lebar_cm,
                "panjang_cm": panjang_cm,
                "ratio": ratio,
                "clothing_width_px": cw,
                "scale_cm_per_px": round(
                    float(_state.size_config.get("scale_cm_per_px", 0.1)), 4
                ),
                "perspective": True,
                "image": size_b64,
            }
        except Exception as exc:
            logger.warning("[SIZE/WARP] Estimation error for '%s': %s", label, exc)
            return {"available": False, "reason": "error"}

    try:
        # ── Step 1: resolve scale and folder bounds at inference resolution ───
        inf_w = img.shape[1]
        if _state.calib_image_width and _state.calib_image_width > 0:
            res_scale = inf_w / _state.calib_image_width
        else:
            res_scale = 1.0

        # Normalise folder x-coords to inference resolution
        x1_norm = (
            int(_state.folder_x1 * res_scale) if _state.folder_x1 is not None else None
        )
        x2_norm = (
            int(_state.folder_x2 * res_scale) if _state.folder_x2 is not None else None
        )

        # Compute cm/px scale at inference resolution
        if _state.scale_cm_per_px and _state.scale_cm_per_px > 0:
            scale_inf = _state.scale_cm_per_px * (1.0 / res_scale)
        elif _state.folder_width_px and _state.folder_width_px > 0:
            folder_w_norm = _state.folder_width_px * res_scale
            scale_inf = FOLDER_WIDTH_CM / folder_w_norm
        else:
            return {"available": False, "reason": "not_configured"}

        # ── Step 2: ROI mask — restrict Canny contour search to folder area only ──
        # Critical: without this, Canny picks up the folding board frame, background,
        # or image border as the "largest" contour instead of the clothing.
        if x1_norm is not None and x2_norm is not None:
            folder_span = x2_norm - x1_norm
            margin = int(
                folder_span * 0.05
            )  # 5% padding so garment edges aren't clipped
            rx1 = max(0, x1_norm - margin)
            rx2 = min(edges.shape[1], x2_norm + margin)
            # Fix 2: batasan vertikal 5%–95% frame agar tepi lantai/background
            # atas-bawah tidak ikut masuk sebagai kandidat kontur.
            ry1 = int(edges.shape[0] * 0.05)
            ry2 = int(edges.shape[0] * 0.95)
            roi_mask = np.zeros_like(edges)
            roi_mask[ry1:ry2, rx1:rx2] = 255
            search_edges = cv2.bitwise_and(edges, roi_mask)
        else:
            search_edges = edges

        # ── Step 3: find contours and select best candidate ──────────────────
        contours, _ = cv2.findContours(
            search_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return {"available": False, "reason": "no_contours_detected"}

        # Fix 3: naikkan threshold ke 5% agar noise dan tepi frame tidak lolos.
        # Threshold 0.5% terlalu kecil — contour kecil bisa jadi "terlebar".
        min_area = img.shape[0] * img.shape[1] * 0.05
        valid = [c for c in contours if cv2.contourArea(c) > min_area]
        if not valid:
            valid = contours  # fallback: keep all if filter is too aggressive

        # ── Anchor + proximity union ──────────────────────────────────────
        # 1. Anchor = kontur terbesar by area (= body utama baju)
        # 2. Sertakan kontur lain yang secara horizontal DEKAT anchor (= lengan)
        # 3. Abaikan kontur jauh (= tepi pelipat, background, lantai)
        # Ini menghindari over-estimation (union semua) dan under-estimation
        # (satu kontur saja).
        anchor = max(valid, key=cv2.contourArea)
        ax, ay, aw, ah = cv2.boundingRect(anchor)
        # Proximity: 12% lebar frame ≈ jarak khas body ke ujung lengan
        prox = int(img.shape[1] * 0.12)
        near = [
            c
            for c in valid
            if cv2.boundingRect(c)[0]
            <= ax + aw + prox  # sisi kiri kontur tidak terlalu jauh ke kanan
            and cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2]
            >= ax - prox  # sisi kanan tidak terlalu jauh ke kiri
        ]
        all_rects = [cv2.boundingRect(c) for c in near]
        cx = min(r[0] for r in all_rects)
        cy = min(r[1] for r in all_rects)
        cw = max(r[0] + r[2] for r in all_rects) - cx
        ch = max(r[1] + r[3] for r in all_rects) - cy

        logger.debug(
            "[SIZE] contours_total=%d  valid=%d  near=%d  bbox=(x=%d,y=%d,w=%d,h=%d)",
            len(contours),
            len(valid),
            len(near),
            cx,
            cy,
            cw,
            ch,
        )

        # ── Step 4: estimate size ───────────────────────────────────
        size, lebar_cm, ratio = estimate_size(float(cw), scale_inf)

        # ── Step 5: render annotated image ────────────────────────────
        annotated = annotate_size(
            img,
            label,
            size,
            ratio,
            lebar_cm=lebar_cm,
            contour_bbox=(cx, cy, cw, ch),
            folder_x1=x1_norm,
            folder_x2=x2_norm,
            debug=SIZE_DEBUG_MODE,
        )
        _, buf = cv2.imencode(".jpg", annotated, _JPEG_QUALITY)
        size_b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "available": True,
            "size": size,
            "lebar_cm": lebar_cm,
            "ratio": ratio,
            "clothing_width_px": cw,
            "scale_cm_per_px": round(scale_inf, 4),
            "image": size_b64,
        }

    except Exception as exc:
        logger.warning("[SIZE] Estimation error for '%s': %s", label, exc)
        return {"available": False, "reason": "error"}


def run_fold(
    img: np.ndarray, edges: np.ndarray, label: str, size: str | None = None
) -> dict:
    """
    Canny-based fold planner — draws professional fold lines on the garment image.

    Accepts pre-decoded frame and pre-computed Canny edges from _decode_and_canny()
    so this step shares the decode + Canny work with run_size_estimation.

    Parameters
    ----------
    size : Estimated clothing size ("S"|"M"|"L"|"XL"|"XXL"|None).
           When L/XL/XXL, extra outer fold lines are drawn on each side.
    """
    if not _state.fold_enabled:
        return {"available": False}
    if label in _FOLD_SKIP_LABELS:
        return {"available": False}

    try:
        result_img, fold_lines = draw_fold_lines(img, label, size=size)

        _, buf = cv2.imencode(".jpg", result_img, _JPEG_QUALITY)
        fold_b64 = base64.b64encode(buf).decode("utf-8")
        _, cbuf = cv2.imencode(".jpg", edges, _JPEG_QUALITY)
        canny_b64 = base64.b64encode(cbuf).decode("utf-8")

        return {
            "available": True,
            "image": fold_b64,
            "canny_image": canny_b64,
            "fold_lines": fold_lines,
            "angle": 0.0,
        }

    except Exception as exc:
        logger.warning("[FOLD] Pipeline error for '%s': %s", label, exc)
        return {"available": False}


# ----------------------------------------------------------------
# FastAPI lifespan — load all resources once at startup
# ----------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. ONNX model
    model_path = Path(MODEL_PATH)
    labels_path = Path(LABELS_PATH)

    if not model_path.exists():
        raise RuntimeError(f"[STARTUP] Model not found: {model_path.resolve()}")
    if not labels_path.exists():
        raise RuntimeError(f"[STARTUP] Labels not found: {labels_path.resolve()}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _cpu = multiprocessing.cpu_count() or 2
    _threads = INTRA_OP_NUM_THREADS if INTRA_OP_NUM_THREADS > 0 else min(_cpu, 4)
    opts.intra_op_num_threads = _threads
    opts.inter_op_num_threads = 1

    _state.session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    _state.input_name = _state.session.get_inputs()[0].name
    _state.labels = labels_path.read_text(encoding="utf-8").strip().splitlines()
    _state.start_time = time.time()
    logger.info(
        "[STARTUP] YOLO model loaded | classes=%s | threshold=%.2f | threads=%d/%d",
        _state.labels,
        _state.conf_threshold,
        _threads,
        _cpu,
    )

    # 2. Fold pipeline (native OpenCV — no pickle)
    if FOLD_ENABLED and not _CV2_AVAILABLE:
        logger.warning(
            "[STARTUP] opencv-python not installed — fold pipeline disabled."
        )
    elif FOLD_ENABLED:
        _state.fold_enabled = True
        logger.info("[STARTUP] Native Canny fold pipeline enabled.")
    else:
        logger.info("[STARTUP] Fold pipeline disabled (FOLD_ENABLED=false).")

    # 3. MQTT
    if MQTT_ENABLED and _MQTT_MODULE_AVAILABLE:
        _state.mqtt = ClothbotMQTT(
            host=MQTT_HOST,
            port=MQTT_PORT,
            username=MQTT_USER,
            password=MQTT_PASSWORD,
            topic=MQTT_TOPIC,
        )
        _state.mqtt.connect()
        logger.info("[STARTUP] MQTT initialised | broker=%s:%d", MQTT_HOST, MQTT_PORT)
    elif MQTT_ENABLED and not _MQTT_MODULE_AVAILABLE:
        logger.warning("[STARTUP] mqtt_publisher.py not found — MQTT disabled.")
    else:
        logger.info("[STARTUP] MQTT disabled (MQTT_ENABLED=false).")

    # 4. Size estimation
    if SIZE_ESTIMATION_ENABLED and not _SIZE_MODULE_AVAILABLE:
        logger.warning(
            "[STARTUP] size_estimator/visualization/calibration modules unavailable — size estimation disabled."
        )
    elif SIZE_ESTIMATION_ENABLED and not _CV2_AVAILABLE:
        logger.warning(
            "[STARTUP] opencv-python not installed — size estimation disabled."
        )
    elif SIZE_ESTIMATION_ENABLED:
        cfg = load_folder_config(Path(CONFIG_PATH))
        if cfg:
            _state.folder_width_px = cfg["folder_width_px"]
            _state.folder_x1 = cfg["folder_x1"]
            _state.folder_x2 = cfg["folder_x2"]
            _state.calib_image_width = cfg.get("calib_image_width")
            _state.scale_cm_per_px = cfg.get("scale_cm_per_px")
            _state.size_enabled = True
            # Also load raw config dict for perspective warp mode
            try:
                _state.size_config = _size_load_config(CONFIG_PATH)
            except Exception:
                _state.size_config = None
            has_persp = bool(
                _state.size_config and _state.size_config.get("has_perspective")
            )
            logger.info(
                "[STARTUP] Size estimation enabled | folder_width=%.1f px  scale=%.4f cm/px"
                "  calib_res=%s  perspective=%s  debug=%s",
                _state.folder_width_px or 0,
                _state.scale_cm_per_px or 0.0,
                f"{_state.calib_image_width}px"
                if _state.calib_image_width
                else "unknown (re-calibrate)",
                has_persp,
                SIZE_DEBUG_MODE,
            )
        else:
            logger.warning(
                "[STARTUP] Size estimation: config.json not found or missing 'folder_width_px'. "
                "POST /calibrate or run calibration.py to configure."
            )
    else:
        logger.info(
            "[STARTUP] Size estimation disabled (SIZE_ESTIMATION_ENABLED=false)."
        )

    yield

    logger.info("[SHUTDOWN] Releasing resources.")
    _state.session = None
    if _state.mqtt:
        _state.mqtt.disconnect()
        _state.mqtt = None


# ----------------------------------------------------------------
# Application instance
# ----------------------------------------------------------------

app = FastAPI(
    title="ClothBot Inference API",
    description="YOLOv8l-cls ONNX + Folding Planner v3.2 (Canny)",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next) -> Response:
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "[REQUEST] %s %s → %d  (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ----------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------


@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "model": "yolov8l-cls-onnx",
        "fold_enabled": _state.fold_enabled,
        "classes": _state.labels,
        "version": "2.0.0",
        "uptime_sec": round(time.time() - _state.start_time, 2),
    }


@app.get("/health")
def health() -> dict:
    snap = _state.stats_snapshot()
    return {
        "status": "healthy",
        "model_loaded": _state.session is not None,
        "fold_enabled": _state.fold_enabled,
        "inference_count": _state.inference_count,
        "avg_latency_ms": snap["avg_latency_ms"],
        "mqtt": _state.mqtt.status() if _state.mqtt else {"enabled": False},
        "size_enabled": _state.size_enabled,
        "folder_width_px": _state.folder_width_px,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    """
    Classify a clothing image and optionally run the fold planner.

    Accepts multipart/form-data with field 'image' (JPEG/PNG/WEBP, max 5 MB).
    Returns YOLO classification result and, when confident, the Canny fold
    visualization as a base64-encoded PNG.
    """
    content_type = image.content_type or ""
    if not content_type.startswith("image/"):
        _state.record(0.0, success=False)
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Send JPEG, PNG, or WEBP.",
        )

    raw = await image.read()

    if not raw:
        _state.record(0.0, success=False)
        raise HTTPException(status_code=400, detail="Empty image file received.")

    if len(raw) > MAX_FILE_BYTES:
        _state.record(0.0, success=False)
        raise HTTPException(
            status_code=413,
            detail=f"Image {len(raw) / 1024:.1f} KB exceeds 5 MB limit.",
        )

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        original_size = list(img.size)
    except Exception as exc:
        _state.record(0.0, success=False)
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    # ---- YOLO inference ----
    t0 = time.perf_counter()
    try:
        tensor = preprocess(img)
        raw_out = run_inference(tensor)
        result = postprocess(raw_out)
    except Exception as exc:
        _state.record(0.0, success=False)
        logger.error("[PREDICT] YOLO error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    yolo_ms = round((time.perf_counter() - t0) * 1000, 3)
    _state.record(yolo_ms, success=True)

    label = result["predicted_class"]

    # ---- Shared CV2 decode + Canny (once, reused by fold & size) ----
    cv2_img, cv2_edges = None, None
    if result["is_confident"] and label != "null" and _CV2_AVAILABLE:
        cv2_img, cv2_edges = _decode_and_canny(raw)

    # ---- Size estimation (runs first so size is available for fold lines) ----
    size_data = {"available": False}
    size_ms = None
    if cv2_img is not None:
        t2 = time.perf_counter()
        size_data = run_size_estimation(cv2_img, cv2_edges, label)
        size_ms = round((time.perf_counter() - t2) * 1000, 3)

    # ---- Fold pipeline (uses size to draw double fold lines for L/XL/XXL) ----
    fold_data = {"available": False}
    fold_ms = None
    if cv2_img is not None:
        t1 = time.perf_counter()
        estimated_size = size_data.get("size") if size_data.get("available") else None
        fold_data = run_fold(cv2_img, cv2_edges, label, size=estimated_size)
        fold_ms = round((time.perf_counter() - t1) * 1000, 3)

    # ---- MQTT publish ----
    mqtt_published = False
    if (
        _state.mqtt
        and result["is_confident"]
        and label != "null"
        and label in SERVO_MAP
    ):
        mqtt_published = _state.mqtt.publish_command(
            label=label,
            steps=SERVO_MAP[label]["steps"],
        )
        if not mqtt_published:
            logger.warning("[PREDICT] MQTT publish failed for label '%s'", label)

    return {
        **result,
        "inference_time_ms": yolo_ms,
        "fold_time_ms": fold_ms,
        "image_received_size": original_size,
        "mqtt_published": mqtt_published,
        "fold": fold_data,
        "size": size_data,
        "size_time_ms": size_ms,
    }


@app.get("/metrics")
def metrics() -> dict:
    snap = _state.stats_snapshot()
    return {
        "uptime_sec": round(time.time() - _state.start_time, 2),
        "total_requests": _state.total_requests,
        "success_requests": _state.success_requests,
        "error_requests": _state.error_requests,
        "avg_latency_ms": snap["avg_latency_ms"],
        "min_latency_ms": snap["min_latency_ms"],
        "max_latency_ms": snap["max_latency_ms"],
        "model_path": MODEL_PATH,
        "classes": _state.labels,
        "threshold": round(_state.conf_threshold, 5),
        "fold_enabled": _state.fold_enabled,
    }


@app.post("/calibrate")
async def calibrate(image: UploadFile = File(...)) -> dict:
    """
    Calibrate the folding board (pelipat pakaian) reference width.

    Upload an image of the **empty** folding board in frame.
    Detects vertical edges using Canny + HoughLinesP, saves result to
    config.json, and hot-reloads the running size estimation state.

    On success returns detected folder_x1, folder_x2, and folder_width_px.
    """
    if not _CV2_AVAILABLE:
        raise HTTPException(status_code=503, detail="opencv-python not available.")
    if not _SIZE_MODULE_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Size estimation modules not available."
        )

    content_type = image.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Send JPEG, PNG, or WEBP.",
        )

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image file received.")
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image {len(raw) / 1024:.1f} KB exceeds 5 MB limit.",
        )

    try:
        nparr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=422, detail="Cannot decode image.")

        result = detect_folder_edges(frame)
        if result is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Folder edges not detected. "
                    "Ensure the folding board is clearly visible with good contrast, "
                    "or use calibration.py --width <pixels> for manual override."
                ),
            )

        folder_x1, folder_x2 = result
        folder_width = folder_x2 - folder_x1
        calib_image_width = frame.shape[1]

        cfg_path = Path(CONFIG_PATH)
        config_data = {
            "folder_width_px": folder_width,
            "folder_x1": folder_x1,
            "folder_x2": folder_x2,
            "calib_image_width": calib_image_width,
        }
        if cfg_path.exists():
            try:
                old = json.loads(cfg_path.read_text(encoding="utf-8"))
                if "size_thresholds" in old:
                    config_data["size_thresholds"] = old["size_thresholds"]
            except (OSError, json.JSONDecodeError):
                pass

        config_data["folder_width_cm"] = FOLDER_WIDTH_CM
        config_data["scale_cm_per_px"] = round(FOLDER_WIDTH_CM / folder_width, 4)
        cfg_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        # Hot-reload running state
        _state.folder_width_px = float(folder_width)
        _state.folder_x1 = folder_x1
        _state.folder_x2 = folder_x2
        _state.calib_image_width = calib_image_width
        _state.size_enabled = True
        _state.scale_cm_per_px = (
            FOLDER_WIDTH_CM / folder_width if folder_width > 0 else None
        )
        # Hot-reload raw config dict (perspective fields preserved if present)
        try:
            _state.size_config = _size_load_config(CONFIG_PATH)
        except Exception:
            _state.size_config = config_data

        logger.info(
            "[CALIBRATE] Config saved | w=%d px  x1=%d  x2=%d  calib_res=%d px",
            folder_width,
            folder_x1,
            folder_x2,
            calib_image_width,
        )

        return {
            "status": "ok",
            "folder_width_px": folder_width,
            "folder_x1": folder_x1,
            "folder_x2": folder_x2,
            "calib_image_width": calib_image_width,
            "scale_cm_per_px": round(FOLDER_WIDTH_CM / folder_width, 4),
            "config_path": str(cfg_path.resolve()),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[CALIBRATE] Error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Calibration error: {exc}")


@app.post("/calibrate-perspective")
async def calibrate_perspective_web(
    image: UploadFile = File(...),
    points: str = Form(...),
    preview_only: bool = Form(False),
) -> dict:
    """
    Kalibrasi perspektif 4 titik via web browser.

    Menerima foto pelipat kosong dan 4 koordinat sudut yang diklik user di browser.
    Server menghitung perspective transform matrix, menyimpan ke config.json,
    dan mengembalikan preview gambar hasil warp.

    Body (multipart/form-data):
      image        : foto pelipat kosong (JPEG/PNG/WEBP, maks 5 MB)
      points       : JSON string [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                     urutan: TL → TR → BR → BL
                     koordinat ternormalisasi 0.0–1.0 relatif terhadap dimensi
                     gambar yang ditampilkan di browser
      preview_only : jika True, hanya kembalikan preview tanpa menyimpan config
                     (default: False — simpan dan preview)

    Returns:
      status        : "ok"
      preview_image : base64 JPEG gambar hasil warp (900×560 px)
      src_points    : [[x,y],...] koordinat dalam piksel asli gambar
      dst_size      : [900, 560]
      scale_cm_per_px : 0.1
      saved         : bool — apakah config.json diupdate
    """
    if not _CV2_AVAILABLE:
        raise HTTPException(status_code=503, detail="opencv-python tidak tersedia.")
    if not _SIZE_MODULE_AVAILABLE:
        raise HTTPException(
            status_code=503, detail="Modul size estimation tidak tersedia."
        )

    # ── Validate image ────────────────────────────────────────────────────────
    content_type = image.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Format tidak didukung: '{content_type}'. Kirim JPEG, PNG, atau WEBP.",
        )
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File gambar kosong.")
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Ukuran gambar {len(raw) / 1024:.1f} KB melebihi batas 5 MB.",
        )

    # ── Decode image ──────────────────────────────────────────────────────────
    try:
        nparr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Tidak dapat membaca gambar: {exc}"
        )

    img_h, img_w = frame.shape[:2]

    # ── Parse and validate points ─────────────────────────────────────────────
    try:
        pts_raw = json.loads(points)
        if not isinstance(pts_raw, list) or len(pts_raw) != 4:
            raise ValueError("Harus tepat 4 titik")
        # Validate each point
        for i, p in enumerate(pts_raw):
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError(f"Titik {i + 1} harus berupa [x, y]")
            if not (0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0):
                raise ValueError(
                    f"Titik {i + 1} koordinat harus antara 0.0 dan 1.0, dapat: {p}"
                )
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Format points tidak valid: {exc}. "
            "Kirim JSON array 4 titik ternormalisasi: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]",
        )

    # ── Scale normalized points → pixel coordinates ───────────────────────────
    src_pts_px = [
        [round(float(p[0]) * img_w), round(float(p[1]) * img_h)] for p in pts_raw
    ]

    # ── Compute perspective transform ─────────────────────────────────────────
    DST_W, DST_H = 900, 560  # 900px=90cm, 560px=56cm → scale = 0.1 cm/px
    dst_pts = np.float32(
        [
            [0, 0],
            [DST_W - 1, 0],
            [DST_W - 1, DST_H - 1],
            [0, DST_H - 1],
        ]
    )
    try:
        src_np = np.float32(src_pts_px)
        M = cv2.getPerspectiveTransform(src_np, dst_pts)
        warped = cv2.warpPerspective(frame, M, (DST_W, DST_H))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Gagal menghitung perspective transform: {exc}. "
            "Pastikan 4 titik membentuk quadrilateral yang valid.",
        )

    # ── Encode warped preview ─────────────────────────────────────────────────
    _, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 88])
    preview_b64 = base64.b64encode(buf).decode("utf-8")

    scale_cm_per_px = round(FOLDER_WIDTH_CM / DST_W, 4)  # 90 / 900 = 0.1

    if not preview_only:
        # ── Extend config.json (preserve old fields) ──────────────────────────
        cfg_path = Path(CONFIG_PATH)
        existing: dict = {}
        if cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

        new_fields: dict = {
            "folder_width_cm": FOLDER_WIDTH_CM,
            "folder_height_cm": 56.0,
            "scale_cm_per_px": scale_cm_per_px,
            "src_points": src_pts_px,
            "dst_size": [DST_W, DST_H],
            "has_perspective": True,
        }
        merged = {**existing, **new_fields}
        cfg_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")

        # ── Hot-reload state ──────────────────────────────────────────────────
        _state.scale_cm_per_px = scale_cm_per_px
        _state.size_enabled = True
        try:
            _state.size_config = _size_load_config(CONFIG_PATH)
        except Exception:
            _state.size_config = merged

        logger.info(
            "[CALIB-PERSP] Saved | src=%s  scale=%.4f  img=%dx%d",
            src_pts_px,
            scale_cm_per_px,
            img_w,
            img_h,
        )

    return {
        "status": "ok",
        "preview_image": preview_b64,
        "src_points": src_pts_px,
        "dst_size": [DST_W, DST_H],
        "scale_cm_per_px": scale_cm_per_px,
        "image_size": [img_w, img_h],
        "saved": not preview_only,
    }


@app.get("/ui")
def ui() -> FileResponse:
    """Web UI — served from static/index.html."""
    return FileResponse("static/index.html")


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=1, log_level="info")

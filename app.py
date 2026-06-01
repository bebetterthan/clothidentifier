"""
app.py — ClothBot Inference API v2.0
Pipeline: Image → YOLOv8l-cls (ONNX) → Folding Planner v3.2 (Canny)
"""

import io
import os
import base64
import pickle
import tempfile
import threading
import time
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

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

# ----------------------------------------------------------------
# Thread limits — must be set before onnxruntime is imported
# ----------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("ONNXRUNTIME_CPU_NUM_THREADS", "2")

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
MODEL_PATH      = os.getenv("MODEL_PATH",      "./best.onnx")
LABELS_PATH     = os.getenv("LABELS_PATH",     "./labels.txt")
CONF_THRESHOLD  = float(os.getenv("CONF_THRESHOLD", "0.70"))
FOLD_MODEL_PATH = os.getenv("FOLD_MODEL_PATH", "./model/fold_pipeline.pkl")
FOLD_ENABLED    = os.getenv("FOLD_ENABLED",    "true").lower() == "true"
MAX_FILE_BYTES  = 5 * 1024 * 1024   # 5 MB
INPUT_SIZE      = 224

MQTT_ENABLED  = os.getenv("MQTT_ENABLED",  "true").lower() == "true"
MQTT_HOST     = os.getenv("MQTT_HOST",     "localhost")
MQTT_PORT     = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER     = os.getenv("MQTT_USER",     "clothbot")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_TOPIC    = os.getenv("MQTT_TOPIC",    "clothbot/servo/command")


# ----------------------------------------------------------------
# Application state
# ----------------------------------------------------------------

@dataclass
class AppState:
    session:        Optional[ort.InferenceSession] = None
    input_name:     str                            = ""
    labels:         list                           = field(default_factory=list)
    conf_threshold: float                          = CONF_THRESHOLD
    start_time:     float                          = field(default_factory=time.time)
    fold_pipeline:  Optional[dict]                 = None
    fold_enabled:   bool                           = False
    mqtt:           Optional[object]               = None   # ClothbotMQTT

    total_requests:   int   = 0
    success_requests: int   = 0
    error_requests:   int   = 0
    inference_count:  int   = 0
    _latencies:       list  = field(default_factory=list)
    _lock:            threading.Lock = field(default_factory=threading.Lock)

    def record(self, latency_ms: float, success: bool) -> None:
        with self._lock:
            self.total_requests += 1
            if success:
                self.success_requests += 1
                self.inference_count  += 1
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
                "min_latency_ms": round(float(np.min(lats)),  3),
                "max_latency_ms": round(float(np.max(lats)),  3),
            }
        return {"avg_latency_ms": 0.0, "min_latency_ms": 0.0, "max_latency_ms": 0.0}


_state = AppState()


# ----------------------------------------------------------------
# YOLO inference pipeline
# ----------------------------------------------------------------

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0).astype(np.float32)


def run_inference(tensor: np.ndarray) -> np.ndarray:
    return _state.session.run(None, {_state.input_name: tensor})[0]


def postprocess(raw_output: np.ndarray) -> dict:
    probs = raw_output[0]
    idx   = int(np.argmax(probs))
    conf  = round(float(probs[idx]), 5)
    return {
        "predicted_class": _state.labels[idx],
        "confidence":      conf,
        "is_confident":    conf >= _state.conf_threshold,
        "threshold":       round(_state.conf_threshold, 5),
        "probabilities":   {
            _state.labels[i]: round(float(p), 5)
            for i, p in enumerate(probs)
        },
    }


# ----------------------------------------------------------------
# Folding pipeline (Canny + geometry)
# ----------------------------------------------------------------

_FOLD_SKIP_LABELS = {"null"}


def _garment_bbox(edges, h, w):
    """Return (x_min, x_max, y_min, y_max, cx, cy) from Canny edges."""
    import numpy as np
    ys, xs = np.where(edges > 0)
    if len(ys) < 20:
        return int(w*.1), int(w*.9), int(h*.1), int(h*.9), w//2, h//2
    x_min = max(0,   int(np.percentile(xs,  2)) - 4)
    x_max = min(w-1, int(np.percentile(xs, 98)) + 4)
    y_min = max(0,   int(np.percentile(ys,  2)) - 4)
    y_max = min(h-1, int(np.percentile(ys, 98)) + 4)
    return x_min, x_max, y_min, y_max, (x_min+x_max)//2, (y_min+y_max)//2


def _dashed_line(img, pt1, pt2, color, thickness, dash=18, gap=10):
    """Draw a dashed line between pt1 and pt2."""
    import numpy as np
    dx, dy = pt2[0]-pt1[0], pt2[1]-pt1[1]
    length = max(1, int(np.hypot(dx, dy)))
    step   = dash + gap
    for i in range(0, length, step):
        s = i / length
        e = min(1.0, (i + dash) / length)
        p1 = (int(pt1[0] + dx*s), int(pt1[1] + dy*s))
        p2 = (int(pt1[0] + dx*e), int(pt1[1] + dy*e))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _fold_arrow(img, pt, direction, color, size):
    """Draw a small filled triangle arrow. direction: 'up','down','left','right'."""
    x, y = pt
    if direction == "up":
        pts = [(x, y-size), (x-size, y+size//2), (x+size, y+size//2)]
    elif direction == "down":
        pts = [(x, y+size), (x-size, y-size//2), (x+size, y-size//2)]
    elif direction == "left":
        pts = [(x-size, y), (x+size//2, y-size), (x+size//2, y+size)]
    else:  # right
        pts = [(x+size, y), (x-size//2, y-size), (x-size//2, y+size)]
    import numpy as np
    cv2.fillPoly(img, [np.array(pts, np.int32)], color)


def _fold_label(img, text, pt, color, scale):
    """Draw text with dark background for readability."""
    font  = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(1, int(scale * 1.5))
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = pt
    # background rect
    cv2.rectangle(img, (x-3, y-th-3), (x+tw+3, y+bl+1),
                  (20, 20, 20), -1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _fold_guides(edges, h, w, label):
    """
    Step-by-step fold guides based on retail/KonMari methodology.
    Positions relative to garment bounding box from Canny edges.

    Rasio berdasarkan panduan:
      baju_lengan_pendek : vertikal 30%/70%, horizontal 60%, 30%
      baju_lengan_panjang: vertikal 35%/65%, diagonal lengan, horizontal 57%, 30%
      celana_panjang     : vertikal 50%, horizontal 60%, 35%
      celana_pendek      : vertikal 50%, horizontal 50%
    """
    import numpy as np
    x0, x1, y0, y1, cx, cy = _garment_bbox(edges, h, w)
    gw  = x1 - x0
    gh  = y1 - y0
    th  = max(2, int(min(h, w) * 0.005))
    asz = max(8, int(min(h, w) * 0.025))

    ORANGE = (255, 100,   0)
    CYAN   = (  0, 210, 255)
    GREEN  = ( 80, 230,  80)
    YELLOW = (255, 215,   0)

    guides = []

    if label == "baju_lengan_pendek":
        # Vertikal: sisi kiri x=30%, kanan x=70%
        lx  = x0 + int(gw * 0.30)
        rx  = x0 + int(gw * 0.70)
        # Horizontal: lipat bawah y=60%, lipat lagi y=30%
        yb1 = y0 + int(gh * 0.60)
        yb2 = y0 + int(gh * 0.30)

        guides = [
            {"pt1": (lx, y0), "pt2": (lx, y1), "color": ORANGE, "th": th + 1,
             "label": "1 Lipat Kiri",
             "arrows": [((lx + (cx - lx) // 2, cy), "right")]},

            {"pt1": (rx, y0), "pt2": (rx, y1), "color": ORANGE, "th": th + 1,
             "label": "2 Lipat Kanan",
             "arrows": [((rx - (rx - cx) // 2, cy), "left")]},

            {"pt1": (x0, yb1), "pt2": (x1, yb1), "color": CYAN, "th": th,
             "label": "3 Lipat Bawah",
             "arrows": [((cx, yb1 + int(gh * 0.10)), "up")]},

            {"pt1": (x0, yb2), "pt2": (x1, yb2), "color": GREEN, "th": max(1, th - 1),
             "label": "4 Lipat Lagi",
             "arrows": [((cx, yb2 + int(gh * 0.08)), "up")]},
        ]

    elif label == "baju_lengan_panjang":
        # Vertikal lebih sempit karena lengan dilipat ke dalam: 35%/65%
        lx  = x0 + int(gw * 0.35)
        rx  = x0 + int(gw * 0.65)
        # Diagonal lengan: (5%,15%)→(35%,40%) kiri, mirror kanan
        dl1 = (x0 + int(gw * 0.05), y0 + int(gh * 0.15))
        dl2 = (x0 + int(gw * 0.35), y0 + int(gh * 0.40))
        dr1 = (x0 + int(gw * 0.95), y0 + int(gh * 0.15))
        dr2 = (x0 + int(gw * 0.65), y0 + int(gh * 0.40))
        # Horizontal: y=57%, y=30%
        yb1 = y0 + int(gh * 0.57)
        yb2 = y0 + int(gh * 0.30)

        guides = [
            # Diagonal lengan dulu (step 1 & 2)
            {"pt1": dl1, "pt2": dl2, "color": YELLOW, "th": th,
             "label": "1 Lipat Lengan Kiri",
             "arrows": [((dl2[0] - int(gw * 0.05), dl2[1] - int(gh * 0.05)), "right")]},

            {"pt1": dr1, "pt2": dr2, "color": YELLOW, "th": th,
             "label": "2 Lipat Lengan Kanan",
             "arrows": [((dr2[0] + int(gw * 0.05), dr2[1] - int(gh * 0.05)), "left")]},

            # Vertikal badan (step 3 & 4)
            {"pt1": (lx, y0), "pt2": (lx, y1), "color": ORANGE, "th": th + 1,
             "label": "3 Lipat Sisi Kiri",
             "arrows": [((lx + (cx - lx) // 2, cy), "right")]},

            {"pt1": (rx, y0), "pt2": (rx, y1), "color": ORANGE, "th": th + 1,
             "label": "4 Lipat Sisi Kanan",
             "arrows": [((rx - (rx - cx) // 2, cy), "left")]},

            # Horizontal (step 5 & 6)
            {"pt1": (x0, yb1), "pt2": (x1, yb1), "color": CYAN, "th": th,
             "label": "5 Lipat Bawah",
             "arrows": [((cx, yb1 + int(gh * 0.10)), "up")]},

            {"pt1": (x0, yb2), "pt2": (x1, yb2), "color": GREEN, "th": max(1, th - 1),
             "label": "6 Lipat Lagi",
             "arrows": [((cx, yb2 + int(gh * 0.08)), "up")]},
        ]

    elif label == "celana_panjang":
        # Vertikal tengah x=50%, horizontal y=60% lalu y=35%
        yh1 = y0 + int(gh * 0.60)
        yh2 = y0 + int(gh * 0.35)

        guides = [
            {"pt1": (cx, y0), "pt2": (cx, y1), "color": ORANGE, "th": th + 1,
             "label": "1 Lipat Tengah",
             "arrows": [((cx + gw // 6, cy), "right")]},

            {"pt1": (x0, yh1), "pt2": (x1, yh1), "color": CYAN, "th": th,
             "label": "2 Kaki ke Pinggang",
             "arrows": [((cx, yh1 + int(gh * 0.10)), "up")]},

            {"pt1": (x0, yh2), "pt2": (x1, yh2), "color": GREEN, "th": max(1, th - 1),
             "label": "3 Lipat Lagi",
             "arrows": [((cx, yh2 + int(gh * 0.08)), "up")]},
        ]

    elif label == "celana_pendek":
        # Vertikal x=50%, horizontal y=50%
        yh = y0 + int(gh * 0.50)

        guides = [
            {"pt1": (cx, y0), "pt2": (cx, y1), "color": ORANGE, "th": th + 1,
             "label": "1 Lipat Tengah",
             "arrows": [((cx + gw // 6, cy), "right")]},

            {"pt1": (x0, yh), "pt2": (x1, yh), "color": CYAN, "th": th,
             "label": "2 Lipat Bawah",
             "arrows": [((cx, yh + int(gh * 0.12)), "up")]},
        ]

    for g in guides:
        g["asz"] = asz
    return guides


def run_fold(raw_bytes: bytes, label: str) -> dict:
    """Canny-based fold planner — step-by-step garment folding guide."""
    if not _state.fold_enabled:
        return {"available": False}
    if label in _FOLD_SKIP_LABELS:
        return {"available": False}

    try:
        import numpy as np

        nparr = np.frombuffer(raw_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"available": False}

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        guides     = _fold_guides(edges, h, w, label)
        result_img = img.copy()
        fold_lines: list = []
        lbl_scale  = max(0.35, min(0.55, w / 900))

        for g in guides:
            bgr = (g["color"][2], g["color"][1], g["color"][0])
            _dashed_line(result_img, g["pt1"], g["pt2"], bgr, g["th"])
            for apt, adir in g.get("arrows", []):
                _fold_arrow(result_img, apt, adir, bgr, g["asz"])
            # label near first endpoint
            lx = g["pt1"][0] + 4
            ly = g["pt1"][1] - 6
            _fold_label(result_img, g["label"], (lx, ly), g["color"], lbl_scale)
            fold_lines.append({
                "from":  list(g["pt1"]),
                "to":    list(g["pt2"]),
                "color": list(g["color"]),
                "label": g["label"],
            })

        _, buf    = cv2.imencode(".png", result_img)
        fold_b64  = base64.b64encode(buf).decode("utf-8")
        _, cbuf   = cv2.imencode(".png", edges)
        canny_b64 = base64.b64encode(cbuf).decode("utf-8")

        return {
            "available":   True,
            "image":       fold_b64,
            "canny_image": canny_b64,
            "fold_lines":  fold_lines,
            "angle":       0.0,
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
    model_path  = Path(MODEL_PATH)
    labels_path = Path(LABELS_PATH)

    if not model_path.exists():
        raise RuntimeError(f"[STARTUP] Model not found: {model_path.resolve()}")
    if not labels_path.exists():
        raise RuntimeError(f"[STARTUP] Labels not found: {labels_path.resolve()}")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1

    _state.session    = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    _state.input_name = _state.session.get_inputs()[0].name
    _state.labels     = labels_path.read_text(encoding="utf-8").strip().splitlines()
    _state.start_time = time.time()
    logger.info(
        "[STARTUP] YOLO model loaded | classes=%s | threshold=%.2f",
        _state.labels, _state.conf_threshold,
    )

    # 2. Fold pipeline (native OpenCV — no pickle)
    if FOLD_ENABLED and not _CV2_AVAILABLE:
        logger.warning("[STARTUP] opencv-python not installed — fold pipeline disabled.")
    elif FOLD_ENABLED:
        _state.fold_enabled = True
        logger.info("[STARTUP] Native Canny fold pipeline enabled.")
    else:
        logger.info("[STARTUP] Fold pipeline disabled (FOLD_ENABLED=false).")

    # 3. MQTT
    if MQTT_ENABLED and _MQTT_MODULE_AVAILABLE:
        _state.mqtt = ClothbotMQTT(
            host=MQTT_HOST, port=MQTT_PORT,
            username=MQTT_USER, password=MQTT_PASSWORD,
            topic=MQTT_TOPIC,
        )
        _state.mqtt.connect()
        logger.info("[STARTUP] MQTT initialised | broker=%s:%d", MQTT_HOST, MQTT_PORT)
    elif MQTT_ENABLED and not _MQTT_MODULE_AVAILABLE:
        logger.warning("[STARTUP] mqtt_publisher.py not found — MQTT disabled.")
    else:
        logger.info("[STARTUP] MQTT disabled (MQTT_ENABLED=false).")

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
    title       = "ClothBot Inference API",
    description = "YOLOv8l-cls ONNX + Folding Planner v3.2 (Canny)",
    version     = "2.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)


@app.middleware("http")
async def request_logger(request: Request, call_next) -> Response:
    t0       = time.perf_counter()
    response = await call_next(request)
    elapsed  = (time.perf_counter() - t0) * 1000
    logger.info(
        "[REQUEST] %s %s → %d  (%.1f ms)",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


# ----------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------

@app.get("/")
def root() -> dict:
    return {
        "status":       "ok",
        "model":        "yolov8l-cls-onnx",
        "fold_enabled": _state.fold_enabled,
        "classes":      _state.labels,
        "version":      "2.0.0",
        "uptime_sec":   round(time.time() - _state.start_time, 2),
    }


@app.get("/health")
def health() -> dict:
    snap = _state.stats_snapshot()
    return {
        "status":          "healthy",
        "model_loaded":    _state.session is not None,
        "fold_enabled":    _state.fold_enabled,
        "inference_count": _state.inference_count,
        "avg_latency_ms":  snap["avg_latency_ms"],
        "mqtt":            _state.mqtt.status() if _state.mqtt else {"enabled": False},
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
        img           = Image.open(io.BytesIO(raw)).convert("RGB")
        original_size = list(img.size)
    except Exception as exc:
        _state.record(0.0, success=False)
        raise HTTPException(status_code=422, detail=f"Cannot decode image: {exc}")

    # ---- YOLO inference ----
    t0 = time.perf_counter()
    try:
        tensor  = preprocess(img)
        raw_out = run_inference(tensor)
        result  = postprocess(raw_out)
    except Exception as exc:
        _state.record(0.0, success=False)
        logger.error("[PREDICT] YOLO error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    yolo_ms = round((time.perf_counter() - t0) * 1000, 3)
    _state.record(yolo_ms, success=True)

    label = result["predicted_class"]

    # ---- Fold pipeline ----
    fold_data = {"available": False}
    fold_ms   = None
    if result["is_confident"] and label != "null":
        t1        = time.perf_counter()
        fold_data = run_fold(raw, label)
        fold_ms   = round((time.perf_counter() - t1) * 1000, 3)

    # ---- MQTT publish ----
    mqtt_published = False
    if _state.mqtt and result["is_confident"] and label != "null" and label in SERVO_MAP:
        mqtt_published = _state.mqtt.publish_command(
            label=label,
            steps=SERVO_MAP[label]["steps"],
        )
        if not mqtt_published:
            logger.warning("[PREDICT] MQTT publish failed for label '%s'", label)

    return {
        **result,
        "inference_time_ms":   yolo_ms,
        "fold_time_ms":        fold_ms,
        "image_received_size": original_size,
        "mqtt_published":      mqtt_published,
        "fold":                fold_data,
    }


@app.get("/metrics")
def metrics() -> dict:
    snap = _state.stats_snapshot()
    return {
        "uptime_sec":       round(time.time() - _state.start_time, 2),
        "total_requests":   _state.total_requests,
        "success_requests": _state.success_requests,
        "error_requests":   _state.error_requests,
        "avg_latency_ms":   snap["avg_latency_ms"],
        "min_latency_ms":   snap["min_latency_ms"],
        "max_latency_ms":   snap["max_latency_ms"],
        "model_path":       MODEL_PATH,
        "fold_model_path":  FOLD_MODEL_PATH,
        "classes":          _state.labels,
        "threshold":        round(_state.conf_threshold, 5),
        "fold_enabled":     _state.fold_enabled,
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

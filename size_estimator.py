"""
size_estimator.py — Size estimation logic for ClothBot

Estimates clothing size (S/M/L/XL/XXL) by converting the detected clothing
width in pixels to centimetres using a calibrated scale (cm per pixel) derived
from a physical folder (pelipat pakaian) whose real-world width is known.

Perspective Warp Support
------------------------
When a config.json produced by calibration.py contains ``has_perspective=True``
and valid ``src_points`` / ``dst_size`` fields, the module can:

  1. Compute a perspective-transform matrix (``get_warp_matrix``).
  2. Warp an arbitrary BGR frame to a rectified, top-down view (``warp_frame``).
  3. Detect the widest clothing contour inside that rectified frame
     (``get_largest_contour_bbox``).
  4. Estimate size using a scale derived entirely from the warped-space geometry,
     where ``scale = folder_width_cm / dst_width_px`` (``run_size_estimation``).

When no perspective metadata is present the pipeline degrades gracefully to the
original pixel-scale approach, preserving full backward compatibility.

OpenCV / NumPy dependency
-------------------------
``numpy`` is imported at module level (lightweight).
``cv2`` is imported *inside* functions that need it so that the module can be
imported in environments where OpenCV is not installed (pure-logic callers only).

No OpenCV dependency is required for: ``load_folder_config``, ``estimate_size``,
``is_size_enabled``, or ``load_config``.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("clothbot")

# Physical width of the pelipat (folder) used as calibration reference.
FOLDER_WIDTH_CM: float = 90.0

# Default size thresholds expressed in absolute centimetres.
# Ordered largest-first so the first matching entry wins.
# Can be overridden via "size_thresholds" key in config.json.
DEFAULT_SIZE_MAP_CM: list[tuple[str, float]] = [
    ("XXL", 68.0),
    ("XL", 59.0),
    ("L", 54.0),
    ("M", 49.0),
    ("S", 0.0),
]

# Labels for which size estimation is applicable
SIZE_ENABLED_LABELS: frozenset[str] = frozenset(
    {
        "baju_lengan_panjang",
        "baju_lengan_pendek",
        "celana_panjang",
        "celana_pendek",
    }
)

# Module-level size map (mutated when config overrides are loaded)
_size_map_cm: list[tuple[str, float]] = list(DEFAULT_SIZE_MAP_CM)


def load_folder_config(config_path: Path) -> Optional[dict]:
    """
    Load calibration data from config.json.

    Expected JSON structure::

        {
          "folder_width_px":   320,          // pixel width from calibration image
          "scale_cm_per_px":   0.28125,      // preferred — skips the division below
          "folder_width_cm":   90.0,         // optional, defaults to FOLDER_WIDTH_CM
          "folder_x1":         160,          // optional — left edge x coord
          "folder_x2":         480,          // optional — right edge x coord
          "calib_image_width": 1280,         // optional — resolution normalisation
          "size_thresholds": {               // optional — override DEFAULT_SIZE_MAP_CM
            "XXL": 68.0,
            "XL":  59.0,
            "L":   54.0,
            "M":   49.0,
            "S":    0.0
          }
        }

    Resolution of ``scale_cm_per_px``
    ----------------------------------
    1. If ``scale_cm_per_px`` is present in the JSON it is used directly.
    2. Otherwise it is computed as ``folder_width_cm / folder_width_px``.
    3. If neither ``scale_cm_per_px`` nor ``folder_width_px`` are present,
       the function returns ``None``.

    Returns
    -------
    dict with normalised fields on success, None if file is missing or malformed.
    The returned dict always includes ``scale_cm_per_px`` and ``folder_width_cm``
    in addition to the pixel-space fields.
    """
    global _size_map_cm
    if not config_path.exists():
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[SIZE] Cannot read config.json: %s", exc)
        return None

    # Guard: need at least one of these two to derive scale.
    if "scale_cm_per_px" not in data and "folder_width_px" not in data:
        logger.warning(
            "[SIZE] config.json must contain 'scale_cm_per_px' or 'folder_width_px'"
        )
        return None

    # Resolve folder_width_cm (real-world reference width)
    try:
        folder_width_cm = float(data.get("folder_width_cm", FOLDER_WIDTH_CM))
    except (TypeError, ValueError) as exc:
        logger.warning("[SIZE] Invalid 'folder_width_cm' value: %s", exc)
        folder_width_cm = FOLDER_WIDTH_CM

    # Resolve scale_cm_per_px — prefer the explicit value when present
    if "scale_cm_per_px" in data:
        try:
            scale_cm_per_px = float(data["scale_cm_per_px"])
        except (TypeError, ValueError) as exc:
            logger.warning("[SIZE] Invalid 'scale_cm_per_px' value: %s", exc)
            return None
    else:
        try:
            folder_width_px = float(data["folder_width_px"])
        except (TypeError, ValueError) as exc:
            logger.warning("[SIZE] Invalid 'folder_width_px' value: %s", exc)
            return None
        if folder_width_px <= 0:
            logger.warning("[SIZE] 'folder_width_px' must be > 0")
            return None
        scale_cm_per_px = folder_width_cm / folder_width_px

    # Override size map when provided in config.
    # Always reset to defaults first so stale values from a previous load don't persist.
    # After parsing, sort descending by min_cm so the first match is always the
    # largest applicable size (required for correctness).
    _size_map_cm = list(DEFAULT_SIZE_MAP_CM)
    if "size_thresholds" in data:
        try:
            custom = {k: float(v) for k, v in data["size_thresholds"].items()}
            _size_map_cm = sorted(custom.items(), key=lambda kv: kv[1], reverse=True)
            logger.info("[SIZE] Custom size map (cm) loaded: %s", _size_map_cm)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "[SIZE] Invalid 'size_thresholds' in config, using defaults: %s", exc
            )

    # fmt: off
    return {
        "folder_width_px":    float(data["folder_width_px"]) if "folder_width_px" in data else None,
        "folder_width_cm":    folder_width_cm,
        "scale_cm_per_px":    scale_cm_per_px,
        "folder_x1":          int(data["folder_x1"]) if "folder_x1" in data else None,
        "folder_x2":          int(data["folder_x2"]) if "folder_x2" in data else None,
        # Width of the image used during calibration — required for resolution
        # normalisation when inference images have a different resolution.
        "calib_image_width":  int(data["calib_image_width"]) if "calib_image_width" in data else None,
    }
    # fmt: on


def estimate_size(
    clothing_width_px: float, scale_cm_per_px: float
) -> tuple[str, float, float]:
    """
    Estimate clothing size from its pixel width and a cm-per-pixel scale.

    Parameters
    ----------
    clothing_width_px : Width of clothing bounding rect in pixels.
    scale_cm_per_px   : Calibrated scale derived from folder calibration
                        (``folder_width_cm / folder_width_px``).

    Returns
    -------
    (size, lebar_cm, ratio)
        size      : one of "S" | "M" | "L" | "XL" | "XXL"
        lebar_cm  : estimated clothing width in centimetres, rounded to 1 d.p.
        ratio     : lebar_cm / FOLDER_WIDTH_CM, rounded to 3 d.p. (debug/logging)

    Notes
    -----
    Thresholds are checked largest-first; first match is returned.
    Default mapping (in cm)::

        lebar_cm ≥ 68 → XXL
        lebar_cm ≥ 59 → XL
        lebar_cm ≥ 54 → L
        lebar_cm ≥ 49 → M
        lebar_cm ≥  0 → S
    """
    if scale_cm_per_px <= 0:
        logger.warning(
            "[SIZE] scale_cm_per_px is %s — cannot estimate size", scale_cm_per_px
        )
        return "S", 0.0, 0.0

    lebar_cm = clothing_width_px * scale_cm_per_px
    ratio = lebar_cm / FOLDER_WIDTH_CM  # for debug/logging

    for size, min_cm in _size_map_cm:
        if lebar_cm >= min_cm:
            return size, round(lebar_cm, 1), round(ratio, 3)

    return "S", round(lebar_cm, 1), round(ratio, 3)


def is_size_enabled(label: str) -> bool:
    """Return True if size estimation is applicable for the given YOLO label."""
    return label in SIZE_ENABLED_LABELS


# ---------------------------------------------------------------------------
# Perspective warp helpers (new — backward-compatible additions)
# ---------------------------------------------------------------------------


def load_config(path: "str | Path" = "config.json") -> dict:
    """
    Muat config.json lengkap termasuk field perspektif.

    Raise FileNotFoundError dengan pesan jelas jika file tidak ditemukan.
    Berbeda dengan load_folder_config(), fungsi ini:
    - Raise exception (tidak return None) jika file tidak ada
    - Return raw dict config lengkap (termasuk src_points, has_perspective, dll.)
    - Juga memuat module-level _size_map_cm via load_folder_config() sebagai side-effect

    Returns
    -------
    dict — raw config JSON, dengan semua field yang tersimpan di file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"config.json tidak ditemukan di '{p.resolve()}'. "
            "Jalankan calibration.py terlebih dahulu."
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FileNotFoundError(
            f"config.json rusak atau tidak valid JSON: {exc}"
        ) from exc

    # Side-effect: update module-level _size_map_cm via existing loader
    load_folder_config(p)

    return data


def get_warp_matrix(
    config: dict,
) -> "tuple[np.ndarray | None, tuple[int, int] | None]":
    """
    Ekstrak perspective transform matrix dari config.

    Returns (M, dst_size) jika has_perspective=True dan src_points tersedia,
    (None, None) jika tidak (fallback ke mode tanpa warp).
    """
    import cv2  # local import: cv2/numpy optional at module level

    if not config.get("has_perspective"):
        return None, None

    src_raw = config.get("src_points")
    dst_raw = config.get("dst_size")

    if not src_raw or len(src_raw) != 4 or not dst_raw or len(dst_raw) != 2:
        logger.warning(
            "[SIZE] has_perspective=True tapi src_points/dst_size tidak lengkap — "
            "fallback ke mode tanpa warp"
        )
        return None, None

    try:
        src_pts = np.array(src_raw, dtype=np.float32)
        dst_w, dst_h = int(dst_raw[0]), int(dst_raw[1])
        dst_pts = np.array(
            [
                [0, 0],
                [dst_w - 1, 0],
                [dst_w - 1, dst_h - 1],
                [0, dst_h - 1],
            ],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return M, (dst_w, dst_h)
    except Exception as exc:
        logger.warning("[SIZE] Gagal menghitung warp matrix: %s", exc)
        return None, None


def warp_frame(
    frame: "np.ndarray",
    M: "np.ndarray | None",
    dst_size: "tuple[int, int] | None",
) -> "np.ndarray":
    """
    Terapkan perspective warp pada frame.

    Jika M is None, kembalikan frame asli tanpa modifikasi.
    """
    if M is None or dst_size is None:
        return frame
    import cv2

    return cv2.warpPerspective(frame, M, dst_size)


def get_largest_contour_bbox(
    frame: "np.ndarray",
) -> "tuple[int, int, int, int] | None":
    """
    Jalankan adaptive Canny pada frame, kembalikan bounding rect kontur terlebar.

    Didesain untuk bekerja pada frame yang sudah di-warp (perspektif sudah lurus),
    maupun frame asli.

    Pada warped frame, margin 3% dipotong dari semua sisi sebelum Canny
    sehingga border warp (artefak tepi hasil getPerspectiveTransform) tidak
    ikut terdeteksi sebagai kontur garmen.

    Returns
    -------
    (x, y, w, h) dari kontur garmen, atau None jika tidak ditemukan.
    """
    import cv2

    h, w = frame.shape[:2]

    # Non-uniform margins:
    # - y-axis (atas/bawah warped = sisi lengan baju): kecil (3%) agar lebar dada
    #   terukur penuh, tidak terpotong.
    # - x-axis (kiri/kanan warped = area kerah/hem dalam portrait): besar (8%) agar
    #   bar mesin di atas baju dan meja di bawah baju ter-exclude dari deteksi.
    #   Bar mesin biasanya ~50-70px dari kiri warped; m_x=8%*900=72px meng-exclude-nya.
    m_y = max(2, int(h * 0.03))  # 3% of warped height  (~17px)
    m_x = max(m_y, int(w * 0.08))  # 8% of warped width   (~72px)
    inner = frame[m_y : h - m_y, m_x : w - m_x]

    # CLAHE pada channel L (LAB) untuk normalisasi kontras lokal.
    # Ini membantu Canny menemukan tepi di badan baju yang memiliki
    # warna uniform (abu-abu) dan kontras rendah terhadap latar belakang.
    lab = cv2.cvtColor(inner, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    inner_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(inner_eq, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Canny via Otsu threshold
    otsu_thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo = max(10.0, otsu_thresh * 0.33)
    hi = max(20.0, otsu_thresh * 0.66)
    edges = cv2.Canny(blur, lo, hi)

    # Fallback: morphological gradient jika Canny tidak menemukan tepi
    if np.count_nonzero(edges) < 50:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gradient = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        _, edges = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Horizontal-dominant dilation: expand ke kiri-kanan (arah panjang baju)
    # TANPA expand signifikan ke atas-bawah (arah lebar dada).
    # Kernel (25 × 3): lebar horizontal 25px untuk bridge gap antar fitur baju,
    # tinggi vertikal 3px (minimal) agar tidak menjangkau bar mesin / meja.
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    edges_dilated = cv2.dilate(edges, dil_kernel, iterations=3)

    contours, _ = cv2.findContours(
        edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Fallback ke edges asli jika dilation menghilangkan semua kontur
    if not contours:
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    if not contours:
        return None

    # Filter kontur terlalu kecil (noise)
    ih, iw = inner.shape[:2]
    min_area = ih * iw * 0.005
    valid = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid:
        valid = contours

    # Global union seluruh kontur valid.
    # Di warped frame, seluruh area = papan pelipat = area baju.
    # Tidak ada objek asing di luar board, jadi semua kontur valid = bagian baju.
    # Ini menggantikan anchor+proximity yang gagal mendeteksi badan baju
    # karena proximity hanya horizontal dan tidak mencakup badan baju yang jauh.
    rects = [cv2.boundingRect(c) for c in valid]
    ux1 = min(r[0] for r in rects)
    ux2 = max(r[0] + r[2] for r in rects)
    uy1 = min(r[1] for r in rects)
    uy2 = max(r[1] + r[3] for r in rects)

    # Kembalikan koordinat relatif ke frame asli (tambah kembali margin non-uniform)
    return (ux1 + m_x, uy1 + m_y, ux2 - ux1, uy2 - uy1)


def run_size_estimation(
    frame: "np.ndarray",
    config: dict,
) -> dict:
    """
    Pipeline estimasi ukuran lengkap: [warp →] Canny → kontur → estimasi ukuran.

    Jika config mengandung has_perspective=True dan src_points valid:
      - Frame di-warp terlebih dahulu (perspektif diluruskan)
      - Canny + kontur dijalankan pada frame warped
      - Scale diambil dari warped space (folder_width_cm / dst_size[0])

    Jika tidak ada perspektif:
      - Frame digunakan apa adanya
      - Scale diambil dari config (scale_cm_per_px atau folder_width_px)

    Parameters
    ----------
    frame  : BGR frame asli (tidak dimodifikasi)
    config : Dict hasil load_config() atau dict config langsung

    Returns
    -------
    dict dengan field:
      - "size"         : str | None  — "S"|"M"|"L"|"XL"|"XXL" atau None
      - "lebar_cm"     : float       — lebar pakaian dalam cm
      - "bbox"         : (x,y,w,h) | None — bounding rect kontur terlebar
      - "warped_frame" : np.ndarray  — frame setelah warp (= frame asli jika no warp)
    """
    M, dst_size = get_warp_matrix(config)
    warped = warp_frame(frame, M, dst_size)

    bbox = get_largest_contour_bbox(warped)
    if bbox is None:
        return {
            "size": None,
            "lebar_cm": 0.0,
            "bbox": None,
            "warped_frame": warped,
        }

    _, _, bbox_w, bbox_h = bbox

    # Tentukan scale berdasarkan mode
    if M is not None and dst_size is not None:
        # Warped mode: scale = folder_height_cm / dst_height_px
        # Shirt is placed collar-up; chest width maps to vertical axis in warped frame.
        # dst_size[1] = 560px = 56cm → scale = 56/560 = 0.1 cm/px (same as horizontal)
        folder_height_cm = float(config.get("folder_height_cm", 56.0))
        scale = folder_height_cm / dst_size[1]
        measure_px = float(bbox_h)
    else:
        # Non-warp mode: measure horizontal width as before
        scale = float(config.get("scale_cm_per_px") or 0.0)
        if scale <= 0 and config.get("folder_width_px"):
            folder_w_cm = float(config.get("folder_width_cm", FOLDER_WIDTH_CM))
            scale = folder_w_cm / float(config["folder_width_px"])
        if scale <= 0:
            logger.warning("[SIZE] scale_cm_per_px tidak tersedia di config")
            return {
                "size": None,
                "lebar_cm": 0.0,
                "bbox": bbox,
                "warped_frame": warped,
            }
        measure_px = float(bbox_w)

    size, lebar_cm, _ = estimate_size(measure_px, scale)

    return {
        "size": size,
        "lebar_cm": lebar_cm,
        "bbox": bbox,
        "warped_frame": warped,
    }

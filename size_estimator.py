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
#
# Threshold ini dikalibrasi berdasarkan deteksi brightness segmentation
# pada papan pelipat hitam 65cm×90cm.
# Referensi: baju M → chest ≈40cm, length ≈55cm.
# Gunakan SIZE_DEBUG_MODE=true untuk melihat nilai d: dan p: aktual,
# lalu sesuaikan threshold berikut.
DEFAULT_SIZE_MAP_CM: list[tuple[str, float]] = [
    ("XXL", 58.0),  # chest ≥ 58cm
    ("XL", 50.0),  # chest ≥ 50cm
    ("L", 45.0),  # chest ≥ 45cm
    ("M", 38.0),  # chest ≥ 38cm  (ref: baju M = ~40cm)
    ("S", 0.0),  # chest < 38cm
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

# Length-based size thresholds (shirt collar-to-hem, cm).
# Dikalibrasi berdasarkan: baju M = ~55cm panjang.
DEFAULT_LENGTH_MAP_CM: list[tuple[str, float]] = [
    ("XXL", 68.0),  # panjang ≥ 68cm
    ("XL", 62.0),  # panjang ≥ 62cm
    ("L", 58.0),  # panjang ≥ 58cm
    ("M", 52.0),  # panjang ≥ 52cm  (ref: baju M = ~55cm)
    ("S", 0.0),  # panjang < 52cm
]

_length_map_cm: list[tuple[str, float]] = list(DEFAULT_LENGTH_MAP_CM)

# Ordered size list for rank comparison
_SIZE_ORDER: list[str] = ["S", "M", "L", "XL", "XXL"]


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


def estimate_size_combined(
    chest_cm: float,
    length_cm: float,
) -> tuple[str, float, float]:
    """
    Estimasi ukuran menggunakan dua dimensi: lebar dada dan panjang baju.

    Strategi: klasifikasi masing-masing dimensi secara independen, ambil
    ukuran yang LEBIH BESAR dari keduanya.

    Ini mengatasi chest yang ter-cap oleh lebar board (65cm): shirt L yang
    chest-nya clips di ~63cm (→M), tapi panjangnya 70cm (→L), final = L.

    Parameters
    ----------
    chest_cm  : Lebar dada estimasi dalam cm (dari bbox_h di warped frame).
    length_cm : Panjang baju estimasi dalam cm (dari bbox_w di warped frame).

    Returns
    -------
    (size, chest_cm, length_cm)
    """
    # Classify by chest
    chest_size = "S"
    for size, min_cm in _size_map_cm:
        if chest_cm >= min_cm:
            chest_size = size
            break

    # Classify by length
    length_size = "S"
    for size, min_cm in _length_map_cm:
        if length_cm >= min_cm:
            length_size = size
            break

    # Take the larger of the two
    chest_idx = _SIZE_ORDER.index(chest_size)
    length_idx = _SIZE_ORDER.index(length_size)
    final_size = _SIZE_ORDER[max(chest_idx, length_idx)]

    logger.debug(
        "[SIZE/COMBINED] chest=%.1fcm→%s  length=%.1fcm→%s  final=%s",
        chest_cm,
        chest_size,
        length_cm,
        length_size,
        final_size,
    )
    return final_size, round(chest_cm, 1), round(length_cm, 1)


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
    Deteksi bounding rect pakaian pada warped frame.

    Strategi utama: Brightness Segmentation (L channel LAB).
    Papan pelipat berwarna HITAM, pakaian lebih TERANG → mudah dipisahkan
    dengan Otsu threshold pada channel L.
    Pendekatan ini kebal terhadap lubang papan yang merusak Canny/density scan.

    Fallback: Canny + density scan jika brightness segmentation gagal
    (mis. baju gelap di atas papan gelap).

    Returns
    -------
    (x, y, w, h) koordinat relatif ke frame asli, atau None.
    """
    import cv2

    h, w = frame.shape[:2]

    # Non-uniform margins:
    # - m_y kecil (3%): jaga pengukuran lebar dada penuh di arah vertikal
    # - m_x besar (8%): exclude bar mesin (kiri warped) dan meja (kanan warped)
    m_y = max(2, int(h * 0.03))
    m_x = max(m_y, int(w * 0.08))
    inner = frame[m_y : h - m_y, m_x : w - m_x]
    ih, iw = inner.shape[:2]

    # ── Stage 1: Brightness Segmentation ───────────────────────────────────
    # Papan pelipat hitam vs pakaian yang lebih terang → Otsu pada L channel
    # Keunggulan: kebal terhadap lubang papan, tekstur papan, dan noise Canny.
    lab = cv2.cvtColor(inner, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]  # L = kecerahan (0=hitam, 255=putih)

    otsu_val, bright_mask = cv2.threshold(
        l_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ── Hapus kardus coklat/tan dari bright_mask (SEBELUM morph ops) ────────
    # Root cause overestimate: kardus di papan ikut terdeteksi sebagai "baju"
    # karena brightness-nya mirip. Kardus memiliki b* tinggi (kuning/coklat)
    # dalam ruang warna LAB, sedangkan baju putih/terang memiliki b* netral.
    #
    # Skala OpenCV LAB: b*=128 netral, >128 kuning, <128 biru.
    # Threshold 138 = 10 unit di atas netral → tangkap kardus coklat/tan
    # tanpa menghapus baju putih, abu, atau warna lain yang lebih jenuh tapi
    # tidak masuk range kuning-coklat (misal: baju merah, biru, hijau).
    #
    # Dilakukan SEBELUM morph ops agar close kernel tidak menjembatani
    # piksel kardus ke piksel baju yang berdekatan.
    b_ch = lab[:, :, 2]  # b* channel
    _, cardboard_mask = cv2.threshold(b_ch, 138, 255, cv2.THRESH_BINARY)
    bright_mask = cv2.bitwise_and(bright_mask, cv2.bitwise_not(cardboard_mask))
    logger.debug(
        "[BBOX/CARDBOARD] b*>138 pixels removed: %d",
        int(np.count_nonzero(cardboard_mask)),
    )

    # Morphological closing untuk mengisi lubang dalam area baju
    # (misal: kancing, logo, lipatan yang lebih gelap dari kain)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, close_k)
    # Remove isolated small bright spots (bukan baju)
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, open_k)

    # Cari kontur pada bright mask
    contours_b, _ = cv2.findContours(
        bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter: ambil kontur dengan area cukup besar (bukan noise)
    min_area = ih * iw * 0.05  # minimal 5% area inner frame = baju
    valid_b = [c for c in contours_b if cv2.contourArea(c) > min_area]

    if valid_b:
        # Gunakan kontur terbesar (= baju utama)
        largest_b = max(valid_b, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest_b)
        logger.debug(
            "[BBOX/BRIGHT] otsu=%.0f  contours=%d  bbox=(%d,%d,%d,%d)",
            otsu_val,
            len(valid_b),
            bx,
            by,
            bw,
            bh,
        )
        return (bx + m_x, by + m_y, bw, bh)

    # ── Stage 2: Fallback — Canny + Density Scan ──────────────────────────
    # Digunakan saat brightness segmentation gagal:
    # - Baju warna gelap di atas papan gelap (kontras L rendah)
    # - Threshold Otsu tidak berhasil memisahkan baju dari papan
    logger.debug("[BBOX] Brightness segmentation gagal, fallback ke Canny+density")

    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    otsu_t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, max(10.0, otsu_t * 0.33), max(20.0, otsu_t * 0.66))

    if np.count_nonzero(edges) < 50:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        _, edges = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    edges = cv2.dilate(edges, dil_k, iterations=2)

    # Density scan dengan threshold lebih tinggi untuk filter lubang papan
    min_density = max(20, min(iw, ih) // 20)
    row_counts = np.sum(edges > 0, axis=1)
    col_counts = np.sum(edges > 0, axis=0)
    valid_rows = np.where(row_counts >= min_density)[0]
    valid_cols = np.where(col_counts >= min_density)[0]
    if len(valid_rows) < 2:
        valid_rows = np.where(row_counts > 0)[0]
    if len(valid_cols) < 2:
        valid_cols = np.where(col_counts > 0)[0]
    if len(valid_rows) < 2 or len(valid_cols) < 2:
        return None

    uy1, uy2 = int(valid_rows[0]), int(valid_rows[-1])
    ux1, ux2 = int(valid_cols[0]), int(valid_cols[-1])
    return (ux1 + m_x, uy1 + m_y, ux2 - ux1, uy2 - uy1)


def _measure_shirt_slices(
    warped: "np.ndarray",
    bbox: "tuple[int, int, int, int]",
) -> "tuple[float, float]":
    """
    Ukur lebar dada dan panjang baju menggunakan pendekatan SLICE.

    Masalah dengan full-bbox measurement:
    - Chest (bbox_h): termasuk ujung lengan di papan → terlalu besar
    - Length (bbox_w): termasuk kardus di tepi warped → terlalu besar

    Solusi slice:
    - CHEST  : ukur vertical extent pada slice x = 15-35% dari kerah
               (posisi dada, jauh dari ujung lengan di tepi warped).
    - LENGTH : ukur horizontal extent pada slice y = 35-65% dari bbox
               (badan tengah, menghindari kardus yang ada di tepi kanan warped).

    Di tiap slice, hitung pixel terang per baris/kolom dan hanya sertakan
    baris/kolom yang memiliki density ≥ 40% dari lebar slice
    (threshold ini menyaring ujung lengan tipis & kardus yang terpisah).

    Returns
    -------
    (chest_px, length_px) dalam koordinat warped frame.
    """
    import cv2

    h, w = warped.shape[:2]
    m_y = max(2, int(h * 0.03))
    m_x = max(m_y, int(w * 0.08))
    inner = warped[m_y : h - m_y, m_x : w - m_x]
    ih, iw = inner.shape[:2]

    # Bright mask (same Otsu on L channel) + hapus kardus coklat
    lab = cv2.cvtColor(inner, cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    _, bright = cv2.threshold(l_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Hapus kardus (b* > 138) sama seperti di get_largest_contour_bbox.
    # Wajib dilakukan di sini karena bright mask dihitung ulang secara independen.
    b_ch = lab[:, :, 2]
    _, cardboard_mask_s = cv2.threshold(b_ch, 138, 255, cv2.THRESH_BINARY)
    bright = cv2.bitwise_and(bright, cv2.bitwise_not(cardboard_mask_s))

    # Closing kecil untuk mengisi lubang pada kain (sama dengan Stage 1 di get_largest_contour_bbox)
    close_ks = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, close_ks)

    # Translate bbox to inner-frame coordinates
    bx, by, bw, bh = bbox
    bx_i = max(0, bx - m_x)
    by_i = max(0, by - m_y)
    bw_i = min(bw, iw - bx_i)
    bh_i = min(bh, ih - by_i)

    # ── CHEST: cari segmen kontigu terpanjang per kolom ────────────────────
    # Root cause lama: active_r[-1] - active_r[0] mengukur span TOTAL dari
    # piksel terang paling atas ke paling bawah, termasuk frame kayu papan
    # pelipat yang muncul sebagai strip cerah di atas DAN bawah warped frame.
    # Hasilnya selalu ≈ 611px (tinggi inner frame) berapapun ukuran baju.
    #
    # Fix: gunakan SEGMEN KONTIGU TERPANJANG dari baris aktif per kolom.
    # Gap gelap antara frame kayu (tepi) dan baju (tengah) memisahkan segmen,
    # sehingga hanya badan baju yang terukur.
    chest_vals: list[int] = []
    scan_x_start = bx_i + int(bw_i * 0.10)
    scan_x_end = bx_i + int(bw_i * 0.80)
    # Minimal 20% bbox height atau 100px (≈10cm) agar noise tidak masuk.
    min_chest_px = max(100, int(bh_i * 0.20))
    for xi in range(scan_x_start, min(scan_x_end, iw), 4):
        col = bright[by_i : by_i + bh_i, max(0, xi) : min(iw, xi + 4)]
        if col.size == 0:
            continue
        row_active = np.any(col > 0, axis=1)
        # Temukan semua segmen kontigu: [start, end) per segmen
        padded = np.concatenate(([False], row_active, [False]))
        seg_starts = np.where(~padded[:-1] & padded[1:])[0]
        seg_ends = np.where(padded[:-1] & ~padded[1:])[0]
        if len(seg_starts):
            max_seg = int(np.max(seg_ends - seg_starts))
            if max_seg >= min_chest_px:
                chest_vals.append(max_seg)

    # Fallback bertingkat jika scan tidak menghasilkan cukup nilai
    if len(chest_vals) >= 5:
        chest_px = float(np.percentile(chest_vals, 15))
    elif len(chest_vals) >= 1:
        # Terlalu sedikit sampel → pakai median agar lebih stabil
        chest_px = float(np.median(chest_vals))
    else:
        # Tidak ada sampel sama sekali → estimasi konservatif 65% bbox height
        chest_px = float(bh_i * 0.65)

    # ── LENGTH: horizontal slice at 35-65% of bbox height (body centre) ────
    # At the vertical centre of the shirt, only the shirt body is present
    # (sleeve tips are at the top/bottom of the warped frame).
    # Cardboard visible at the right edge of warped is separated by a dark
    # board gap → column density drops to 0 before reaching the cardboard.
    ly1 = by_i + int(bh_i * 0.35)
    ly2 = by_i + int(bh_i * 0.65)
    ly1 = max(0, min(ly1, ih - 1))
    ly2 = max(ly1 + 1, min(ly2, ih))

    length_px = float(bw_i * 0.65)  # fallback konservatif 65% bbox width
    if ly2 > ly1:
        l_slice = bright[ly1:ly2, bx_i : bx_i + bw_i]
        if l_slice.size > 0:
            col_cnts = np.sum(l_slice > 0, axis=0)
            min_d = max(1, int((ly2 - ly1) * 0.40))
            col_active_arr = col_cnts >= min_d
            # Gunakan SEGMEN KONTIGU TERPANJANG (bukan active_c[-1] - active_c[0])
            # untuk menghindari kardus / ubin lantai yang terpisah oleh gap gelap.
            padded_c = np.concatenate(([False], col_active_arr, [False]))
            seg_s = np.where(~padded_c[:-1] & padded_c[1:])[0]
            seg_e = np.where(padded_c[:-1] & ~padded_c[1:])[0]
            if len(seg_s):
                max_seg_l = int(np.max(seg_e - seg_s))
                if max_seg_l > 0:
                    length_px = float(max_seg_l)

    logger.debug(
        "[SLICE] chest_vals=%d  chest=%.0fpx=%.1fcm  length=%.0fpx=%.1fcm",
        len(chest_vals),
        chest_px,
        chest_px * 0.1,
        length_px,
        length_px * 0.1,
    )
    return chest_px, length_px


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
    measured_bbox = None  # diisi di warped mode

    # Tentukan scale berdasarkan mode
    if M is not None and dst_size is not None:
        # Warped mode: gunakan SLICE-BASED measurement.
        # bbox dari get_largest_contour_bbox = full bright region (termasuk lengan & kardus)
        # → JANGAN langsung pakai bbox_h / bbox_w untuk pengukuran.
        # _measure_shirt_slices mengukur:
        #   chest_px : vertical extent di slice 15-35% dari kerah (posisi dada saja)
        #   length_px: horizontal extent di slice 35-65% dari bbox (badan tengah saja)
        folder_height_cm = float(config.get("folder_height_cm", 65.0))
        folder_width_cm_val = float(config.get("folder_width_cm", FOLDER_WIDTH_CM))
        scale_h = folder_height_cm / dst_size[1]  # 65/650 = 0.1 cm/px
        scale_w = folder_width_cm_val / dst_size[0]  # 90/900 = 0.1 cm/px

        chest_px, length_px = _measure_shirt_slices(warped, bbox)
        chest_cm = chest_px * scale_h
        length_cm = length_px * scale_w

        # Faktor koreksi terpisah per dimensi (diatur di config.json).
        # scale_correction_chest  : koreksi lebar dada  (chest_cm)
        # scale_correction_length : koreksi panjang baju (length_cm)
        # scale_correction_factor : fallback global jika yang spesifik tidak ada
        #
        # Cara hitung: correction = aktual_cm / terbaca_cm
        # Contoh: chest terbaca 58.8cm, aktual 40cm → 40/58.8 = 0.680
        global_corr = float(config.get("scale_correction_factor", 1.0))
        chest_corr = float(config.get("scale_correction_chest", global_corr))
        length_corr = float(config.get("scale_correction_length", global_corr))
        chest_cm = round(chest_cm * chest_corr, 1)
        length_cm = round(length_cm * length_corr, 1)
        logger.debug(
            "[CORRECTION] chest_corr=%.3f  length_corr=%.3f  "
            "chest=%.1fcm  length=%.1fcm",
            chest_corr,
            length_corr,
            chest_cm,
            length_cm,
        )

        size, lebar_cm, panjang_cm = estimate_size_combined(chest_cm, length_cm)

        # Build measured_bbox in warped space:
        # represents the MEASURED shirt body (not raw detected region).
        # Centered on detected bbox, width=length_px, height=chest_px.
        # Used by app.py for accurate display on original frame.
        bx_r, by_r, bw_r, bh_r = bbox
        shirt_cx = bx_r + bw_r / 2
        shirt_cy = by_r + bh_r / 2
        m_bx = max(0, int(shirt_cx - length_px / 2))
        m_by = max(0, int(shirt_cy - chest_px / 2))
        m_bw = min(int(length_px), dst_size[0] - m_bx)
        m_bh = min(int(chest_px), dst_size[1] - m_by)
        measured_bbox = (m_bx, m_by, m_bw, m_bh)
    else:
        # Non-warp mode: measure horizontal width only (backward compat)
        scale = float(config.get("scale_cm_per_px") or 0.0)
        if scale <= 0 and config.get("folder_width_px"):
            folder_w_cm = float(config.get("folder_width_cm", FOLDER_WIDTH_CM))
            scale = folder_w_cm / float(config["folder_width_px"])
        if scale <= 0:
            logger.warning("[SIZE] scale_cm_per_px tidak tersedia di config")
            return {
                "size": None,
                "lebar_cm": 0.0,
                "panjang_cm": 0.0,
                "bbox": bbox,
                "warped_frame": warped,
            }
        size, lebar_cm, _ = estimate_size(float(bbox_w), scale)
        panjang_cm = 0.0

    return {
        "size": size,
        "lebar_cm": lebar_cm,
        "panjang_cm": panjang_cm,
        "bbox": bbox,
        "measured_bbox": measured_bbox
        if (M is not None and dst_size is not None)
        else None,
        "warped_frame": warped,
    }

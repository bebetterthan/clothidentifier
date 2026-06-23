"""
visualization.py — Size estimation overlay helpers for ClothBot

Provides OpenCV drawing utilities for visualising size estimation results
on top of frames. All operations are performed on copies (non-destructive)
unless stated otherwise.

New additions (v2):
  - ``COLOR_MAP``          — updated per-size BGR colors for the new API.
  - ``draw_reference_lines`` — draw folder reference lines from a config dict.
  - ``draw_size_result``   — draw a bounding box + size label in-place.

Requires: opencv-python or opencv-python-headless
"""

import cv2
import numpy as np

# ------------------------------------------------------------------
# Color constants  (BGR format for OpenCV)
# ------------------------------------------------------------------

# Per-size bounding box / label colors
SIZE_COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "XXL": (0, 0, 200),  # dark red
    "XL": (0, 80, 255),  # orange-red
    "L": (0, 165, 255),  # orange
    "M": (0, 255, 0),  # green
    "S": (255, 180, 0),  # blue-ish
}

# Alias ekspor untuk API baru — warna sedikit diperbarui mengikuti spesifikasi prompt
COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "XXL": (0, 0, 255),  # merah
    "XL": (0, 80, 255),  # oranye-merah
    "L": (0, 165, 255),  # oranye
    "M": (0, 200, 0),  # hijau
    "S": (255, 100, 0),  # biru
}

# Folder (pelipat) reference line color — golden yellow
FOLDER_LINE_COLOR: tuple[int, int, int] = (0, 200, 255)


# ------------------------------------------------------------------
# Primitive drawing helpers
# ------------------------------------------------------------------


def _draw_folder_lines(
    frame: np.ndarray,
    folder_x1: int,
    folder_x2: int,
    alpha: float = 0.80,
) -> None:
    """
    Draw vertical reference lines at the folder edges in-place.

    Lines are blended at *alpha* onto *frame* so they don't obscure details.
    """
    h = frame.shape[0]
    overlay = frame.copy()
    cv2.line(overlay, (folder_x1, 0), (folder_x1, h), FOLDER_LINE_COLOR, 2, cv2.LINE_AA)
    cv2.line(overlay, (folder_x2, 0), (folder_x2, h), FOLDER_LINE_COLOR, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_contour_bbox(
    frame: np.ndarray,
    contour_bbox: tuple[int, int, int, int],
    size: str,
    alpha: float = 0.80,
) -> None:
    """
    Draw the detected clothing bounding box in-place.

    Parameters
    ----------
    contour_bbox : (x, y, w, h) as returned by cv2.boundingRect
    """
    x, y, w, h = contour_bbox
    color = SIZE_COLOR_MAP.get(size, (255, 255, 255))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_size_pill(
    frame: np.ndarray,
    label: str,
    size: str,
    ratio: float,
    lebar_cm: float = 0.0,
    debug: bool = False,
) -> None:
    """
    Draw a pill-style size label in the top-left corner of *frame* in-place.

    Normal mode  : "baju_lengan_pendek — M  (56.3 cm)"
    Debug mode   : "baju_lengan_pendek — M  (56.3 cm | ratio 0.63)"

    The cm value is only appended when *lebar_cm* > 0.
    """
    color = SIZE_COLOR_MAP.get(size, (255, 255, 255))
    text = f"{label} \u2014 {size}"

    if lebar_cm > 0 and debug:
        text += f"  ({lebar_cm:.1f} cm | ratio {ratio:.2f})"
    elif lebar_cm > 0:
        text += f"  ({lebar_cm:.1f} cm)"
    elif debug:
        text += f"  (ratio {ratio:.2f})"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2

    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    px, py = 10, 7  # horizontal / vertical padding
    x0, y0 = 12, 12

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + tw + 2 * px, y0 + th + bl + 2 * py),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Colored text
    cv2.putText(
        frame,
        text,
        (x0 + px, y0 + py + th),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def draw_reference_lines(frame: np.ndarray, config: dict) -> None:
    """
    Gambar garis vertikal kuning sebagai indikator area referensi di frame asli.

    Membaca ``folder_x1`` dan ``folder_x2`` dari dict config.
    Tidak melakukan apa-apa jika salah satu field tidak tersedia.

    Modifikasi in-place — tidak mengembalikan nilai.
    """
    x1 = config.get("folder_x1")
    x2 = config.get("folder_x2")
    if x1 is None or x2 is None:
        return
    _draw_folder_lines(frame, int(x1), int(x2))


def draw_size_result(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    size: str,
    lebar_cm: float,
    debug: bool = False,
) -> None:
    """
    Gambar bounding box berwarna + label ukuran pada frame secara in-place.

    Parameters
    ----------
    frame    : BGR frame yang akan dianotasi (dimodifikasi langsung).
    bbox     : (x, y, w, h) dari Canny contour.
    label    : Nama kelas YOLO (mis. "baju_lengan_pendek").
    size     : Ukuran estimasi: "S" | "M" | "L" | "XL" | "XXL".
    lebar_cm : Lebar pakaian estimasi dalam cm.
    debug    : Jika True, tampilkan nilai lebar_cm di label.

    Warna bounding box mengikuti ``COLOR_MAP``.
    Format label: ``"{label} - {size}"`` atau ``"{label} - {size} ({lebar_cm}cm)"`` saat debug.
    """
    x, y, w, h = bbox
    color = COLOR_MAP.get(size, (255, 255, 255))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

    text = f"{label} - {size}"
    if debug:
        text += f" ({lebar_cm:.1f}cm)"

    # Posisi teks: di atas bounding box, dengan clamp agar tidak keluar frame
    ty = max(y - 10, 15)
    cv2.putText(
        frame, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )


def annotate_size(
    frame: np.ndarray,
    label: str,
    size: str,
    ratio: float,
    contour_bbox: tuple[int, int, int, int] | None = None,
    folder_x1: int | None = None,
    folder_x2: int | None = None,
    lebar_cm: float = 0.0,
    debug: bool = False,
) -> np.ndarray:
    """
    Produce a fully-annotated copy of *frame* with size estimation overlays.

    Drawing layers (bottom → top):
      1. Folder reference lines   (drawn if folder_x1 and folder_x2 provided)
      2. Clothing contour bbox    (drawn if contour_bbox provided)
      3. Size label pill          (always drawn in top-left corner)

    Parameters
    ----------
    frame        : BGR source image (not modified — a copy is returned).
    label        : YOLO class name (e.g. "baju_lengan_pendek").
    size         : Estimated size: "S" | "M" | "L" | "XL" | "XXL".
    ratio        : clothing_width_px / folder_width_px.
    contour_bbox : (x, y, w, h) bounding rect of largest Canny contour, optional.
    folder_x1    : Left edge of folding board in pixels, optional.
    folder_x2    : Right edge of folding board in pixels, optional.
    lebar_cm     : Estimated clothing width in centimetres; shown when > 0.
    debug        : Show ratio value in the label text.

    Returns
    -------
    np.ndarray — Annotated copy of the input frame.
    """
    result = frame.copy()

    if folder_x1 is not None and folder_x2 is not None:
        _draw_folder_lines(result, folder_x1, folder_x2)

    if contour_bbox is not None:
        _draw_contour_bbox(result, contour_bbox, size)

    _draw_size_pill(result, label, size, ratio, lebar_cm=lebar_cm, debug=debug)

    return result

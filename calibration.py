"""
calibration.py — Folding board (pelipat pakaian) calibration tool for ClothBot

Detects the left and right vertical edges of the folding board using
Canny edge detection + Probabilistic Hough Transform, then saves the
result to config.json for use by the size estimation pipeline.

Usage examples
--------------
    # Auto-detect from image
    python calibration.py --image /path/to/empty_folder_frame.jpg

    # Auto-detect, write to custom config location
    python calibration.py --image frame.jpg --config /srv/clothbot/config.json

    # Manual override (skips auto-detection)
    python calibration.py --width 320

    # Manual with x-positions (for visualization reference lines)
    python calibration.py --width 320 --x1 160 --x2 480

    # Interactive camera calibration (click mode)
    python calibration.py --camera          # uses camera index 0
    python calibration.py --camera 1        # uses camera index 1

    # Kalibrasi perspektif 4 titik (klik marker di live feed)
    python calibration.py --perspective     # kamera index 0
    python calibration.py --perspective 1   # kamera index 1
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Physical width of the reference object (pelipat + kardus)
FOLDER_WIDTH_CM: float = 90.0  # lebar board saat dipakai (horizontal) = 90 cm
FOLDER_HEIGHT_CM: float = 56.0  # tinggi board saat dipakai (vertikal) = 56 cm
PERSPECTIVE_DST_W: int = 900  # warped output width  (900px = 90cm → 0.1 cm/px)
PERSPECTIVE_DST_H: int = 560  # warped output height (560px = 56cm)

# ----------------------------------------------------------------
# Core detection
# ----------------------------------------------------------------


def detect_folder_edges(frame: np.ndarray) -> tuple[int, int] | None:
    """
    Detect the left and right vertical edges of the folding board.

    Algorithm
    ---------
    1. Canny edge detection
    2. Probabilistic Hough Transform to find line segments
    3. Filter for near-vertical lines (|dx| / |dy| < 0.2)
    4. Split x-midpoints into left-half / right-half groups
    5. Return median of each group as (folder_x1, folder_x2)

    Parameters
    ----------
    frame : BGR image as np.ndarray

    Returns
    -------
    (folder_x1, folder_x2) pixel x-coordinates, or None if detection fails.
    """
    h, w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(h * 0.3),
        maxLineGap=15,
    )

    if lines is None:
        return None

    # Collect x-midpoints of near-vertical segments
    vertical_xs: list[float] = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > 0 and (dx / dy) < 0.2:
            vertical_xs.append((x1 + x2) / 2.0)

    if len(vertical_xs) < 2:
        return None

    # Partition into left-half and right-half of the image
    mid = w / 2.0
    left_xs = [x for x in vertical_xs if x < mid]
    right_xs = [x for x in vertical_xs if x >= mid]

    if not left_xs or not right_xs:
        return None

    folder_x1 = int(round(float(np.median(left_xs))))
    folder_x2 = int(round(float(np.median(right_xs))))
    return folder_x1, folder_x2


# ----------------------------------------------------------------
# Config I/O helpers
# ----------------------------------------------------------------


def _load_existing_config(config_path: Path) -> dict:
    """Return existing config dict (preserves 'size_thresholds' if present)."""
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_calibration(
    config_path: Path,
    folder_width_px: int,
    folder_x1: int | None = None,
    folder_x2: int | None = None,
    calib_image_width: int | None = None,
) -> dict:
    """
    Write calibration data to config.json.

    Preserves any existing 'size_thresholds' key so custom thresholds
    are not wiped by a re-calibration.

    Also computes and stores CM-based fields:
      - ``folder_width_cm`` : physical width of the reference object (cm).
      - ``scale_cm_per_px`` : conversion factor derived from
        ``FOLDER_WIDTH_CM / folder_width_px``.

    Parameters
    ----------
    calib_image_width : Width in pixels of the image used for calibration.
                        Saved so the inference pipeline can normalise clothing
                        widths captured at a different resolution.

    Returns the saved config dict.
    """
    existing = _load_existing_config(config_path)

    if folder_width_px <= 0:
        raise ValueError(
            f"folder_width_px must be positive, got {folder_width_px}. "
            "Check that two distinct vertical edges were detected."
        )

    scale_cm_per_px = round(FOLDER_WIDTH_CM / folder_width_px, 4)

    config: dict = {"folder_width_px": folder_width_px}
    if folder_x1 is not None:
        config["folder_x1"] = folder_x1
    if folder_x2 is not None:
        config["folder_x2"] = folder_x2
    if calib_image_width is not None and calib_image_width > 0:
        config["calib_image_width"] = calib_image_width
    config["folder_width_cm"] = FOLDER_WIDTH_CM
    config["scale_cm_per_px"] = scale_cm_per_px
    if "size_thresholds" in existing:
        config["size_thresholds"] = existing["size_thresholds"]

    # Preserve perspective calibration fields so a regular re-calibration
    # does not wipe a previously-saved perspective calibration.
    for _key in ("src_points", "dst_size", "has_perspective", "folder_height_cm"):
        if _key in existing:
            config[_key] = existing[_key]

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"[CALIB] scale_cm_per_px={scale_cm_per_px}  "
        f"(folder_width_cm={FOLDER_WIDTH_CM} / folder_width_px={folder_width_px})"
    )
    return config


# ----------------------------------------------------------------
# High-level calibration entry points
# ----------------------------------------------------------------


def calibrate_from_image(
    image_path: str,
    config_path: str = "config.json",
) -> dict:
    """
    Run auto-calibration from an image file and write config.json.

    Saves pixel-based fields (``folder_width_px``, ``folder_x1``,
    ``folder_x2``, ``calib_image_width``) as well as CM-based fields
    (``folder_width_cm``, ``scale_cm_per_px``).

    Parameters
    ----------
    image_path  : Path to a frame showing the empty folding board.
    config_path : Output path for the JSON config.

    Returns
    -------
    Saved config dict.

    Raises
    ------
    FileNotFoundError  : if image_path cannot be read.
    RuntimeError       : if edge detection fails.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    _, w = frame.shape[:2]
    print(f"[CALIB] Image loaded: {w} px wide")

    result = detect_folder_edges(frame)
    if result is None:
        raise RuntimeError(
            "Folder edges not detected. "
            "Ensure the folding board is clearly visible with good contrast. "
            "Re-run with --width <pixels> for manual override."
        )

    folder_x1, folder_x2 = result
    folder_width = folder_x2 - folder_x1

    print(
        f"[CALIB] Detected edges → x1={folder_x1}  x2={folder_x2}  "
        f"width={folder_width} px"
    )

    config = save_calibration(
        Path(config_path),
        folder_width,
        folder_x1,
        folder_x2,
        calib_image_width=w,
    )
    print(f"[CALIB] Config saved → {config_path}")
    return config


def calibrate_manual(
    folder_width_px: int,
    folder_x1: int | None = None,
    folder_x2: int | None = None,
    config_path: str = "config.json",
) -> dict:
    """
    Save a manually specified folder width (and optional edge positions)
    to config.json. No image required.

    Also saves CM-based fields (``folder_width_cm``, ``scale_cm_per_px``)
    derived from the global ``FOLDER_WIDTH_CM`` constant.

    Returns saved config dict.
    """
    config = save_calibration(Path(config_path), folder_width_px, folder_x1, folder_x2)
    print(f"[CALIB] Manual config saved → {config_path}  (width={folder_width_px} px)")
    return config


def calibrate_camera(
    camera_index: int = 0,
    config_path: str = "config.json",
) -> dict:
    """
    Interactive camera-based calibration: click left and right edges of the
    folding board on a live camera feed.

    Controls:
        Left-click — mark left/right edge (first click = left, second = right)
        S          — save calibration and exit
        R          — reset points and start over
        Q          — quit without saving

    Returns saved config dict on success.
    Raises RuntimeError if camera cannot be opened or user quits without saving.
    """
    WIN = "Kalibrasi — Klik tepi KIRI lalu tepi KANAN area referensi"
    points: list[int] = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append(x)
            print(f"[CALIB] Titik {len(points)}: x={x} px")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. "
            "Check --camera value and that the camera is connected."
        )

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_click)

    print(f"[CALIB] Camera opened. Klik tepi KIRI lalu tepi KANAN area referensi.")
    print(f"[CALIB] Tombol: S=simpan  R=reset  Q=keluar")

    saved_config = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw clicked lines
        for px in points:
            cv2.line(frame, (px, 0), (px, frame.shape[0]), (0, 255, 255), 2)

        if len(points) == 2:
            lebar_px = abs(points[1] - points[0])
            skala = FOLDER_WIDTH_CM / lebar_px
            cv2.putText(
                frame,
                f"Lebar: {lebar_px}px = {FOLDER_WIDTH_CM}cm | Skala: {skala:.4f} cm/px",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Tekan S untuk simpan, R untuk reset",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
        else:
            remaining = 2 - len(points)
            cv2.putText(
                frame,
                f"Klik {'tepi KIRI' if len(points) == 0 else 'tepi KANAN'} area referensi ({remaining} titik lagi)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and len(points) == 2:
            lebar_px = abs(points[1] - points[0])
            h, w = frame.shape[:2]
            saved_config = save_calibration(
                Path(config_path),
                folder_width_px=lebar_px,
                folder_x1=min(points),
                folder_x2=max(points),
                calib_image_width=w,
            )
            print(f"[CALIB] ✓ Kalibrasi disimpan → {config_path}")
            print(f"[CALIB]   {saved_config}")
            break
        elif key == ord("r"):
            points.clear()
            print("[CALIB] Reset — klik ulang tepi KIRI dan KANAN")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved_config is None:
        raise RuntimeError("Kalibrasi dibatalkan — tidak ada data yang disimpan.")
    return saved_config


def calibrate_perspective(
    camera_index: int = 0,
    config_path: str = "config.json",
) -> dict:
    """
    Kalibrasi perspektif 4 titik untuk ClothBot.

    Instruksi klik: TL → TR → BR → BL (ikuti urutan marker X biru di sudut pelipat).

    Kontrol:
        Klik kiri — tandai titik sudut (urutan TL→TR→BR→BL)
        S          — simpan kalibrasi dan keluar
        R          — reset, klik ulang dari awal
        Q          — keluar tanpa simpan

    Jika config.json sudah ada, field lama dipertahankan (extend, bukan replace).

    Returns saved config dict on success.
    Raises RuntimeError if camera cannot be opened or user quits without saving.
    """
    WIN_CAM = "Kalibrasi Perspektif — Klik 4 Marker (TL→TR→BR→BL)"
    WIN_WARP = "Preview Warp (Koreksi Perspektif)"

    CORNER_LABELS = ["TL", "TR", "BR", "BL"]
    # BGR colors per corner for visual feedback
    CORNER_COLORS = [
        (0, 0, 255),  # TL — merah
        (0, 128, 255),  # TR — oranye
        (0, 255, 0),  # BR — hijau
        (255, 0, 0),  # BL — biru
    ]

    dst_size = (PERSPECTIVE_DST_W, PERSPECTIVE_DST_H)
    dst_pts = np.float32(
        [
            [0, 0],
            [PERSPECTIVE_DST_W - 1, 0],
            [PERSPECTIVE_DST_W - 1, PERSPECTIVE_DST_H - 1],
            [0, PERSPECTIVE_DST_H - 1],
        ]
    )

    src_pts: list[list[int]] = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(src_pts) < 4:
            src_pts.append([x, y])
            label = CORNER_LABELS[len(src_pts) - 1]
            print(f"[CALIB] Titik {len(src_pts)} ({label}): ({x}, {y})")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Tidak dapat membuka kamera index {camera_index}. "
            "Periksa nilai --perspective dan pastikan kamera terhubung."
        )

    cv2.namedWindow(WIN_CAM)
    cv2.setMouseCallback(WIN_CAM, on_click)

    print("[CALIB] Klik 4 marker sudut pelipat secara berurutan: TL → TR → BR → BL")
    print("[CALIB] Tombol: S=simpan  R=reset  Q=keluar")

    saved_config: dict | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()

        # Draw clicked points + labels
        for i, (px, py) in enumerate(src_pts):
            cv2.circle(display, (px, py), 8, CORNER_COLORS[i], -1)
            cv2.circle(display, (px, py), 9, (255, 255, 255), 1)
            cv2.putText(
                display,
                CORNER_LABELS[i],
                (px + 12, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                CORNER_COLORS[i],
                2,
            )

        if len(src_pts) == 4:
            # Draw border polygon
            poly = np.array(src_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [poly], True, (0, 255, 255), 2)

            # Compute and show warp preview
            M = cv2.getPerspectiveTransform(np.float32(src_pts), dst_pts)
            warped = cv2.warpPerspective(frame, M, dst_size)
            cv2.imshow(WIN_WARP, warped)

            cv2.putText(
                display,
                "Tekan S untuk simpan, R untuk reset",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            remaining = 4 - len(src_pts)
            next_corner = CORNER_LABELS[len(src_pts)]
            cv2.putText(
                display,
                f"Klik marker {next_corner} ({remaining} titik lagi)",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        cv2.imshow(WIN_CAM, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and len(src_pts) == 4:
            h, w = frame.shape[:2]

            # scale in warped space is always FOLDER_WIDTH_CM / PERSPECTIVE_DST_W
            scale_warped = round(FOLDER_WIDTH_CM / PERSPECTIVE_DST_W, 4)

            # Load existing config to preserve old fields (e.g. folder_width_px)
            existing = _load_existing_config(Path(config_path))

            new_fields: dict = {
                "folder_width_cm": FOLDER_WIDTH_CM,
                "folder_height_cm": FOLDER_HEIGHT_CM,
                "scale_cm_per_px": scale_warped,
                "src_points": [list(map(int, p)) for p in src_pts],
                "dst_size": list(dst_size),
                "has_perspective": True,
            }
            merged = {**existing, **new_fields}
            Path(config_path).write_text(json.dumps(merged, indent=2), encoding="utf-8")
            saved_config = merged

            print(f"✓ Kalibrasi perspektif disimpan ke {config_path}")
            print(f"  scale_cm_per_px (warped) = {scale_warped}")
            print(f"  src_points = {src_pts}")
            break

        elif key == ord("r"):
            src_pts.clear()
            cv2.destroyWindow(WIN_WARP)
            print("[CALIB] Reset — klik ulang 4 marker dari awal")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved_config is None:
        raise RuntimeError(
            "Kalibrasi perspektif dibatalkan — tidak ada data yang disimpan."
        )
    return saved_config


# ----------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ClothBot — calibrate folding board (pelipat pakaian) reference width.\n"
            "Run once with the folder visible in frame, no clothing on top."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--image",
        metavar="PATH",
        help="Path to an image of the empty folder in frame (auto-detect).",
    )
    src.add_argument(
        "--width",
        type=int,
        metavar="PX",
        help="Manual folder width in pixels (skips auto-detection).",
    )
    src.add_argument(
        "--camera",
        type=int,
        nargs="?",
        const=0,
        metavar="INDEX",
        help="Interactive camera calibration (click mode). INDEX = camera device index (default: 0).",
    )
    src.add_argument(
        "--perspective",
        type=int,
        nargs="?",
        const=0,
        metavar="INDEX",
        help=(
            "Kalibrasi perspektif 4 titik dengan klik marker sudut (TL→TR→BR→BL). "
            "INDEX = indeks kamera (default: 0)."
        ),
    )

    parser.add_argument(
        "--x1",
        type=int,
        metavar="PX",
        default=None,
        help="(Optional, with --width) Left edge x-coordinate in pixels.",
    )
    parser.add_argument(
        "--x2",
        type=int,
        metavar="PX",
        default=None,
        help="(Optional, with --width) Right edge x-coordinate in pixels.",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        metavar="PATH",
        help="Output config file path (default: config.json).",
    )

    args = parser.parse_args()

    try:
        if args.width is not None:
            calibrate_manual(
                args.width,
                folder_x1=args.x1,
                folder_x2=args.x2,
                config_path=args.config,
            )
        elif args.camera is not None:
            calibrate_camera(camera_index=args.camera, config_path=args.config)
        elif args.perspective is not None:
            calibrate_perspective(
                camera_index=args.perspective, config_path=args.config
            )
        else:
            calibrate_from_image(args.image, config_path=args.config)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
